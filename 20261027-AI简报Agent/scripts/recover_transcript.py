"""从检查点恢复缺失的会话记录。

什么时候需要它：一次运行还在进行时切了页面、关了浏览器、或者进程被杀，
界面那条"写会话记录"的语句就不会执行——但 Agent 跑在后台事件循环上，
往往已经把这一轮跑完了。于是数据在检查点里，界面上只剩一句提问。

检查点（runtime/archives/<档案>/checkpoints.db）由 LangGraph 写，存的是完整消息状态，
所以回答正文与工具调用序列都能取回来。这个脚本干的就是把它搬到会话库。

用法：
    .venv/Scripts/python.exe scripts/recover_transcript.py --list
    .venv/Scripts/python.exe scripts/recover_transcript.py --archive c8068f81 --thread f0aa1dc6
    .venv/Scripts/python.exe scripts/recover_transcript.py --all      # 扫描所有档案的孤儿提问
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from core import archive, config, history  # noqa: E402


def _orphans(archive_id: str) -> list[dict]:
    """会话库里只有 user 消息、没有 assistant 消息的会话。"""

    db = config.sessions_db(archive_id)
    if not db.exists():
        return []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT s.thread_id, s.title, s.updated_at, "
            "  SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END) AS u, "
            "  SUM(CASE WHEN m.role='assistant' THEN 1 ELSE 0 END) AS a "
            "FROM sessions s JOIN messages m ON m.thread_id = s.thread_id "
            "GROUP BY s.thread_id HAVING a = 0"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _read_messages(archive_id: str, thread_id: str) -> list:
    """从检查点里取这一轮的消息列表。

    两级来源，缺一不可：

    1. `checkpoints.channel_values["messages"]` —— 常规 LangGraph 图会把消息存在这里；
    2. `writes` 表里 `channel='messages'` 的行 —— **本项目实际走的是这一条**。
       deepagents 的 state 把 messages 标成了不随检查点落盘（`channel_values` 里只有
       todos / memory_contents / __pregel_tasks 这些），消息只以"每步写入"的形式留在
       writes 表。实测一次 10 分钟的运行有 82 条消息写入，顺序按 rowid 就是时间序。

    读的是临时副本：Saver 初始化会建表，直接开只读连接会失败；而改动原库
    不该是恢复脚本的副作用。
    """

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    src = config.checkpoints_db(archive_id)
    if not src.exists():
        return []

    serde = JsonPlusSerializer()
    with tempfile.TemporaryDirectory(prefix="cp-recover-") as tmp:
        copy = Path(tmp) / "cp.db"
        shutil.copy2(src, copy)
        for suffix in ("-wal", "-shm"):
            side = Path(str(src) + suffix)
            if side.exists():
                shutil.copy2(side, Path(str(copy) + suffix))
        conn = sqlite3.connect(copy, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            tup = SqliteSaver(conn).get_tuple({"configurable": {"thread_id": thread_id}})
            if tup:
                msgs = list(tup.checkpoint.get("channel_values", {}).get("messages", []))
                if msgs:
                    return msgs

            rows = conn.execute(
                "SELECT type, value FROM writes "
                "WHERE thread_id=? AND checkpoint_ns='' AND channel='messages' "
                "ORDER BY rowid",
                (thread_id,),
            ).fetchall()
        finally:
            conn.close()

    out: list = []
    seen: set = set()
    for row in rows:
        try:
            value = serde.loads_typed((row["type"], row["value"]))
        except Exception:  # noqa: BLE001  单条坏了不该让整次恢复失败
            continue
        for msg in (value if isinstance(value, list) else [value]):
            key = (type(msg).__name__, getattr(msg, "id", None), str(getattr(msg, "content", ""))[:120])
            if key in seen:
                continue
            seen.add(key)
            out.append(msg)
    return out


def _rebuild(messages: list) -> tuple[str, list[dict]]:
    """从消息列表还原（回答正文, 工具调用序列）。

    正文取最后一条 AI 消息的 text；工具序列按时间顺序把 tool_calls 与 ToolMessage 配对。
    还原出来的步骤只有"工具名 + 参数 + 返回字数"，没有实时那次的全量内容——
    这一点要如实反映，不要假装和原始轨迹一样。
    """

    from langchain_core.messages import AIMessage, ToolMessage

    steps: list[dict] = []
    answer = ""
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in (msg.tool_calls or []):
                steps.append({"type": "tool_call", "name": call.get("name"),
                              "args": call.get("args") or {}, "depth": 0})
            text = _text(msg.content)
            if text.strip():
                answer = text  # 只保留最后一段有内容的 AI 文本
        elif isinstance(msg, ToolMessage):
            steps.append({"type": "tool_result", "name": msg.name or "",
                          "chars": len(_text(msg.content)), "depth": 0})
    return answer.strip(), steps


def _text(content) -> str:
    """消息内容可能是 str，也可能是 content blocks 列表。"""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
        return "".join(out)
    return str(content or "")


def recover(archive_id: str, thread_id: str, dry_run: bool = False) -> dict:
    db = config.sessions_db(archive_id)
    existing = history.load_messages(thread_id, db)
    if any(m["role"] == "assistant" for m in existing):
        return {"ok": False, "reason": "该会话已有助手回复，未改动"}

    messages = _read_messages(archive_id, thread_id)
    if not messages:
        return {"ok": False, "reason": "检查点里没有这个 thread 的消息"}

    answer, steps = _rebuild(messages)
    if not answer:
        return {"ok": False, "reason": "检查点里没有可用的助手正文"}

    if not dry_run:
        history.save_message(thread_id, "assistant", answer, steps, db_path=db)
    return {"ok": True, "chars": len(answer), "steps": len(steps), "answer": answer}


def _resolve_thread(archive_id: str, prefix: str) -> str:
    """允许只给前几位：会话 id 是 uuid，肉眼抄全称容易错。"""

    db = config.sessions_db(archive_id)
    if not db.exists():
        return prefix
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
    try:
        rows = [r[0] for r in conn.execute("SELECT thread_id FROM sessions")]
    finally:
        conn.close()
    exact = [t for t in rows if t == prefix]
    if exact:
        return exact[0]
    hits = [t for t in rows if t.startswith(prefix)]
    return hits[0] if len(hits) == 1 else prefix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", help="档案 id；默认扫描全部")
    ap.add_argument("--thread", help="会话 id（可只给前缀）；不给则处理该档案所有孤儿会话")
    ap.add_argument("--all", action="store_true", help="扫描所有档案")
    ap.add_argument("--list", action="store_true", help="只列出，不写")
    ap.add_argument("--dry-run", action="store_true", help="只报告能恢复什么")
    args = ap.parse_args()

    targets: list[tuple[str, str]] = []
    archive_ids = [args.archive] if args.archive else (
        [a["id"] for a in archive.list_archives()] if args.all or args.list else [archive.active()]
    )

    for aid in archive_ids:
        if not aid:
            continue
        if args.thread:
            targets.append((aid, _resolve_thread(aid, args.thread)))
            continue
        for row in _orphans(aid):
            targets.append((aid, row["thread_id"]))

    if not targets:
        print("没有需要恢复的会话")
        return 0

    for aid, tid in targets:
        title = next((m["content"][:30] for m in history.load_messages(tid, config.sessions_db(aid))
                      if m["role"] == "user"), "")
        if args.list:
            print(f"  {aid}/{tid[:8]}  {title}")
            continue
        res = recover(aid, tid, dry_run=args.dry_run)
        tag = "可恢复" if args.dry_run else "已恢复"
        if res["ok"]:
            print(f"  {tag} {aid}/{tid[:8]}  正文 {res['chars']} 字、工具步骤 {res['steps']} 条  ← {title}")
        else:
            print(f"  跳过 {aid}/{tid[:8]}  {res['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
