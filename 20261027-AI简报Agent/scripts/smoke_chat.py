"""冒烟测试：不启动界面，直接跑一轮对话，把事件流打印出来。

用途是验收「骨架能跑」——不需要浏览器，也能看到工具是否真的被调用。

**它和界面走同一条执行路径**（`agent/runner.py`），所以"写了报告却没交付"的自动补救
在这里同样生效。如果这里没过，界面里也不会过——两边不会出现"界面能跑、脚本不能跑"。

用法：
    .venv/Scripts/python.exe scripts/smoke_chat.py
    .venv/Scripts/python.exe scripts/smoke_chat.py -q "今天几号？再算 (3.5-2)/3.5*100"
    .venv/Scripts/python.exe scripts/smoke_chat.py -a <档案> -q "..."      # 指定档案
    .venv/Scripts/python.exe scripts/smoke_chat.py --info                  # 只看装配信息
    .venv/Scripts/python.exe scripts/smoke_chat.py --no-retry              # 复现"跳步"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):  # Windows 终端默认 GBK
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from agent import runner  # noqa: E402
from agent.builder import build_agent, close_agent  # noqa: E402
from agent.events import stream_events  # noqa: E402
from core import archive, config, history  # noqa: E402
from tools.search import reset_quota  # noqa: E402


class InlineRuntime:
    """把 Run 接到当前事件循环上。

    Run 是为"界面在另一个线程"设计的，靠 runtime.submit 起协程；
    脚本里已经在事件循环内，直接 ensure_future 即可，不必再开线程。
    """

    def __init__(self, agent, archive_id: str):
        self.agent = agent
        self.archive = archive_id

    def submit(self, coro):
        return asyncio.ensure_future(coro)


def make_factory():
    """把事件打印出来，同时原样传下去。"""

    def factory(runtime, question, thread_id):
        async def gen():
            async for ev in stream_events(runtime.agent, question, thread_id):
                kind = ev["type"]
                if kind == "token":
                    print(ev["content"], end="", flush=True)
                elif kind == "thought":
                    print(f"\n[旁白] {ev['content'][:80]}", flush=True)
                elif kind == "tool_call":
                    indent = "  " * ev.get("depth", 0)
                    print(f"\n{indent}[工具] {ev['name']} "
                          f"{json.dumps(ev['args'], ensure_ascii=False)[:160]}")
                elif kind == "tool_result":
                    indent = "  " * ev.get("depth", 0)
                    print(f"{indent}  -> {len(ev.get('content') or '')} 字")
                elif kind == "error":
                    print(f"\n[错误] {ev['content']}")
                yield ev

        return gen()

    return factory


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", default="今天是几号？再算一下 (3.5-2)/3.5*100")
    ap.add_argument("-t", "--thread", default=None)
    ap.add_argument("-a", "--archive", default=None, help="档案 id，默认用当前档案")
    ap.add_argument("--info", action="store_true", help="只打印装配信息，不发请求")
    ap.add_argument("--no-retry", action="store_true", help="关掉交付补救，用于复现跳步")
    args = ap.parse_args()

    archive_id = args.archive or archive.active()
    reset_quota()

    agent, info = await build_agent(archive_id)
    print("装配信息：")
    print(json.dumps(info, ensure_ascii=False, indent=2))

    if args.info:
        await close_agent(agent)
        return 0

    thread_id = args.thread or str(uuid.uuid4())
    db = config.sessions_db(archive_id)
    history.save_message(thread_id, "user", args.question, db_path=db)
    print(f"\n档案：{archive_id}　会话：{thread_id}\n问题：{args.question}\n" + "-" * 70)

    run = runner.start(InlineRuntime(agent, archive_id), args.question, thread_id,
                       db, archive_id, stream_factory=make_factory(),
                       deliver_retry=not args.no_retry)

    while not run.done.is_set():
        await asyncio.sleep(0.2)

    await close_agent(agent)

    print("\n" + "-" * 70)
    calls = sum(1 for s in run.steps if s.get("type") == "tool_call")
    print(f"工具调用 {calls} 次 ｜ 正文 {len(run.answer)} 字")
    print(f"交付：{'已通过 publish_brief' if run.published else '未交付'}"
          + ("（已自动补救一次）" if run.retried else ""))
    if run.error:
        print(f"错误：{run.error}")
    if not run.published:
        print("提示：本期没有落盘。跑 `scripts/audit_archive.py` 看缺了什么。")
    print(f"会话已写入 {db.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
