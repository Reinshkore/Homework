"""体检一个档案：这期的证据、判定、自检与产出到底怎么样。

用途是横向比模型：同一句提问、同一个窗口、不同模型，产出的差异要看同一组数字。
所以指标固定，不接受"我觉得这次写得更好"。

用法：
    .venv/Scripts/python.exe scripts/audit_archive.py --archive <id>
    .venv/Scripts/python.exe scripts/audit_archive.py --all
    .venv/Scripts/python.exe scripts/audit_archive.py --archive <id> --log runtime/verify_x.log
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from core import archive, config, lint, store  # noqa: E402


def _briefs(aid: str) -> list[Path]:
    d = config.briefs_dir(aid)
    if not d.exists():
        return []
    return sorted((p for p in d.glob("*.md") if not p.name.startswith("_")),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def _log_stats(path: Path) -> dict:
    """从 smoke_chat 的日志里取工具次数与耗时（日志里有时间戳吗？没有，按行数兜底）。"""

    if not path or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    calls = len(re.findall(r"\[工具\]", text))
    errors = len(re.findall(r"\[错误\]", text))
    return {"tool_calls": calls, "errors": errors}


def audit(aid: str, log_path: Path | None = None) -> dict:
    meta = archive.read_meta(aid)
    out: dict = {"archive": aid, "name": meta["name"], "created": meta["created_at"]}

    db = config.store_db(aid)
    pid = store.latest_period(db) if db.exists() else None
    out["period"] = pid or "-"
    if pid:
        s = store.query_stats(db, pid)
        clusters = store.query_clusters(db, pid)
        out.update({
            "model": s.get("model") or "未知",
            "evidence": s["total"],
            "by_type": dict(s["by_type"]),
            "coverage": f"{s['coverage']}/6",
            "primary": s["primary"],
            "primary_rate": f"{int(s['primary_rate'] * 100)}%",
            "judged": s["judged"],
            "in_report": s["in_report"],
            "clusters": len(clusters),
            "multi_source_clusters": sum(1 for c in clusters if c["source_count"] > 1),
        })

    briefs = _briefs(aid)
    if briefs:
        latest = briefs[0]
        text = latest.read_text(encoding="utf-8")
        out["brief"] = latest.name
        out["chars"] = len(re.findall(r"[\u4e00-\u9fff]", text))
        out["term_anchors"] = len(re.findall(r"^`[^`]{1,20}`：", text, re.M))
        out["event_blocks"] = len(re.findall(r"^\*\*发生了什么\*\*", text, re.M))
        tier = pid.split("|") if pid else []
        aud = tier[1] if len(tier) > 1 else "学习者"
        length = tier[2] if len(tier) > 2 else "标准"
        res = lint.lint(text, aud, length, 7, period_id=pid or "", db_path=db)
        out["lint_pass"] = res["pass"]
        out["lint_errors"] = res["error_count"]
        out["lint_warnings"] = res["warning_count"]
        out["lint_detail"] = [f"{v['rule']} {v['detail']}" for v in res["violations"]][:6]

    ledger = config.ledger_file(aid)
    if ledger.exists():
        out["ledger_entries"] = len(re.findall(r"^## ", ledger.read_text(encoding="utf-8"), re.M))

    if log_path:
        out.update(_log_stats(log_path))
    return out


def show(a: dict) -> None:
    print(f"档案 {a['archive']}（{a['name']}，建于 {a['created']}）")
    print(f"  期      {a.get('period', '-')}")
    if a.get("model"):
        print(f"  模型    {a['model']}")
    if a.get("evidence"):
        print(f"  证据    {a['evidence']} 条｜类型 {a.get('coverage')} {a.get('by_type')}")
        print(f"  判定    已判定 {a['judged']}｜入选 {a['in_report']}")
        print(f"  簇      {a['clusters']} 个（多信源 {a.get('multi_source_clusters', 0)} 个）")
        print(f"  一手    {a['primary']} 条 / {a['primary_rate']}")
    if a.get("brief"):
        print(f"  正文    {a['brief']}｜{a['chars']} 字｜术语锚点 {a['term_anchors']}｜事件 {a['event_blocks']}")
        print(f"  自检    {'通过' if a['lint_pass'] else '未通过'}（error {a['lint_errors']} / warning {a['lint_warnings']}）")
        for line in a.get("lint_detail", []):
            print(f"       · {line}")
    if a.get("ledger_entries") is not None:
        print(f"  账本    {a['ledger_entries']} 期")
    if a.get("tool_calls") is not None:
        print(f"  工具    {a['tool_calls']} 次（日志错误 {a['errors']} 条）")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--log", help="对应的 smoke_chat 日志，用来统计工具调用次数")
    args = ap.parse_args()

    ids = [args.archive] if args.archive else [m["id"] for m in archive.list_archives()]
    for aid in ids:
        if not archive.exists(aid):
            print(f"档案 {aid} 不存在")
            continue
        show(audit(aid, Path(args.log) if args.log else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
