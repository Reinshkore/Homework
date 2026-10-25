"""证据库验收脚本（M3 的判据）。

不联网、不调模型，用一组人造数据把四条规则逐一压一遍：
  1. url 去重与标题去重各拦一类重复
  2. 窗口外的条目被拒，48 小时内的标为边界条目
  3. 技能排除的域名在入库层被拦
  4. 「一手」只认 官方域名 + 发布页 + 真的抓过正文

跑法：.venv/Scripts/python.exe scripts/probe_store.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from core import config, store  # noqa: E402

DB = config.RUNTIME_DIR / "probe_store.db"


def d(offset: int) -> str:
    return (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")


def main() -> int:
    if DB.exists():
        DB.unlink()
    for suffix in ("-wal", "-shm"):
        p = Path(str(DB) + suffix)
        if p.exists():
            p.unlink()

    end, start = d(0), d(-7)
    print(f"窗口：{start} .. {end}\n")

    items = [
        # 正常条目：官方发布页 + 抓过正文 → 应判为一手
        dict(title="OpenAI 发布新一代推理模型", url="https://openai.com/blog/new-reasoning-model",
             date=d(-2), subject="OpenAI", event_type="T1", site="OpenAI", fetched_chars=3200),
        # 正常条目：二手媒体
        dict(title="MiniMax M3.1 Flash 开源模型权重，总参数 428B",
             url="https://www.ithome.com/1/016/800.htm", date=d(-3),
             subject="MiniMax", event_type="T3", site="IT之家", fetched_chars=0),
        # url 重复（带跟踪参数）
        dict(title="OpenAI 发布新一代推理模型（转载）",
             url="https://openai.com/blog/new-reasoning-model?utm_source=weibo", date=d(-2),
             subject="OpenAI", event_type="T1", fetched_chars=0),
        # 标题重复（跨站转载）
        dict(title="MiniMax M3.1 Flash 开源模型权重，总参数 428B- DoNews",
             url="https://www.donews.com/news/detail/1/6751322.html", date=d(-3),
             subject="MiniMax", event_type="T3", site="DoNews", fetched_chars=0),
        # 窗口外（10 天前）
        dict(title="上个月的旧闻", url="https://example.com/old", date=d(-20),
             subject="某公司", event_type="T2", fetched_chars=0),
        # 边界条目（窗口起点前 1 天）
        dict(title="边界：窗口前一天的发布", url="https://example.com/boundary", date=d(-8),
             subject="某公司", event_type="T2", fetched_chars=0),
        # 技能排除域名
        dict(title="A Paper on Agents", url="https://arxiv.org/abs/2610.00001", date=d(-2),
             subject="某实验室", event_type="T2", fetched_chars=0),
        # 缺日期
        dict(title="没有日期的条目", url="https://example.com/nodate", date="",
             subject="某公司", event_type="T1"),
        # 事件类型不合法
        dict(title="类型写错", url="https://example.com/badtype", date=d(-1),
             subject="某公司", event_type="T9"),
        # 官方域名但只是 docs 页 → 不算一手
        dict(title="某模型 API 文档", url="https://platform.minimax.io/docs/guides/text-generation",
             date=d(-2), subject="MiniMax", event_type="T4", fetched_chars=2000),
    ]

    pid = store.ensure_period(DB, start, end, "学习者", "标准")
    res = store.add_evidence(DB, pid, items, start, end)

    print(f"period_id = {pid}")
    print(f"入库 {res['counts']['added']} ｜ 去重 {res['counts']['skipped']} ｜ 拒绝 {res['counts']['rejected']}\n")

    print("—— 被拒的理由 ——")
    for r in res["rejected"]:
        print(f"  - {r['title'][:40]:<42} {r['reason']}")
    print("\n—— 被判重复的 ——")
    for s in res["skipped_duplicate"]:
        print(f"  - {s['url'][:60]:<62} {s['reason']}")

    print("\n—— 入库结果 ——")
    for a in res["added"]:
        flag = "一手" if a["is_primary"] else "  "
        bnd = " 边界" if a["boundary"] else ""
        print(f"  #{a['id']:<3} {a['event_type']}  {flag}{bnd}  {a['title'][:44]}")

    stats = store.query_stats(DB, pid)
    print("\n—— stats ——")
    print(f"  total={stats['total']}  coverage={stats['coverage']}/6  primary={stats['primary']}"
          f"（{stats['primary_rate']}）  within_48h={stats['within_48h']}  boundary={stats['boundary']}")
    print(f"  by_type={ {k: v for k, v in stats['by_type'].items() if v} }")

    # ---- 断言 ----
    print("\n—— 判定 ——")
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append(ok)
        print(f"  [{'通过' if ok else '失败'}] {name}" + (f"  {detail}" if detail else ""))

    check("10 条里入 4 条（拒 4、重 2）", res["counts"]["added"] == 4, f"实际 {res['counts']['added']}")
    check("url 重复被拦", any("url 重复" in s["reason"] for s in res["skipped_duplicate"]))
    check("标题重复被拦", any("标题" in s["reason"] for s in res["skipped_duplicate"]))
    check("窗口外被拒", any("窗口" in r["reason"] for r in res["rejected"]))
    check("arXiv 被拒", any("排除" in r["reason"] for r in res["rejected"]))
    check("缺日期被拒", any("缺日期" in r["reason"] for r in res["rejected"]))
    check("非法事件类型被拒", any("T1–T6" in r["reason"] for r in res["rejected"]))
    check("边界条目被放行并标记", stats["boundary"] == 1, f"boundary={stats['boundary']}")

    rows = store.query_rows(DB, pid)
    primary_titles = [r["title"] for r in rows if r["is_primary"]]
    check("只有官方发布页算一手", primary_titles == ["OpenAI 发布新一代推理模型"], f"{primary_titles}")
    check("官方 docs 页不算一手",
          all(r["is_primary"] == 0 for r in rows if "/docs/" in r["url"]))
    check("48h 内条数正确", stats["within_48h"] >= 1, f"within_48h={stats['within_48h']}")

    # 簇合并 + 单信源簇数
    ids = [r["id"] for r in rows][:2]
    store.update_evidence(DB, ids, {"cluster_id": 1, "displacement": "能力", "radius": 4,
                                    "certainty": "A", "in_report": 1})
    store.update_evidence(DB, [r["id"] for r in rows][2:4], {"cluster_id": 2, "displacement": "成本",
                                                             "radius": 3, "certainty": "B",
                                                             "in_report": 1})
    clusters = store.query_clusters(DB, pid)
    check("簇聚合正确", len(clusters) == 2, f"{[c['cluster_id'] for c in clusters]}")
    check("stats 反映已判定数", store.query_stats(DB, pid)["judged"] == 4)

    print(f"\n{'全部通过' if all(checks) else '有失败项'}（{sum(checks)}/{len(checks)}）")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
