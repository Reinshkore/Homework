"""证据库 MCP 服务器（FastMCP，stdio）。

为什么放独立进程而不是进程内工具：
  1. 证据库需要独占写；检查点库（data/checkpoints.db）由 Agent 主进程持有，
     同进程写两个库容易在并发时抢锁。
  2. 证据库的物理路径在 /runtime/，Agent 的虚拟文件系统对该目录是禁读禁写的
     ——模型不能直接改库，只能通过这三个工具。这是「权限隔离」的实际落点。

启动方式由 mcp_servers.json 决定：python mcp/brief_store.py --archive <id>（cwd = 仓库根）。
`--archive` 由 agent/builder.py 在装配时追加 —— 档案是运行时才知道的，
所以这个进程的库路径、简报目录、账本路径全都跟着这个参数走，不读全局常量。
单独调试：python mcp/brief_store.py --archive default 会等待 stdin，用 MCP Inspector 更顺手。
"""

import sys
from datetime import datetime
from pathlib import Path

# 以 `python mcp/brief_store.py` 启动时 sys.path[0] 是 mcp/，导入不到 core
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from core import archive, config, lint, settings, store  # noqa: E402


def _archive_arg() -> str:
    """手工解析，不用 argparse：argparse 遇到未知参数会直接退出进程，
    而 MCP 客户端可能给这个进程传别的参数，那样会静默连不上。"""

    argv = sys.argv[1:]
    if "--archive" in argv:
        i = argv.index("--archive")
        if i + 1 < len(argv):
            return argv[i + 1].strip() or config.DEFAULT_ARCHIVE
    return config.DEFAULT_ARCHIVE


ARCHIVE = _archive_arg()
DB = config.store_db(ARCHIVE)
BRIEFS = config.briefs_dir(ARCHIVE)
LEDGER = config.ledger_file(ARCHIVE)

srv = FastMCP("brief-store", log_level="WARNING")


@srv.tool()
def evidence_add(items: list[dict], window_start: str, window_end: str,
                 audience: str = "", length: str = "", reset: bool = False) -> dict:
    """把检索到的候选事件写入证据库。**采集完立刻调用，不要把候选只留在对话里。**

    工具会替你做的事（不要自己做，也不要绕过）：
    - 按 url 与标题双重去重（同一篇稿子被不同站点转载只留一条）
    - 拒绝日期落在时间窗外的条目（48 小时内的标为边界条目放行）
    - 按域名与页面性质判定「一手」，不采信任何权威度评分

    Args:
        items: 候选事件列表，每项包含：
            title 标题（必填）
            url 原文链接（必填，http/https）
            date 发布日期 YYYY-MM-DD（必填，不确定就丢弃这条，不要猜）
            subject 主体，公司或机构名（必填）
            event_type 事件类型，T1–T6（必填）：
                T1 基础模型 / T2 能力解锁 / T3 成本与可及性
                T4 推理与部署 / T5 算力与硬件 / T6 治理与生态
            site 信源名，可留空（从域名推）
            snippet 摘要，最多 240 字
            fetched_chars 若你实际调过 fetch_page，填它返回的 chars，否则 0
                ——「一手」只认 fetch 过的官方发布页，没抓过就不算
            d_event 事件发生日，与 date 不同时才填
        window_start: 窗口起始日 YYYY-MM-DD
        window_end: 窗口结束日 YYYY-MM-DD
        audience: 受众档位，可选 非专业 / 学习者 / 业内
        length: 篇幅档位，可选 精简 / 标准 / 深度
        reset: 同一窗口同一档位重跑时传 true，先清空该期旧证据再写入。
            增量采集（边搜边入库）必须留 false，否则会擦掉前面几批。
    """

    pid = store.ensure_period(DB, window_start, window_end, audience, length,
                              model=settings.current_model())
    return store.add_evidence(DB, pid, items, window_start, window_end, reset=reset)


@srv.tool()
def evidence_query(period_id: str = "", view: str = "rows", event_type: str = "",
                   in_report: int = -1, cluster_id: int = -1, limit: int = 60) -> dict:
    """读证据库。判定阶段用它取候选表，不要凭记忆复述。

    Args:
        period_id: 留空则用最近一期
        view:
            rows     瘦表，一行一条（不含摘要）。判定位移与簇合并时用它
            stats    统计：类型覆盖 K/6、一手率、48h 内条数、边界条目数、已判定数
            clusters 按簇聚合，含每个簇的信源数。检查「单信源最多 2 簇」时用它
            sites    每个信源贡献了几个簇
        event_type: 只看某一类事件，T1–T6；留空为全部
        in_report: 1 只看已入选正文的，0 只看未入选的，-1 全部
        cluster_id: 只看某个簇
        limit: rows 最多返回条数，默认 60
    """

    pid = period_id or store.latest_period(DB)
    if not pid:
        return {"error": "还没有任何一期，先调用 evidence_add"}

    if view == "stats":
        return store.query_stats(DB, pid)
    if view == "clusters":
        return {"period_id": pid, "clusters": store.query_clusters(DB, pid)}
    if view == "sites":
        return {"period_id": pid, "sites": store.site_cluster_counts(DB, pid)}
    return {
        "period_id": pid,
        "rows": store.query_rows(
            DB, pid,
            event_type=event_type or None,
            in_report=None if in_report < 0 else in_report,
            cluster_id=cluster_id if cluster_id >= 0 else None,
            limit=limit,
        ),
    }


@srv.tool()
def evidence_update(ids: list[int], fields: dict) -> dict:
    """把判定结果写回证据库。判定与写回必须配套——只在对话里下结论，库里仍是空的。

    可改字段：
        displacement  五类位移之一：能力 / 成本 / 可及性 / 供应链 / 约束，或 none / unsure
        radius        影响半径 1–5
        certainty     确定性 A / B / C
        cluster_id    簇编号（同一事件的多条证据填同一个整数）
        form          fermenting（后续有新事实）/ echo（只有评论与转载）
        in_report     1 入选正文，0 未入选
        boundary      边界条目标记
        note          判定理由，一句话

    Args:
        ids: 要更新的证据 id 列表
        fields: 字段字典，例如 {"displacement": "成本", "radius": 4, "cluster_id": 1}
    """

    return store.update_evidence(DB, ids, fields)


@srv.tool()
def brief_lint(markdown: str, audience: str = "学习者", length: str = "标准",
               window_days: int = 7, period_id: str = "") -> dict:
    """单独跑格式校验，不落盘。一般不需要直接调——`publish_brief` 内部会先跑一遍。

    它查的是有客观答案的东西：条数、字数、字段是否齐、链接能不能在证据库里找到。
    术语是否切档、判断是否空洞这类需要语义判断的，交给 critic 子代理。

    Args:
        markdown: 简报正文全文（含头部档位行）
        audience: 实际采用的受众档位
        length: 实际采用的篇幅档位
        window_days: 时间窗天数
        period_id: 留空则用最近一期
    """

    pid = period_id or store.latest_period(DB) or ""
    return lint.lint(markdown, audience, length, window_days, period_id=pid, db_path=DB)


@srv.tool()
def publish_brief(markdown: str, ledger_entry: str, audience: str = "学习者",
                  length: str = "标准", window_days: int = 7,
                  period_id: str = "") -> dict:
    """**交付本期简报：自检 → 落盘 → 记账，一次完成。不调这个工具，本期就没有产出。**

    为什么做成一个工具而不是三步文件操作：实测过一次，模型写完分析就直接在对话里
    出报告，既没写文件也没跑自检，还在结尾声称「文件路径：/outputs/briefs/...」。
    没有返回值的步骤，模型会以为自己做过了。这个工具把三步绑成一个有明确成败的动作。

    流程：
      1. 先跑格式校验。不通过 → **不写任何文件**，返回 violations 让你修；最多改两轮。
      2. 通过 → 写简报正文，并把 ledger_entry 追加到账本。
      3. 返回 published=true 与文件路径。只有拿到 true，本期才算交付。

    写法与路径由工具决定，不要自己用 write_file 拼：宿主按档案分目录，
    简报落在 /outputs/briefs/<档案>/,账本落在 /docs/ledger/<档案>/。以返回的
    `path` 与 `ledger` 为准。

    Args:
        markdown: 简报正文全文，含头部档位行。不要带「以下是简报」这类前言。
        ledger_entry: 账本片段，按 `assets/report-skeleton.md` 第九节的格式写（主线判断、
            位移类型、依赖假设、推翻条件、悬置事项、检索缺口）。工具只负责追加，不改写。
        audience: 实际采用的受众档位
        length: 实际采用的篇幅档位
        window_days: 时间窗天数
        period_id: 留空则用最近一期
    """

    pid = period_id or store.latest_period(DB) or ""
    result = lint.lint(markdown, audience, length, window_days, period_id=pid, db_path=DB)
    if not result["pass"]:
        return {
            "published": False,
            "violations": result["violations"],
            "measured": result["measured"],
            "quota": result["quota"],
            "next": "按 violations 逐条修改 markdown 后再次调用本工具；最多两轮。"
                    "两轮后仍有 error，在正文尾部加一行「本期未通过自检：N 条」再提交。",
        }

    BRIEFS.mkdir(parents=True, exist_ok=True)
    filename = f"ai-brief-{audience}-{length}-{datetime.now():%Y-%m-%d}.md"
    path = BRIEFS / filename
    path.write_text(markdown, encoding="utf-8")

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    head = "# ai-brief 判断账本\n\n每期追加一段，不覆盖、不回改历史条目。\n\n"
    if not LEDGER.exists():
        LEDGER.write_text(head, encoding="utf-8")
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write("\n" + ledger_entry.strip() + "\n")

    return {
        "published": True,
        "archive": {"id": ARCHIVE, "name": archive.name_of(ARCHIVE)},
        "path": f"/outputs/briefs/{ARCHIVE}/{filename}",
        "bytes": len(markdown.encode("utf-8")),
        "chars": result["measured"]["chars"],
        "violations": result["violations"],
        "ledger": config.ledger_virtual_path(ARCHIVE),
        "next": "本期已交付。回复用户：一句话摘要 + 档位 + 文件路径，不要把全文再贴一遍。",
    }


if __name__ == "__main__":
    srv.run(transport="stdio")
