"""证据库：把「候选事件」从上下文搬到表里。

为什么非搬不可，三个具体后果：

1. 上下文预算。标准档要 6–8 个簇，候选通常是它的 3–5 倍。每条带摘要就是一两万字，
   而技能还允许「扩窗 7→14→30」重跑一轮检索。
2. 规则不可执行。技能的「同一信源只算一票、单信源最多供 2 簇、复用簇要标注」
   是跨条目统计，模型在几十条里数不过来，而且数错了没人能发现。
3. 不可审计。答辩时问「这条证据哪来的」，不能答「模型说它搜到了」。

表结构刻意保持瘦：证据只存判定需要的字段，不存全文——全文在需要时用 fetch_page 取。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from core import config
from core.sources import classify, is_publish_page, title_key, url_norm

EVENT_TYPES = ("T1", "T2", "T3", "T4", "T5", "T6")
DISPLACEMENTS = ("能力", "成本", "可及性", "供应链", "约束", "none", "unsure")

SCHEMA = """
CREATE TABLE IF NOT EXISTS period (
    id          TEXT PRIMARY KEY,
    window_start TEXT NOT NULL,
    window_end   TEXT NOT NULL,
    audience    TEXT NOT NULL DEFAULT '',
    length      TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    period_id    TEXT NOT NULL,
    d_event      TEXT,
    d_public     TEXT NOT NULL,
    subject      TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    title        TEXT NOT NULL,
    title_norm   TEXT NOT NULL,
    url          TEXT NOT NULL,
    url_norm     TEXT NOT NULL,
    site         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    publish_page INTEGER NOT NULL DEFAULT 0,
    fetched_chars INTEGER NOT NULL DEFAULT 0,
    is_primary   INTEGER NOT NULL DEFAULT 0,
    snippet      TEXT NOT NULL DEFAULT '',
    boundary     INTEGER NOT NULL DEFAULT 0,
    cluster_id   INTEGER,
    displacement TEXT,
    radius       INTEGER,
    certainty    TEXT,
    form         TEXT,
    in_report    INTEGER NOT NULL DEFAULT 0,
    note         TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    UNIQUE(period_id, url_norm)
);

CREATE INDEX IF NOT EXISTS idx_ev_period ON evidence(period_id, event_type);
CREATE INDEX IF NOT EXISTS idx_ev_title  ON evidence(period_id, title_norm);
"""


@contextmanager
def _connect(db_path: Path):
    """必须是上下文管理器。

    `with sqlite3.connect(...) as conn` 只提交事务、**不关闭连接**——
    写库时看起来一切正常，但在 Windows 上会留下文件锁，
    外部进程（比如备份脚本、测试的临时目录清理）删不掉这个库。
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    # 老库补列：CREATE TABLE IF NOT EXISTS 不会给已存在的表加字段。
    # 记下模型是为了回答"这期是哪个模型跑的"——换过模型之后，光看时间分不出来。
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(period)")}
    if "model" not in cols:
        conn.execute("ALTER TABLE period ADD COLUMN model TEXT NOT NULL DEFAULT ''")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def period_id_of(window_start: str, window_end: str, audience: str = "", length: str = "") -> str:
    """一期的标识。同一天同档位重跑会落进同一个 period，这是有意的——
    重跑覆盖旧候选比产生两份互不相干的证据更不容易出错。"""

    return f"{window_start}..{window_end}|{audience or '-'}|{length or '-'}"


def ensure_period(db_path: Path, window_start: str, window_end: str,
                  audience: str = "", length: str = "", model: str = "") -> str:
    pid = period_id_of(window_start, window_end, audience, length)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO period(id, window_start, window_end, audience, length, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET model = CASE WHEN excluded.model != '' "
            "THEN excluded.model ELSE period.model END",
            (pid, window_start, window_end, audience, length, model, _now()),
        )
    return pid


# ------------------------------------------------------------------ 入库
def add_evidence(db_path: Path, period_id: str, items: list[dict],
                 window_start: str, window_end: str, reset: bool = False) -> dict:
    """批量入库。去重键是 url_norm 与 title_norm 两个：前者拦同一页面的不同链接，
    后者拦同一篇稿子被不同站点转载。

    窗口判定在入库时做，不信检索参数——实测 DeepSeek 有 5% 的结果会越出窗口。

    reset=True 时先清空该期已有证据。用途只有一个：同一个窗口同一个档位重跑时保持幂等。
    增量采集（边搜边入库）用默认值 False，否则第二次入库会把第一批擦掉。
    """

    added, skipped, rejected = [], [], []
    with _connect(db_path) as conn:
        if reset:
            conn.execute("DELETE FROM evidence WHERE period_id=?", (period_id,))
        for raw in items:
            title = (raw.get("title") or "").strip()
            url = (raw.get("url") or "").strip()
            date = (raw.get("date") or "").strip()[:10]
            subject = (raw.get("subject") or "").strip()
            etype = (raw.get("event_type") or "").strip().upper()

            if not (title and url and subject and etype):
                rejected.append({"title": title[:60], "reason": "缺 title / url / subject / event_type"})
                continue
            if etype not in EVENT_TYPES:
                rejected.append({"title": title[:60], "reason": f"event_type 必须是 T1–T6，收到 {etype}"})
                continue
            if not date:
                # 技能：日期不确定直接丢弃。让模型挑一个日期比丢掉更危险。
                rejected.append({"title": title[:60], "reason": "缺日期，按技能规则丢弃"})
                continue

            u_norm, t_norm = url_norm(url), title_key(title)
            kind = classify(url)
            if kind == "excluded":
                # 技能把 arXiv / Papers with Code 列为排除项。检索层已经滤过一遍，
                # 这里再拦一次：模型可能凭记忆手填条目，不能只靠上游。
                rejected.append({"title": title[:60], "reason": f"域名被技能排除（{kind}）"})
                continue
            if conn.execute(
                "SELECT id FROM evidence WHERE period_id=? AND url_norm=?", (period_id, u_norm)
            ).fetchone():
                skipped.append({"url": url, "reason": "url 重复"})
                continue
            if t_norm and conn.execute(
                "SELECT id, url FROM evidence WHERE period_id=? AND title_norm=?", (period_id, t_norm)
            ).fetchone():
                skipped.append({"url": url, "reason": "标题重复（跨站转载）"})
                continue

            boundary = 0
            if not (window_start <= date <= window_end):
                # 边界条目：发生在窗口前 48 小时内、但窗口内才可查证（技能 pipeline 六之二）
                delta = (datetime.strptime(window_start, "%Y-%m-%d")
                         - datetime.strptime(date, "%Y-%m-%d")).days
                if 0 < delta <= 2:
                    boundary = 1
                else:
                    rejected.append({"title": title[:60],
                                     "reason": f"日期 {date} 在窗口 {window_start}–{window_end} 之外"})
                    continue

            fetched = int(raw.get("fetched_chars") or 0)
            primary = int(kind == "primary" and is_publish_page(url) and fetched >= 200)

            cur = conn.execute(
                """INSERT INTO evidence(period_id, d_event, d_public, subject, event_type,
                        title, title_norm, url, url_norm, site, kind, publish_page,
                        fetched_chars, is_primary, snippet, boundary, displacement,
                        radius, certainty, note, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    period_id, (raw.get("d_event") or date)[:10], date, subject, etype,
                    title, t_norm, url, u_norm, (raw.get("site") or url_norm(url)).strip()[:40],
                    kind, int(is_publish_page(url)), fetched, primary,
                    (raw.get("snippet") or "").strip()[:240], boundary,
                    raw.get("displacement"), raw.get("radius"), raw.get("certainty"),
                    (raw.get("note") or "")[:200], _now(),
                ),
            )
            added.append({"id": cur.lastrowid, "title": title[:60], "event_type": etype,
                          "is_primary": bool(primary), "boundary": bool(boundary)})

    return {"period_id": period_id, "added": added, "skipped_duplicate": skipped,
            "rejected": rejected,
            "counts": {"added": len(added), "skipped": len(skipped), "rejected": len(rejected)}}


# ------------------------------------------------------------------ 查询
def query_rows(db_path: Path, period_id: str, event_type: str | None = None,
               in_report: int | None = None, cluster_id: int | None = None,
               limit: int = 60) -> list[dict]:
    sql = ("SELECT id, d_event, d_public, subject, event_type, title, site, kind,"
           " is_primary, boundary, cluster_id, displacement, radius, certainty, form,"
           " in_report, url FROM evidence WHERE period_id=?")
    args: list = [period_id]
    if event_type:
        sql += " AND event_type=?"
        args.append(event_type)
    if in_report is not None:
        sql += " AND in_report=?"
        args.append(int(in_report))
    if cluster_id is not None:
        sql += " AND cluster_id=?"
        args.append(int(cluster_id))
    sql += " ORDER BY d_public DESC, id ASC LIMIT ?"
    args.append(max(1, min(int(limit or 60), 200)))

    with _connect(db_path) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def query_stats(db_path: Path, period_id: str) -> dict:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM evidence WHERE period_id=?", (period_id,)).fetchall()
        period = conn.execute("SELECT * FROM period WHERE id=?", (period_id,)).fetchone()

    total = len(rows)
    by_type = {t: 0 for t in EVENT_TYPES}
    for r in rows:
        by_type[r["event_type"]] = by_type.get(r["event_type"], 0) + 1

    within_48h = 0
    if period:
        cutoff = datetime.strptime(period["window_end"], "%Y-%m-%d").toordinal() - 2
        within_48h = sum(1 for r in rows if datetime.strptime(r["d_public"], "%Y-%m-%d").toordinal() >= cutoff)

    sites = {r["site"] for r in rows}
    primary = sum(1 for r in rows if r["is_primary"])
    return {
        "period_id": period_id,
        "model": (period["model"] if period else "") or "",
        "window": {"start": period["window_start"], "end": period["window_end"]} if period else None,
        "total": total,
        "by_type": by_type,
        "coverage": sum(1 for v in by_type.values() if v),  # 类型覆盖 K/6
        "unique_sites": len(sites),
        "primary": primary,
        "primary_rate": round(primary / total, 2) if total else 0.0,
        "within_48h": within_48h,
        "boundary": sum(1 for r in rows if r["boundary"]),
        "echo": sum(1 for r in rows if r["form"] == "echo"),
        "judged": sum(1 for r in rows if r["displacement"]),
        "in_report": sum(1 for r in rows if r["in_report"]),
        "max_cluster": max([r["cluster_id"] or 0 for r in rows], default=0),
    }


def query_clusters(db_path: Path, period_id: str) -> list[dict]:
    """按簇聚合，顺带算出「单信源贡献了几簇」——技能的 2 簇上限靠它判。"""

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, cluster_id, event_type, site, d_public, title, form, in_report "
            "FROM evidence WHERE period_id=? AND cluster_id IS NOT NULL",
            (period_id,),
        ).fetchall()

    groups: dict[int, list] = {}
    for r in rows:
        groups.setdefault(r["cluster_id"], []).append(r)

    out = []
    for cid, items in sorted(groups.items()):
        sites = [i["site"] for i in items]
        out.append({
            "cluster_id": cid,
            "event_type": items[0]["event_type"],
            "first_date": min(i["d_public"] for i in items),
            "evidence_ids": [i["id"] for i in items],
            "source_count": len(set(sites)),
            "sites": sorted(set(sites)),
            "form": items[0]["form"] or "unknown",
            "in_report": any(i["in_report"] for i in items),
            "titles": [i["title"][:50] for i in items][:3],
        })
    return out


def site_cluster_counts(db_path: Path, period_id: str) -> dict:
    """每个信源贡献了几个簇。技能要求单信源 ≤2，超了要换源。"""

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT site, COUNT(DISTINCT cluster_id) AS n FROM evidence "
            "WHERE period_id=? AND cluster_id IS NOT NULL GROUP BY site ORDER BY n DESC",
            (period_id,),
        ).fetchall()
    return {r["site"]: r["n"] for r in rows}


# ------------------------------------------------------------------ 更新
ALLOWED_FIELDS = {"cluster_id", "displacement", "radius", "certainty", "form",
                  "in_report", "boundary", "note", "d_event", "subject", "event_type"}


def update_evidence(db_path: Path, ids: list[int], fields: dict) -> dict:
    bad = [k for k in fields if k not in ALLOWED_FIELDS]
    if bad:
        return {"ok": False, "error": f"不允许改动的字段：{bad}", "allowed": sorted(ALLOWED_FIELDS)}
    if not ids:
        return {"ok": False, "error": "ids 不能为空"}

    sets = ", ".join(f"{k}=?" for k in fields)
    args = list(fields.values())
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE evidence SET {sets} WHERE id IN ({','.join('?' * len(ids))})",
            (*args, *[int(i) for i in ids]),
        )
        changed = cur.rowcount
    return {"ok": True, "updated": changed, "fields": list(fields)}


def latest_period(db_path: Path) -> str | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT id FROM period ORDER BY created_at DESC LIMIT 1").fetchone()
    return row["id"] if row else None
