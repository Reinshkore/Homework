"""简报的机械校验。

存在理由：`quality.md` 有 24 条自检，但「模型自查」在长上下文末尾必然漏。
实测过一次——首期报告 8 条事件只写了 1 个术语锚点、篇幅超配额 46%、
声称 8 个事件簇而库里是 15 个单条簇。这三条都有客观答案，代码判得比模型准。

分工：这里只判**形式**（条数、字数、字段、链接是否存在于库）。
判断内容好坏（术语是否切档、判断是否空洞）交给 critic 子代理——那需要语义判断。
"""

from __future__ import annotations

import re
from pathlib import Path

from core import store
from core.sources import url_norm

# 篇幅配额（output-spec.md 第一节）
QUOTA = {
    "精简": {"main": 1, "clusters": (3, 4), "chars": (600, 900), "inquiry": (2, 2)},
    "标准": {"main": 2, "clusters": (6, 8), "chars": (1200, 2000), "inquiry": (2, 3)},
    "深度": {"main": 3, "clusters": (10, 12), "chars": (2500, 4000), "inquiry": (3, 3)},
}

# 短窗（≤3 天）下调主线数与簇数（output-spec.md 一之一）
SHORT_WINDOW = {
    "精简": {"main": 1, "clusters": (3, 4)},
    "标准": {"main": 1, "clusters": (4, 5)},
    "深度": {"main": 2, "clusters": (6, 8)},
}

NAMED_RULES = {
    "L1": "头部档位行",
    "L2": "主线数",
    "L3": "全文字数",
    "L4": "事件条数",
    "L5": "术语锚点",
    "L6": "头部簇数与库一致",
    "L7": "主线四要素",
    "L8": "事实段措辞",
    "L9": "链接可溯源",
    "L10": "思考题结构",
    "L11": "头部簇数配额",
    "L12": "退路引用的条目存在",
}

_HEADER_AUD = re.compile(r"档位：\s*(非专业|学习者|业内)")
_HEADER_LEN = re.compile(r"档位：[^\n]*?×\s*(精简|标准|深度)")
_HEADER_COVER = re.compile(r"类型覆盖：\s*(\d+)\s*/\s*6")
_HEADER_CLUSTER = re.compile(r"事件簇：\s*(\d+)\s*个")
_MAINLINE_HEAD = re.compile(r"^###\s*主线\s*\d+\s*[—\-]", re.M)
_EVENT_HEAD = re.compile(r"^###\s*\d+\.\s", re.M)
_TERM_LINE = re.compile(r"^`[^`]{1,20}`：", re.M)
_LINK = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")
_CJK = re.compile(r"[\u4e00-\u9fff]")
# 「将」在中文里有两个意思：将要（未来）与「把」。只抓前者的固定搭配，
# 否则「DeepSeek 将框架开源」这种已完成的事实会被误判成预测。
_FORBIDDEN_FACT = re.compile(
    r"(即将|意味着|将会|将于|将在|将成为|将推出|将发布|将上线|将实现|将带来|将提升|将下降|将取代|将面临)"
)
_INQUIRY_TYPE = re.compile(r"\*\*(校验型|外推型|反证型)\*\*")

# 「本期思考」整节。退路判据只在这一节里查——正文别处出现的「第 N 条」不算。
_INQUIRY_SEC = re.compile(r"^#+\s*本期思考(.*?)(?=^#+\s|\Z)", re.S | re.M)
# 编号条目：`4. **非专业档**　<问题>`。标签是稳定标识，序号随档位变。
_INQUIRY_ITEM = re.compile(r"^\s*(\d+)\.\s*\*\*([^*\n]{1,12})\*\*", re.M)
_REF_BY_NUM = re.compile(r"见(?:下)?第\s*(\d+)\s*条")
_REF_BY_TAG = re.compile(r"见(?:下方)?[「『]?([\u4e00-\u9fff]{2,6}档)[」』]?(?:那一条|那条)")


def _quota(length: str, window_days: int) -> dict:
    base = dict(QUOTA.get(length) or QUOTA["标准"])
    if window_days <= 3:
        base.update(SHORT_WINDOW.get(length, {}))
    return base


def _blocks(text: str, head_re: re.Pattern) -> list[str]:
    """按标题切块，返回每个块的正文（不含标题行）。"""
    marks = [(m.start(), m.end()) for m in head_re.finditer(text)]
    out = []
    for i, (_, end) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append(text[end:stop])
    return out


def lint(markdown: str, audience: str, length: str, window_days: int,
         period_id: str = "", db_path: Path | None = None) -> dict:
    text = markdown or ""
    q = _quota(length, window_days)
    violations: list[dict] = []

    def bad(rule: str, detail: str, severity: str = "error") -> None:
        violations.append({"rule": rule, "name": NAMED_RULES.get(rule, rule),
                           "severity": severity, "detail": detail})

    # ---- L1 头部档位行 ----
    aud_m = _HEADER_AUD.search(text)
    len_m = _HEADER_LEN.search(text)
    if not aud_m or not len_m:
        bad("L1", "缺档位行。格式：档位：<受众> × <篇幅>（缺省注「未指定」）")
    else:
        if aud_m.group(1) != audience:
            bad("L1", f"头部受众写「{aud_m.group(1)}」，实际档位是「{audience}」")
        if len_m.group(1) != length:
            bad("L1", f"头部篇幅写「{len_m.group(1)}」，实际档位是「{length}」")

    # ---- L2 主线数 ----
    mains = _blocks(text, _MAINLINE_HEAD)
    if len(mains) != q["main"]:
        bad("L2", f"{length}档（{window_days} 天窗）应写 {q['main']} 条主线，实际 {len(mains)} 条")

    # ---- L3 全文字数 ----
    chars = len(_CJK.findall(text))
    lo, hi = q["chars"]
    if not lo <= chars <= hi:
        # 报差额，不报区间。实测模型会自己算减法，但算完只肯删二三百字再试一次——
        # 2026-10-27 那期从 5476 一路削到 4016，八次都差一点。直接把"还剩多少"给出来。
        if chars > hi:
            bad("L3", f"正文 {chars} 字，{length}档配额 {lo}–{hi} 字，"
                      f"超上限 {chars - hi} 字——一次删够，不要每次只删几十字再提交")
        else:
            bad("L3", f"正文 {chars} 字，{length}档配额 {lo}–{hi} 字，还差 {lo - chars} 字")

    # ---- L4 事件条数 ----
    events = _blocks(text, _EVENT_HEAD)
    clo, chi = q["clusters"]
    if not clo <= len(events) <= chi:
        bad("L4", f"事件明细 {len(events)} 条，{length}档（{window_days} 天窗）配额 {clo}–{chi} 条")

    # ---- L5 术语锚点（学习者档逐条要求）----
    if audience == "学习者":
        missing = [i + 1 for i, b in enumerate(events) if not _TERM_LINE.search(b)]
        if missing:
            bad("L5", f"学习者档每条事件需一句术语锚点，第 {missing} 条缺")

    # ---- L6 头部簇数与库一致 ----
    # 比的是「入选的簇数」，不是库里全部簇数。候选簇 11 个、只用了 7 个是正常的，
    # 头部该写 7——写 11 等于把没进正文的候选也算成本期事件。
    declared = _HEADER_CLUSTER.search(text)
    if db_path and period_id:
        clusters = store.query_clusters(db_path, period_id)
        real = len([c for c in clusters if c["in_report"]])
        if declared:
            if int(declared.group(1)) != real:
                bad("L6", f"头部写事件簇 {declared.group(1)} 个，库里入选的簇是 {real} 个"
                          f"（全部候选簇 {len(clusters)} 个）")
        elif real:
            bad("L6", "头部缺「事件簇：N 个」", "warning")
        if declared and int(declared.group(1)) == real:
            singles = [c for c in clusters if c["in_report"] and c["source_count"] == 1]
            if len(clusters) > 1 and real and len(singles) == real:
                bad("L6", f"{real} 个入选簇全部只有单一信源，合并很可能没做", "warning")

    # ---- L11 头部簇数在配额内 ----
    if declared:
        clo, chi = q["clusters"]
        if not clo <= int(declared.group(1)) <= chi:
            bad("L11", f"头部事件簇 {declared.group(1)} 个，{length}档（{window_days} 天窗）"
                       f"配额 {clo}–{chi} 个")

    # ---- L7 主线四要素 ----
    for i, b in enumerate(mains, 1):
        has_evidence = ("证据" in b) or ("依据" in b)
        has_refute = ("推翻条件" in b) or ("我可能错在" in b)
        has_assume = "依赖假设" in b
        missing = []
        if not has_evidence:
            missing.append("证据")
        if not has_refute:
            missing.append("推翻条件")
        if not has_assume and length != "精简":
            missing.append("依赖假设")
        if missing:
            bad("L7", f"主线 {i} 缺：{'、'.join(missing)}")

    # ---- L8 事实段措辞 ----
    fact_hits = []
    for i, b in enumerate(events, 1):
        m = re.search(r"\*\*发生了什么\*\*：(.+)", b)
        if m and _FORBIDDEN_FACT.search(m.group(1)):
            fact_hits.append(i)
    if fact_hits:
        bad("L8", f"第 {fact_hits} 条的事实段出现「将/即将/意味着」，事实段只写已发生")

    # ---- L9 链接可溯源 ----
    if db_path and period_id:
        known = {url_norm(r["url"]) for r in store.query_rows(db_path, period_id, limit=200)}
        unknown = []
        for link in _LINK.findall(text):
            if url_norm(link) not in known:
                unknown.append(link)
        if unknown:
            bad("L9", f"{len(unknown)} 个链接在证据库里查不到，疑似编造或未入库：{unknown[:3]}")

    # ---- L10 思考题结构 ----
    kinds = _INQUIRY_TYPE.findall(text)
    lo_i, hi_i = q["inquiry"]
    main_kinds = [k for k in kinds if k in ("校验型", "外推型", "反证型")]
    if not lo_i <= len(main_kinds) <= hi_i:
        bad("L10", f"主档思考题 {len(main_kinds)} 条，{length}档配额 {lo_i}–{hi_i} 条")
    if main_kinds.count("反证型") != 1:
        bad("L10", f"「反证型」应恰好 1 条，实际 {main_kinds.count('反证型')} 条")

    # ---- L12 退路引用的条目必须存在 ----
    # 规范允许「按卡住的原因选退路」，但退路指向的条目数随档位变：业内 5 条
    # （含学习者档、非专业档各 1 条），学习者只有 4 条。2026-10-27 有一期学习者档
    # 写「见第 5 条」，而那一期只有 4 条——原因不是模型乱编，是参考模板里写死的
    # 是业内档序号，照抄就指到空气上。这条完全有客观答案，不该靠模型自觉。
    sec = _INQUIRY_SEC.search(text)
    if sec:
        body = sec.group(1)
        items = {int(m.group(1)): m.group(2).strip() for m in _INQUIRY_ITEM.finditer(body)}
        for m in _REF_BY_NUM.finditer(body):
            n = int(m.group(1))
            if n not in items:
                bad("L12", f"退路写「见第 {n} 条」，但本期思考只有 {max(items) if items else 0} 条"
                           f"（编号：{sorted(items)}）——改用标签引用（如「见下方非专业档那条」）")
        labels = "、".join(items.values())
        for m in _REF_BY_TAG.finditer(body):
            tag = m.group(1)
            if tag not in labels:
                bad("L12", f"退路写「见{tag}那条」，但本期思考里没有这个标签的条目"
                           f"（现有：{labels or '无编号条目'}）")

    cover_m = _HEADER_COVER.search(text)
    measured = {
        "chars": chars,
        "mainlines": len(mains),
        "events": len(events),
        "inquiry": len(main_kinds),
        "coverage": int(cover_m.group(1)) if cover_m else None,
        "clusters_declared": int(declared.group(1)) if declared else None,
    }
    errors = [v for v in violations if v["severity"] == "error"]
    return {
        "pass": not errors,
        "quota": {"main": q["main"], "clusters": list(q["clusters"]), "chars": list(q["chars"])},
        "measured": measured,
        "violations": violations,
        "error_count": len(errors),
        "warning_count": len(violations) - len(errors),
        "next": ("全部通过" if not errors else
                 "按 violations 逐条修正后复检；最多两轮，仍有违规就标注「本期未通过自检：N 条」照常输出"),
    }
