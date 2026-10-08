"""规则抽取层（"规则优先、模型补充"中的规则侧）。

    由客户模板 JSON 驱动：
      - 表格块：表头经 aliases 归一成标准字段，每行一条记录片段
      - 文本块：按模板 text_patterns 逐块匹配，同一字段多值视为冲突
    片段按姓名（或证件号）聚合成记录 —— 对应"信息分散、人工聚合"的痛点。
    """
import re

from .mapping import normalize_label, normalize_value


def _chunk(text, template):
    strat = template.get("text_chunk", "blank")
    if strat == "marker":
        marker = template.get("chunk_marker", r"^【")
        chunks, cur = [], []
        for line in text.splitlines():
            if re.search(marker, line) and cur:
                chunks.append("\n".join(cur))
                cur = []
            cur.append(line)
        if cur:
            chunks.append("\n".join(cur))
        return [c for c in chunks if c.strip()]
    # 默认按空行分段
    parts = re.split(r"\n\s*\n", text)
    return [p for p in parts if p.strip()]


def _table_fragments(block, template, source):
    frags = []
    header = block.get("header") or []
    keys = [normalize_label(h, template) for h in header]
    for row in block.get("rows") or []:
        fields = {}
        for i, cell in enumerate(row):
            if i >= len(keys) or not keys[i]:
                continue
            val = str(cell or "").strip()
            if not val:
                continue
            fields[keys[i]] = {
                "value": normalize_value(keys[i], val),
                "raw": val, "source": source, "rule_hit": True,
            }
        if fields:
            frags.append({"fields": fields, "source": source, "conflicts": []})
    return frags


def _text_fragments(text, source, template):
    frags = []
    pats = template.get("text_patterns") or []
    if not pats:
        return frags
    for chunk in _chunk(text, template):
        fields = {}
        conflicts = []
        for p in pats:
            fkey = p["field"]
            if fkey in fields:
                continue
            hits = []
            for m in re.finditer(p["re"], chunk):
                try:
                    v = m.group("v")
                except (IndexError, ValueError):
                    v = m.group(0)
                hits.append((str(v).strip(), m.group(0)))
            distinct = []
            for v, raw in hits:
                if v not in [d[0] for d in distinct]:
                    distinct.append((v, raw))
            if not distinct:
                continue
            v, raw = distinct[0]
            fields[fkey] = {
                "value": normalize_value(fkey, v),
                "raw": raw, "source": source, "rule_hit": True,
            }
            if len(distinct) > 1:
                conflicts.append({
                    "field": fkey, "rule": "multi_value_conflict", "level": "error",
                    "message": f"同一段落内 {fkey} 出现多个取值：" + " / ".join(d[0] for d in distinct),
                })
        # 片段必须带有身份（姓名或证件号）才参与聚合
        if fields and ("name" in fields or "id_no" in fields):
            frags.append({"fields": fields, "source": source, "conflicts": conflicts})
    return frags


def extract_rules(parsed, template):
    """返回 (片段列表, 块内冲突列表)。跨源冲突在 merge 阶段产生。"""
    fragments = []
    for att in parsed["attachments"]:
        source = att.get("name", "附件")
        for block in att.get("blocks", []):
            if block["type"] == "table":
                fragments.extend(_table_fragments(block, template, source))
            else:
                fragments.extend(_text_fragments(block["text"], source, template))
    # 邮件正文也是文本来源
    for block in parsed["body"]["blocks"]:
        fragments.extend(_text_fragments(block["text"], parsed["body"]["name"], template))
    return fragments


def merge_fragments(fragments):
    """按姓名（其次证件号）聚合片段为记录；跨片段同字段取值不同记为冲突。"""
    order, merged, conflicts = [], {}, []
    for i, fr in enumerate(fragments):
        f = fr["fields"]
        key = (f.get("name") or {}).get("value") or (f.get("id_no") or {}).get("value")
        if not key:
            if len(f) >= 2:
                key = f"#frag{i}"
            else:
                continue
        if key not in merged:
            merged[key] = {"key": key, "fields": {}, "conflicts": []}
            order.append(key)
        rec = merged[key]
        for fk, fv in f.items():
            if fk in rec["fields"]:
                old = rec["fields"][fk]["value"]
                if old != fv["value"]:
                    conflicts.append({
                        "record": key, "field": fk, "rule": "cross_source_conflict",
                        "level": "error",
                        "message": f"多来源取值不一致：{old}（{rec['fields'][fk]['source']}） / "
                                   f"{fv['value']}（{fv['source']}），请人工确认",
                    })
            else:
                rec["fields"][fk] = fv
        for c in fr.get("conflicts", []):
            c["record"] = key
            rec["conflicts"].append(c)
    return [merged[k] for k in order], conflicts
