"""转置流水线：解析 → 规则抽取 → LLM 抽取（live/cache/offline）→ 聚合合并 → 校验 → 评分。"""
import json
import pathlib

from .schema import FIELDS, FIELD_BY_KEY
from .mapping import get_template, apply_derived
from .parse import parse_body, parse_attachment, try_ocr
from .rules_extract import extract_rules, merge_fragments
from .extract import get_llm_records
from .confidence import score_field
from .validate import validate_record, clean

SAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "samples"
META = json.loads((SAMPLES_DIR / "meta.json").read_text(encoding="utf-8"))


def list_samples():
    return [{"sample_id": m["sample_id"], "template_id": m["template_id"], "desc": m["desc"]}
            for m in META]


def get_sample(sample_id):
    for m in META:
        if m["sample_id"] == sample_id:
            return m
    raise KeyError(f"样本不存在：{sample_id}")


def build_text(parsed):
    parts = []
    for b in parsed["body"]["blocks"]:
        parts.append(f"【邮件正文】\n{b['text']}")
    for att in parsed["attachments"]:
        for b in att.get("blocks", []):
            if b["type"] == "text":
                parts.append(f"【附件 {att['name']}】\n{b['text']}")
            else:
                rows = ["\t".join(b["header"])] + ["\t".join(r) for r in b["rows"]]
                parts.append(f"【附件 {att['name']}】\n" + "\n".join(rows))
    return "\n\n".join(parts)


def transpose(template_id, sample_id, force_llm=False):
    meta = get_sample(sample_id)
    sdir = SAMPLES_DIR / meta["dir"]
    parsed = {
        "body": parse_body(sdir / meta["files"]["body"]),
        "attachments": [parse_attachment(sdir / a) for a in meta["files"]["attachments"]],
    }

    # 降级判定：扫描件（无文字层）
    scanned_atts = [a for a in parsed["attachments"] if "scanned" in a.get("flags", [])]
    degradation = None
    if scanned_atts:
        got_ocr = [try_ocr(a) for a in scanned_atts]
        if not any(got_ocr):
            degradation = ("附件为扫描件（无文字层）：已走降级路径，扫描内容未自动抽取，"
                           "请人工处理或接入 OCR 引擎")

    template = get_template(template_id)

    # 1) 规则抽取 + 按人聚合
    fragments = extract_rules(parsed, template)
    rule_records, cross_conflicts = merge_fragments(fragments)

    # 2) LLM 抽取（三层降级）
    text = build_text(parsed)
    llm_records, mode = get_llm_records(sample_id, text, template, force=force_llm)

    # 3) 模型结果合并进规则记录（规则优先，模型补充，不一致记冲突）
    llm_conflicts = []
    if llm_records:
        norm_llm = []
        for r in llm_records:
            if isinstance(r, dict):
                norm_llm.append({k: clean(v) for k, v in r.items()
                                 if k in FIELD_BY_KEY and clean(v)})
        used = set()
        for rec in rule_records:
            rid = rec["fields"].get("id_no", {}).get("value")
            rname = rec["fields"].get("name", {}).get("value")
            match = None
            for j, lr in enumerate(norm_llm):
                if j in used:
                    continue
                if rid and clean(lr.get("id_no")) == rid:
                    match = j
                    break
                if rname and clean(lr.get("name")) == rname:
                    match = j
                    break
            if match is None:
                continue
            used.add(match)
            for fk, lv in norm_llm[match].items():
                existing = rec["fields"].get(fk)
                if existing is None:
                    rec["fields"][fk] = {"value": lv, "source": "模型", "raw": "",
                                         "rule_hit": False, "llm_state": "only"}
                elif existing["value"] == lv:
                    existing["llm_state"] = "agree"
                else:
                    existing["llm_state"] = "differ"
                    llm_conflicts.append({
                        "record": rec["key"], "field": fk,
                        "rule": "rule_llm_conflict", "level": "warning",
                        "message": f"规则抽取为「{existing['value']}」，模型抽取为「{lv}」，请人工确认",
                    })
        for j, lr in enumerate(norm_llm):
            if j in used:
                continue
            key = clean(lr.get("name")) or clean(lr.get("id_no")) or f"#llm{j}"
            rule_records.append({
                "key": key,
                "fields": {fk: {"value": lv, "source": "模型", "raw": "",
                                "rule_hit": False, "llm_state": "only"}
                           for fk, lv in lr.items()},
                "conflicts": [],
            })

    # 4) 校验 + 置信度评分
    records_out = []
    for idx, rec in enumerate(rule_records):
        values = {f["key"]: (clean(rec["fields"].get(f["key"], {}).get("value")) or None)
                  for f in FIELDS}
        issues = validate_record(idx, values)
        extra_issues = []
        for c in cross_conflicts:
            if c["record"] == rec["key"]:
                extra_issues.append({k: v for k, v in c.items() if k != "record"} | {"record": idx})
        for c in rec.get("conflicts", []):
            extra_issues.append({k: v for k, v in c.items() if k != "record"} | {"record": idx})
        for c in llm_conflicts:
            if c["record"] == rec["key"]:
                extra_issues.append({k: v for k, v in c.items() if k != "record"} | {"record": idx})
        issues.extend(extra_issues)

        fields_out = {}
        for f in FIELDS:
            fk = f["key"]
            fv = rec["fields"].get(fk, {})
            value = clean(fv.get("value")) or None
            score, level, reasons = score_field(
                fk, value, bool(fv.get("rule_hit")), fv.get("llm_state"), mode)
            fields_out[fk] = {
                "value": value, "source": fv.get("source", ""), "raw": fv.get("raw", ""),
                "rule_hit": bool(fv.get("rule_hit")), "llm_state": fv.get("llm_state"),
                "score": score, "level": level, "reasons": reasons, "edited": False,
            }
        if any(i["level"] == "error" for i in issues):
            status = "error"
        elif any(i["level"] == "warning" for i in issues):
            status = "warning"
        else:
            status = "ok"
        records_out.append({
            "key": rec["key"], "fields": fields_out, "values": values,
            "derived": apply_derived(values), "issues": issues,
            "extra_issues": extra_issues, "status": status,
        })

    return {
        "sample_id": sample_id, "template_id": template_id, "mode": mode,
        "degradation": degradation,
        "flags": ["scanned"] if scanned_atts else [],
        "raw_text": text, "records": records_out,
    }


def revalidate(records_in, mode="offline"):
    """人工修正后重新校验评分。

    records_in: [{key, fields: {fk: {value, source, raw, rule_hit, llm_state}},
                  edited: [fk...], extra_issues: [...]}]
    edited 字段置信度记为 1.0（人工确认）。
    """
    out = []
    for idx, r in enumerate(records_in):
        edited = set(r.get("edited") or [])
        fmeta = r.get("fields", {})
        values, fields_out = {}, {}
        for f in FIELDS:
            fk = f["key"]
            m = fmeta.get(fk) or {}
            value = clean(m.get("value")) or None
            values[fk] = value
            score, level, reasons = score_field(
                fk, value, bool(m.get("rule_hit")), m.get("llm_state"), mode,
                edited=fk in edited)
            fields_out[fk] = {
                "value": value, "source": m.get("source", ""), "raw": m.get("raw", ""),
                "rule_hit": bool(m.get("rule_hit")), "llm_state": m.get("llm_state"),
                "score": score, "level": level, "reasons": reasons,
                "edited": fk in edited,
            }
        issues = validate_record(idx, values)
        extra = []
        for e in r.get("extra_issues") or []:
            extra.append({k: v for k, v in e.items() if k != "record"} | {"record": idx})
        issues.extend(extra)
        if any(i["level"] == "error" for i in issues):
            status = "error"
        elif any(i["level"] == "warning" for i in issues):
            status = "warning"
        else:
            status = "ok"
        out.append({
            "key": r.get("key", ""), "fields": fields_out, "values": values,
            "derived": apply_derived(values), "issues": issues,
            "extra_issues": extra, "status": status,
        })
    return out
