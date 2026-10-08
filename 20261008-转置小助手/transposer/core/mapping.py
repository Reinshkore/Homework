"""客户模板加载与字段映射引擎：换客户只改 JSON 配置，不改代码。"""
import json
import pathlib
import re

from .validate import parse_date

TEMPLATES_DIR = pathlib.Path(__file__).resolve().parent.parent / "templates"


def load_templates():
    out = {}
    for p in sorted(TEMPLATES_DIR.glob("*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        out[t["customer_id"]] = t
    return out


def get_template(template_id):
    return load_templates()[template_id]


def normalize_label(label, template):
    """把客户模板里的列名/标签归一成标准字段 key，找不到返回 None。"""
    l = re.sub(r"[\s　]", "", str(label or "")).lower()
    if not l:
        return None
    aliases = template.get("aliases", {})
    # 先精确匹配
    for key, cands in aliases.items():
        for a in cands:
            if re.sub(r"[\s　]", "", a).lower() == l:
                return key
    # 再包含匹配
    for key, cands in aliases.items():
        for a in cands:
            a2 = re.sub(r"[\s　]", "", a).lower()
            if a2 and a2 in l:
                return key
    return None


def normalize_value(key, value):
    """按字段类型做值归一：日期转 ISO、证件号大写、基数取数字。"""
    v = str(value or "").strip()
    if key == "id_no":
        return v.upper()
    if key == "effective_date":
        d = parse_date(v)
        return d.isoformat() if d else v
    if key == "base":
        m = re.search(r"\d+(?:\.\d+)?", v)
        return m.group(0) if m else v
    return v


def apply_derived(values):
    """派生字段：社保起缴月 = 生效日期所在月。"""
    d = parse_date(values.get("effective_date") or "")
    return {"social_month": d.strftime("%Y-%m") if d else ""}
