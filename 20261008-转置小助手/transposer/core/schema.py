"""标准报岗表字段字典与全局口径假设。"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

FIELDS = [
    {"key": "name", "label": "姓名", "required": True, "type": "string"},
    {"key": "id_no", "label": "证件号", "required": True, "type": "id_no"},
    {"key": "position", "label": "岗位", "required": True, "type": "string"},
    {"key": "region", "label": "地区", "required": False, "type": "string"},
    {"key": "base", "label": "社保基数", "required": False, "type": "number"},
    {"key": "move_type", "label": "入离职类型", "required": True, "type": "enum", "enum": ["入职", "离职"]},
    {"key": "effective_date", "label": "生效日期", "required": True, "type": "date"},
]

DERIVED_FIELDS = [
    {"key": "social_month", "label": "社保起缴月", "type": "month", "source": "effective_date"},
]

FIELD_BY_KEY = {f["key"]: f for f in FIELDS}
FIELD_LABELS = {f["key"]: f["label"] for f in FIELDS}
FIELD_KEYS = [f["key"] for f in FIELDS]

ASSUMPTIONS = json.loads((ROOT / "config" / "assumptions.json").read_text(encoding="utf-8"))
