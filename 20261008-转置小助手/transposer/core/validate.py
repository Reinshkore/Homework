"""校验规则库：全部为确定性纯函数，每条规则可单测。"""
import re
from datetime import datetime

from .schema import FIELDS, FIELD_BY_KEY, ASSUMPTIONS

ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
ID_CHECK_MAP = "10X98765432"

DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日", "%Y.%m.%d")


def clean(v):
    if v is None:
        return ""
    return str(v).strip()


def validate_id_no(v):
    """GB 11643 十八位证件号校验（ISO 7064 MOD 11-2）。"""
    v = clean(v).upper()
    if not re.fullmatch(r"\d{17}[0-9X]", v):
        return False
    s = sum(int(v[i]) * ID_WEIGHTS[i] for i in range(17))
    return ID_CHECK_MAP[s % 11] == v[17]


def parse_date(v):
    v = clean(v)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def id_birthdate(v):
    v = clean(v).upper()
    if not re.fullmatch(r"\d{17}[0-9X]", v):
        return None
    try:
        return datetime.strptime(v[6:14], "%Y%m%d").date()
    except ValueError:
        return None


def _issue(record, field, rule, level, message):
    return {"record": record, "field": field, "rule": rule, "level": level, "message": message}


def validate_record(index, values):
    """对一个员工记录跑全部校验规则，返回 Issue 列表。

    values: {field_key: str|None}，记录号 index 仅用于定位。
    """
    issues = []
    label_of = lambda k: FIELD_BY_KEY[k]["label"] if k in FIELD_BY_KEY else k

    # 规则1：必填非空
    for f in FIELDS:
        v = clean(values.get(f["key"]))
        if f["required"] and not v:
            issues.append(_issue(index, f["key"], "required", "error",
                                 f"{f['label']}为必填字段，原文未提供"))

    # 规则2：证件号校验位
    id_no = clean(values.get("id_no"))
    if id_no and not validate_id_no(id_no):
        issues.append(_issue(index, "id_no", "id_checksum", "error",
                             "证件号校验位不通过（GB 11643 / ISO 7064 MOD 11-2）"))

    # 规则3：生效日期可解析
    eff_raw = clean(values.get("effective_date"))
    eff = parse_date(eff_raw) if eff_raw else None
    if eff_raw and eff is None:
        issues.append(_issue(index, "effective_date", "date_format", "error",
                             f"生效日期无法解析：{eff_raw}"))

    # 规则4：枚举合法
    mt = clean(values.get("move_type"))
    if mt and mt not in FIELD_BY_KEY["move_type"]["enum"]:
        issues.append(_issue(index, "move_type", "enum", "error",
                             f"入离职类型超出枚举范围：{mt}（允许：入职 / 离职）"))

    # 规则5：社保基数为数字且落在假设区间
    base = clean(values.get("base"))
    if base:
        try:
            num = float(base)
            lo, hi = ASSUMPTIONS["base_range"]
            if not (lo <= num <= hi):
                issues.append(_issue(index, "base", "base_range", "warning",
                                     f"社保基数 {num} 超出假设区间 [{lo}, {hi}]，请确认口径"))
        except ValueError:
            issues.append(_issue(index, "base", "base_number", "error",
                                 f"社保基数不是数字：{base}"))

    # 规则6：入职记录应有社保基数
    if mt == "入职" and not base:
        issues.append(_issue(index, "base", "entry_needs_base", "warning",
                             "入职记录缺少社保基数，请与客户确认"))

    # 规则7：生效日期落在合理窗口（假设口径）
    if eff:
        win_lo = datetime.strptime(ASSUMPTIONS["effective_date_window"][0], "%Y-%m-%d").date()
        win_hi = datetime.strptime(ASSUMPTIONS["effective_date_window"][1], "%Y-%m-%d").date()
        if not (win_lo <= eff <= win_hi):
            issues.append(_issue(index, "effective_date", "date_window", "warning",
                                 f"生效日期 {eff.isoformat()} 超出假设窗口 [{win_lo}, {win_hi}]"))

    # 规则8：用工年龄（证件号出生日期 vs 生效日期）
    if eff and id_no:
        birth = id_birthdate(id_no)
        if birth:
            age = (eff - birth).days / 365.25
            if age < ASSUMPTIONS["min_employment_age"]:
                issues.append(_issue(index, "id_no", "employment_age", "warning",
                                     f"按证件号推算用工年龄约 {age:.1f} 岁，未满 "
                                     f"{ASSUMPTIONS['min_employment_age']} 周岁，请人工核实"))

    return issues
