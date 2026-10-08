"""置信度评分：确定性规则加权，不用 LLM 自报。

    格式匹配 0.5 + 规则命中 0.3 + 模型一致 0.2
    离线模式（无 LLM 可用）封顶 0.75，显式降级。
    """
from .schema import FIELD_BY_KEY
from .validate import validate_id_no, parse_date, clean

LEVELS = {"green": "正常", "amber": "待复核", "red": "强制人工"}


def format_ok(field_def, value):
    t = field_def["type"]
    v = clean(value)
    if not v:
        return False
    if t == "string":
        return True
    if t == "id_no":
        return validate_id_no(v)
    if t == "date":
        return parse_date(v) is not None
    if t == "enum":
        return v in field_def.get("enum", [])
    if t == "number":
        try:
            float(v)
            return True
        except ValueError:
            return False
    return True


def score_field(field_key, value, rule_hit, llm_state, mode, edited=False):
    """返回 (score, level, reasons)。llm_state: agree / differ / only / None。"""
    f = FIELD_BY_KEY[field_key]
    v = clean(value)

    if edited:
        return 1.0, "green", ["人工确认"]

    if not v:
        return 0.0, "red", ["字段缺失"]

    reasons = []
    score = 0.0
    if format_ok(f, v):
        score += 0.5
        reasons.append("格式匹配")
    else:
        reasons.append("格式不符")

    if rule_hit:
        score += 0.3
        reasons.append("规则命中")

    if llm_state == "agree":
        score += 0.2
        reasons.append("模型一致")
    elif llm_state == "differ":
        reasons.append("模型不一致")
    elif llm_state == "only":
        reasons.append("仅模型判断")

    if mode == "offline":
        score = min(score, 0.75)
        reasons.append("离线模式")

    level = "green" if score >= 0.8 else ("amber" if score >= 0.5 else "red")
    return round(score, 2), level, reasons
