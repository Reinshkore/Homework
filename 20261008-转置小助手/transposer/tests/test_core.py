import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core.validate import validate_id_no, parse_date, validate_record, id_birthdate
from core.confidence import score_field
from core.mapping import normalize_label, apply_derived, load_templates


def _mkid(region="310101", birth="19900307", seq="001"):
    W = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    M = "10X98765432"
    body = f"{region}{birth}{seq}"
    s = sum(int(body[i]) * W[i] for i in range(17))
    return body + M[s % 11]


VALID_ID = _mkid()


# ---------- 证件号校验 ----------

def test_id_no_valid():
    assert validate_id_no(VALID_ID)


def test_id_no_case_insensitive_x():
    assert validate_id_no(VALID_ID[:-1] + "x" if VALID_ID[-1] == "X" else VALID_ID)


def test_id_no_wrong_checksum():
    wrong = VALID_ID[:-1] + ("0" if VALID_ID[-1] != "0" else "1")
    assert not validate_id_no(wrong)


def test_id_no_bad_length():
    assert not validate_id_no("123")
    assert not validate_id_no("X" * 18)


# ---------- 日期解析 ----------

def test_parse_date_formats():
    assert parse_date("2026-03-16").isoformat() == "2026-03-16"
    assert parse_date("2026/3/16").isoformat() == "2026-03-16"
    assert parse_date("2026年3月16日").isoformat() == "2026-03-16"
    assert parse_date("2026.03.16").isoformat() == "2026-03-16"


def test_parse_date_invalid():
    assert parse_date("2026/02/30") is None      # 2 月 30 日不存在
    assert parse_date("2026年13月5日") is None    # 13 月不存在
    assert parse_date("") is None


def test_id_birthdate():
    assert id_birthdate(VALID_ID).isoformat() == "1990-03-07"
    assert id_birthdate("abc") is None


# ---------- 记录级校验 ----------

def _base_values(**over):
    v = {"name": "张三", "id_no": VALID_ID, "position": "工程师", "region": "上海",
         "base": "8000", "move_type": "入职", "effective_date": "2026-04-01"}
    v.update(over)
    return v


def test_valid_record_no_issues():
    assert validate_record(0, _base_values()) == []


def test_required_missing():
    issues = validate_record(0, _base_values(position=None))
    assert any(i["rule"] == "required" and i["field"] == "position" for i in issues)


def test_enum_invalid():
    issues = validate_record(0, _base_values(move_type="调岗"))
    assert any(i["rule"] == "enum" for i in issues)


def test_bad_checksum():
    wrong = VALID_ID[:-1] + ("0" if VALID_ID[-1] != "0" else "1")
    issues = validate_record(0, _base_values(id_no=wrong))
    assert any(i["rule"] == "id_checksum" and i["level"] == "error" for i in issues)


def test_entry_needs_base_warning():
    issues = validate_record(0, _base_values(base=None))
    assert any(i["rule"] == "entry_needs_base" and i["level"] == "warning" for i in issues)


def test_underage_warning():
    young = _mkid(birth="20120506", seq="234")
    issues = validate_record(0, _base_values(id_no=young))
    assert any(i["rule"] == "employment_age" and i["level"] == "warning" for i in issues)


def test_base_out_of_range():
    issues = validate_record(0, _base_values(base="999999"))
    assert any(i["rule"] == "base_range" for i in issues)


def test_base_not_number():
    issues = validate_record(0, _base_values(base="八千"))
    assert any(i["rule"] == "base_number" and i["level"] == "error" for i in issues)


def test_date_window():
    issues = validate_record(0, _base_values(effective_date="2019-01-01"))
    assert any(i["rule"] == "date_window" for i in issues)


# ---------- 置信度 ----------

def test_score_full_green():
    score, level, _ = score_field("id_no", VALID_ID, True, "agree", "llm_live")
    assert score == 1.0 and level == "green"


def test_score_offline_capped_amber():
    score, level, _ = score_field("id_no", VALID_ID, True, None, "offline")
    assert score == 0.75 and level == "amber"


def test_score_bad_format_capped():
    # 格式不符：0.5 不给；规则命中 0.3；模型一致 0.2 → 0.5（配合校验层的 error 才定性为红）
    score, level, _ = score_field("id_no", VALID_ID[:-1] + "0", True, "agree", "llm_live")
    assert (score, level) == (0.5, "amber")


def test_score_missing_red():
    score, level, _ = score_field("name", None, False, None, "llm_live")
    assert score == 0.0 and level == "red"


def test_score_edited_green():
    score, level, reasons = score_field("id_no", "whatever", False, None, "offline", edited=True)
    assert score == 1.0 and level == "green" and reasons == ["人工确认"]


def test_score_llm_only_amber():
    score, level, _ = score_field("name", "张三", False, "only", "llm_live")
    assert score == 0.5 and level == "amber"


# ---------- 模板映射 ----------

def test_load_templates():
    ts = load_templates()
    assert set(ts) == {"CUST_A", "CUST_B", "CUST_C"}


def test_normalize_label_exact_and_fuzzy():
    t = load_templates()["CUST_A"]
    assert normalize_label("员工姓名", t) == "name"
    assert normalize_label("社保缴纳基数", t) == "base"
    assert normalize_label("（备注）", t) is None


def test_apply_derived():
    assert apply_derived({"effective_date": "2026-04-01"})["social_month"] == "2026-04"
    assert apply_derived({"effective_date": None})["social_month"] == ""
