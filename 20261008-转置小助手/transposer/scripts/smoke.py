"""端到端冒烟：跑全部样本，核对记录数与预期问题项是否命中。

    运行：python scripts/smoke.py
    """
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pipeline import transpose

# 每个样本的预期：记录数 + 应命中的规则名集合
EXPECT = {
    "A-01": {"records": 3, "rules": set()},
    "A-02": {"records": 3, "rules": {"id_checksum", "date_format", "required", "enum"}},
    "B-01": {"records": 2, "rules": set()},
    "B-02": {"records": 2, "rules": {"date_format", "entry_needs_base"}},
    "C-01": {"records": 2, "rules": set()},
    "C-02": {"records": 2, "rules": {"cross_source_conflict", "employment_age", "entry_needs_base"}},
    "S-01": {"records": 2, "rules": {"required"}, "degradation": True},
}


def main():
    all_ok = True
    for sid, exp in EXPECT.items():
        result = transpose(_tpl(sid), sid)
        recs = result["records"]
        rules = {i["rule"] for r in recs for i in r["issues"]}
        n = len(recs)
        ok_rec = n == exp["records"]
        ok_rules = exp["rules"] <= rules
        ok_deg = (result["degradation"] is not None) == bool(exp.get("degradation", False))
        # 正常样本不应有任何 error
        ok_clean = not exp["rules"] and not any(
            i["level"] == "error" for r in recs for i in r["issues"]) if not exp["rules"] else True
        status = "PASS" if (ok_rec and ok_rules and ok_deg and ok_clean) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"[{status}] {sid}  mode={result['mode']}  records={n}/{exp['records']}  rules={sorted(rules)}")
        for r in recs:
            name = r["values"].get("name")
            print(f"    - {name}  status={r['status']}  "
                  f"conf=" + ",".join(f"{k}:{r['fields'][k]['score']}" for k in ("name", "id_no", "effective_date")))
            for i in r["issues"]:
                print(f"        [{i['level']}] {i['rule']}: {i['message']}")
        if result["degradation"]:
            print(f"    ! 降级：{result['degradation']}")
    print("\n全部通过" if all_ok else "\n存在未命中项，见上")
    sys.exit(0 if all_ok else 1)


def _tpl(sid):
    from core.pipeline import list_samples
    for s in list_samples():
        if s["sample_id"] == sid:
            return s["template_id"]
    raise KeyError(sid)


if __name__ == "__main__":
    main()
