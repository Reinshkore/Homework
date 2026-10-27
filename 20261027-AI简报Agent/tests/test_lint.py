"""brief_lint 的离线测试。

重点在两条「反例」：`将` 作「把」不能误报；簇数已经对不上时不再补一条同义警告。
误报多了，这套检查会被当成噪音跳过——那它等于不存在。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import lint, store  # noqa: E402

URL = "https://example.com/a"


def make_report(mains: int = 2, events: int = 6, pad: int = 750, clusters: int = 6) -> str:
    """造一份结构及格的学习者×标准档报告。pad 是补白字数，用来压字数规则。"""
    p = [
        "# AI 行业动向研判 · 2026-10-24",
        "",
        f"档位：学习者 × 标准（未指定）　覆盖期：2026-10-17 — 2026-10-24"
        f"　事件簇：{clusters} 个　位移事件：{mains} 个　类型覆盖：1/6",
        "",
    ]
    for i in range(1, mains + 1):
        p += [
            f"### 主线 {i} — 这是个可被证伪的判断陈述第{i}条",
            "",
            f"- **证据**：事件{i}（量子位，10-20）",
            "- **依赖假设**：企业愿意为成本迁移承担运维风险",
            "- **推翻条件**：闭源方案在关键场景保持显著质量优势",
            "",
        ]
    for i in range(1, events + 1):
        p += [
            f"### {i}. 某个事件标题第{i}条 [L3]",
            "",
            f"2026-10-20｜量子位｜[原文](https://example.com/{i})",
            "",
            "**发生了什么**：某主体把内部推理框架全面开源。",
            "",
            "**改动在哪一层**：推理与部署层。",
            "",
            "**为什么能做到**：支持专家并行与推测解码。",
            "",
            "**还差什么**：目前仅支持单一模型。",
            "",
            "`推理框架`：管理模型如何使用算力的中间层软件。",
            "",
        ]
    p += [
        "## 本期思考",
        "",
        "1. **校验型** 这条判断有多少支撑来自厂商口径？",
        "2. **反证型** 找一个反例：有没有实验室在设置限制后撤回？",
        "",
        "> 若答不上来，见下第 3 条。",
        "",
        "3. **非专业档** 便宜的是调用费，贵的是时间，哪个更值钱？",
    ]
    text = "\n".join(p)
    if pad:
        text += "\n\n" + "填" * pad
    return text


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"
        self.pid = store.ensure_period(self.db, "2026-10-17", "2026-10-24", "学习者", "标准")
        store.add_evidence(
            self.db, self.pid,
            [
                dict(title=f"事件{ i }标题", url=f"https://example.com/{i}", date="2026-10-20",
                     subject="某公司", event_type="T1")
                for i in range(1, 7)
            ],
            "2026-10-17", "2026-10-24",
        )
        for i, row in enumerate(store.query_rows(self.db, self.pid), 1):
            store.update_evidence(self.db, [row["id"]], {"cluster_id": i, "in_report": 1})

    def tearDown(self):
        self.tmp.cleanup()

    def run_lint(self, md: str):
        return lint.lint(md, "学习者", "标准", 7, period_id=self.pid, db_path=self.db)

    def rules(self, md: str) -> list[str]:
        return [v["rule"] for v in self.run_lint(md)["violations"]]


class TestClean(Base):
    def test_clean_passes(self):
        r = self.run_lint(make_report())
        self.assertTrue(r["pass"], msg=str(r["violations"]))
        self.assertEqual(r["measured"]["mainlines"], 2)
        self.assertEqual(r["measured"]["events"], 6)


class TestEachRule(Base):
    def test_L2_mainline_count(self):
        self.assertIn("L2", self.rules(make_report(mains=1)))

    def test_L3_too_long(self):
        self.assertIn("L3", self.rules(make_report(pad=1800)))

    def test_L3_too_short(self):
        self.assertIn("L3", self.rules(make_report(pad=0)))

    def test_L4_event_count(self):
        self.assertIn("L4", self.rules(make_report(events=3)))

    def test_L5_missing_term(self):
        md = make_report().replace("`推理框架`：管理模型如何使用算力的中间层软件。", "")
        self.assertIn("L5", self.rules(md))

    def test_L6_cluster_mismatch(self):
        self.assertIn("L6", self.rules(make_report(clusters=7)))

    def test_L11_cluster_quota(self):
        self.assertIn("L11", self.rules(make_report(clusters=12)))

    def test_L7_missing_refute(self):
        md = make_report().replace("- **推翻条件**：闭源方案在关键场景保持显著质量优势", "")
        self.assertIn("L7", self.rules(md))

    def test_L8_future_tense(self):
        md = make_report().replace("把内部推理框架全面开源", "即将开源推理框架")
        self.assertIn("L8", self.rules(md))

    def test_L9_fake_url(self):
        md = make_report().replace("https://example.com/1)", "https://example.com/made-up)")
        self.assertIn("L9", self.rules(md))

    def test_L10_refute_count(self):
        md = make_report().replace("2. **反证型**", "2. **校验型**")
        self.assertIn("L10", self.rules(md))

    def test_L12_dangling_number(self):
        """退路写「见第 5 条」，但本期只有 3 条——2026-10-27 那期的形态。"""

        md = make_report().replace("见下第 3 条", "见下第 5 条")
        self.assertIn("L12", self.rules(md))

    def test_L12_dangling_label(self):
        """用标签引用也一样查：指向本档没有的那一档，同样是空指针。"""

        md = make_report().replace("见下第 3 条", "见下方「学习者档」那条")
        self.assertIn("L12", self.rules(md))


class TestFalsePositives(Base):
    def test_l12_quiet_when_target_exists(self):
        """引用存在的条目不该报——误报多了这套检查会被当噪音跳过。"""

        self.assertNotIn("L12", self.rules(make_report()))
        md = make_report().replace("见下第 3 条", "见下方「非专业档」那条")
        self.assertNotIn("L12", self.rules(md), "标签存在却报错，是误报")

    def test_will_as_ba(self):
        """「将 A 开源」是将=把，已完成的事实，不能当成未来时。"""
        self.assertNotIn("L8", self.rules(make_report()))

    def test_no_duplicate_cluster_verdict(self):
        """簇数已经对不上时，不再补一条同义的"全是单条簇"警告。"""
        l6 = [v for v in self.run_lint(make_report(clusters=8))["violations"] if v["rule"] == "L6"]
        self.assertEqual(len(l6), 1, msg=str(l6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
