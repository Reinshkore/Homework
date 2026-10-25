"""检索层的离线测试：不联网，只验纯函数与前置校验。

为什么单独测这些：它们一旦错了，错得无声。
去重错了 → 同一事件被算成两条独立证据，簇数虚高；
转义没清 → 反斜杠照抄进正文；
配额错了 → 要么模型无限重检，要么一期只能搜两次。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.search as search  # noqa: E402
from core import config  # noqa: E402


class TestSources(unittest.TestCase):
    def test_url_norm_strips_tracking_and_www(self):
        a = "https://www.ithome.com/1/016/800.htm?utm_source=x&wid=123#top"
        b = "https://ithome.com/1/016/800.htm?wid=123"
        self.assertEqual(search.url_norm(a), search.url_norm(b))

    def test_title_key_ignores_suffix_and_punctuation(self):
        from core.sources import title_key

        a = title_key("MiniMax M3.1 Flash 开源模型权重，总参数 428B")
        b = title_key("MiniMax M3.1 Flash 开源模型权重，总参数 428B- DoNews")
        self.assertEqual(a, b)

    def test_classify(self):
        self.assertEqual(search.classify("https://openai.com/blog/x"), "primary")
        self.assertEqual(search.classify("https://www.qbitai.com/2026/10/1.html"), "secondary")
        self.assertEqual(search.classify("https://arxiv.org/abs/1234"), "excluded")
        self.assertEqual(search.classify("https://random.example.com/a"), "other")

    def test_publish_page_rejects_docs(self):
        from core.sources import is_publish_page

        self.assertTrue(is_publish_page("https://minimax.io/blog/m3-1-flash"))
        self.assertFalse(is_publish_page("https://platform.minimax.io/docs/guides/text-generation"))
        self.assertFalse(is_publish_page("https://openai.com/index/"))  # 首页不算发布页


class TestDedupe(unittest.TestCase):
    def test_drops_cross_site_repost_and_excluded(self):
        hits = [
            search._hit("https://m.163.com/a/1", "KV Cache 砍掉99%，单卡吞吐飙6倍", "2026-10-21", "网易", "x"),
            search._hit("https://c.m.163.com/a/1", "KV Cache 砍掉99%，单卡吞吐飙6倍", "2026-10-21", "网易", "x"),
            search._hit("https://arxiv.org/abs/1", "Some Paper", "2026-10-21", "arXiv", "x"),
            search._hit("https://openai.com/blog/x", "New model", "2026-10-20", "OpenAI", "x"),
        ]
        out, dropped = search._dedupe(hits)
        self.assertEqual(len(out), 2)
        self.assertEqual(dropped, 2)
        self.assertNotIn("excluded", [h["kind"] for h in out])


class TestQuota(unittest.TestCase):
    def setUp(self):
        search.reset_quota()
        self._orig = config.SEARCH_QUOTA

    def tearDown(self):
        config.SEARCH_QUOTA = self._orig
        search.reset_quota()

    def test_blocks_after_limit(self):
        config.SEARCH_QUOTA = 0
        r = search.web_search("任意查询")
        self.assertEqual(r["hits"], [])
        self.assertIn("配额", r["error"])
        self.assertEqual(r["quota"]["remaining"], 0)


class TestValidation(unittest.TestCase):
    def test_bad_lang(self):
        r = search.web_search("x", lang="jp")
        self.assertIn("lang", r["error"])

    def test_bad_date(self):
        r = search.web_search("x", start="2026/10/17", end="2026-10-24")
        self.assertIn("YYYY-MM-DD", r["error"])

    def test_empty_query(self):
        self.assertIn("不能为空", search.web_search("   ")["error"])

    def test_english_never_degrades(self):
        """英文维度失败必须报错，不能悄悄换成中文结果。"""
        orig = config.SEARCH_PROVIDER
        config.SEARCH_PROVIDER = "mock"  # mock 的空快照必然失败
        try:
            r = search.web_search("inference chip", lang="en")
        finally:
            config.SEARCH_PROVIDER = orig
        self.assertFalse(r["degraded"])
        self.assertIn("不要用中文结果顶替", r["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
