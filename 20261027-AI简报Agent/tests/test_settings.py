"""模型清单与 .env 读写的离线测试（不联网）。

三条要守住的性质：
  1. 改 `.env` 不能冲掉注释与别的键——那是唯一记录"这个键为什么存在"的地方；
  2. 密钥回显只能是 `sk-…a1b2`，任何路径都不能把完整值送进页面；
  3. `reload_env` 之后，进程里的 `config` 常量要跟着变。不改就等于"保存成功但没生效"。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import config, settings  # noqa: E402

SAMPLE = """# 注释要保住
DASHSCOPE_API_KEY=sk-old-dashscope-0000
# 检索说明
SEARCH_PROVIDER=deepseek
SEARCH_QUOTA=24
"""


class TestSettings(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "runtime").mkdir(parents=True)
        self._saved = {k: getattr(config, k) for k in ("ROOT", "RUNTIME_DIR")}
        self._env = {k: os.environ.get(k)
                     for k in ("SEARCH_QUOTA", "SEARCH_PROVIDER", "LLM_MODEL")}
        config.ROOT = self.root
        config.RUNTIME_DIR = self.root / "runtime"
        (self.root / ".env").write_text(SAMPLE, encoding="utf-8")

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(config, key, value)
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        config.refresh()
        self.tmp.cleanup()

    # ---- 清单
    def test_model_registry_shape(self):
        self.assertIn(settings.DEFAULT_MODEL, settings.MODELS)
        for mid, spec in settings.MODELS.items():
            self.assertTrue(spec["base_url"].startswith("http"), mid)
            self.assertIn(spec["key_env"], settings.EDITABLE_ENV, mid)

    def test_unknown_model_rejected(self):
        with self.assertRaises(RuntimeError):
            settings.set_model("gpt-9-ultra")

    def test_set_and_read_model(self):
        settings.set_model("deepseek-flash")
        self.assertEqual(settings.current_model(), "deepseek-flash")
        settings.set_model("qwen3.8-max")
        self.assertEqual(settings.current_model(), "qwen3.8-max")

    def test_corrupt_settings_falls_back(self):
        """settings.json 写坏了不能让界面打不开：退回到 .env 的 LLM_MODEL 或缺省值。"""

        (self.root / "runtime" / "settings.json").write_text("{这不是 json", encoding="utf-8")
        self.assertIn(settings.current_model(), settings.MODELS)

    def test_legacy_llm_model_is_respected(self):
        """旧版只认 .env 的 LLM_MODEL，升级后要继续生效；但清单外的名字要挡掉。"""

        os.environ["LLM_MODEL"] = "qwen3.8-max"
        self.assertEqual(settings.current_model(), "qwen3.8-max")

        os.environ["LLM_MODEL"] = "gpt-4o"
        self.assertEqual(settings.current_model(), settings.DEFAULT_MODEL)

    def test_model_ready_reflects_key(self):
        os.environ.pop("DASHSCOPE_API_KEY", None)
        ok, why = settings.model_ready("qwen3.7-plus")
        self.assertFalse(ok)
        self.assertIn("DASHSCOPE_API_KEY", why)

    # ---- 脱敏
    def test_mask_never_leaks_full_value(self):
        secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
        masked = settings.mask(secret)
        self.assertNotIn("abcdefghij", masked)
        self.assertTrue(masked.startswith("sk-"))
        self.assertTrue(masked.endswith(secret[-4:]))

    def test_mask_short_and_empty(self):
        self.assertEqual(settings.mask(""), "未配置")
        self.assertIn("过短", settings.mask("sk-123"))

    # ---- .env 改写
    def test_update_env_preserves_comments_and_other_keys(self):
        changed = settings.update_env({"SEARCH_QUOTA": "30"})
        self.assertEqual(changed, ["SEARCH_QUOTA"])
        text = (self.root / ".env").read_text(encoding="utf-8")
        self.assertIn("# 注释要保住", text)
        self.assertIn("# 检索说明", text)
        self.assertIn("DASHSCOPE_API_KEY=sk-old-dashscope-0000", text)
        self.assertIn("SEARCH_QUOTA=30", text)
        self.assertEqual(settings.read_env()["SEARCH_QUOTA"], "30")

    def test_update_env_appends_new_key(self):
        settings.update_env({"BOCHA_API_KEY": "sk-bocha"})
        text = (self.root / ".env").read_text(encoding="utf-8")
        self.assertTrue(text.rstrip().endswith("BOCHA_API_KEY=sk-bocha"))

    def test_update_env_is_idempotent(self):
        settings.update_env({"SEARCH_QUOTA": "30"})
        second = settings.update_env({"SEARCH_QUOTA": "30"})
        self.assertEqual(second, [])

    def test_read_env_ignores_comments_and_blank(self):
        env = settings.read_env()
        self.assertEqual(set(env), {"DASHSCOPE_API_KEY", "SEARCH_PROVIDER", "SEARCH_QUOTA"})

    # ---- 生效
    def test_reload_env_updates_process(self):
        settings.update_env({"SEARCH_QUOTA": "9", "SEARCH_PROVIDER": "bocha"})
        settings.reload_env()
        self.assertEqual(config.SEARCH_QUOTA, 9)
        self.assertEqual(config.SEARCH_PROVIDER, "bocha")


if __name__ == "__main__":
    unittest.main(verbosity=2)
