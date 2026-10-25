"""界面的无头验收（S1 视觉与导航、S2 会话管理）。

用 Streamlit 官方的 `AppTest` 跑真正的 app.py，不是模拟：
它会执行整个脚本、返回可交互的元素，所以"按钮点了会怎样"能被断言。

页面切换用环境变量 `APP_PAGE` 指定默认页——`AppTest.switch_page` 只认文件路径，
对函数式页面无效，所以每个页面各跑一次脚本。

注意两点：
- 比其余测试慢（每个用例要装配一次 Agent + 起一次 MCP 进程，约 10 秒）
- 会在真实的 data/sessions.db 里建临时会话，用例结束自己删掉
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import archive, config, history, settings  # noqa: E402

try:
    from streamlit.testing.v1 import AppTest

    HAS_APPTEST = True
except Exception:  # noqa: BLE001
    HAS_APPTEST = False

APP = str(ROOT / "app.py")


def build(page: str = "chat") -> "AppTest":
    os.environ["APP_PAGE"] = page
    try:
        return AppTest.from_file(APP, default_timeout=180).run()
    finally:
        os.environ.pop("APP_PAGE", None)


def click(at, label: str):
    return next(b for b in at.button if b.label == label).click().run()


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit.testing.v1")
class TestUi(unittest.TestCase):
    def setUp(self):
        # 界面默认落在 default 档案；别的用例若把它切走了，这里要拨回来，
        # 否则临时会话建在 default，而界面显示的是另一个档案，按钮根本找不到。
        archive.set_active(config.DEFAULT_ARCHIVE)
        self.db = config.sessions_db(config.DEFAULT_ARCHIVE)
        self.tid = history.new_thread_id()
        history.save_message(self.tid, "user", "临时会话·界面测试", db_path=self.db)
        self.before = len(history.list_sessions(self.db))

    def tearDown(self):
        history.delete_session(self.tid, db_path=self.db)
        os.environ.pop("APP_PAGE", None)

    def test_all_pages_render(self):
        at = build("chat")
        self.assertEqual(at.exception, [], msg=str(at.exception))
        self.assertEqual(len(at.chat_input), 1, "对话页必须有输入框")

        for page, expect in (("evidence", "dataframe"), ("ledger", "dataframe"),
                             ("briefs", "selectbox"), ("settings", "radio")):
            at = build(page)
            self.assertEqual(at.exception, [], msg=f"{page}: {at.exception}")
            self.assertGreaterEqual(len(getattr(at, expect)), 1, msg=f"{page} 页缺少 {expect}")
            self.assertEqual(len(at.chat_input), 0, f"{page} 页不该有输入框")

    def test_settings_never_prints_secret(self):
        """设置页可以显示 `sk-…a1b2`，但任何一个元素里都不能出现完整密钥。"""

        at = build("settings")
        self.assertEqual(at.exception, [], msg=str(at.exception))
        secret = settings.read_env().get("DEEPSEEK_API_KEY", "")
        if len(secret) < 10:
            self.skipTest("本机没有配置 DeepSeek 密钥")

        parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
        parts += [t.value for t in at.text_input] + [str(t.placeholder) for t in at.text_input]
        blob = "\n".join(str(p) for p in parts)
        self.assertNotIn(secret, blob, "设置页把完整密钥显示出来了")
        self.assertIn("…", blob, "应当显示脱敏后的密钥提示")

    def test_settings_has_no_chat_input(self):
        at = build("settings")
        self.assertEqual(len(at.chat_input), 0)

    def test_badge_shows_full_window(self):
        """档位徽章用自定义卡片，覆盖期必须是完整起止日期，不能被截断。"""
        at = build("chat")
        badge = next(m.value for m in at.markdown if "覆盖期" in m.value)
        self.assertRegex(badge, r"\d{4}-\d{2}-\d{2} → \d{4}-\d{2}-\d{2}")
        self.assertIn("模型", badge, "徽章要显示这一期是哪个模型跑的")

    def test_delete_needs_confirmation(self):
        at = build("chat")
        at.button(key=f"del-default-{self.tid}").click().run()
        # 只是打开弹窗，不能已删
        self.assertEqual(len(history.list_sessions(self.db)), self.before)
        self.assertIn("_pending", at.session_state)

        click(at, "取消")
        self.assertEqual(len(history.list_sessions(self.db)), self.before)
        self.assertNotIn("_pending", at.session_state)

        at.button(key=f"del-default-{self.tid}").click().run()
        click(at, "确认删除")
        self.assertEqual(len(history.list_sessions(self.db)), self.before - 1)
        self.assertNotIn("_pending", at.session_state)

    def test_rename(self):
        at = build("chat")
        at.button(key=f"rnb-default-{self.tid}").click().run()
        at.text_input[0].set_value("改过的标题").run()
        click(at, "保存")
        titles = [s["title"] for s in history.list_sessions(self.db) if s["thread_id"] == self.tid]
        self.assertEqual(titles, ["改过的标题"])
        self.assertEqual(at.exception, [])

    def test_create_archive_from_sidebar(self):
        """从侧栏建档 → 界面切到新档案 → 徽章显示新档案名。

        断言用"新增了哪一个"，不用"总数等于 2"：上一次跑崩在这条断言上时，
        真正的原因是上一轮残留了目录，而断言只说了数字不对，指向了错的地方。
        """

        import shutil as _sh

        if archive.count() >= archive.MAX_ARCHIVES:
            self.skipTest(f"档案已满 {archive.MAX_ARCHIVES} 个，先清理 runtime/archives/ 再跑这条")
        before = {a["id"] for a in archive.list_archives()}

        at = build("chat")
        at.button(key="arch-new").click().run()
        at.text_input[0].set_value("界面建的档案").run()
        click(at, "创建")

        added = {a["id"] for a in archive.list_archives()} - before
        self.assertEqual(len(added), 1, f"应新增 1 个档案，实际新增 {added}")
        made = added.pop()
        try:
            badge = next(m.value for m in at.markdown if "档案" in m.value and "覆盖期" not in m.value)
            self.assertIn("界面建的档案", badge)
            self.assertEqual(at.exception, [])
        finally:
            # 界面为新档案建过 AgentRuntime，checkpoints.db 开着 → 不关就删不掉
            rt = at.session_state.get("_rt")
            if rt is not None:
                rt.close()
            try:
                archive.delete(made)
            except RuntimeError:
                pass
            _sh.rmtree(config.archive_dir(made), ignore_errors=True)
            _sh.rmtree(config.briefs_dir(made), ignore_errors=True)
            _sh.rmtree(config.ledger_dir(made), ignore_errors=True)
            archive.set_active(config.DEFAULT_ARCHIVE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
