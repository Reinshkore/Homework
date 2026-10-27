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
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import runner  # noqa: E402
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


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit.testing.v1")
class TestTraceFold(unittest.TestCase):
    """执行过程折叠栏：状态存得住、能被侧栏一键收起。

    这两条以前都做不到：
      - 折叠栏是无状态 expander，任何重跑都按 `expanded=` 初值重建，用户展开的
        会被打回去；
      - 运行中用 `st.status`，它的 `update()` 在不传 expanded 时会 ClearField，
        前端收到的语义是"折叠"——进度每 0.4 秒刷新一次，就等于每 0.4 秒收回一次。
    """

    def setUp(self):
        archive.set_active(config.DEFAULT_ARCHIVE)
        self.db = config.sessions_db(config.DEFAULT_ARCHIVE)
        self.tid = history.new_thread_id()
        history.save_message(self.tid, "user", "临时会话·折叠栏", db_path=self.db)
        history.save_message(
            self.tid, "assistant", "回答正文", db_path=self.db,
            steps=[{"type": "tool_call", "name": "web_search", "args": {"q": "x"}},
                   {"type": "tool_result", "name": "web_search", "chars": 12}],
        )
        msgs = history.load_messages(self.tid, self.db)
        self.mid = next(m["id"] for m in msgs if m["role"] == "assistant")
        self.key = f"_trace-msg-{self.mid}"

    def tearDown(self):
        history.delete_session(self.tid, db_path=self.db)
        os.environ.pop("APP_PAGE", None)

    def _open(self):
        at = build("chat")
        at.session_state["threads"] = {config.DEFAULT_ARCHIVE: self.tid}
        return at.run()

    def _folded(self, at) -> bool:
        box = next(e for e in at.expander if e.label == "执行过程")
        return bool(box.proto.expanded)

    def test_state_survives_rerun(self):
        at = self._open()
        self.assertFalse(self._folded(at), "默认应为折叠")

        at.session_state[self.key] = True
        at.run()
        self.assertTrue(self._folded(at), "重跑后应保持展开")

        at.run()
        self.assertTrue(self._folded(at), "再重跑一次仍应展开")

    def test_collapse_all_from_sidebar(self):
        at = self._open()
        at.session_state[self.key] = True
        at.run()
        self.assertTrue(self._folded(at), "先展开，才谈得上收起")

        click(at, "⏬ 全部折叠执行过程")
        self.assertFalse(self._folded(at), "侧栏一键折叠应把展开的栏收起")
        self.assertEqual(at.exception, [])


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit.testing.v1")
class TestChatSubmit(unittest.TestCase):
    """输入框提交这一支：落库 + 启动运行。用桩替代 runner.start，不真跑 Agent。"""

    def setUp(self):
        archive.set_active(config.DEFAULT_ARCHIVE)
        self.db = config.sessions_db(config.DEFAULT_ARCHIVE)
        self.tid = history.new_thread_id()
        history.save_message(self.tid, "user", "临时会话·提交", db_path=self.db)

    def tearDown(self):
        history.delete_session(self.tid, db_path=self.db)
        os.environ.pop("APP_PAGE", None)

    def test_submit_saves_message_and_starts_run(self):
        from unittest import mock

        at = build("chat")
        at.session_state["threads"] = {config.DEFAULT_ARCHIVE: self.tid}
        at.run()

        q = "做一份最近一周的 AI 行业简报"
        with mock.patch("agent.runner.start") as started:
            at.chat_input[0].set_value(q).run()

        self.assertEqual(at.exception, [], msg=str(at.exception))
        self.assertEqual(started.call_count, 1, "提交应只启动一次运行")

        args = started.call_args.args
        self.assertEqual(args[1], q, "第二个参数是提问原文")
        self.assertEqual(args[2], self.tid, "第三个参数是当前会话 thread_id")
        self.assertEqual(args[4], config.DEFAULT_ARCHIVE, "第五个参数是当前档案")

        last = history.load_messages(self.tid, self.db)[-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(last["content"], q, "提问要先落库，再启动运行")


class _FakeRun:
    """够用的假 Run：只实现界面读得到的那几个属性，不碰模型。

    用它是因为"运行中"这条路径没法用真跑验证——一期十几分钟、要花额度。
    但它是改动风险最大的地方：进度每 0.4 秒刷新一次，任何一次刷新把折叠栏
    收回都会复现"展开后快速回弹"。
    """

    def __init__(self, thread_id: str, db: Path, tools: int = 2):
        self.thread_id = thread_id
        self.db = db
        self.question = "假提问：跑一期"
        self.steps: list[dict] = []
        self.text = ""
        self.answer = ""
        self.error = ""
        self.started_at = time.monotonic()
        self.done = threading.Event()
        self.observed_running = False
        self._tools = tools
        self._started = False

    @property
    def running(self) -> bool:
        # 界面第一次问"还跑着吗"的时候才开始计时：装配 Agent 要几秒，
        # 若在构造时就起线程，等界面读到它时早就跑完了，测试会假通过。
        if not self._started:
            self._started = True
            threading.Thread(target=self._tick, daemon=True).start()
        return not self.done.is_set()

    def calls(self) -> int:
        # 界面在"未完成"状态下问过次数，就说明运行中的面板确实渲染过
        if not self.done.is_set():
            self.observed_running = True
        return sum(1 for s in self.steps if s["type"] == "tool_call")

    def _tick(self) -> None:
        for i in range(self._tools):
            time.sleep(0.5)
            self.steps.append({"type": "tool_call", "name": f"tool_{i}", "args": {"i": i}})
        self.answer = "假回答正文"
        # 真 Run 在这时候自己落库，界面不负责记录——假货也要照做，
        # 否则跑完 rerun 会走"历史消息"路径却读不到这条消息
        history.save_message(self.thread_id, "assistant", self.answer, self.steps,
                             db_path=self.db)
        self.done.set()


@unittest.skipUnless(HAS_APPTEST, "需要 streamlit.testing.v1")
class TestRunningPanel(unittest.TestCase):
    """运行中 → 跑完 → 落成历史消息，整条路径不抛异常、且收起来。"""

    def setUp(self):
        archive.set_active(config.DEFAULT_ARCHIVE)
        self.db = config.sessions_db(config.DEFAULT_ARCHIVE)
        self.tid = history.new_thread_id()
        history.save_message(self.tid, "user", "假提问：跑一期", db_path=self.db)

    def tearDown(self):
        runner.RUNS.pop(self.tid, None)
        history.delete_session(self.tid, db_path=self.db)
        os.environ.pop("APP_PAGE", None)

    def test_running_then_history(self):
        fake = _FakeRun(self.tid, self.db)
        runner.RUNS[self.tid] = fake

        at = AppTest.from_file(APP, default_timeout=180)
        at.session_state["threads"] = {config.DEFAULT_ARCHIVE: self.tid}
        at.run()

        self.assertEqual(at.exception, [], msg=str(at.exception))
        self.assertTrue(fake.observed_running,
                        "运行中的面板没被渲染过——这条测试就没验到东西")
        self.assertEqual(fake.calls(), 2, "两次工具调用都要进轨迹")

        # 跑完后走历史路径：执行过程应当存在且是折叠的
        box = next(e for e in at.expander if e.label == "执行过程")
        self.assertFalse(box.proto.expanded, "跑完应自动折叠，不需要用户滚回顶部")

        body = "\n".join(m.value for m in at.markdown)
        self.assertIn("假回答正文", body, "正文要落在历史消息里")

    def test_running_panel_has_no_status(self):
        """回归守卫：运行中的执行过程不能用 st.status。

        `StatusContainer.update()` 在 expanded=None 时会 ClearField("expanded")，
        前端收到的语义是"折叠"。进度每 0.4 秒刷新一次，就等于每 0.4 秒把用户
        刚展开的栏打回去——这是"展开后快速回弹"的根因，不是观感问题。
        """

        src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("st.status(", src,
                         "运行中的执行过程请用 trace_box：st.status.update() 会清掉 expanded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
