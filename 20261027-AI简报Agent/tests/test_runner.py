"""运行器的离线测试。

要复现的那次故障：运行还在进行时切了页面，Streamlit 中断脚本，页面里那句
"写助手回复"永远不执行 —— 运行照常完成、简报照常落盘，会话库里却只剩提问。

所以这里的核心断言只有一条：**没人消费这个 Run，助手回复也必须落库**。
用假运行时 + 假事件流，不联网、不起 Agent。
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import runner  # noqa: E402
from core import history  # noqa: E402


class FakeRuntime:
    """只提供 Run 需要的两样东西：archive 与 submit。"""

    def __init__(self, archive: str = "default"):
        self.archive = archive

    def submit(self, coro):
        def go():
            try:
                asyncio.run(coro)
            except Exception:  # noqa: BLE001
                pass

        thread = threading.Thread(target=go, daemon=True)
        thread.start()
        return thread


def stream_of(events: list[dict], gate: threading.Event | None = None):
    async def gen(_runtime, _question, _thread):
        for ev in events:
            if gate is not None:
                while not gate.is_set():
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0)
            yield ev

    return gen


def stream_raising(message: str = "boom"):
    async def gen(_runtime, _question, _thread):
        yield {"type": "tool_call", "name": "web_search", "args": {}, "depth": 0}
        raise RuntimeError(message)

    return gen


EVENTS = [
    {"type": "tool_call", "name": "web_search", "args": {"query": "x"}, "depth": 0},
    {"type": "tool_result", "name": "web_search", "chars": 120, "depth": 0},
    {"type": "thought", "content": "我先看看……"},
    {"type": "token", "content": "正在写"},
    {"type": "token", "content": "报告"},
    {"type": "answer", "content": "最终回答：两件事。"},
]


def stream_scripted(rounds: list[list[dict]]):
    """按轮次喂事件：第 1 次调用给 rounds[0]，第 2 次给 rounds[1]……"""

    state = {"n": 0}

    def factory(_runtime, _question, _thread):
        async def gen():
            events = rounds[min(state["n"], len(rounds) - 1)]
            state["n"] += 1
            for ev in events:
                await asyncio.sleep(0)
                yield ev

        return gen()

    return factory, state


PUBLISH_OK = {"type": "tool_result", "name": "publish_brief",
              "content": "{'published': True, 'path': '/outputs/briefs/x/a.md'}", "chars": 80}


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sessions.db"
        self.tid = history.new_thread_id()
        runner.RUNS.clear()

    def tearDown(self):
        runner.RUNS.clear()
        self.tmp.cleanup()

    def _wait(self, run, timeout: float = 10.0) -> None:
        self.assertTrue(run.done.wait(timeout), "运行没有在预期时间内完成")

    def test_assistant_saved_even_when_nobody_consumes(self):
        """核心用例：界面撒手不管，记录也在。"""

        run = runner.start(FakeRuntime(), "做一份简报", self.tid, self.db, "default",
                           stream_factory=stream_of(EVENTS))
        # 故意不看 run.steps、不读 run.text —— 模拟页面被切走
        self._wait(run)

        msgs = history.load_messages(self.tid, self.db)
        self.assertEqual([m["role"] for m in msgs], ["assistant"])
        self.assertEqual(msgs[0]["content"], "最终回答：两件事。")

    def test_steps_recorded(self):
        run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default",
                           stream_factory=stream_of(EVENTS))
        self._wait(run)
        kinds = [s["type"] for s in history.load_messages(self.tid, self.db)[0]["steps"]]
        self.assertEqual(kinds, ["tool_call", "tool_result"])
        self.assertEqual(run.calls(), 1)

    def test_thought_not_in_body(self):
        run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default",
                           stream_factory=stream_of(EVENTS))
        self._wait(run)
        body = history.load_messages(self.tid, self.db)[0]["content"]
        self.assertNotIn("我先看看", body)

    def test_only_tokens_still_saved(self):
        """模型只吐 token、没给 answer 事件时，用累积的 token 当正文。"""

        events = [{"type": "token", "content": "增量"}, {"type": "token", "content": "正文"}]
        run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default",
                           stream_factory=stream_of(events))
        self._wait(run)
        self.assertEqual(history.load_messages(self.tid, self.db)[0]["content"], "增量正文")

    def test_error_still_writes_a_message(self):
        """报错也要落一条。空白会让人以为记录又丢了，报错本身是信息。"""

        run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default",
                           stream_factory=stream_raising("connection reset"))
        self._wait(run)
        msgs = history.load_messages(self.tid, self.db)
        self.assertEqual(len(msgs), 1)
        self.assertIn("运行异常中止", msgs[0]["content"])
        self.assertIn("connection reset", msgs[0]["content"])

    def test_error_not_swallowed_by_partial_text(self):
        """有正文也必须写异常。

        2026-10-27 的终止就是这个丢法：`body = answer or text` 拿到的是过程旁白，
        "有正文就不写错误"那条分支把 GraphRecursionError 吃掉了。用户看到的是
        话说到一半停住，既没有产物，也没有原因——比报错更糟。
        """

        def factory(_runtime, _question, _thread):
            async def gen():
                yield {"type": "token", "content": "我先读一下技能定义，再规划后续步骤。"}
                yield {"type": "tool_call", "name": "publish_brief", "args": {}, "depth": 0}
                raise RuntimeError("Recursion limit of 300 reached")

            return gen()

        run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default",
                           stream_factory=factory)
        self._wait(run)
        body = history.load_messages(self.tid, self.db)[0]["content"]
        self.assertIn("我先读一下技能定义", body, "旁白不该被丢掉")
        self.assertIn("步数用尽", body, "异常必须出现在正文里，且说人话")
        self.assertIn("Recursion limit", body, "原生信息也要留，便于追")

    def test_abort_after_repeated_publish_failures(self):
        """交付反复不过就止损，并把最后一版存成草稿。

        实测形态：8 次 publish_brief，字数 5476 → 4016（上限 4000），第 9 次撞步数上限，
        正文只活在上下文里。止损的意义是"至少留下能接着改的东西"。

        12 次失败必须放在**同一轮**里：`_work` 最多跑两轮（正轮 + 一次补救），
        按轮次喂事件根本走不到 12 次——第一版测试就是这么写错的。
        """

        fail = ("{'published': False, 'violations': "
                "[{'rule': 'L3', 'name': '全文字数', 'severity': 'error', "
                "'detail': '正文 4016 字，深度档配额 2500–4000 字'}], 'measured': {}}")
        events: list[dict] = []
        for i in range(runner.PUBLISH_MAX_TRIES + 3):
            events.append({"type": "tool_call", "name": "publish_brief", "depth": 0,
                           "args": {"markdown": f"# 草稿 {i}\n\n正文"}})
            events.append({"type": "tool_result", "name": "publish_brief",
                           "content": fail, "chars": 500})

        with mock.patch("agent.runner.config.briefs_dir",
                        return_value=Path(self.tmp.name) / "briefs"):
            run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default",
                               stream_factory=stream_of(events))
            self._wait(run)

        self.assertTrue(run.aborted, "应在达到次数上限时止损")
        self.assertFalse(run.published)
        self.assertEqual(run.publish_tries, runner.PUBLISH_MAX_TRIES,
                         f"止损后不该再记次数，上限是 {runner.PUBLISH_MAX_TRIES}")

        body = history.load_messages(self.tid, self.db)[0]["content"]
        self.assertIn("都没通过自检", body)
        self.assertIn("4016", body, "要告诉人差在哪")

        drafts = list((Path(self.tmp.name) / "briefs").glob("_未交付-*.md"))
        self.assertEqual(len(drafts), 1, f"应恰好留一份草稿，实际 {drafts}")
        self.assertIn(f"# 草稿 {runner.PUBLISH_MAX_TRIES - 1}",
                      drafts[0].read_text(encoding="utf-8"),
                      "存的应是最后一次提交的那版")

    def test_busy_until_finished(self):
        gate = threading.Event()
        run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default",
                           stream_factory=stream_of(EVENTS, gate=gate))
        self.assertTrue(runner.busy())
        self.assertTrue(runner.busy("default"))
        self.assertFalse(runner.busy("other"))
        gate.set()
        self._wait(run)
        self.assertFalse(runner.busy())

# ---- 交付补救
    def test_retry_when_report_written_but_not_delivered(self):
        """qwen3.7-plus 实测过的形态：写完 3784 字报告，就是不调 publish_brief。"""

        long_text = [{"type": "token", "content": "报告正文" * 120}]  # 480 字
        rounds = [
            [{"type": "tool_call", "name": "web_search", "args": {}, "depth": 0}, *long_text],
            [{"type": "tool_call", "name": "publish_brief", "args": {}, "depth": 0},
             PUBLISH_OK, {"type": "answer", "content": "已交付：/outputs/briefs/x/a.md"}],
        ]
        factory, state = stream_scripted(rounds)
        run = runner.start(FakeRuntime(), "做一份简报", self.tid, self.db, "default",
                           stream_factory=factory)
        self._wait(run)

        self.assertTrue(run.retried, "应当自动补救一次")
        self.assertTrue(run.published)
        self.assertEqual(state["n"], 2)
        body = history.load_messages(self.tid, self.db)[0]["content"]
        self.assertIn("已交付", body, "落库的是补救轮的回答，不是原始长文")

    def test_no_retry_when_already_published(self):
        rounds = [[{"type": "tool_call", "name": "publish_brief", "args": {}, "depth": 0},
                   PUBLISH_OK, {"type": "answer", "content": "报告" * 300}]]
        factory, state = stream_scripted(rounds)
        run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default", stream_factory=factory)
        self._wait(run)
        self.assertFalse(run.retried)
        self.assertEqual(state["n"], 1)

    def test_no_retry_for_short_answer(self):
        """一问一答不该被追问打搅。"""

        factory, state = stream_scripted([[{"type": "answer", "content": "今天是 10 月 25 日"}]])
        run = runner.start(FakeRuntime(), "今天几号", self.tid, self.db, "default",
                           stream_factory=factory)
        self._wait(run)
        self.assertFalse(run.retried)
        self.assertEqual(state["n"], 1)

    def test_retry_keeps_original_when_model_declines(self):
        """补救轮只回「无需交付」时，退回原稿，别把它当正文存下来。"""

        long_text = "报告正文" * 120
        rounds = [
            [{"type": "answer", "content": long_text}],
            [{"type": "answer", "content": "无需交付"}],
        ]
        factory, _ = stream_scripted(rounds)
        run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default", stream_factory=factory)
        self._wait(run)
        self.assertFalse(run.published)
        self.assertTrue(run.retried)
        self.assertEqual(history.load_messages(self.tid, self.db)[0]["content"], long_text)

    def test_retry_can_be_disabled(self):
        long_text = "报告正文" * 120
        factory, state = stream_scripted([[{"type": "answer", "content": long_text}]])
        run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default",
                           stream_factory=factory, deliver_retry=False)
        self._wait(run)
        self.assertFalse(run.retried)
        self.assertEqual(state["n"], 1)

    def test_timeout_aborts_and_records(self):
        """卡住的调用必须有上限：qwen3.8-max 实测在一次模型调用上挂了 37 分钟。"""

        def hanging(_runtime, _question, _thread):
            async def gen():
                yield {"type": "tool_call", "name": "web_search", "args": {}, "depth": 0}
                await asyncio.sleep(30)  # 永远等不到
                yield {"type": "answer", "content": "不该到这里"}

            return gen()

        run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default",
                           stream_factory=hanging, timeout_s=0.4)
        self._wait(run, timeout=8)

        self.assertIn("超时", run.error)
        body = history.load_messages(self.tid, self.db)[0]["content"]
        self.assertIn("超时", body)
        self.assertFalse(runner.busy())

    def test_registry_returns_same_run(self):
        run = runner.start(FakeRuntime(), "q", self.tid, self.db, "default",
                           stream_factory=stream_of(EVENTS))
        self._wait(run)
        self.assertIs(runner.get(self.tid), run)


if __name__ == "__main__":
    unittest.main(verbosity=2)
