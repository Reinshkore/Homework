"""一次运行的完整生命周期，**独立于页面**。

为什么必须独立：原来流式循环写在页面函数里，切页面会让 Streamlit 中断整个脚本，
循环被丢弃——而"写会话记录"那句在循环之后，于是永远不执行。实测后果：
运行照常完成、简报照常落盘（Agent 跑在自己的事件循环上，不受页面影响），
但会话库里只剩一句提问，助手回复丢失。

所以这层把三件事绑在一起，谁都不能半途退出：
    开始 → 跑到底 → 落库（成功、报错、无输出都落一条）

界面只负责"看"：它读 Run 的 steps / text，切走再回来还能接着看。
多条消息的写入用独立的 sqlite 连接，与界面线程无关。
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from agent.events import stream_events
from core import history

# 跑完的 Run 留一小段，便于界面切回来时能立刻重绘；超出的按时间淘汰。
_MAX_KEEP = 20

RUNS: dict[str, "Run"] = {}


# 补救用的第二轮指令。只在"模型显然写完了报告却没交付"时追加一次。
#
# 为什么需要它：交付是可检出的（有没有 publish_brief 调用、返回里 published 是否为真），
# 所以不必指望每个模型都守规矩。实测 qwen3.7-plus 跑完 45 次工具调用、正文全写在对话里，
# 就是不调交付工具——同一份技能 deepseek-flash 会调。这种差异靠改提示词治不干净，
# 靠"事后检出 + 追问一次"能治。
RETRY_PROMPT = (
    "你上一轮把简报写在了对话里，但没有调用 publish_brief，因此这一期没有产出文件、"
    "账本也没有记录。现在请立刻调用 publish_brief 交付：markdown 用你刚写好的全文，"
    "ledger_entry 按 assets/report-skeleton.md 第九节的格式补一段（主线判断、位移类型、"
    "依赖假设、推翻条件、悬置事项、检索缺口）。不要重写报告，也不要再检索。"
    "如果你确认用户只要对话里的内容、不需要落盘，就只回复四个字：无需交付。"
)

_RETRY_MIN_CHARS = 400  # 短回答不是"写了报告没交付"，别拿追问去打搅它

# 一次运行的墙钟上限。实测 qwen3.8-max 在一次模型调用上卡了 37 分钟：数据库与日志都静止，
# 界面永远停在"思考中"，用户分不清是慢还是死了。标准档正常 9–13 分钟，25 分钟足够。
DEFAULT_TIMEOUT_S = float(__import__("os").getenv("RUN_TIMEOUT_MIN") or 25) * 60


class Run:
    def __init__(self, runtime, question: str, thread_id: str, db_path: Path,
                 archive_id: str = "", stream_factory=None, deliver_retry: bool = True,
                 timeout_s: float | None = None):
        self.runtime = runtime
        self.question = question
        self.thread_id = thread_id
        self.db_path = Path(db_path)
        self.archive = archive_id or getattr(runtime, "archive", "")
        self.steps: list[dict] = []
        self.text = ""
        self.answer = ""
        self.error = ""
        self.published = False
        self.retried = False
        self.expect_publish = deliver_retry
        self.timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        self.started_at = time.monotonic()
        self.done = threading.Event()
        self._stream = stream_factory or (
            lambda rt, q, t: stream_events(rt.agent, q, t)
        )

    # ---- 事件累积（在 Agent 的事件循环线程上执行）
    def _on_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "token":
            self.text += ev["content"]
        elif kind == "thought":
            return  # 旁白不进正文
        elif kind in ("tool_call", "tool_result"):
            if kind == "tool_result" and ev.get("name") == "publish_brief":
                body = str(ev.get("content") or "")
                if "'published': True" in body or '"published": true' in body.lower():
                    self.published = True
            self.steps.append(ev)
        elif kind == "error":
            self.error = ev["content"]
        elif kind == "answer":
            self.answer = ev["content"]

    def _finish(self) -> None:
        """无论如何都要落一条助手消息。

        报错也落：用户切回来时看到"出错了：…"，比看到一片空白有用——
        空白会让人以为记录又丢了，而报错本身是信息。
        """

        body = self.answer or self.text
        if not body:
            body = f"出错了：{self.error}" if self.error else "（没有生成回答）"
        self.answer = body
        try:
            history.save_message(self.thread_id, "assistant", body, self.steps,
                                 db_path=self.db_path)
        except Exception as exc:  # noqa: BLE001  落库失败也不该让 Run 卡在未完成
            self.error = self.error or f"写会话记录失败：{exc}"
        finally:
            self.done.set()

    # ---- 启动
    def start(self) -> "Run":
        async def consume() -> None:
            try:
                async with asyncio.timeout(self.timeout_s):
                    await self._work()
            except TimeoutError:
                mins = int(self.timeout_s // 60)
                self.error = f"运行超时（{mins} 分钟）已中止。检索与判定结果都还在证据库里，可以下期接着用。"
            except Exception as exc:  # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self._finish()

        try:
            self.runtime.submit(consume())
        except Exception as exc:  # noqa: BLE001  运行时的循环已关闭之类
            self.error = f"启动失败：{exc}"
            self._finish()
        return self

    async def _work(self) -> None:
        """一轮正常流程 + 一次交付补救。超时由外层 asyncio.timeout 管。"""

        await self._round(self.question)
        if not self._needs_retry():
            return

        self.retried = True
        keep = self.answer or self.text
        self.text, self.answer = "", ""
        await self._round(RETRY_PROMPT)
        # 补救的判据是"有没有交付"，不是"回答有多长"：
        # 模型可能只回一句"已交付：路径"，那是对的；也可能回"无需交付"，那就退回原稿。
        # 用长度判断会把前者误判成后者。
        if not self.published:
            self.text = keep
            self.answer = keep

    async def _round(self, question: str) -> None:
        async for ev in self._stream(self.runtime, question, self.thread_id):
            self._on_event(ev)

    def _needs_retry(self) -> bool:
        """写了长报告、却没交付——补救一次的判据。"""

        if not self.expect_publish or self.published:
            return False
        body = (self.answer or self.text).strip()
        if len(body) < _RETRY_MIN_CHARS:
            return False
        return True

    # ---- 给界面读
    @property
    def running(self) -> bool:
        return not self.done.is_set()

    def calls(self) -> int:
        return sum(1 for s in self.steps if s.get("type") == "tool_call")


def start(runtime, question: str, thread_id: str, db_path: Path, archive_id: str = "",
          stream_factory=None, deliver_retry: bool = True,
          timeout_s: float | None = None) -> Run:
    run = Run(runtime, question, thread_id, db_path, archive_id,
              stream_factory=stream_factory, deliver_retry=deliver_retry,
              timeout_s=timeout_s).start()
    RUNS[thread_id] = run
    _prune()
    return run


def get(thread_id: str) -> Run | None:
    return RUNS.get(thread_id)


def busy(archive_id: str | None = None) -> bool:
    """有没有正在跑的运行。界面用它挡住"切档案/删档案/新会话"。

    切档案会关闭当前运行时，连带停掉它的事件循环——正在跑的运行会被腰斩。
    这跟"运行中禁止改环境变量"是同一条道理：别在别人干活时拆他的桌子。
    """

    return any(r.running and (archive_id is None or r.archive == archive_id)
               for r in RUNS.values())


def _prune() -> None:
    if len(RUNS) <= _MAX_KEEP:
        return
    finished = sorted((r for r in RUNS.values() if not r.running),
                      key=lambda r: r.steps[-1].get("_ts", 0) if r.steps else 0)
    for run in finished[:len(RUNS) - _MAX_KEEP]:
        RUNS.pop(run.thread_id, None)
