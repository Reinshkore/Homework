"""Agent 运行时：一个档案一个 Agent，切档案时整体换掉。

为什么不用 `st.cache_resource`：它按参数分别缓存，切过的档案会一直留在内存里，
每个都占一条后台线程与一个 MCP 子进程 —— 那正是"独占后台进程"，用户明确不要。

所以生命周期在这里显式管理：
    切档案 → close 旧的（关检查点连接、停事件循环）→ 建新的

`get_runtime(state, archive_id)` 里的 state 只要是个像 dict 的对象就行
（Streamlit 的 session_state 就是），这样本模块不依赖 streamlit，
scripts/smoke_chat.py 也能直接用 AgentRuntime。
"""

from __future__ import annotations

import asyncio
import threading

from agent.builder import build_agent, close_agent
from core import config


class AgentRuntime:
    def __init__(self, archive_id: str = config.DEFAULT_ARCHIVE, spinner=None):
        self.archive = archive_id
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name=f"agent-loop-{archive_id}"
        )
        self._thread.start()
        self._closed = False
        self.agent, self.info = self.call(build_agent(archive_id))

    def call(self, coro):
        """把协程丢进本运行时的事件循环并等结果。界面线程与服务线程不共享 loop，
        这也是 Agent 与 SQLite 连接能被安全复用的原因。"""

        if self._closed:
            raise RuntimeError("运行时已关闭，请重新获取")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def submit(self, coro):
        """不等结果，返回 concurrent.futures.Future。流式对话用它，边跑边取事件。"""

        if self._closed:
            raise RuntimeError("运行时已关闭，请重新获取")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self) -> None:
        """关检查点连接 → 停循环。先关连接再停循环，否则连接永远关不上。"""

        if self._closed:
            return
        self._closed = True
        try:
            self.call_sync(close_agent(self.agent))
        except Exception:  # noqa: BLE001  关闭失败不该阻塞切换
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass

    def call_sync(self, coro) -> None:
        """close 内部用：此时 _closed 已经置位，绕开 call 的守卫。"""

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.result(timeout=10)


def get_runtime(state, archive_id: str = config.DEFAULT_ARCHIVE) -> AgentRuntime:
    """取当前档案的运行时；档案变了就先关旧的。

    state 必须持久（session_state / dict），否则每次调用都会重建 Agent，
    每次重建要重新起 MCP 子进程，界面会卡十秒。
    """

    rt = state.get("_rt")
    if isinstance(rt, AgentRuntime) and rt.archive == archive_id and not rt._closed:
        return rt
    if rt is not None:
        try:
            rt.close()
        except Exception:  # noqa: BLE001
            pass
    rt = AgentRuntime(archive_id)
    state["_rt"] = rt
    return rt
