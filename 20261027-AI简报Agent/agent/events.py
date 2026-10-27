"""把 Agent 的执行过程归一化成扁平事件，供界面层消费。

同时订阅 LangGraph 的两种流：
    messages  模型逐 token 输出，用于打字机效果
    updates   每个节点结束后的完整状态，用于拿到工具调用与返回
配合 subgraphs=True，子代理内部的步骤也会被带出来，用 depth 区分层级（0=主 Agent）。

与 demo5 的关键差异：`answer` 只装最终回答，工具调用前的旁白单独走 `thought`。
demo5 把两者拼在一起，聊天场景无所谓，但简报要落盘、要贴进幻灯片，
正文里混进"我先把六类都检索一遍"就是脏数据。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

# LangGraph 的 super-step 上限。一个工具调用大约吃掉 2 步（模型节点 + 工具节点），
# 所以这个数约等于"能跑多少次工具调用"。
#
# 实测定档（2026-10-27）：深度档一期 64 次工具调用 → step 到 160 被打断，
# 而当时正文只差 16 字就过线。160 是 demo 留下的默认值，对"检索 + 逐条判定 +
# 反复交付"这种长任务偏紧，抬到 300（约 130 次调用）。真正的兜底是 runner 的
# 25 分钟墙钟，不是这个数——它只防死循环，不该变成"活儿干到一半被掐"。
RECURSION_LIMIT = 300


def _iter_messages(update):
    """从 stream_mode="updates" 的单个节点更新中取出新消息。"""

    if not isinstance(update, dict) or update.get("messages") is None:
        return []
    msgs = getattr(update["messages"], "value", update["messages"])  # 兼容 Overwrite 包装
    return msgs if isinstance(msgs, list) else [msgs]


def _short(value, limit: int = 400) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[:limit] + "…（已截断）"


async def stream_events(agent, question: str, thread_id: str):
    """运行一轮对话，产出事件字典：

    {"type": "token",       "content": str}                      主 Agent 流式文字
    {"type": "thought",     "content": str}                      工具调用前的旁白（不进正文）
    {"type": "tool_call",   "depth": int, "name": str, "args": dict}
    {"type": "tool_result", "depth": int, "name": str, "content": str}
    {"type": "answer",      "content": str}                      最终回答（仅正文）
    {"type": "error",       "content": str}                      异常的可读描述
    """

    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    final, buffer = "", ""

    try:
        async for namespace, mode, chunk in agent.astream(
            {"messages": [{"role": "user", "content": question}]},
            config=cfg,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            depth = len(namespace)

            if mode == "messages":
                msg, _meta = chunk
                # 只把主 Agent 的文字做流式展示；子代理的过程不打扰读者
                if depth == 0 and isinstance(msg, AIMessageChunk) and msg.text:
                    buffer += msg.text
                    yield {"type": "token", "content": msg.text}
                continue

            for update in chunk.values():
                for msg in _iter_messages(update):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        if depth == 0 and buffer.strip():
                            yield {"type": "thought", "content": buffer.strip()}
                            buffer = ""
                        for call in msg.tool_calls:
                            yield {
                                "type": "tool_call",
                                "depth": depth,
                                "name": call["name"],
                                "args": call.get("args") or {},
                            }
                    elif isinstance(msg, ToolMessage):
                        raw = msg.text or ""
                        yield {
                            "type": "tool_result",
                            "depth": depth,
                            "name": msg.name,
                            "content": _short(raw),
                            # chars 是截断前的真实长度。界面要显示"返回多少字"，
                            # 显示截断后的 400 会误导——那只是展示上限。
                            "chars": len(raw),
                        }
                    elif isinstance(msg, AIMessage) and depth == 0 and msg.text:
                        final = msg.text
    except Exception as exc:  # noqa: BLE001
        # 返回结构化错误而不是抛出去：界面要能显示"哪里失败了、下一步怎么办"
        yield {"type": "error", "content": f"{type(exc).__name__}: {exc}"}
        return

    yield {"type": "answer", "content": (final or buffer).strip()}
