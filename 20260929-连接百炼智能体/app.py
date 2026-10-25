"""
百炼 Managed Agents · Streamlit 本地网页示例

与同目录的 Node 版（server.mjs + public/index.html）调用的是同一套 API、同一个会话：
    浏览器(Streamlit) -> 本地 Python -> 百炼 Managed Agents API

依赖：streamlit、requests（见 requirements.txt）
运行：streamlit run app.py
配置：读取同目录 .env（DASHSCOPE_API_KEY / BAILIAN_WORKSPACE_ID / BAILIAN_SESSION_ID）

官方文档：
  - API 总览与认证   https://help.aliyun.com/zh/model-studio/managed-agents-api-overview
  - 会话事件流（SSE） https://help.aliyun.com/zh/model-studio/managed-agents-event-stream
  - 订阅 Event 流     https://help.aliyun.com/zh/model-studio/event-sse-stream
"""
import json
import os
import time
from pathlib import Path

import requests
import streamlit as st


# ---------- 1. 读取 .env（零依赖迷你解析器） ----------
def load_env(file: Path) -> None:
    if not file.exists():
        return
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)  # 真实环境变量优先于 .env


load_env(Path(__file__).parent / ".env")

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
WORKSPACE = os.environ.get("BAILIAN_WORKSPACE_ID", "")
REGION = os.environ.get("BAILIAN_REGION", "cn-beijing")
SESSION = os.environ.get("BAILIAN_SESSION_ID", "")

# Managed Agents 基地址按「工作空间 + 地域」拼装
BASE = f"https://{WORKSPACE}.{REGION}.maas.aliyuncs.com/api/v1/agentstudio"


# ---------- 2. 调用百炼：发送消息并流式读取事件 ----------
def stream_events(prompt: str):
    """
    向会话发送一条用户消息，并以生成器形式产出事件字典：
      {"type": "delta",  "text": "..."}                      文本增量
      {"type": "info",   "text": "..."}                      工具调用等提示
      {"type": "status", "status": "...", "stop_reason": {}}  会话状态
      {"type": "error",  "message": "..."}                    错误
    """
    auth = {"Authorization": f"Bearer {API_KEY}"}
    stream_url = f"{BASE}/sessions/{SESSION}/events/stream"
    post_url = f"{BASE}/sessions/{SESSION}/events"

    try:
        # 关键顺序：先建立事件流长连接，再发送消息，否则可能错过早期事件
        with requests.get(
            stream_url,
            headers={**auth, "Accept": "text/event-stream"},
            params={"event_deltas[]": "message"},  # 开启文本增量流
            stream=True,
            timeout=(15, 300),
        ) as upstream:
            if upstream.status_code != 200:
                yield {"type": "error",
                       "message": f"订阅事件流失败 HTTP {upstream.status_code} {upstream.text[:300]}"}
                return

            # 发送用户消息（同一会话即自动延续多轮上下文）
            resp = requests.post(
                post_url,
                headers={**auth, "Content-Type": "application/json"},
                json={"input": [{"role": "user", "type": "message",
                                 "content": [{"type": "text", "text": prompt}]}]},
                timeout=30,
            )
            if resp.status_code != 200:
                yield {"type": "error",
                       "message": f"发送消息失败 HTTP {resp.status_code} {resp.text[:300]}"}
                return

            streamed_ids = set()  # 已通过增量帧下发的事件 id，用于与完整消息去重
            for raw in upstream.iter_lines():  # 按字节读取，自行解码 UTF-8
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue  # 忽略 event: / id: / :注释（: connected、:keepalive）

                try:
                    ev = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue

                etype = ev.get("type")
                blocks = ev.get("content") or []

                if etype == "event_start":
                    inner = ev.get("event") or {}
                    if inner.get("type") == "message" and inner.get("id"):
                        streamed_ids.add(inner["id"])

                elif etype == "event_delta":
                    text = ((ev.get("delta") or {}).get("content") or {}).get("text")
                    if text:
                        yield {"type": "delta", "text": text}

                elif etype == "message":
                    if ev.get("role") == "assistant" and ev.get("id") not in streamed_ids:
                        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                        if text:
                            yield {"type": "delta", "text": text}

                elif etype == "tool_call":
                    name = ((blocks[0] or {}).get("data") or {}).get("name") if blocks else None
                    if name:
                        yield {"type": "info", "text": f"调用工具：{name}"}

                elif etype == "session_status":
                    data = ((blocks[0] or {}).get("data") or {}) if blocks else {}
                    status = data.get("session_status")
                    yield {"type": "status", "status": status, "stop_reason": data.get("stop_reason")}
                    if status in ("idle", "terminated"):
                        return  # 本轮结束

                elif etype == "error":
                    yield {"type": "error",
                           "message": (ev.get("error") or {}).get("message", "未知运行期错误")}

    except requests.exceptions.RequestException as exc:
        yield {"type": "error", "message": f"网络请求异常：{exc}"}


# ---------- 3. 页面 ----------
st.set_page_config(page_title="百炼 Managed Agents · Streamlit 示例", page_icon="🤖")
st.title("百炼 Managed Agents · Streamlit 本地网页示例")

missing = [name for name, val in (
    ("DASHSCOPE_API_KEY", API_KEY),
    ("BAILIAN_WORKSPACE_ID", WORKSPACE),
    ("BAILIAN_SESSION_ID", SESSION),
) if not val]

if missing:
    st.error("缺少配置：" + "、".join(missing) + "。请在 .env 中填写后重启。")
    st.stop()

with st.sidebar:
    st.subheader("配置")
    st.caption("会话 ID（多轮上下文由平台侧维护）")
    st.code(SESSION, language=None)
    st.caption(f"端点：{BASE}")
    if st.button("清空本地显示"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("输入消息，回车发送")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        box = st.empty()      # 承载流式文本
        acc = ""
        last_paint = 0.0
        for ev in stream_events(prompt):
            kind = ev.get("type")
            if kind == "delta":
                acc += ev["text"]
                now = time.time()
                if now - last_paint > 0.05:   # 轻微节流，避免过于频繁重绘
                    box.markdown(acc + "▋")
                    last_paint = now
            elif kind == "info":
                st.info(ev["text"])
            elif kind == "error":
                st.error(ev["message"])
            elif kind == "status":
                if (ev.get("stop_reason") or {}).get("type") == "requires_action":
                    st.warning("本次需要人工审批工具调用，本示例未实现审批交互。")

        box.markdown(acc if acc else "_（无文本回复）_")

    if acc:
        st.session_state.messages.append({"role": "assistant", "content": acc})
