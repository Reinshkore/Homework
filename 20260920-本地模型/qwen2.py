"""
Qwen3-0.6B 本地对话 Demo（Streamlit 版）— 修正版

运行：streamlit run qwen2.py
"""

import os

# ---------------------------------------------------------------
# 离线模式：避免每次启动都去 HuggingFace 联网校验导致卡住
# 必须写在 import transformers 之前
# ---------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

MODEL_PATH = r"C:\Users\xnito\Reishkurs\LLM\models\Qwen--Qwen3-0.6B\snapshots\master"

# 模型的上下文总长度上限（来自 config.json 的 max_position_embeddings）
MAX_CONTEXT = 32768
# 单次回复最多生成多少 token（注意：这只是上限，不是目标）
MAX_NEW_TOKENS = 2048


@st.cache_resource
def load_model():
    """加载模型。cache_resource 保证整个 Streamlit 会话只加载一次。"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,  # 让权重按 bf16 加载，省内存、更快
    )
    model.eval()  # 推理模式
    return tokenizer, model


tokenizer, model = load_model()


def build_inputs(history):
    """
    【关键修正】把完整的多轮历史交给 apply_chat_template。

    原代码每次都只传 [{"role":"user","content":prompt}]，
    等于模型完全看不到之前的对话，界面上的聊天记录只是给人看的摆设。
    """
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    # 转成张量后再裁掉超出上下文上限的部分（保留最新内容，即从左侧截断）
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    if inputs.input_ids.shape[1] > MAX_CONTEXT - MAX_NEW_TOKENS:
        keep = MAX_CONTEXT - MAX_NEW_TOKENS
        inputs["input_ids"] = inputs.input_ids[:, -keep:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs.attention_mask[:, -keep:]
    return inputs


def generate_response(history):
    """流式生成回复，逐段 yield 文本。"""
    inputs = build_inputs(history)
    prompt_len = inputs.input_ids.shape[1]

    # TextIteratorStreamer 让生成过程在后台线程跑，主线程可以边收边显示
    # 这样才是真正的流式，而不是"等全部生成完再假装一个字一个字打出来"
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated = ""
    for chunk in streamer:
        generated += chunk
        yield generated, prompt_len
    thread.join()


# ---------------------------------------------------------------
# 界面
# ---------------------------------------------------------------
st.set_page_config(page_title="Qwen3 Chat", page_icon="💬")
st.title("Qwen3 Chat")
st.caption("本地部署 · Qwen3-0.6B · CPU")

# 侧边栏：方便调试和清空
with st.sidebar:
    st.header("设置")
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"上下文上限：{MAX_CONTEXT} tokens")
    st.caption(f"单次最多生成：{MAX_NEW_TOKENS} tokens")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("你想问点什么？"):
    # 1. 先把用户消息存进历史并立即显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 生成助手回复
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        # 传完整历史（不含正在生成的这条助手消息）
        for partial, _plen in generate_response(st.session_state.messages):
            full_response = partial
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    # 3. 回复完成后，整条存进历史
    #    这正是原代码最坑的地方：如果在生成过程中就会话重跑，
    #    assistant 消息还没写进历史，而 chat_input 的值已经没了
    #    —— 用户输入凭空消失。先写历史、后生成，就不会丢。
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
