# =============================================================================
# 示例 3.5  ——  在 3.4 的基础上，加入「多轮记忆」
# =============================================================================
#
# 【这一版解决了什么】
#   3.3 和 3.4 都有一个共同缺陷：
#       generate_response 里构造的 messages 列表【永远只有当前这一句】，
#       模型完全看不到之前的对话 —— 界面上显示的历史只是摆设。
#   3.5 的改进很朴素：把之前的对话【真的拼进去】，一起喂给模型。
#
#   同时它还给用户一个开关，让用户自己决定"记几轮"。
#
# 【与前两版的关系】
#   3.3 = 同步假流式
#   3.4 = 真流式（线程 + 队列）
#   3.5 = 真流式 + 多轮记忆   ← 本文件，功能最完整的一版
#
# 【前置知识】
#   本文件同时用到了三块知识，缺一不可：
#     · 线程与队列   →  《前置知识.md》第 2 章
#     · 大模型推理   →  《前置知识.md》第 3 章
#     · Streamlit    →  《前置知识.md》第 4 章
#
# 【本文件相对 3.4 的代码差异清单】（建议对照 3.4 一起读）
#   1. CustomStreamer 多了 self.is_first_chunk 标记
#   2. think 清理的写法略有不同（split 参数不一样）
#   3. generate_response_stream 多了 history 参数
#   4. 侧边栏多了"记忆设置"（复选框 + 记忆轮数滑块）
#   5. session_state 多了 memory_enabled / memory_rounds 两个键
#   6. 主线程多了 thread.join()
#
# 【环境说明】
#   路径 "../models/Qwen3-0.6B" 是教程作者机器上的相对路径。
#   你自己的实际路径是：
#       C:\Users\xnito\Reishkurs\LLM\models\Qwen--Qwen3-0.6B\snapshots\master
# =============================================================================

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Thread
# 线程类，让生成在后台跑。详见《前置知识.md》2.4。

from queue import Queue, Empty
# Queue  = 线程安全的队列（跨线程传送带）。
# Empty  = "限时取数据但队列为空"时抛出的异常。
# 详见《前置知识.md》2.6 / 2.9。

import time
# 计时用。

# 【注意】3.4 里那行没用的 "import platform" 在 3.5 里被删掉了，是个进步。


# 加载模型和分词器
@st.cache_resource
def load_model():
    # 缓存模型，避免每次脚本重跑都重新加载 1.5GB 权重。
    # 详见《前置知识.md》4.3。

    model_name = "../models/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, model


tokenizer, model = load_model()


class CustomStreamer:
    # 这个类的作用同 3.4：充当"模型"与"界面"之间的中间人。
    # transformers 每生成一个 token 就回调一次 put()，把 token 交给我们。
    # 详见《前置知识.md》2.10 / 5.1。

    def __init__(self, queue):
        # 构造函数，创建实例时自动执行。
        # self 代表"这个实例自己"，用 self.xxx = ... 把数据挂上去。
        # 详见《前置知识.md》1.4。

        self.queue = queue
        # 存下"传送带"。put() 跑在生成线程里，而取数据的主线程也要用同一个队列，
        # 这就是两个线程能通信的原因。详见《前置知识.md》2.6 / 2.7。

        self.first_token_time = None
        # 记录首 token 到达时间，None 表示还没记录。

        self.generation_start = time.perf_counter()
        # 高精度计时器的起点（在实例创建时取，也就是生成开始之前）。

        self.full_response = ""  # 存储完整响应
        # 累积每一次收到的片段。

        self.is_first_chunk = True  # 标记是否为第一个chunk
        # 【★ 3.5 新增 ★】
        # 一个布尔标记，用来判断"这是不是第一次收到的数据"。
        # 为什么要判断？因为第一次收到的东西最脏 —— 里面混着 prompt 和各种标记，
        # 需要特殊处理（见下面 put 里的用法）。
        # 处理过一次之后立刻置为 False，后续就都走常规路径了。

    def put(self, value):
        # ---------------------------------------------------------------------
        # 【牢记这一点，否则会看不懂下面的清洗逻辑】
        # put() 会被反复调用，但【第一次非常特殊】：
        #   第 1 次 put：收到的是【整个 prompt】（用户问题 + 模板标记），形状如 (1, 18)
        #   第 2 次起  ：每次只收到【一个新 token】，形状 (1,)
        # 实测：生成 15 个新 token 时，put 被调用 16 次。
        # 详见《前置知识.md》5.2。
        # ---------------------------------------------------------------------

        # 记录第一个token到达时间
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        # 【注意】第一次进入 put 时记录的其实是"prompt 到达的时刻"，
        # 不是真正的首 token 延迟。所以下面那个 get_first_token_latency()
        # 算出来的数字没有实际意义（实测约 2.8 毫秒）。

        # 解码tokens
        tokens = value.cpu().numpy()
        # value 是 PyTorch 张量：.cpu() 保证在内存里，.numpy() 转成 NumPy 数组。

        text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        # tokens[0] 取"批"里的第 0 条（去掉多余的一维）。
        # skip_special_tokens=True 丢掉 <|im_start|> 这类控制标记。
        # 第一次调用时，这里得到的是整段 prompt 的文字。

        # 移除 </think> 之前的内容（如果存在）
        if '</think>' in text:
            text = text.split('</think>', 1)[1]
        # 作用：丢掉思考和 </think> 本身，只保留正式回答。
        #
        # 【和 3.4 的细微差别】
        #   3.4 写的是  split('</think>')[-1]       取"最后一段"
        #   3.5 写的是  split('</think>', 1)[1]     限制只切一次，取"第二段"
        # 当字符串里只有一个 </think> 时，两者结果完全一样。
        # 只有当出现多个 </think> 时才会有区别，实际场景里几乎不会遇到。
        #
        # 【一个连带效果】这段代码还顺手挡住了 prompt 回显：
        #   因为模板渲染出的 prompt 末尾正好是 "<think>\n\n</think>\n\n"，
        #   第一次 put 收到的整段 prompt 被切掉前半部分后只剩空白，
        #   于是用户不会看到自己的问题被重复显示。
        #   但这份"能用"依赖模板渲染结果的巧合，而不是有意设计。
        #   详见《前置知识.md》5.2。

        # 第一个chunk中，移除 "Assistant:" 之前的内容
        if self.is_first_chunk:
            assistant_idx = text.rfind("Assistant:")
            if assistant_idx >= 0:
                text = text[assistant_idx + len("Assistant:"):]
            self.is_first_chunk = False
        # 【3.5 新增的处理，但它实际上不起作用】
        #
        # 逐句看：
        #   text.rfind("Assistant:")  从右往左找最后一次出现的 "Assistant:"，
        #                              找不到就返回 -1。
        #   if assistant_idx >= 0:     找到了才做切分。
        #   text[assistant_idx + len("Assistant:"):]
        #                              从 "Assistant:" 之后开始截取，
        #                              也就是丢掉角色前缀，只留正文。
        #
        # 【为什么它不起作用】
        # 实际渲染出来的文本是小写的 "assistant\n"（换行结尾，没有冒号和空格），
        # 因为 <|im_start|> 这种特殊标记已经被 skip_special_tokens 去掉了。
        # 大写带冒号的 "Assistant:" 根本不会出现，
        # 所以 rfind 永远返回 -1，这个 if 分支永远不执行。
        # 换句话说：真正挡住 prompt 的是上面那段 </think> 清理，
        # 而不是这段 is_first_chunk 逻辑。

        cleaned_text = text.strip()
        # 去掉首尾空白。

        self.full_response += cleaned_text
        # 累积。

        self.queue.put(cleaned_text)
        # 【★ 跨线程传数据 ★】把这一片文字放进队列，等主线程来取。
        # Queue 内部有锁，两个线程同时读写是安全的。
        # 详见《前置知识.md》2.6 / 2.7。

    def end(self):
        # 生成正常结束时，transformers 会自动调用一次。
        self.queue.put(None)
        # 【哨兵值】放一个 None 表示"说完了"。
        # 不能用"队列空"表示结束，因为模型还在算下一个 token 时队列也是空的。
        # 详见《前置知识.md》2.8。

    def get_first_token_latency(self):
        if self.first_token_time:
            return (self.first_token_time - self.generation_start) * 1000  # 单位为毫秒
        return 0
        # 【注意】这个函数在 3.5 里依然从未被调用，是死代码。
        # 3.4 里就有这个问题，3.5 没有改。


def generate_response_stream(prompt, queue, max_tokens, temperature, history=None):
    # =========================================================================
    # 【★ 跑在【子线程】里 ★】
    # =========================================================================
    # 职责：专心生成，每生成一点就往 queue 里丢。
    # 它完全不碰界面 —— 界面由主线程负责。这叫"职责分离"。
    #
    # 【3.5 新增参数】history=None
    #   = 之前的对话记录（一个消息列表）。
    #   写成 =None 表示"默认不传"，这是 Python 里指定默认值的写法。
    #   这样调用方可以只传前四个参数，history 就自动是 None（等于不用记忆）。
    #   详见《前置知识.md》1.2。
    # =========================================================================

    generation_start = time.perf_counter()
    # 记录开始时间。【注意】这个变量后面没被用到，是冗余代码（3.4 里也一样）。

    # 准备输入：包含历史记录（如果启用）
    if history:
        messages = history + [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    # 【★ 这就是 3.5 的核心改动 ★】
    #
    # if history:  这个写法利用了 Python 的"真值判断"：
    #              history 如果是 None 或空列表 []，都算"假"，走 else 分支。
    #
    # 两个分支的差别：
    #   有历史时：messages = 历史消息 + 当前问题
    #             列表相加 = 把两个列表拼成一个新列表。
    #   无历史时：messages = 只有当前问题（和 3.3/3.4 一样）
    #
    # 拼好之后，下面 apply_chat_template 会把整串对话一起转成模型能读的格式。
    # 这样模型才真正"看得到"之前聊过什么。
    #
    # 一个细节：当前这个问题是【在这里显式拼上去】的。
    # 而主线程准备 history 时特意把当前问题排除掉了（见下面的 messages[:-1]），
    # 所以不会出现"同一句话被拼两遍"的重复。

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        # 关闭 Qwen3 的思考模式。传 False 后模板会给一个【空的】think 块，
        # 模型就不真的思考了，直接作答。
        # 实测渲染末尾是：<|im_start|>assistant\n<think>\n\n</think>\n\n
        # 详见《前置知识.md》3.4。
    )
    # apply_chat_template：把消息列表套用模型规定的模板，拼成一整段字符串。
    # 每个模型的模板都不一样，Qwen3 用 <|im_start|> / <|im_end|> 划分说话人边界。
    # 详见《前置知识.md》3.4。

    model_inputs = tokenizer([text], return_tensors="pt")
    # 编码成 PyTorch 张量。外面套方括号 = "一批里有 1 条"。
    # 详见《前置知识.md》3.5。

    prompt_tokens = len(model_inputs.input_ids[0])
    # 输入有多少个 token。【注意】这个变量后面没被用到，是冗余代码。

    # 初始化流式处理器
    streamer = CustomStreamer(queue)
    # 创建中间人，把"传送带"交给它。

    # 生成回复
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_tokens,
        # 来自侧边栏滑块。
        temperature=temperature,
        # 来自侧边栏滑块。
        streamer=streamer,
        # 【★ 这一行让流式生效★】generate 每生成一步就回调 streamer.put()。
        do_sample=True if temperature > 0 else False
        # 温度 >0 时用随机采样，否则用贪心解码。
        # 【注意】滑块最小值是 0.1，所以这里永远为 True，else 分支走不到。
        # 另外传了 temperature 会覆盖模型自带的 temperature=0.6 等参数。
        # 详见《前置知识.md》3.8。
    )

    # 完整解码用于日志输出
    full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # 【注意】这里解码的是"输入+输出"整个序列，没有切掉输入部分（3.3 用 [len(...):] 切了）。
    # 因为这里只用于在终端打日志，不显示给用户，所以把 prompt 也打出来无妨。

    if '</think>' in full_response:
        full_response = full_response.split('</think>', 1)[1]
    for prefix in ['User:', 'Assistant:']:
        if prefix in full_response:
            full_response = full_response.split(prefix, 1)[-1]
    # 同样的清理，用于终端日志。
    # 【注意】'User:' / 'Assistant:' 这些前缀不会出现，所以这个循环是无效代码。
    # 【另一个注意】当有历史记录时，full_response 里会出现多个 </think>
    # （每轮对话的模板里都有一个空的 think 块），
    # 这里只切掉第一个，所以控制台打出来的日志可能会包含历史内容。不影响界面显示。

    print(full_response.strip())
    # 打印到运行 streamlit 的那个终端窗口。
    print('*' * 40)
    # 星号分隔线。


# Streamlit 界面

st.title("Qwen3 Chat")
st.write("By WJW")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 对话历史。详见《前置知识.md》4.2。

if "memory_enabled" not in st.session_state:
    st.session_state.memory_enabled = False
    # 【3.5 新增】记忆功能的总开关，默认关闭。
    # 为什么要放进 session_state？
    #   因为 Streamlit 每次交互都会重跑整个脚本。
    #   如果只是写成普通变量 memory_enabled = False，
    #   那么用户在侧边栏打开开关之后，下一次重跑又会被重置成 False —— 开关就失效了。
    #   放进 session_state 才能跨重跑保持用户的选择。
    # 详见《前置知识.md》4.2。

if "memory_rounds" not in st.session_state:
    st.session_state.memory_rounds = 3
    # 【3.5 新增】要记住几轮对话，默认 3 轮。同样需要跨重跑保持。

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    # 逐条画成聊天气泡。

# 侧边栏配置
with st.sidebar:
    # with 块里创建的控件都会放进左侧面板。

    st.header("⚙️ 模型配置")

    max_tokens = st.slider("最大生成长度", 100, 2048, 1024)
    # 滑块：标签, 最小值, 最大值, 默认值。

    temperature = st.slider("温度系数", 0.1, 2.0, 1.0, 0.1)
    # 第五个参数 0.1 是步长。

    # 记忆设置
    st.divider()
    # 画一条水平分隔线，让侧边栏分组更清楚。

    st.header("🧠 记忆设置")

    st.session_state.memory_enabled = st.checkbox("启用长期记忆", value=st.session_state.memory_enabled)
    # 复选框（打勾框）。
    # 第二个参数 value=... 指定"当前是否勾选"，返回值是勾选状态（True/False）。
    #
    # 【关键写法】把返回值【赋值回 session_state】。
    # 这样做的意义：用户勾选后，这个 True 被存进 session_state，
    # 脚本重跑时 value=st.session_state.memory_enabled 会读到 True，
    # 复选框保持勾选状态，不会被重置。这就是"受控组件"的用法。

    st.session_state.memory_rounds = st.slider("记忆轮数", 1, 10, st.session_state.memory_rounds,
                                               disabled=not st.session_state.memory_enabled,
                                               help="控制模型记住之前的对话轮数")
    # 记忆轮数滑块：范围 1~10，默认值取 session_state 里的当前值。
    # 第三、四个关键字参数的含义：
    #   disabled=not st.session_state.memory_enabled
    #       当记忆开关没打开时（memory_enabled 是 False，not False = True），
    #       这个滑块被禁用（变灰、拖不动）。
    #       这是一种界面联动：没开记忆就没必要调轮数。
    #   help="..."
    #       鼠标悬停在控件上时显示的提示文字。
    # 注意它没有写步长参数，所以默认步长是 1，只能选整数。
    # 同样把返回值写回 session_state 以保持状态。

    if st.button("🧹 清空对话"):
        st.session_state.messages = []
        st.rerun()
        # 清空历史并立刻重跑脚本，让界面马上更新。
        # 【注意】这里只清了 messages，没有重置 memory_enabled / memory_rounds。
        # 详见《前置知识.md》4.1。

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 海象运算符 := ：赋值并同时判断非空。
    # 等价于 prompt = st.chat_input(...) 然后 if prompt:
    # 详见《前置知识.md》1.7。

    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 先把用户这句话存进历史。

    with st.chat_message("user"):
        st.markdown(prompt)
    # 把用户这句话显示成一个气泡。

    # 生成助手回复
    with st.chat_message("assistant"):
        placeholder = st.empty()
        # 空占位符，会被反复覆盖写入，实现打字机效果。
        # 详见《前置知识.md》4.5。

        full_response = ""
        queue = Queue()
        # 【★ 传送带 ★】每问一个问题都新建一个队列，
        # 避免上一次对话的残留数据混进这一次。

        # 构建历史上下文（如果启用记忆）
        history = None
        if st.session_state.memory_enabled and len(st.session_state.messages) > 1:
            history_messages = st.session_state.messages[:-1]  # 排除当前问题
            keep_count = min(st.session_state.memory_rounds * 2, len(history_messages))
            history = history_messages[-keep_count:]
        # 【★ 这是 3.5 主线程里最需要理解的一段 ★】
        # 目的：从完整历史里，挑出"最近 N 轮"对话，拼成要传给模型的 history。
        # 逐行拆解：
        #
        # (1) history = None
        #     默认不用记忆。传 None 给子线程时，子线程会走 else 分支只用当前问题。
        #
        # (2) if st.session_state.memory_enabled and len(st.session_state.messages) > 1:
        #     两个条件同时满足才构建历史：
        #       · 用户打开了记忆开关
        #       · 消息数大于 1 —— 为什么？
        #         因为上面刚把当前问题 append 进去，所以此刻列表里至少有 1 条。
        #         如果正好只有 1 条，说明这是第一轮对话，压根没有"历史"可记。
        #
        # (3) history_messages = st.session_state.messages[:-1]
        #     【切片】[start:end] 是 Python 的切片语法，end 位置不包含在内。
        #     [:-1] 意思是"从头取到倒数第一个【之前】"，也就是丢掉最后一条。
        #     丢掉的那条正是【当前这个问题】—— 因为它马上要在子线程里
        #     被显式拼上去，这里不排除就会重复两次。
        #     所以此刻 history_messages = 往前的所有 user/assistant 消息。
        #
        # (4) keep_count = min(st.session_state.memory_rounds * 2, len(history_messages))
        #     【为什么乘 2？】因为一轮对话 = 1 条 user + 1 条 assistant = 2 条消息。
        #       记住 3 轮 → 需要保留 3×2 = 6 条消息。
        #     用 min(..., len(...)) 是为了取两者中较小的：
        #       万一历史还不够 6 条（比如刚聊了 2 轮），就按实际条数来，
        #       否则下面切片会越界取到空列表。
        #
        # (5) history = history_messages[-keep_count:]
        #     切片取"最后 keep_count 条"。
        #     [-6:] 表示"从倒数第 6 个开始，一直到结尾"。
        #     即只保留最近的 6 条，更早的都丢掉。
        #
        #   为什么要限制轮数，而不是把所有历史都传进去？
        #     因为模型的上下文长度是有上限的（Qwen3-0.6B 是 40960 个 token），
        #     而且输入越长，速度越慢。所以必须做"最近若干轮"的截断。
        #     详见《前置知识.md》3.7。

        # 启动生成线程
        thread = Thread(
            target=generate_response_stream,
            # target 写的是函数名本身，【没有括号】。
            # 加括号是"立刻调用"，不加括号才是"交给线程去调用"。
            # 这是最容易搞错的地方之一。
            args=(prompt, queue, max_tokens, temperature, history)
            # 参数元组，顺序必须和函数定义一致。
            # 【注意】history 也传进去了 —— 这就是"多轮记忆"真正生效的地方。
            # 【关键】queue 是同一个对象，所以子线程和主线程共用一条传送带。
            # 详见《前置知识.md》2.4。
        )
        thread.start()
        # 【★ 启动线程 ★】这一行之后函数就在另一条执行流里跑，
        # 主线程【立刻】继续往下执行，不会等它。
        # 这就是"边生成边显示"能实现的原因。
        # 详见《前置知识.md》2.4。

        # 流式输出响应
        while True:
            # 主线程在这里扮演"消费者"：不断从队列取数据、刷新界面。
            try:
                chunk = queue.get(timeout=0.1)
                # 取出队列里的第一片文字。
                # timeout=0.1 = 最多等 0.1 秒，超时就抛 Empty 异常。
                # 详见《前置知识.md》2.9。

                if chunk is None:
                    break
                # 收到哨兵值 None = 子线程说"我说完了"，跳出循环。
                # 【注意】这里用 is None 而不是 == None，
                # 因为 None 是单例对象，判断"是不是同一个对象"用 is 更符合习惯。
                # 详见《前置知识.md》2.8。

                full_response += chunk
                # 拼接。

                placeholder.markdown(full_response + "▌")
                # 覆盖写入占位符，"▌"是假装的光标。
                # 【真流式的观感就来自这一步】：文字一片片长出来。

            except Empty:
                continue
                # 0.1 秒没取到数据，属于正常情况（模型还在算下一个 token）。
                # 空转一轮，回到循环开头再取。

        placeholder.markdown(full_response.strip())
        # 去掉光标符号，显示最终干净的结果。

        thread.join()
        # 【3.5 新增】等待子线程彻底结束。
        #
        # 为什么在这里加它？
        #   因为退出上面那个 while 循环的条件是"收到 None"。
        #   而 None 是 end() 在生成【即将结束】时放进去的，
        #   此刻子线程可能还在收尾（比如执行后面那几行 print 日志）。
        #   join() 会阻塞主线程、【直到子线程真正跑完】才继续往下走。
        #   这样能保证"保存到历史"时，子线程确实已经完成了。
        #
        # 【注意】join() 不加超时参数会一直等。
        #   如果子线程出了异常，反而可能等很久。
        #   详见《前置知识.md》2.4。

    # 保存助手回复
    st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})
    # 存进历史，这样下一轮就有记忆可用了。
    # 【注意】依然是"生成完成后才写入"，所以 3.3 提到的
    # "生成期间重跑会丢输入"的风险在 3.5 里仍然存在。


# =============================================================================
# 【本示例小结】
#   3.5 相比 3.4 的实质进步只有一个：多轮记忆真的生效了。
#     做法：把最近 N 轮对话拼进 messages，一起交给 apply_chat_template。
#     配套：侧边栏提供开关和轮数控制，用 session_state 保持状态。
#
#   三个版本一张表：
#     ┌──────┬────────────┬────────────┬──────────┐
#     │      │ 是否真流式 │ 是否有多轮 │ 是否卡界面│
#     ├──────┼────────────┼────────────┼──────────┤
#     │ 3.3  │    否      │    否      │   是     │
#     │ 3.4  │    是      │    否      │   否     │
#     │ 3.5  │    是      │    是      │   否     │
#     └──────┴────────────┴────────────┴──────────┘
#
#   到这里功能就完整了。剩下的是工程细节，可以自己动手改：
#     ① 三个文件里的模型路径都要改成你自己的绝对路径
#     ② "生成期间重跑会丢输入"：把 assistant 消息的入库时机提前
#     ③ 子线程异常时界面会永久卡死：给 while 循环加异常兜底
#     ④ 首 token 延迟的计算方式要改成"跳过第一次 put 之后才算"
#     ⑤ 输入很长时没有截断保护，可能超过 40960 的上下文上限
# =============================================================================
