# =============================================================================
# 示例 3.4  ——  引入「线程 + 队列」，实现真正的流式输出
# =============================================================================
#
# 【这一版解决了什么】
#   3.3 的问题是：必须等模型把整段话全部生成完，才能显示第一个字。
#   3.4 要做的只有一件事：
#       让"生成"这件事在【另一个线程】里跑，主线程同时在旁边"边收边显示"。
#
# 【本文件的知识密度最高】
#   真正的难点不是 transformers，而是两个通用编程概念：
#     ① 线程（Thread）   —— 让两件事"同时"进行
#     ② 队列（Queue）    —— 让两个线程安全地传数据
#   这两个概念跟大模型毫无关系，是操作系统/编程语言层面的通用知识。
#   请务必先读《前置知识.md》第 2 章「并发与线程」，否则下面的代码会看不懂。
#
# 【与 3.3 / 3.5 的关系】
#   3.3 = 同步假流式   →   本文件 = 真流式   →   3.5 = 真流式 + 多轮记忆
#
# 【环境说明】
#   同 3.3：路径 "../models/Qwen3-0.6B" 是教程作者机器上的相对路径。
#   你自己的实际路径是：
#       C:\Users\xnito\Reishkurs\LLM\models\Qwen--Qwen3-0.6B\snapshots\master
# =============================================================================

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Thread
# Thread = 线程类。用它可以把一个函数丢到"另一条执行流"里去跑。
# 详见《前置知识.md》2.2 / 2.4。

from queue import Queue, Empty
# Queue = 线程安全的队列，专门用来在两个线程之间传数据。
# Empty = 一个异常类型，当"限时取数据但没取到"时会抛出它。
# 详见《前置知识.md》2.6 / 2.9。

import time
# 用来做计时（测量性能）。

import platform
# 【注意】这个模块导入了但整个文件里从未使用，是多余的导入。
# 属于可以删掉的"死导入"。


# 加载模型和分词器
@st.cache_resource
def load_model():
    # 装饰器作用同 3.3：把加载好的模型缓存起来，避免每次重跑都重新加载 1.5GB 权重。
    # 详见《前置知识.md》4.3。

    model_name = "../models/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, model


tokenizer, model = load_model()


class CustomStreamer:
    # =========================================================================
    # 【本文件的核心】一个"自定义流式处理器"
    # =========================================================================
    # 它是干什么的？—— 它是"模型"和"你的界面"之间的【中间人】。
    #
    # 为什么需要中间人？
    #   model.generate 生成文字时，是"一个 token 一个 token"往外吐的。
    #   如果不装这个中间人，这些 token 只会闷在 generate 内部，
    #   等全部生成完才一起返回给你 —— 那就还是 3.3 的假流式。
    #
    #   transformers 提供了一个【回调机制】(callback)：
    #     只要传给 generate 一个带 put() 方法的对象，
    #     它每生成一个 token 就会自动调用一次 put() 把它交给你。
    #   这个 CustomStreamer 就是那个被回调的对象。
    #
    # 【关键认知】这里的 put 不是你主动调用的，是 transformers 在生成过程中
    # "回头调用"你的方法。这类"你留个方法给它、它到时机自己叫"的设计就叫回调。
    # 详见《前置知识.md》2.10 / 5.1。
    # =========================================================================

    def __init__(self, queue):
        # __init__ 是"构造函数"，创建实例时自动执行，用来做初始化。
        # self 代表"这个实例自己"，通过 self.xxx = ... 把数据挂到实例上，
        # 这样其它方法里也能读到自己存的值。
        # 详见《前置知识.md》1.4。

        self.queue = queue
        # 把外面传进来的队列存下来。
        # 【为什么要把队列存进来？】
        # 因为 put() 是在【生成线程】里被调用的，而界面在【主线程】。
        # 两个线程要交换数据，就需要一个双方都能访问的"公传送带"，
        # 这个传送带就是 queue。
        # 详见《前置知识.md》2.6 / 2.7。

        self.first_token_time = None
        # 用来记录"第一个 token 是什么时候到的"，初始为 None 表示还没记录。

        self.generation_start = time.perf_counter()
        # time.perf_counter() 是"高精度计时器"，返回一个不断增大的浮点数。
        # 注意：它在【创建实例时】就取了一次时间，也就是"生成开始前"。
        # 两个时间点相减就是耗时。

        self.full_response = ""  # 存储完整响应
        # 把每次收到的片段累积起来，最终得到完整回答。

    def put(self, value):
        # ---------------------------------------------------------------------
        # 【★ 最重要的一个认知 ★】
        # 这个方法会被 transformers 反复调用，但【第一次调用非常特殊】：
        #
        #   第 1 次 put：收到的是【整个 prompt】（用户问题 + 模板标记），
        #                实测形状是 (1, 18) 这种"多元素"张量。
        #   第 2 次起  ：每次只收到【一个新生成的 token】，形状 (1,)。
        #
        # 实测：生成 15 个新 token 时，put 一共被调用 16 次
        #       （1 次 prompt + 15 次 token）。
        #
        # 这个特性带来一个陷阱：如果你把第一次收到的东西原样显示出去，
        # 用户就会看到自己的问题被"回显"了一遍。
        # 所以下面必须做"清洗"。详见《前置知识.md》5.2。
        # ---------------------------------------------------------------------

        # 记录第一个token到达时间
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        # 【注意】这里是"第一次进入 put"时记录时间，而第一次 put 收到的是 prompt。
        # 所以这个值实际记的是"prompt 进入 streamer 的时刻"，
        # 并不是"第一个生成 token 到达的时刻"。用下面的
        # get_first_token_latency() 算出来的耗时会小得没有意义（实测约 2.8 毫秒）。

        # 解码tokens
        tokens = value.cpu().numpy()
        # value 是一个 PyTorch 张量。
        #   .cpu()   如果张量在显卡上，先搬回内存（这里本来就是 CPU，等于没动）
        #   .numpy() 转成 NumPy 数组，方便用下标取值

        text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        # 取第 [0] 个（去掉"批"这一维），解码成文字。
        # 第一次调用时，这里得到的是整段 prompt 的文字形式。

        # 移除 <think> 标签之前的内容
        if '<think>' in text:
            text = text.split('</think>')[-1]
        elif '</think>' in text:
            text = text.split('</think>')[-1]
        # 这两段在做什么：模型输出里可能带"思考过程"（Qwen3 的 <think>...</think> 标签）。
        # 这段代码想只保留 </think> 之后的部分，也就是丢掉思考过程、只留正式回答。
        #
        # 【问题 1】两个分支做的事完全一样（都是 split('</think>')[-1]），
        #          所以 if / elif 的区分毫无意义，可以合并成一句：
        #              if '</think>' in text: text = text.split('</think>')[-1]
        #
        # 【问题 2】它为什么能顺便挡住 prompt 回显？
        #          纯属巧合。因为下面 enable_thinking=False 渲染出的 prompt 末尾
        #          正好是 "<think>\n\n</think>\n\n"，
        #          于是第一次 put 收到的 prompt 被 split('</think>')[-1] 切掉了前半段，
        #          只剩下末尾的空字符串，prompt 就显示不出来了。
        #          但这份"能用"依赖模板渲染结果的巧合：
        #          一旦把 enable_thinking 打开，prompt 就会被原样回显给用户。
        #
        # 【问题 3】更根本的做法应该是在这里主动丢弃第一次调用，
        #          而不是依赖上面这种巧合。

        # 清理前缀如 'User:', 'Assistant:'
        for prefix in ['User:', 'Assistant:']:
            if prefix in text:
                text = text.split(prefix, 1)[-1]
        # 【这是无效代码】
        # 它想清理 "User:" / "Assistant:" 这类角色前缀。
        # 但实际渲染出来的文本是 "user\n" / "assistant\n"（小写、无冒号），
        # 因为 <|im_start|> 这种特殊标记已经被 skip_special_tokens 去掉了。
        # 带冒号的形式根本不会出现，所以这个循环永远不会命中。

        cleaned_text = text.strip()
        # 去掉首尾空白。

        self.full_response += cleaned_text
        # 累积到完整回答里。

        self.queue.put(cleaned_text)
        # 【★ 关键一步 ★】把这一小片文字放进队列。
        # 这是"跨线程传数据"的动作：本方法跑在生成线程里，
        # 而队列另一头是主线程，主线程正等着取数据去刷新界面。
        # Queue 内部自带锁，所以两个线程同时读写不会出乱子。
        # 详见《前置知识.md》2.6 / 2.7。

    def end(self):
        # 生成结束时，transformers 会自动调用一次这个方法。
        self.queue.put(None)
        # 【哨兵值模式】
        # 往队列里放一个 None，意思是"我说完了"。
        # 为什么不用"队列空了"来表示结束？因为队列空只代表"暂时没数据"，
        # 模型还在算下一个 token 时队列就是空的，主线程不能因此就以为结束了。
        # 所以需要一个明确的"结束信号"，这个信号就叫哨兵值(sentinel)。
        # 详见《前置知识.md》2.8。

    def get_first_token_latency(self):
        # 计算"首 token 延迟"（单位毫秒）。
        if self.first_token_time:
            return (self.first_token_time - self.generation_start) * 1000  # 单位为毫秒
        return 0
        # 【问题】这个函数定义了，但整个文件里【从未被调用过】，是死代码。
        # 而且如前面所说，它测的其实是"prompt 进入 streamer 的时刻"，
        # 不是真正的首 token 延迟。


def generate_response_stream(prompt, queue, max_tokens, temperature):
    # =========================================================================
    # 【★ 这个函数跑在【子线程】里，不是主线程 ★】
    # =========================================================================
    # 它的职责：专心做生成，每生成一点就把文字丢进 queue。
    # 它完全不碰界面 —— 界面由主线程负责。这叫"职责分离"。
    #
    # 参数 queue 是从外面传进来的同一个队列对象：
    #   子线程往里面写  ←→  主线程从里面读
    #   这个队列就是两个线程之间的"传送带"。
    # =========================================================================

    generation_start = time.perf_counter()
    # 记录一个开始时间。
    # 【注意】这个局部变量后面没有被使用（CustomStreamer 里另有一份自己的计时），
    # 属于冗余代码。

    # 准备输入
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        # 【Qwen3 特有的参数】关闭"思考模式"。
        # Qwen3 默认会先输出一段思考过程（包在 <think>...</think> 里）再给答案。
        # 传 False 后，模板会直接给一个【空的】think 块，模型就不真的思考了，直接作答。
        #
        # 实测渲染结果末尾是：
        #   <|im_start|>assistant\n<think>\n\n</think>\n\n
        # 这个"空 think 块"正是上面 put() 里那段清理逻辑能生效的原因。
        # 详见《前置知识.md》3.4。
    )
    # 【注意】这里 messages 依然只有当前这一句 —— 没有传历史。
    # 所以 3.4 的多轮对话仍然是失效的，这是 3.5 要补的功能。

    model_inputs = tokenizer([text], return_tensors="pt")
    # 编码成张量。
    # 注意这里没有再传 add_special_tokens=False（3.3 里传了），
    # 也就是用了默认值 True。因为文本里已经带特殊标记、且分词器通常
    # 不会重复添加 BOS，实际结果一般没差别。
    # 注意这里也没有 .to(model.device)：在 CPU 上没问题，
    # 但如果以后把模型放到显卡上跑，这里会报"设备不一致"的错。

    prompt_tokens = len(model_inputs.input_ids[0])
    # 算一下输入有多少个 token。
    # 【注意】这个变量后面也没有被使用，是冗余代码。

    # 初始化流式输出器
    streamer = CustomStreamer(queue)
    # 创建中间人，把"传送带"交给它。
    # 之后 transformers 每生成一个 token 就会调用 streamer.put(...)。

    # 生成并监控
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_tokens,
        # 来自侧边栏滑块的用户设置。
        temperature=temperature,
        # 采样温度，也来自侧边栏。
        streamer=streamer,
        # 【★ 就是这一行让流式生效 ★】
        # 传进去之后，generate 不再"闷头算完再返回"，
        # 而是每步都把新 token 通过回调交给我们。
        do_sample=True if temperature > 0 else False
        # 温度大于 0 时启用"随机采样"，否则用"贪心解码"（每次选概率最大的词）。
        # 【注意】侧边栏滑块的最小值是 0.1，所以这里实际上永远为 True，
        # 那个 else False 分支永远走不到。
        #
        # 【重要副作用】你一旦传了 temperature，就会覆盖模型自带的采样参数
        # （Qwen3-0.6B 自带 temperature=0.6 / top_k=20 / top_p=0.95）。
        # 滑块默认是 1.0，也就是说默认跑的不是模型推荐的参数。
        # 详见《前置知识.md》3.8。
    )

    full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # 把"输入+输出"整个序列解码出来。
    # 【注意】这里没有像 3.3 那样切掉输入部分（3.3 用了 [len(...):]）。
    # 不过这里的用途只是"在控制台打日志"，不用于显示，所以把 prompt 也打出来无妨。

    # 清理完整响应用于显示
    if '<think>' in full_response and '</think>' in full_response:
        full_response = full_response.split('</think>')[-1]
    for prefix in ['User:', 'Assistant:']:
        if prefix in full_response:
            full_response = full_response.split(prefix, 1)[-1]
    # 同样的清理逻辑，用在"整体日志"上。
    # 同样是多余的（'User:' 前缀不会出现）。

    print(full_response.strip())
    # 打印到【运行 streamlit 的那个终端窗口】，方便你在后台看模型到底输出了什么。
    print('*' * 40)
    # 打印 40 个星号当作分隔线。


# Streamlit 界面

st.title("Qwen3 Chat")
st.write("By：WJW")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []
    # session_state 的作用见《前置知识.md》4.2。

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 侧边栏控制
with st.sidebar:
    # st.sidebar 是页面左侧的收起式面板，用来放配置项。
    # with 语句的含义：在这个块里创建的控件，都放到侧边栏里去。

    st.header("⚙️ 模型配置")

    max_tokens = st.slider("最大生成长度", 100, 2048, 1024)
    # 滑块：标签文字, 最小值, 最大值, 默认值。
    # 返回值是用户当前选中的数字。注意它的范围 100~2048 是安全的，
    # 远小于模型 40960 的上下文上限。

    temperature = st.slider("温度系数", 0.1, 2.0, 1.0, 0.1)
    # 第五个参数 0.1 是"步长"，也就是每次拖动增加 0.1。
    # 【注意】默认 1.0 高于模型推荐的 0.6，会把回答变得更随机。

    if st.button("🧹 清空对话"):
        # 按钮：被点击时返回 True（只在那一次重跑时为 True）。
        st.session_state.messages = []
        st.rerun()
        # 立即重新运行脚本，让清空效果马上反映到界面上。
        # 不调用它的话，界面要等下一次交互才会更新。
        # 详见《前置知识.md》4.1。

# 聊天输入
if prompt := st.chat_input("请输入您的问题..."):
    # 海象运算符 :=，见 3.3 的说明。

    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回复
    with st.chat_message("assistant"):
        placeholder = st.empty()
        # 空占位符，会被反复覆盖写入，实现打字机效果。
        # 详见《前置知识.md》4.5。

        full_response = ""
        queue = Queue()
        # 【★ 创建传送带 ★】
        # 注意它是在这里创建的：每个问题都用一个全新的队列，
        # 避免上一次对话残留的数据混进来。
        # 详见《前置知识.md》2.6。

        # 启动生成线程
        thread = Thread(
            target=generate_response_stream,
            # target = 这个线程要去执行哪个函数。
            # 【重要】这里写的是函数名本身，【没有括号】。
            # 加括号就是"立刻调用它"，而不加括号才是"把这个函数交给线程去调用"。
            # 这是 Python 新手最容易搞错的地方之一。

            args=(prompt, queue, max_tokens, temperature)
            # args = 传给那个函数的参数，必须是元组（所以末尾那个逗号不能省）。
            # 【关键】queue 被传进去了，所以子线程和主线程持有的是同一个队列对象。
            # 这是它们能通信的前提。
            # 详见《前置知识.md》2.4。
        )

        thread.start()
        # 【★ 启动线程 ★】
        # 这一行之后，generate_response_stream 就会在【另一条执行流】里开始跑，
        # 而主线程【立刻】继续往下执行，不会等它。
        # 这就是并发的本质：两件事同时推进。
        # 详见《前置知识.md》2.4。

        # 流式输出
        while True:
            # 主线程进入"收数据 → 刷新界面"的循环。
            # 这整个循环叫"消费者"。
            try:
                chunk = queue.get(timeout=0.1)
                # 从队列取一片文字。
                # timeout=0.1 表示"最多等 0.1 秒"。
                # 如果 0.1 秒内队列一直是空的，就抛出 Empty 异常。
                # 【为什么不干脆死等？】
                # 死等的话，如果子线程那边出了问题，主线程会永远卡住、
                # 界面彻底失去响应。给个超时至少还能周期性醒来。
                # 详见《前置知识.md》2.9。

                if chunk is None:
                    break
                # 收到哨兵值 None = 子线程说"我说完了"，退出循环。
                # 详见《前置知识.md》2.8。

                full_response += chunk
                # 把新收到的一片拼到已有文字后面。

                placeholder.markdown(full_response + "▌")
                # 覆盖写入占位符，末尾加"光标"符号。
                # 【这就是真流式的观感来源】：文字一片片长出来。

            except Empty:
                continue
                # 0.1 秒内没取到数据 —— 正常现象（模型还在算下一个 token）。
                # 什么都不做，回到循环开头再取一次。

        # 清理最终响应
        cleaned_response = full_response.strip()
        placeholder.markdown(cleaned_response)
        # 去掉末尾的光标符号，显示最终干净的结果。

    # 保存到历史记录
    st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
    # 【注意】依然是"生成完成后才写入历史"，所以 3.3 里提到的
    # "生成期间重跑会丢失用户输入"的风险在这里依然存在。


# =============================================================================
# 【本示例小结】
#   相比 3.3 新增：
#     1. CustomStreamer 类 —— 通过回调机制，实时接住每一个生成的 token
#     2. Queue            —— 线程安全的"传送带"，让两个线程交换数据
#     3. Thread           —— 让生成在后台跑，主线程得以同时刷新界面
#     4. 侧边栏滑块       —— 让用户能调生成长度和温度
#     5. enable_thinking  —— 关掉模型的思考过程
#
#   相比 3.3 仍未解决的：
#     - 多轮记忆（messages 里依然只有当前一句）        → 3.5 解决
#     - 生成期间重跑会丢输入                          → 一直没解决
#
#   本文件里可以自己动手改进的点（建议先跑通再改）：
#     ① 删掉无用的 import platform
#     ② 把 put() 里重复的两个分支合并
#     ③ 主动丢弃第一次 put（而不是靠 <think> 清理碰巧挡住 prompt 回显）
#     ④ 给 while 循环加上"子线程异常"的兜底，避免界面永久卡死：
#        若 generate_response_stream 抛异常，end() 就不会被调用，
#        队列永远收不到 None，主循环会无限空转、界面不动也不报错。
# =============================================================================
