# =============================================================================
# 示例 3.3  ——  最基础的版本：同步阻塞 + "假装"流式
# =============================================================================
#
# 【这个例子在整条学习路线里的位置】
#   3.3 → 3.4 → 3.5 是三个递进的版本，不是三个并列的例子：
#     3.3（本文件）：能跑通的最简版。用"等全部生成完，再一点点显示"来假装流式。
#     3.4          ：引入「线程 + 队列」，实现真正的边生成边显示。
#     3.5          ：在 3.4 基础上加入「多轮记忆」。
#
#   所以这个文件的真正作用是：让你先看清"不这样做会有什么问题"。
#
# 【前置知识】
#   读这份代码需要：装饰器、函数、类（本文件只用到前两个）。
#   详细讲解见同目录下《前置知识.md》第 1 章。
#
# 【环境说明 —— 重要】
#   下面 load_model() 里的路径 "../models/Qwen3-0.6B" 是教程作者机器上的相对路径。
#   你自己机器上的实际路径是：
#       C:\Users\xnito\Reishkurs\LLM\models\Qwen--Qwen3-0.6B\snapshots\master
#   注意两点：
#     (1) 目录名是 "Qwen--Qwen3-0.6B"（中间有两个连字符），不是 "Qwen3-0.6B"；
#     (2) 真正的模型文件在更深的 snapshots\master 这一层里。
#   如果路径不对，会报 "HFValidationError: Repo id must be in the form ..."，
#   这个报错信息有误导性 —— 它的真实含义是"本地找不到这个路径"。
# =============================================================================

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 关于上面的两行 import -----------------------------------------------
# streamlit：负责网页界面（输入框、聊天气泡、按钮）。
# transformers：HuggingFace 的模型库，负责加载模型和分词器。
# 注意 AutoModelForCausalLM / AutoTokenizer 是"自动类"：
#   你只给一个路径，它会自己去读目录里的 config.json，
#   然后决定该实例化哪个具体的类（这里是 Qwen3ForCausalLM）。
#   CausalLM = Causal Language Model = 因果语言模型，即"根据前文预测下一个词"。


# 初始化模型和分词器
@st.cache_resource
def load_model():
    # --- @st.cache_resource 是什么（本文件第一个难点）----------------------
    # 它是一个"装饰器"，作用是：把这个函数的结果缓存起来。
    #
    # 为什么必须缓存？因为 Streamlit 的一个核心特性是：
    #   用户在网页上每做一次交互（打字、点按钮），整个脚本就会【从头重新执行一遍】。
    # 如果不加这个装饰器，每次交互都要重新加载 1.5GB 的模型权重 —— 根本没法用。
    # 加上之后，Streamlit 发现"这个函数参数没变"，就直接把上次的模型对象还给你。
    #
    # _resource 这个后缀的意思是：缓存的是"资源类对象"（模型、数据库连接），
    # 而不是普通数据。区别在于它不会做序列化拷贝，多个用户会话共享同一个对象。
    # 详细讲解见《前置知识.md》4.3。

    model_name = "../models/Qwen3-0.6B"
    # 相对路径。相对于"你执行 streamlit run 命令时所在的目录"，不是相对于这个文件。
    # 这是新手最容易踩的坑之一，详见文件头的【环境说明】。

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 分词器：负责"文字 ↔ 数字编号"的双向转换。
    #   文字 → 一串整数（叫 token id），模型只认整数。
    #   一串整数 → 文字，叫"解码"。
    # trust_remote_code=True：允许加载模型目录里自带的 Python 代码。
    #   有些新模型（包括 Qwen3）的预处理逻辑不在 transformers 库里，
    #   而是随模型一起下载。不开这个开关就会报错。
    # 详见《前置知识.md》3.1 / 3.2。

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    # 加载模型权重本体。这一步很慢（要读 1.5GB 文件），所以必须靠 cache_resource 缓存。
    # 注意这里没有指定 dtype（数据类型）。模型 config.json 里写的是 bfloat16，
    # 但在 CPU 上加载时，显式写 dtype=torch.bfloat16 更稳妥。

    return tokenizer, model
    # 返回一个元组（tuple）。下面第 10 行用两个变量同时接住它，这叫"解包"。


tokenizer, model = load_model()
# 这行在脚本顶层，意味着【每次脚本重跑都会执行】。
# 但因为 load_model 被 cache_resource 装饰了，只有第一次真加载，之后都是秒返回。
# 这是"把昂贵操作缓存起来"的典型用法。


# 生成回复的函数
def generate_response(prompt):
    # 输入：用户的问题（字符串）。输出：模型的回答（字符串）。
    # 这个函数是【同步阻塞】的 —— 调用它之后，程序会一直卡在这里，
    # 直到模型把整段回答全部生成完才返回。这就是 3.4 要解决的问题。

    messages = [{"role": "user", "content": prompt}]
    # 构造"对话消息列表"。这是大模型领域的事实标准格式：
    #   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    # role 有两种常用值：user（用户说的）/ assistant（模型说的）。
    #
    # 【注意】这里列表里【永远只有一条】消息 —— 也就是当前这一句。
    # 界面上虽然显示出了历史对话，但那些历史【从来没有传给模型】。
    # 结果：模型完全不知道之前聊过什么，多轮对话实际是失效的。
    # 这正是 3.5 要修复的问题（见示例3.5.py 里的 history 参数）。

    text = tokenizer.apply_chat_template(
        messages,              # 上面构造的消息列表
        tokenize=False,        # False = 只拼成字符串，先不要转成数字
        add_generation_prompt=True
        # True = 在字符串末尾追加 "轮到 assistant 说话了" 的提示标记。
        # 不追加的话，模型不知道该"继续说"还是"已经说完了"。
    )
    # apply_chat_template 的作用：把消息列表"套用"模型规定的模板，拼成一整段字符串。
    # 每个模型的模板都不一样。Qwen3 拼出来大概长这样：
    #   <|im_start|>user
    #   请概括一下Qwen3的优点。<|im_end|>
    #   <|im_start|>assistant
    # 其中 <|im_start|> / <|im_end|> 是"特殊标记"，用来划分说话人边界。
    # 所以这一步 = 把你的话翻译成模型的母语格式。
    # 详见《前置知识.md》3.4。

    model_inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)
    # 这一行做了四件事，拆开看：
    #   tokenizer([text])         把字符串编码成 token id
    #   add_special_tokens=False  不要在开头额外加 BOS（起始标记）。
    #                             因为 apply_chat_template 已经加过必要的标记了，再加就重复。
    #   return_tensors="pt"       返回 PyTorch 张量（pt = PyTorch），而不是 Python 列表。
    #                             模型只接受张量。
    #   .to(model.device)         把数据搬到模型所在的设备上。
    #                             这里是 CPU，所以其实没搬动；但如果模型跑在显卡上，
    #                             这一步就是必须的（CPU 和 GPU 内存是分开的）。
    #
    # 注意 [text] 外面套了方括号 —— 加一维，表示"一批里有 1 条"。
    # 模型总是按"批"处理，哪怕只有一条。
    # 详见《前置知识.md》3.5。

    generated_ids = model.generate(
        model_inputs.input_ids,   # 输入的数字序列
        max_new_tokens=1024
        # 最多新生成 1024 个 token。这是个【上限】，不是目标值。
        # 模型正常说完会自己吐出一个"结束标记"(EOS)并停下；
        # 只有它停不下来时，这个上限才会兜住它。
        # 注意：输入长度 + max_new_tokens 不能超过模型的上下文上限
        # （Qwen3-0.6B 是 40960）。设太大不会提速，只会让异常情况更糟。
    )
    # 这是【整个程序最慢的一行】，CPU 上可能要几十秒。
    # 它会阻塞当前线程，期间什么都做不了 —— 这是 3.3 最核心的缺陷。
    # 详见《前置知识.md》3.6。

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    # 这行看起来绕，实际是在"切掉输入部分，只留新生成的"。拆开看：
    #   generated_ids              形状是 (1, 输入长度+新生成长度)，即"一批 1 条"
    #   generated_ids[0]           取第 0 条，去掉"批"这一维
    #   len(model_inputs.input_ids[0])  输入的长度，比如 16
    #   [16:]                      从第 16 个往后切，正好丢掉输入、留下新生成的
    #   .tolist()                  从张量转回 Python 列表（后面 decode 需要）
    #
    # 【为什么要切？】因为 model.generate 返回的是"输入 + 输出"拼在一起的完整序列，
    # 不切的话你会把用户自己问的问题也解码出来重复显示一遍。
    # 详见《前置知识.md》3.6。

    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    # 解码：把数字变回文字。
    # skip_special_tokens=True：丢掉 <|im_start|>、<|im_end|> 这类控制标记，
    #   否则它们会混在回答里显示出来。
    # .strip("\n")：去掉首尾多余的换行。
    # 详见《前置知识.md》3.2。

    return response


# Streamlit 用户界面

st.title("Qwen3 Chat")
# 在页面上显示一个大标题。

st.write("By：WJW")
# 显示一行普通文字。

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []
    # st.session_state 是一个"跨重跑存活的字典"。
    # 前面说过脚本每次交互都从头重跑，普通变量会被清空；
    # session_state 里的东西则会保留下来。
    # 这里判断"如果还没有 messages 这个键，就新建一个空列表"，
    # 保证第一次进入页面时不会因为"键不存在"而报错。
    # 详见《前置知识.md》4.2。

# 从历史中显示聊天消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    # 遍历历史，一条条画成聊天气泡。
    #   with st.chat_message("user")   开启一个"用户气泡"的容器
    #   st.markdown(...)               往这个容器里放内容（markdown 支持 **加粗** 等格式）
    # 因为每次都重画全部历史，所以刷新页面后对话记录还在。

# 接受用户输入
if prompt := st.chat_input("你想问点什么？"):
    # 这里用了海象运算符 :=（Python 3.8+）。
    # 它的含义同时包含两件事：把 st.chat_input() 的返回值赋给 prompt，并且判断它是否非空。
    # 等价于：
    #   prompt = st.chat_input("你想问点什么？")
    #   if prompt:
    #   好处是少写一行，且不用把调用写两遍。
    # st.chat_input 是页面底部的输入框。用户没提交时返回 None，提交了返回用户输入的文字。
    # 所以这个 if 块只在"用户刚提交了消息"的这一次脚本运行中成立。
    # 详见《前置知识.md》4.4。

    # 将用户消息添加到聊天历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 先把用户这句话存进历史（注意是把"当前的 prompt"存进去，但 generate_response
    # 里并没有用这个历史，这是本版本的缺陷）。

    # 在聊天消息容器中显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    # 把用户刚说的话显示成一个气泡。
    # 小疑问：下面重画历史时不是已经画过一次了吗？为什么这里要再画一次？
    # 答案：因为这次脚本运行时，上面那个 for 循环是在我们 append 之前执行的，
    #       历史列表里还没有这一条。所以这里需要补画。

    # 在聊天消息容器中显示助手回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # st.empty() 创建一个"空占位符"，后面可以反复往里面覆盖写内容。
        # 这是做"打字机效果"的关键：同一个位置不断被新的、更长的文本覆盖。
        # 详见《前置知识.md》4.5。

        full_response = ""
        # 累积已显示的文字。

        # 生成回复
        assistant_response = generate_response(prompt)
        # 【★ 这里是 3.3 最大的问题所在 ★】
        # generate_response 是同步阻塞的，这一行执行时程序会一直等，
        # 等模型把整段回答全部生成完，才把完整字符串交回来。
        # 也就是说：用户会先干等几十秒（屏幕上什么都不动），然后才看到文字出现。
        # 这就叫"假流式"——它在时间上并没有省，只是把等待挪到了打字动画之前。
        # 3.4 用「线程 + 队列」来解决这个问题。

        # 模拟流式回复效果
        for chunk in assistant_response.split():
            # 【★ 第二个问题 ★】
            # .split() 不带参数时，按"空白字符"（空格、换行）切分。
            # 但中文句子之间【基本没有空格】！
            # 比如 "杭州是中国浙江省的省会。" 用 .split() 切出来是：
            #   ["杭州是中国浙江省的省会。"]   ← 只有 1 个元素
            # 所以对中文输出而言，这个 for 循环只循环 1 次，
            # 打字机效果【根本不会出现】。
            # 更糟的是下面还硬加了一个空格，会把中文句子里塞进多余空格。

            full_response += chunk + " "
            # 累加。(注意 += 在循环里累加字符串效率较低，长文本会明显变慢。)

            message_placeholder.markdown(full_response + "▌")
            # 把当前累积的文字写进占位符，末尾加一个 "▌" 假装光标。

        message_placeholder.markdown(full_response)
        # 循环结束后，把末尾的 "▌" 去掉，显示最终结果。

    # 将助手回复添加到聊天历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # 【★ 第三个问题 ★】
    # 助手回复是在【生成完成之后】才写入 session_state 的。
    # 如果在生成过程中用户刷新了页面、或者发生了任何导致脚本重跑的事件，
    # 这段代码会从 st.chat_input 重新执行 —— 而此时输入框已经空了，
    # prompt 是 None，整个 if 块被跳过，刚问的问题和正在生成的答案一起消失。
    # 稳妥的做法是"用户消息立即入库、助手消息生成完立即入库"。
    #
    # 另外注意 full_response 的缩进层级：它是在 with 块里定义的，
    # 但 Python 的 with 不产生新作用域，所以这里仍然能访问到。


# =============================================================================
# 【本示例小结】
#   已经做到的：能加载本地模型、能对话、界面能显示历史。
#   三个缺陷（这正是后面两个示例要逐个解决的）：
#     1. 同步阻塞 → 回答前要干等，且页面完全卡住。   → 3.4 用线程解决
#     2. 假流式 → split() 对中文无效，打字机效果失效。→ 3.4 用真流式解决
#     3. 无记忆 → 只传当前一句，多轮对话失效。      → 3.5 用 history 解决
# =============================================================================
