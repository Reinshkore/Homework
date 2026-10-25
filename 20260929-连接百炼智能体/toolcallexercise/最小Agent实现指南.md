# 从「聊天脚本」到「最小 Agent」——实现指南

> 目标：把 [Runtest.py](file:///c:/Users/xnito/Reishkurs/Gits/Homework/20260929/toolcallexercise/Runtest.py) 从「能聊天的循环」升级为「带**上下文存储**、**工具调用**、**系统提示词**、**Skill 封装**的最小 Agent」。
> 使用方式：**本文只做指导，不改你的代码**。你按示例手打一次，每完成一步发我检查，通过后再进入下一步。
> 本文档会随着你的进度**逐步补齐后续步骤的示例代码**——先把第一步做扎实。

---

## 0. 先看现状：你现在的脚本在做什么、有哪些坑

当前 `Runtest.py` 的结构是：`settings` 配置字典 + 全局 `message_list` + 几个函数 + 一个 `while` 主循环。方向是对的，但有几个**必须先修**的问题，它们会直接挡住后面的功能：

| # | 问题 | 说明 | 影响 |
|---|---|---|---|
| 1 | `new()` 没有真正重置全局变量 | 函数里写 `message_list = []` 只是创建了**局部变量**，不会动全局的那个 | 以为清了历史，其实没清 |
| 2 | `reset_system_prompt()` 键名拼错 | 写成了 `settings["sys_ptompt"]`（少了 `r`） | 这行代码实际没生效 |
| 3 | `message_list.append(response.choices[0].message)` 存的是 **SDK 对象** | 不是纯 dict | 一旦要落盘持久化，`json.dump` 直接报 `TypeError: Object of type ChatCompletionMessage is not JSON serializable` |
| 4 | `api_key` 写在代码里且为空 | 密钥硬编码 | 换成环境变量，避免泄露、也免去每次改代码 |
| 5 | `reset_settings()` 把 `reasoning_effort` 从 `high` 改成 `low` | 与初始 `settings` 不一致 | "重置"语义混乱，容易踩坑 |
| 6 | 请求参数有两个来源 | `settings` 里定义的参数又手写在 `create()` 调用里 | 改一处忘一处 |

> 另外确认一下：`"deepseek-flash"` 若不是你所用服务商的真实模型名，会报 `model not found`。示例里我先用官方的 `deepseek-chat`，你按实际可用模型名调整即可。
> 还有 `reasoning_effort` / `extra_body={"thinking": ...}` 是部分服务商特有的参数，官方 DeepSeek 端点不一定支持。**第一步示例先去掉它们**，让焦点放在上下文上；确认可用后再自行加回。

---

## 1. 总计划：实现顺序与理由

顺序不是随意排的，它由**依赖关系**决定：

| 步骤 | 做什么 | 为什么在这个位置 | 验收标准 |
|---|---|---|---|
| **① 上下文存储** | 修好全局状态；消息只存纯 dict；落盘 `history.json`；加裁剪上限 | **地基**。工具调用会产生「assistant 工具请求 + tool 结果」这类多轮消息，上下文管不好，工具调用必然乱 | 重启脚本后历史还在；超限时旧消息被裁剪、system 始终在首位 |
| **② 工具调用** | 定义工具 Schema → 请求带 `tools` → 解析 `tool_calls` → 执行本地函数 → 把结果回填 → 循环直到不再调用 | 建立在①之上：回填结果靠的是「往同一个上下文里追加消息」 | 问一个需要工具的问题，脚本能自动调用本地函数并给出答案 |
| **③ 系统提示词** | 升级 system prompt：工具使用规则、何时调用/不调用、失败处理、记忆约定、输出风格 | 必须在②之后。提示词里要写「怎么用工具」，工具没定型就无从下笔 | 不需要工具时不乱调；需要时正确调；回答风格符合要求 |
| **④ Skill 封装** | 把「工具定义 + 执行函数 + 使用说明」打包成可复用的 Skill 单元 + 注册表；schema 与提示词片段自动生成 | 这是**抽象/重构**，必须建立在②能跑通的具体实现上，否则是过早抽象 | 新增一个能力只需加一个 Skill 定义，主循环零改动 |

**一句话顺序**：**地基（上下文）→ 能力（工具）→ 规范（提示词）→ 抽象（Skill）**。

### 术语约定（先对齐，避免误解）

- **上下文存储**：对话历史的保存与裁剪，能跨脚本重启保留。
- **工具调用**：模型通过 function calling 让本地代码去执行真实动作（查时间、读文件、算数……）。
- **系统提示词**：约束模型行为的 system 消息，这里特指「如何配合工具与上下文工作」的规则。
- **Skill**：把一项能力**打包**成「名字 + 描述 + 参数 + 执行函数 + 用法说明」的可复用单元。

> ⚠️ **需要你确认一件事**：这里的「skill」我理解为**本地能力封装**（类似《深入理解 AI Agent》第 2 章的 Agent Skills 思路）。
> 如果你课程里说的 skill 指的是**百炼平台的 Skill 资源**（`POST /skills` 那套），那是完全不同的另一条路——请告诉我，我改为按平台 API 写。

---

## 2. 第一步（现在做这个）：上下文存储

### 2.1 这一步要做 4 件事

1. **修好全局状态**：所有要改 `message_list` 的函数，显式声明 `global`（或用 `.clear()`）。
2. **消息只存纯 dict**：不再直接 append SDK 返回的对象，而是取 `role` / `content` 重新组装成 dict。
3. **落盘 + 加载**：写 `history.json`，启动时读回。
4. **裁剪上限**：只保留 system + 最近 N 轮，防止上下文无限膨胀（也顺势省 token）。

### 2.2 示例代码（完整可运行，请手打一遍）

```python
import json
import os

from openai import OpenAI

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")
MAX_TURNS = 20  # 除 system 外，最多保留的历史轮数

settings = {
    "base_url": "https://api.deepseek.com",
    "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),  # 密钥走环境变量
    "model": "deepseek-chat",  # 以你实际可用的模型名为准
    "stream": False,
    "sys_prompt": "当前用户是一名学生，正在学习调用模型接口。仅根据用户要求进行回应，不需要自行扩展。",
}

message_list = []  # 全局上下文：始终以 system 消息开头


# ---------- 上下文存储 ----------
def load_history():
    """启动时读取历史；文件不存在或损坏则重新初始化。"""
    global message_list
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                message_list = json.load(f)
        except (json.JSONDecodeError, OSError):
            message_list = []
    # 保证首条永远是 system
    if not message_list or message_list[0].get("role") != "system":
        message_list.insert(0, {"role": "system", "content": settings["sys_prompt"]})


def save_history():
    """把当前上下文写回磁盘。"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(message_list, f, ensure_ascii=False, indent=2)


def trim_history():
    """保留 system + 最近 MAX_TURNS 轮，防止上下文无限增长。"""
    global message_list
    system_msg = message_list[0]
    rest = message_list[1:]
    user_positions = [i for i, m in enumerate(rest) if m.get("role") == "user"]
    if len(user_positions) > MAX_TURNS:
        rest = rest[user_positions[-MAX_TURNS]:]  # 从第 N 近的 user 消息开始截断
    message_list = [system_msg] + rest


def new_session():
    """开启新会话：清空历史并落盘。"""
    global message_list
    message_list = [{"role": "system", "content": settings["sys_prompt"]}]
    save_history()


# ---------- 对话 ----------
def text_in():
    """读取一行输入，返回状态给主循环。"""
    user_input = input("You: ").strip()
    if user_input in ("end", "/exit"):
        return "end"
    if user_input == "/new":
        new_session()
        print("(已开启新会话)")
        return "skip"
    if not user_input:
        return "skip"
    message_list.append({"role": "user", "content": user_input})
    return "ok"


def request_up():
    """发一轮请求，并把回复以纯 dict 形式追加进上下文。"""
    trim_history()
    response = client.chat.completions.create(
        model=settings["model"],
        messages=message_list,
        stream=False,
    )
    reply = response.choices[0].message
    print(f"AI: {reply.content}")
    # 关键：只存纯 dict，才能被 json 序列化
    message_list.append({"role": "assistant", "content": reply.content})
    save_history()


# ---------- 主流程 ----------
if __name__ == "__main__":
    if not settings["api_key"]:
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")

    client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"])

    load_history()
    print(f"(已加载 {len(message_list) - 1} 条历史消息；end 退出，/new 开新会话)")
    print("-" * 40)

    while True:
        state = text_in()
        if state == "end":
            break
        if state == "skip":
            continue
        try:
            request_up()
        except Exception as e:
            print(f"[请求出错] {e}")

    print("Run ends.")
```

### 2.3 与旧版的对照（重点看这几处）

| 旧写法 | 新写法 | 为什么 |
|---|---|---|
| `def new(): message_list=[]` | `def new_session(): global message_list; message_list=[...]` | 不声明 `global` 改不到全局 |
| `append(response.choices[0].message)` | `append({"role":"assistant","content":reply.content})` | SDK 对象不能序列化 |
| `api_key=""` 写死在代码里 | `os.environ.get("DEEPSEEK_API_KEY")` | 密钥不落代码 |
| 无持久化 | `load_history()` / `save_history()` | 跨重启保留上下文 |
| 无上限 | `trim_history()` | 防止上下文无限增长 |

### 2.4 自测清单（做完请逐条验证）

- [ ] 能正常跑起来并对话，无报错。
- [ ] 目录里生成了 `history.json`，内容是「system + 你的对话」的 JSON 数组。
- [ ] **退出后重新运行**，问「我刚才说了什么？」，模型能答出来 → 说明加载生效。
- [ ] 连聊 21 轮以上，打开 `history.json` 检查：最老的消息被裁掉，但 **第一条 system 仍在**。
- [ ] 输入 `/new` 后历史被清空（只剩 system）。

### 2.5 常见坑

1. **忘写 `global`** → 现象是"明明清空了，重启又回来了"。
2. **存了 SDK 对象** → `json.dump` 报 `TypeError: Object of type ChatCompletionMessage is not JSON serializable`。
3. **`content` 为 `None`** → 目前不会出现；但到第二步工具调用时，带 `tool_calls` 的 assistant 消息 `content` 会是 `None`，需要单独处理（第二步会讲）。
4. **裁剪时把 system 一起裁掉** → 角色设定丢失，所以 `trim_history()` 里显式把 system 摘出来保护。
5. **失败的请求不落盘** → 这是有意为之：只有拿到回复的完整轮次才 `save_history()`，避免上下文里留下"只有提问没有回答"的残缺记录。

---

## 3. 后续步骤（占位，等第一步通过后补齐）

- **第二步 · 工具调用**：`tools` 参数、`tool_calls` 解析、执行本地函数、`{"role":"tool","tool_call_id":...}` 回填、多轮循环、`content=None` 的处理。
- **第三步 · 系统提示词**：把工具使用规则、调用时机、失败兜底、记忆约定写进 system prompt。
- **第四步 · Skill 封装**：Skill 数据结构 + 注册表 + 自动生成 tools schema 与提示词片段，主循环零改动即可扩展能力。

> 完成第一步并自查通过后，发我检查。确认无误我再把第二步的详细指导与示例代码补进本文档。
