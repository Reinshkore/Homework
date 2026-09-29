# -*- coding: utf-8 -*-
# 零依赖版：只用 Python 标准库，不 import openai
#
# 适用场景：手机端装不了 openai 库（pip 不可用 / 联网受限）时使用。
# 直连 ollama 原生接口 POST /api/chat，功能与 openai 库写法等价。
#
# 相比 OpenAI 兼容接口的一个额外好处：
#   原生接口支持 think 布尔值，可以彻底关掉思维链；
#   而 OpenAI 兼容接口的 reasoning_effort 只能调级别（low/medium/high），关不掉。

import json
import urllib.request

BASE = "http://10.201.208.97:11434"      # 不要带 /v1，本脚本走原生接口
                                         # 该 IP 由路由器 DHCP 动态分配，连接失败时先在笔记本上执行 ipconfig 查当前 WLAN 的 IPv4 地址并更新这里
MODEL = "qwen3:0.6b"
THINK = False                            # False 关闭思维链；True 或 "low"/"medium"/"high" 开启
TIMEOUT = 300                            # 首次调用要加载模型，留足时间

SYS_PROMPT_DEFAULT = "当前用户是一名学生，正在学习调用模型接口。仅根据用户要求进行回应，不需要自行扩展。"

settings = {
    "base_url": BASE,
    "model": MODEL,
    "think": THINK,
    "stream": False,
    "sys_prompt": SYS_PROMPT_DEFAULT
}

message_list = []


def reset_settings():
    settings["base_url"] = BASE
    settings["model"] = MODEL
    settings["think"] = THINK
    settings["stream"] = False
    return 0


def new():
    message_list.clear()
    message_list.append({"role": "system", "content": settings["sys_prompt"]})
    return 0


def reset_system_prompt():
    settings["sys_prompt"] = SYS_PROMPT_DEFAULT
    return 0


def text_in():
    user_input = input("Please text in:")
    if user_input == "end":
        return "end"
    message_list.append({"role": "user", "content": user_input})
    return 0


def request_up():
    payload = {
        "model": settings["model"],
        "messages": message_list,
        "think": settings["think"],
        "stream": settings["stream"]
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        settings["base_url"] + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    content = result["message"]["content"]
    print(content)
    message_list.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    try:
        reset_settings()
        reset_system_prompt()
        new()
        while True:
            if text_in() == "end":
                break
            request_up()
    except Exception as e:
        print("Error Info:{0}".format(e))

    print("Run ends.")
