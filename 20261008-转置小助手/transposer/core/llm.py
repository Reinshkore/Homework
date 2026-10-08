"""LLM 抽取层（"规则优先、模型补充"中的模型侧）。

    走 OpenAI 兼容接口；key / base / model 均可经 .env 配置。
    未配置 key 时本层不可用，整体自动降级为离线规则模式。
    """
import json
import os

import httpx

from .dotenv import load_env
from .schema import FIELD_LABELS

load_env()


class LLMError(Exception):
    pass


def llm_config():
    return {
        "key": os.environ.get("TRANSPOSER_API_KEY", ""),
        "base": os.environ.get("TRANSPOSER_API_BASE", "https://api.openai.com/v1").rstrip("/"),
        "model": os.environ.get("TRANSPOSER_MODEL", "gpt-4o-mini"),
    }


def llm_available():
    return bool(llm_config()["key"])


def build_prompt(text, template):
    alias_desc = "\n".join(
        f"  {k}（{FIELD_LABELS[k]}）：{' / '.join(template.get('aliases', {}).get(k, []))}"
        for k in FIELD_LABELS
    )
    fields_desc = ", ".join(f"{k}({FIELD_LABELS[k]})" for k in FIELD_LABELS)
    system = (
        "你是入离职信息抽取助手。从邮件文本中为每一位员工抽取一条记录。"
        f"输出字段：{fields_desc}。"
        "move_type 只能是\"入职\"或\"离职\"；effective_date 统一输出为 YYYY-MM-DD；"
        "base 输出为纯数字。原文没有的字段输出 null。"
        "只输出一个 JSON 数组，每个元素是一个员工，不要输出任何解释文字。"
    )
    user = (
        f"客户字段别名参考：\n{alias_desc}\n\n"
        f"邮件文本：\n{text}"
    )
    return system, user


def _parse_llm_json(content):
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    data = json.loads(text)
    if not isinstance(data, list):
        raise LLMError("LLM 返回的不是数组")
    return data


def llm_extract(text, template, timeout=60.0):
    """调用 LLM 抽取，返回 [{field: value}] 列表；失败抛 LLMError。"""
    cfg = llm_config()
    if not cfg["key"]:
        raise LLMError("未配置 TRANSPOSER_API_KEY")
    system, user = build_prompt(text, template)
    resp = httpx.post(
        f"{cfg['base']}/chat/completions",
        headers={"Authorization": f"Bearer {cfg['key']}"},
        json={
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _parse_llm_json(content)
