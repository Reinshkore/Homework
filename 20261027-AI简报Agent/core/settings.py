"""模型清单与运行设置。

模型是**写死的三个**，不是"填任意模型名"。理由：Agent 能不能跑完一期，
取决于它对工具调用格式的稳定性，不是参数规模。清单外的名字写错时，
报错会出现在第 40 次工具调用上——那时已经花了十分钟和几毛钱。
写死三个、每个都实测跑通过，比"支持任意模型"诚实。

服务地址也写死在这里，不放进 `.env`：同一家的 key 只对应一个网关，
让用户配 base_url 只会多一个"填错了但看起来能连"的失败模式。

三家的差异只体现在 `base_url` 与 `key_env` 两个字段，装配层不需要知道厂商是谁。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from core import config

DEFAULT_MODEL = "qwen3.7-plus"

MODELS: dict[str, dict] = {
    "qwen3.7-plus": {
        "label": "Qwen3.7-plus",
        "vendor": "阿里百炼",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "DASHSCOPE_API_KEY",
        "note": "默认。工具调用稳定，中英混合检索表现好",
    },
    "qwen3.8-max": {
        "label": "Qwen3.8-max",
        "vendor": "阿里百炼",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "DASHSCOPE_API_KEY",
        "note": "同百炼。推理更强，长报告的结构更稳，单价更高",
    },
    "deepseek-flash": {
        "label": "deepseek-flash",
        "vendor": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
        "note": "快档。工具调用链更长时更易跳步，跑完一期前建议先小样本试",
    },
}

# 设置页里允许改的键（其余一律不暴露）。值是给人看的说明。
EDITABLE_ENV = {
    "DASHSCOPE_API_KEY": "阿里百炼密钥（Qwen 两个模型都用它）",
    "DEEPSEEK_API_KEY": "DeepSeek 密钥（deepseek-flash 与检索共用）",
    "BOCHA_API_KEY": "博查密钥（中文检索降级备用）",
    "SEARCH_PROVIDER": "主检索源",
    "SEARCH_FALLBACK": "中文检索的降级源，留空则不降级",
    "SEARCH_QUOTA": "每期检索次数上限",
}

SECRET_KEYS = {"DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "BOCHA_API_KEY", "ZHIPU_API_KEY", "TAVILY_API_KEY"}


# ------------------------------------------------------------------ 运行设置
def settings_path() -> Path:
    return config.RUNTIME_DIR / "settings.json"


def load() -> dict:
    path = settings_path()
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001  设置写坏了要能自愈，不能把界面卡死
            data = {}
    model = data.get("model")
    if model not in MODELS:
        model = config.get("LLM_MODEL") if config.get("LLM_MODEL") in MODELS else DEFAULT_MODEL
    return {"model": model}


def save(**changes) -> dict:
    data = {**load(), **changes}
    if data.get("model") not in MODELS:
        data["model"] = DEFAULT_MODEL
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def current_model() -> str:
    return load()["model"]


def set_model(model_id: str) -> dict:
    if model_id not in MODELS:
        raise RuntimeError(f"未知模型 {model_id}，可选：{'、'.join(MODELS)}")
    return save(model=model_id)


def spec(model_id: str | None = None) -> dict:
    return MODELS.get(model_id or current_model()) or MODELS[DEFAULT_MODEL]


def model_ready(model_id: str | None = None) -> tuple[bool, str]:
    """模型能不能用：只看它对应的那个 key 在不在。"""

    s = spec(model_id)
    key = config.get(s["key_env"])
    if not key:
        return False, f"缺 {s['key_env']}"
    return True, ""


def probe(model_id: str | None = None) -> dict:
    """真发一次最小请求，验证 key 与网关。

    为什么需要它：`model_ready` 只查"键在不在"，查不出"键填错了"。
    实测过一次——百炼那一项被填成了 DeepSeek 的 key，长度、前缀全都像，
    只有真发请求才会 401。等到一期跑到第一次模型调用才报错，代价是十分钟。
    """

    import httpx

    mid = model_id or current_model()
    s = spec(mid)
    key = config.get(s["key_env"])
    if not key:
        return {"ok": False, "detail": f"缺 {s['key_env']}"}

    url = f"{s['base_url'].rstrip('/')}/chat/completions"
    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": mid,
                  "messages": [{"role": "user", "content": "只回复两个字：可用"}],
                  "max_tokens": 16},
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}

    if resp.status_code != 200:
        return {"ok": False, "status": resp.status_code, "detail": resp.text[:300]}
    try:
        text = resp.json()["choices"][0]["message"]["content"] or ""
    except Exception:  # noqa: BLE001
        text = ""
    return {"ok": True, "status": 200, "detail": text.strip()[:40]}


# ------------------------------------------------------------------ .env 读写
def env_path() -> Path:
    return config.ROOT / ".env"


def read_env() -> dict[str, str]:
    path = env_path()
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def mask(value: str) -> str:
    """只用于界面回显：`sk-…a1b2`。绝不把完整密钥写进页面。"""

    value = (value or "").strip()
    if not value:
        return "未配置"
    if len(value) <= 8:
        return "已配置（过短，请检查）"
    return f"{value[:3]}…{value[-4:]}"


def update_env(changes: dict[str, str]) -> list[str]:
    """改写 `.env` 里指定的键，**保留注释、顺序与其它键**。

    返回改动的键名。为什么不整份重写：`.env` 里有注释说明每个键的用途，
    重写会把这些说明冲掉，下次谁都不知道 `SEARCH_FALLBACK` 为什么存在。
    """

    path = env_path()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pending = {k: v for k, v in changes.items() if v is not None}
    changed: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.partition("=")[0].strip()
        if key in pending:
            new = f"{key}={pending.pop(key)}"
            if lines[i] != new:
                lines[i] = new
                changed.append(key)

    for key, value in pending.items():
        lines.append(f"{key}={value}")
        changed.append(key)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def reload_env() -> None:
    """把 `.env` 的新值灌回进程环境。

    `load_dotenv` 默认不覆盖已有变量，所以必须 override：否则"改了 key 但没生效"，
    而界面上又显示保存成功——这是最难查的一类问题。
    """

    from dotenv import load_dotenv

    load_dotenv(env_path(), override=True)
    config.refresh()


def clear_env_keys(keys: list[str]) -> None:
    for key in keys:
        os.environ.pop(key, None)
