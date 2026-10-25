"""路径与环境变量的唯一入口。

别的模块不直接读 os.environ、不自己拼路径 —— 改了目录只要动这一个文件，
这也是 demo5 里 history.py 反向 import agent.py 那种依赖要拆掉的原因。

档案（archive）是一套互相隔离的运行态：证据库、会话、检查点、账本、简报各一份。
所以从这一版起，**凡是与运行态有关的路径都必须带档案 id**，不再有模块级常量。
原来那些常量（STORE_DB / LEDGER_DIR）已经删掉：留着会让漏改的地方静默写到
default 档案里，这种错误在界面上看不出来，只会表现为"切了档案但数据没变"。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"            # 旧版会话/检查点目录，迁移后仅作兼容（scripts/migrate_archives.py）
RUNTIME_DIR = ROOT / "runtime"      # 证据库等运行时状态（不追踪）
OUTPUTS_DIR = ROOT / "outputs"      # 简报产出，Agent 唯一的常规写入区
SKILLS_DIR = ROOT / "skills"        # 技能目录
MEMORY_DIR = ROOT / "memory"        # 长期记忆
DOCS_DIR = ROOT / "docs"            # 工作文档（Agent 只读）
BRIEFS_DIR = OUTPUTS_DIR / "briefs"  # 简报根目录，下面按档案分子目录

ARCHIVES_DIR = RUNTIME_DIR / "archives"
DEFAULT_ARCHIVE = "default"

load_dotenv(ROOT / ".env")


def get(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    return value.strip() if value and value.strip() else default


def require(key: str) -> str:
    value = get(key)
    if not value:
        raise RuntimeError(f"缺少环境变量 {key}，请检查 .env（模板见 .env.example）")
    return value


# ------------------------------------------------------------------ 档案路径
def archive_dir(archive_id: str = DEFAULT_ARCHIVE) -> Path:
    return ARCHIVES_DIR / (archive_id or DEFAULT_ARCHIVE)


def store_db(archive_id: str = DEFAULT_ARCHIVE) -> Path:
    """证据库。物理在 /runtime/ 下，Agent 的虚拟文件系统对它禁读禁写——只能走 MCP 工具。"""
    return archive_dir(archive_id) / "store.db"


def sessions_db(archive_id: str = DEFAULT_ARCHIVE) -> Path:
    return archive_dir(archive_id) / "sessions.db"


def checkpoints_db(archive_id: str = DEFAULT_ARCHIVE) -> Path:
    return archive_dir(archive_id) / "checkpoints.db"


def archive_meta(archive_id: str = DEFAULT_ARCHIVE) -> Path:
    return archive_dir(archive_id) / "meta.json"


def archive_memory(archive_id: str = DEFAULT_ARCHIVE) -> Path:
    """档案补充记忆。放在 memory/ 而不是 runtime/：Agent 对 runtime 禁读，
    放进去它也读不到自己的记忆文件。全局口径仍是 memory/AGENTS.md。"""
    return MEMORY_DIR / "archives" / f"{archive_id}.md"


def briefs_dir(archive_id: str = DEFAULT_ARCHIVE) -> Path:
    """简报要能被 Agent 看见并写（/outputs/** 是放行区），所以按档案分子目录，不挪进 runtime。"""
    return BRIEFS_DIR / (archive_id or DEFAULT_ARCHIVE)


def ledger_dir(archive_id: str = DEFAULT_ARCHIVE) -> Path:
    return DOCS_DIR / "ledger" / (archive_id or DEFAULT_ARCHIVE)


def ledger_file(archive_id: str = DEFAULT_ARCHIVE) -> Path:
    return ledger_dir(archive_id) / ".ai-brief-ledger.md"


def ledger_virtual_path(archive_id: str = DEFAULT_ARCHIVE) -> str:
    """Agent 侧看到的路径。权限表按 /docs/ledger/** 放行，子目录一并覆盖。"""
    return f"/docs/ledger/{archive_id or DEFAULT_ARCHIVE}/.ai-brief-ledger.md"


def ensure_dirs(archive_id: str = DEFAULT_ARCHIVE) -> None:
    for d in (ARCHIVES_DIR, archive_dir(archive_id), briefs_dir(archive_id),
              ledger_dir(archive_id), MEMORY_DIR / "archives"):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ 模型
LLM_MODEL = get("LLM_MODEL", "qwen3.7-plus")
SUBAGENT_MODEL = get("SUBAGENT_MODEL")

# ------------------------------------------------------------------ 检索
SEARCH_PROVIDER = get("SEARCH_PROVIDER", "deepseek")
SEARCH_FALLBACK = get("SEARCH_FALLBACK") or None
SEARCH_QUOTA = int(get("SEARCH_QUOTA") or 24)


def search_key(provider: str) -> str | None:
    """按 provider 取密钥。返回 None 表示该源未配置。"""
    env_name = {
        "deepseek": "DEEPSEEK_API_KEY",
        "bocha": "BOCHA_API_KEY",
        "zhipu": "ZHIPU_API_KEY",
        "tavily": "TAVILY_API_KEY",
    }.get(provider)
    return get(env_name) if env_name else None


def refresh() -> None:
    """重读可改的那几个环境变量。

    模块级常量只在 import 时算一次，而设置页会在运行中改 `.env`。
    不调这个函数，界面显示"已保存"、实际进程里还是旧值——最难查的一类问题。
    """

    global LLM_MODEL, SUBAGENT_MODEL, SEARCH_PROVIDER, SEARCH_FALLBACK, SEARCH_QUOTA
    LLM_MODEL = get("LLM_MODEL", "qwen3.7-plus")
    SUBAGENT_MODEL = get("SUBAGENT_MODEL")
    SEARCH_PROVIDER = get("SEARCH_PROVIDER", "deepseek")
    SEARCH_FALLBACK = get("SEARCH_FALLBACK") or None
    SEARCH_QUOTA = int(get("SEARCH_QUOTA") or 24)
