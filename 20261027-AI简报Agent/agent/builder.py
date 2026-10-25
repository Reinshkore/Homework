"""Agent 装配。

把 Deep Agents 的积木拼在一起：

    模型      阿里百炼，OpenAI 兼容接口
    工具      tools/registry.py 的进程内工具 + mcp_servers.json 的 MCP 工具
    技能      /skills/，渐进式披露（只先注入 name + description）
    记忆      /memory/AGENTS.md（全局口径）+ /memory/archives/<id>.md（档案补充），每轮注入
    子代理    critic
    后端      FilesystemBackend，Agent 看到的 "/" 就是仓库根目录
    权限      只允许写 /outputs/** 与 /docs/ledger/**，其余全拒
    检查点    AsyncSqliteSaver，落在 runtime/archives/<id>/checkpoints.db

界面层（app.py）只消费 build_agent(archive_id) 与 events.stream_events()，
不含任何 Agent 知识。

**档案靠一个参数贯穿**：证据库（MCP 子进程的 --archive）、检查点、补充记忆。
漏传任何一处，切档案后都会静默写回 default，表现为"切了但数据没变"。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.prompts import SYSTEM_PROMPT
from agent.subagents import build_subagents
from core import archive, config, settings
from tools.registry import PY_TOOLS


# ------------------------------------------------------------------ 模型
def build_model(model_name: str | None = None) -> ChatOpenAI:
    """按模型清单取 base_url 与密钥。

    清单之外的模型名直接拒绝：写错一个不存在的名字，报错会出现在第 40 次工具调用上，
    那时十分钟和几毛钱已经花掉了。宁可启动时就失败。
    """

    mid = model_name or settings.current_model()
    if mid not in settings.MODELS:
        raise RuntimeError(f"未知模型 {mid}；可选：{'、'.join(settings.MODELS)}")

    spec = settings.MODELS[mid]
    api_key = config.get(spec["key_env"])
    if not api_key:
        raise RuntimeError(
            f"模型 {mid} 需要 {spec['key_env']}，当前未配置。到「设置」页填写，或改选别的模型。"
        )

    return ChatOpenAI(
        model=mid,
        api_key=api_key,
        base_url=spec["base_url"],
        temperature=float(config.get("LLM_TEMPERATURE", "0.3")),
        timeout=180,
        max_retries=2,
    )


# ------------------------------------------------------------------ MCP
def load_mcp_config(archive_id: str = config.DEFAULT_ARCHIVE) -> dict:
    """读取 mcp_servers.json；enabled=false 的跳过，stdio 的 python 换成本解释器。

    `archiveArg: true` 的服务器会追加 `--archive <id>`：档案是运行时才知道的，
    不能写死在 json 里，也不宜用环境变量传（stdio 客户端一旦收到 env 就会整体替换
    子进程环境，PATH 之类全丢）。
    """

    cfg_path = config.ROOT / (config.get("MCP_CONFIG", "mcp_servers.json"))
    if not cfg_path.exists():
        return {}

    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    servers: dict = {}
    for name, conn in raw.get("mcpServers", {}).items():
        if not conn.get("enabled", True):
            continue
        needs_archive = bool(conn.get("archiveArg"))
        conn = {k: v for k, v in conn.items()
                if k not in ("description", "enabled", "archiveArg")}
        conn.setdefault("transport", "stdio")
        if needs_archive:
            conn["args"] = [*conn.get("args", []), "--archive", archive_id]
        if conn["transport"] == "stdio":
            if conn.get("command") == "python":
                conn["command"] = sys.executable
            conn.setdefault("cwd", str(config.ROOT))
        servers[name] = conn
    return servers


async def load_mcp_tools(archive_id: str = config.DEFAULT_ARCHIVE) -> tuple[list, list[str]]:
    """逐个连接已启用的 MCP 服务器。单点失败只记警告，不拖垮全局。"""

    servers = load_mcp_config(archive_id)
    if not servers:
        return [], []

    from langchain_mcp_adapters.client import MultiServerMCPClient

    tools: list = []
    warnings: list[str] = []
    for name, conn in servers.items():
        try:
            client = MultiServerMCPClient({name: conn})
            tools.extend(await client.get_tools())
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"MCP 服务器 [{name}] 加载失败：{exc!r}")
    return tools, warnings


# ------------------------------------------------------------------ 权限
def build_permissions(archive_id: str = config.DEFAULT_ARCHIVE) -> list[FilesystemPermission]:
    """按顺序匹配，第一条命中即生效。顺序写错会静默失效，所以用显式常量。

    1. 先给两个写入区放行（必须在 deny /docs/** 之前，否则永远匹配不到）
    2. 挡住密钥、虚拟环境、运行时数据、git 内部
    3. 工作文档区只读
    4. 其余一律拒绝写

    写入区**按档案收窄**：只放行本档案的简报与账本子目录。档案互不可见是这个功能
    的前提，放行 /outputs/** 等于让 A 档案的报告能覆盖 B 档案的。
    """

    return [
        FilesystemPermission(
            operations=["write"],
            paths=[f"/outputs/briefs/{archive_id}/**", f"/docs/ledger/{archive_id}/**"],
            mode="allow",
        ),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/.env", "/.venv/**", "/data/**", "/runtime/**", "/.git/**"],
            mode="deny",
        ),
        FilesystemPermission(operations=["write"], paths=["/docs/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
    ]


# ------------------------------------------------------------------ 组装
def _discover_skills() -> list[str] | None:
    """有技能才传 /skills/，否则传 None —— 空目录会让模型白读一次。"""

    if not config.SKILLS_DIR.exists():
        return None
    has_skill = any((p / "SKILL.md").exists() for p in config.SKILLS_DIR.iterdir() if p.is_dir())
    return ["/skills/"] if has_skill else None


async def build_checkpointer(archive_id: str = config.DEFAULT_ARCHIVE) -> AsyncSqliteSaver:
    """必须在之后运行 Agent 的同一个事件循环里创建。"""

    import aiosqlite

    config.ensure_dirs(archive_id)
    conn = await aiosqlite.connect(config.checkpoints_db(archive_id))
    return AsyncSqliteSaver(conn)


def build_memory(archive_id: str = config.DEFAULT_ARCHIVE) -> list[str]:
    """全局口径一份 + 档案补充一份。档案补充不存在就只注入全局，
    不要传一个空文件路径——那会让模型读一次空文件，白花一次工具调用。"""

    paths = ["/memory/AGENTS.md"]
    if config.archive_memory(archive_id).exists():
        paths.append(f"/memory/archives/{archive_id}.md")
    return paths


def build_system_prompt(archive_id: str = config.DEFAULT_ARCHIVE) -> str:
    """系统提示词 + 当前档案的具体路径。

    档案 id 是运行时才知道的，技能文档里只能写 `<档案>` 占位。把具体值放在这里，
    模型不用猜、也不需要用 glob 去找账本——猜错的后果是它去读别的档案的账本，
    把两个读者的判断口径混在一起，而这种错误在报告里看不出来。
    """

    name = archive.name_of(archive_id)
    return SYSTEM_PROMPT + f"""
## 当前档案

档案名：{name}（id `{archive_id}`）。证据库、会话、账本与别的档案互相隔离。

- 简报目录：`/outputs/briefs/{archive_id}/`
- 账本：`/docs/ledger/{archive_id}/.ai-brief-ledger.md`
- 档位偏好：`/docs/ledger/{archive_id}/.ai-brief-profile.md`

技能文档里写作 `<档案>` 的路径，一律按上面三行替换。不要读写其它档案的目录。
"""


async def build_agent(archive_id: str = config.DEFAULT_ARCHIVE):
    """返回 (agent, info)。info 是界面与 /info 的唯一数据源，不是日志。"""

    config.ensure_dirs(archive_id)
    mcp_tools, warnings = await load_mcp_tools(archive_id)
    skills = _discover_skills()
    model = settings.current_model()

    agent = create_deep_agent(
        model=build_model(),
        tools=[*PY_TOOLS, *mcp_tools],
        system_prompt=build_system_prompt(archive_id),
        subagents=build_subagents(build_model(config.SUBAGENT_MODEL)),
        skills=skills,
        memory=build_memory(archive_id),
        backend=FilesystemBackend(root_dir=config.ROOT, virtual_mode=True),
        permissions=build_permissions(archive_id),
        middleware=[TodoListMiddleware()],
        checkpointer=await build_checkpointer(archive_id),
        name="ai-brief-analyst",
    )

    info = {
        "archive": archive_id,
        "model": model,
        "custom_tools": [t.__name__ for t in PY_TOOLS],
        "mcp_tools": [t.name for t in mcp_tools],
        "skills": [p.name for p in (config.SKILLS_DIR.iterdir() if skills else []) if p.is_dir()],
        "subagents": ["critic", "general-purpose"],
        "search_provider": config.SEARCH_PROVIDER,
        "search_fallback": config.SEARCH_FALLBACK,
        "write_roots": [f"/outputs/briefs/{archive_id}/", f"/docs/ledger/{archive_id}/"],
        "warnings": warnings,
    }
    return agent, info


async def close_agent(agent) -> None:
    """退出前关闭 SQLite 连接，避免事件循环关闭后后台线程报错。"""

    saver = getattr(agent, "checkpointer", None)
    if isinstance(saver, AsyncSqliteSaver):
        await saver.conn.close()
