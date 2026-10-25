"""档案：一套互相隔离的运行态。

一个档案有自己的证据库、会话、检查点、账本、简报目录与补充记忆。
目录布局（物理）：

    runtime/archives/<id>/meta.json      名称与创建时间
    runtime/archives/<id>/store.db       证据库（MCP 独占写）
    runtime/archives/<id>/sessions.db    会话记录（给人看）
    runtime/archives/<id>/checkpoints.db 对话上下文（给模型看）
    outputs/briefs/<id>/                 简报
    docs/ledger/<id>/                    账本
    memory/archives/<id>.md              档案补充记忆（与 memory/AGENTS.md 的全局口径分开）

三条设计边界，都跟"删了会怎样"有关：

1. **default 档案不可删。** 它是迁移落点，也是删到只剩一个时的兜底。
2. **删档案不删简报。** 简报是交付物，不是运行态；档案没了，报告仍然是那一期的证据。
   所以删除时保留 outputs/briefs/<id>/，往期简报页继续能看到（档案名变成 id）。
3. **补充记忆单独一份，不与全局口径合并。** 全局的 AGENTS.md 装的是产品口径
   （日期不可信、authority 不等于一手……），每档案复制一份的话，改一条要改 N 份，
   必然有一份忘掉——交接文档里"两份技能副本、改错那份"就是这么发生的。
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path

from core import config

MAX_ARCHIVES = 5
DEFAULT_ID = config.DEFAULT_ARCHIVE
DEFAULT_NAME = "默认档案"
def _active_file() -> Path:
    """每次现算，不在模块导入时定死：测试要把 ARCHIVES_DIR 指到临时目录，
    常量会让 set_active 写回真实目录，表现为"切了档案却记不住上次用的哪个"。 """

    return config.ARCHIVES_DIR / ".active"

MEMORY_TEMPLATE = """# 档案记忆：{name}

（这个档案特有的偏好写在这里，比如读者背景、默认档位、常看的公司。
通用口径见 memory/AGENTS.md，不要在这里重复。）
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _count(db_path: Path, sql: str) -> int:
    """只读计数。库可能还不存在、表可能还没建，一律返回 0，不抛。"""

    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        try:
            return int(conn.execute(sql).fetchone()[0])
        finally:
            conn.close()
    except Exception:  # noqa: BLE001  库被占用或表不存在都当 0
        return 0


# ------------------------------------------------------------------ 读写元信息
def read_meta(archive_id: str) -> dict:
    path = config.archive_meta(archive_id)
    name, created = archive_id, ""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            name = (data.get("name") or archive_id).strip()
            created = data.get("created_at") or ""
        except Exception:  # noqa: BLE001  meta 写坏了不该让整个界面打不开
            pass
    return {
        "id": archive_id,
        "name": name,
        "created_at": created,
        "sessions": _count(config.sessions_db(archive_id),
                           "SELECT COUNT(*) FROM sessions"),
        "evidence": _count(config.store_db(archive_id),
                           "SELECT COUNT(*) FROM evidence"),
    }


def write_meta(archive_id: str, name: str, created_at: str = "") -> None:
    path = config.archive_meta(archive_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"id": archive_id, "name": name.strip() or archive_id,
                    "created_at": created_at or _now()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ------------------------------------------------------------------ 查询
def _dirs() -> list[Path]:
    """不触发 ensure_default 的目录扫描。

    create() 要用它数档案数，而 list_archives() 会先 ensure_default → create，
    两个一凑就是无限递归。所以计数这条路径必须绕开 ensure_default。
    """

    if not config.ARCHIVES_DIR.exists():
        return []
    return [p for p in config.ARCHIVES_DIR.iterdir() if p.is_dir()]


def count() -> int:
    return len(_dirs())


def exists(archive_id: str) -> bool:
    return bool(archive_id) and config.archive_dir(archive_id).is_dir()


def ensure_default() -> None:
    if not exists(DEFAULT_ID):
        write_meta(DEFAULT_ID, DEFAULT_NAME)
    config.ensure_dirs(DEFAULT_ID)


def list_archives() -> list[dict]:
    ensure_default()
    out = [read_meta(p.name) for p in _dirs()]
    out.sort(key=lambda a: (a["id"] != DEFAULT_ID, a["created_at"], a["id"]))
    return out


def name_of(archive_id: str) -> str:
    """档案名。删掉的档案只在简报页出现，那里只能显示 id。"""

    return read_meta(archive_id)["name"] if exists(archive_id) else archive_id


def active() -> str:
    """上次使用的档案。查询参数优先，其次这个文件，最后 default。

    单个用户的本机应用，不做多标签页同步；多开浏览器时以 URL 参数为准。
    """

    ensure_default()
    stamp = _active_file()
    if stamp.exists():
        try:
            aid = stamp.read_text(encoding="utf-8").strip()
            if exists(aid):
                return aid
        except Exception:  # noqa: BLE001
            pass
    return DEFAULT_ID


def set_active(archive_id: str) -> None:
    if not exists(archive_id):
        return
    config.ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    _active_file().write_text(archive_id, encoding="utf-8")


# ------------------------------------------------------------------ 增删改
def create(name: str, archive_id: str = "") -> str:
    """新建档案，返回 id。超过上限抛 RuntimeError —— 界面据此提示，不自作主张删旧的。

    `archive_id` 只在迁移时显式指定（default 是固定 id）；正常新建走随机 id，
    避免用户输入的档案名里带空格/中文被当成路径。
    """

    current = count()
    if current >= MAX_ARCHIVES:
        raise RuntimeError(f"档案已达上限 {MAX_ARCHIVES} 个，先删掉不用的再建")

    archive_id = archive_id or uuid.uuid4().hex[:8]
    config.ensure_dirs(archive_id)
    write_meta(archive_id, name or f"档案 {archive_id}")

    mem = config.archive_memory(archive_id)
    if not mem.exists():
        mem.parent.mkdir(parents=True, exist_ok=True)
        mem.write_text(MEMORY_TEMPLATE.format(name=name or archive_id), encoding="utf-8")

    ledger = config.ledger_file(archive_id)
    if not ledger.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(
            "# ai-brief 判断账本\n\n每期追加一段，不覆盖、不回改历史条目。\n\n",
            encoding="utf-8",
        )
    return archive_id


def rename(archive_id: str, name: str) -> None:
    if not exists(archive_id):
        return
    meta = read_meta(archive_id)
    write_meta(archive_id, name, meta["created_at"])


def _rmtree(path: Path) -> bool:
    """删目录并确认真的没了。

    不能只写 `rmtree(ignore_errors=True)`：Windows 上被打开的文件删不掉，
    忽略错误的结果是"界面说删了、磁盘上还在"，下次启动又冒出来一个幽灵档案。
    关连接后文件锁释放有几十毫秒延迟，所以重试几次再判失败。

    第二段兜底是逐文件删：某些运行环境把 `rmtree` 重定向到"移入回收站"，
    目录里只要有一个文件正被占用，整个操作就整体失败（连没被占的也一起放弃），
    这时逐个删能保住大部分，剩下的等下一轮重试。
    """

    for _ in range(6):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return True
        for item in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            try:
                item.rmdir() if item.is_dir() else item.unlink()
            except OSError:
                pass
        try:
            path.rmdir()
        except OSError:
            pass
        if not path.exists():
            return True
        time.sleep(0.3)
    return not path.exists()


def delete(archive_id: str) -> dict:
    """删运行态，留简报。

    返回被删掉的东西，界面用它给用户一句如实的反馈。
    "删了什么、留了什么"必须说清：用户点删除时以为全没了，实际简报还在，
    反过来（以为简报没了，实际还在）会让人怀疑删除没生效。

    前置条件：**不能有 Agent 运行时打开着这个档案的 checkpoints.db**。
    界面在删除当前档案前会先 close 运行时；脚本里如果自己建过 AgentRuntime，也要先关。
    """

    if archive_id == DEFAULT_ID:
        raise RuntimeError("默认档案不能删除")
    if not exists(archive_id):
        raise RuntimeError("档案不存在")

    removed = []
    for label, path in (
        ("运行态目录", config.archive_dir(archive_id)),
        ("账本", config.ledger_dir(archive_id)),
    ):
        if not path.exists():
            continue
        if not _rmtree(path):
            raise RuntimeError(
                f"{label}删除失败：{path.name} 仍被占用。"
                "多半是还有会话在跑或 Agent 未释放，稍后重试；重启应用后再试一定成功。"
            )
        removed.append(label)

    for label, path in (("补充记忆", config.archive_memory(archive_id)),):
        if path.exists():
            path.unlink()
            removed.append(label)

    kept = len(list(config.briefs_dir(archive_id).glob("*.md"))) \
        if config.briefs_dir(archive_id).exists() else 0
    # 没有简报就顺手把空的简报目录删掉。留着的话每建一次档案就在 outputs/briefs/
    # 下留一个空目录，用几次就看不出哪些是真有产出的档案了。
    if kept == 0 and config.briefs_dir(archive_id).exists():
        _rmtree(config.briefs_dir(archive_id))

    if active() == archive_id:
        set_active(DEFAULT_ID)
    return {"removed": removed, "kept_briefs": kept}


def stats_label(archive_id: str) -> str:
    """侧栏/下拉框用的一行摘要。"""
    m = read_meta(archive_id)
    return f"{m['name']}（会话 {m['sessions']}｜证据 {m['evidence']}）"
