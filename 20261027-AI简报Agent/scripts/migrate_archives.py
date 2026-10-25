"""把单档案时代的数据迁进 default 档案。只跑一次，可重复跑（已迁过的会跳过）。

迁移内容：

    runtime/store.db                → runtime/archives/default/store.db
    data/sessions.db                → runtime/archives/default/sessions.db
    data/checkpoints.db*            → runtime/archives/default/checkpoints.db*
    outputs/briefs/*.md             → outputs/briefs/default/*.md
    docs/ledger/.ai-brief-ledger.md → docs/ledger/default/.ai-brief-ledger.md

为什么必须搬而不是复制后再删：会话与检查点靠 thread_id 关联，分居两地会表现为
"点开历史会话，模型却说不知道之前聊过什么"。所以三样一起搬，且保持同名。

`runtime/snapshots/` 里的旧快照不迁——它们是旧布局的备份，结构已经不匹配，
留着当历史档案即可，`snapshot.py list` 会显示它们没有档案标记。

用法：
    .venv/Scripts/python.exe scripts/migrate_archives.py
    .venv/Scripts/python.exe scripts/migrate_archives.py --dry-run
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from core import archive, config  # noqa: E402

DRY = "--dry-run" in sys.argv

# 物理路径 → 目标（都是相对 ROOT 的展示名）
FILE_MOVES = [
    (config.RUNTIME_DIR / "store.db", "store.db"),
    (config.DATA_DIR / "sessions.db", "sessions.db"),
    (config.DATA_DIR / "checkpoints.db", "checkpoints.db"),
]

DEFAULT_MEMORY = """# 档案记忆：默认档案

本档案是主档案。默认档位：学习者 × 标准，窗口 7 天。
"""


def _move(src: Path, dst: Path, moves: list[str], skips: list[str]) -> None:
    """搬文件（连带 -wal / -shm）。目标已存在就跳过，不覆盖。"""

    if not src.exists():
        return
    if dst.exists():
        skips.append(f"{src.relative_to(ROOT)} （目标已存在，跳过）")
        return
    rel = f"{src.relative_to(ROOT)} → {dst.relative_to(ROOT)}"
    if DRY:
        moves.append(rel)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        s = Path(str(src) + suffix)
        if s.exists():
            shutil.move(str(s), str(dst) + suffix)
    moves.append(rel)


def main() -> int:
    aid = config.DEFAULT_ARCHIVE
    moves: list[str] = []
    skips: list[str] = []

    if not archive.exists(aid):
        if DRY:
            print(f"[dry] 会新建档案 {aid}")
        else:
            archive.create(archive.DEFAULT_NAME, archive_id=aid)
            print(f"已建档案 {aid}")

    for src, name in FILE_MOVES:
        _move(src, config.archive_dir(aid) / name, moves, skips)

    # 简报：根目录的散文件搬进 default 子目录
    root_briefs = config.BRIEFS_DIR
    if root_briefs.exists():
        for p in sorted(root_briefs.glob("*.md")):
            _move(p, config.briefs_dir(aid) / p.name, moves, skips)

    # 账本
    old_ledger = config.DOCS_DIR / "ledger" / ".ai-brief-ledger.md"
    _move(old_ledger, config.ledger_file(aid), moves, skips)

    # 档案记忆：没有就写一份带默认档位的
    mem = config.archive_memory(aid)
    if not mem.exists():
        if DRY:
            print(f"[dry] 会写 {mem.relative_to(config.ROOT)}")
        else:
            mem.parent.mkdir(parents=True, exist_ok=True)
            mem.write_text(DEFAULT_MEMORY, encoding="utf-8")
            moves.append(f"新建 {mem.relative_to(config.ROOT)}")

    config.ensure_dirs(aid)
    archive.set_active(aid)

    print(("【演练】" if DRY else "") + "迁移结果：")
    for m in moves:
        print(f"  ✓ {m}")
    for s in skips:
        print(f"  · {s}")
    if not moves and not skips:
        print("  没有可迁的内容（可能已经迁过）")

    print("\n当前目录：")
    for label, path in (
        ("档案", config.archive_dir(aid)),
        ("证据库", config.store_db(aid)),
        ("会话", config.sessions_db(aid)),
        ("检查点", config.checkpoints_db(aid)),
        ("简报", config.briefs_dir(aid)),
        ("账本", config.ledger_file(aid)),
        ("记忆", config.archive_memory(aid)),
    ):
        mark = "存在" if path.exists() else "缺失"
        print(f"  {label:<4} {path.relative_to(ROOT)}  [{mark}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
