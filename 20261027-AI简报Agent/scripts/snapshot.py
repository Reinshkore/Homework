"""测试状态快照：备份 / 清空 / 恢复。

为什么需要它：反复测试会互相污染，而且污染方式是隐蔽的——
  · 同一个窗口同一个档位重跑，evidence 会累加上一期的条目，簇数与一手率跟着变；
  · 账本里留下两条同日记录，下一期 Step 0 回检时读到的是同期，不是上期；
  · 简报目录里同名报告被覆盖，你没法对比"改之前"和"改之后"。

**每一份快照只管一个档案**：证据库、账本、简报都在档案目录下，
跨档案一起备份会把别的档案也清掉，那正好破坏了档案隔离。

用法：
    .venv/Scripts/python.exe scripts/snapshot.py save before-lint
    .venv/Scripts/python.exe scripts/snapshot.py list
    .venv/Scripts/python.exe scripts/snapshot.py clear
    .venv/Scripts/python.exe scripts/snapshot.py restore before-lint
    .venv/Scripts/python.exe scripts/snapshot.py -a <档案> save x   # 指定档案
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from core import archive, config  # noqa: E402

SNAP_DIR = config.RUNTIME_DIR / "snapshots"


def targets(archive_id: str) -> dict[str, Path]:
    """需要一起搬的三块：证据库、账本、简报（该档案的）。"""

    return {
        "store": config.store_db(archive_id),
        "ledger": config.ledger_dir(archive_id),
        "briefs": config.briefs_dir(archive_id),
    }


def _copy(src: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out) if out.is_dir() else out.unlink()
    if src.is_dir():
        shutil.copytree(src, out)
    else:
        shutil.copy2(src, out)
        for suffix in ("-wal", "-shm"):
            side = Path(str(src) + suffix)
            if side.exists():
                shutil.copy2(side, Path(str(out) + suffix))


def save(archive_id: str, name: str | None = None) -> Path:
    stamp = datetime.now().strftime("%H%M%S")
    dest = SNAP_DIR / (name or f"snap-{stamp}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ARCHIVE").write_text(archive_id, encoding="utf-8")
    done = []
    for label, src in targets(archive_id).items():
        if src.exists():
            _copy(src, dest / label)
            done.append(label)
    print(f"已备份档案 {archive_id} 到 {dest.relative_to(ROOT)}：{', '.join(done) or '（无内容）'}")
    return dest


def clear(archive_id: str) -> None:
    for label, path in targets(archive_id).items():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
        elif path.exists():
            path.unlink()
    print(f"已清空档案 {archive_id} 的证据库、账本与简报目录（原状态请用 restore 恢复）")


def restore(name: str, archive_id: str = "") -> None:
    src = SNAP_DIR / name
    if not src.exists():
        print(f"找不到快照 {src.relative_to(ROOT)}")
        return
    # 快照记录了它属于哪个档案；恢复时默认回到原档案，避免把 A 的测试数据灌进 B
    stamp = src / "ARCHIVE"
    if not archive_id and stamp.exists():
        archive_id = stamp.read_text(encoding="utf-8").strip()
    archive_id = archive_id or config.DEFAULT_ARCHIVE
    for label, target in targets(archive_id).items():
        s = src / label
        if not s.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(s, target)
        else:
            if target.exists():
                target.unlink()
            shutil.copy2(s, target)
            for suffix in ("-wal", "-shm"):
                side = Path(str(s) + suffix)
                if side.exists():
                    shutil.copy2(side, Path(str(target) + suffix))
    print(f"已从 {src.relative_to(ROOT)} 恢复到档案 {archive_id}")


def list_snaps() -> None:
    if not SNAP_DIR.exists():
        print("还没有快照")
        return
    for p in sorted(SNAP_DIR.iterdir()):
        if not p.is_dir():
            continue
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        parts = [d.name for d in p.iterdir() if d.is_dir()]
        stamp = p / "ARCHIVE"
        aid = stamp.read_text(encoding="utf-8").strip() if stamp.exists() else "?"
        print(f"  {p.name:<20} {size/1024:6.1f} KB  档案={aid}  {', '.join(parts)}")


def main() -> int:
    args = sys.argv[1:]
    archive_id = ""
    if "-a" in args:
        i = args.index("-a")
        archive_id = args[i + 1] if i + 1 < len(args) else ""
        args = args[:i] + args[i + 2:]
    args = args or ["list"]
    archive_id = archive_id or archive.active()
    cmd = args[0]
    if cmd == "save":
        save(archive_id, args[1] if len(args) > 1 else None)
    elif cmd == "clear":
        save(archive_id, "auto-before-clear")
        clear(archive_id)
    elif cmd == "restore":
        if len(args) < 2:
            print("用法：snapshot.py restore <name>")
            return 1
        restore(args[1], archive_id)
    elif cmd == "list":
        list_snaps()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
