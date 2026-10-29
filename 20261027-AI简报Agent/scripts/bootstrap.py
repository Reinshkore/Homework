"""一条命令把运行环境装好。

**为什么不把 `.venv` 传上仓库**：它是 468 MB / 25 257 个文件，而且 `.venv/pyvenv.cfg`
里写死了本机的绝对路径（`home = C:\\Users\\...\\python\\versions\\3.13.12`）——
换台机器、甚至换个目录都跑不起来。虚拟环境是生成物，不是源码；仓库该存的是
生成它的配方，也就是这个脚本加两份 requirements。

用法（在仓库根目录）：

    python scripts/bootstrap.py            # 建 .venv，按 requirements.lock.txt 精确复现
    python scripts/bootstrap.py --loose    # 按 requirements.txt 装（换 Python 大版本或换系统时用）
    python scripts/bootstrap.py --check    # 只体检，不动任何东西
    python scripts/bootstrap.py --index https://mirrors.aliyun.com/pypi/simple/

装完还要做一件事：把 `.env.example` 复制成 `.env` 并填入密钥（脚本会自动复制一份空的）。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
REQ = ROOT / "requirements.txt"
LOCK = ROOT / "requirements.lock.txt"
ENV = ROOT / ".env"
ENV_SAMPLE = ROOT / ".env.example"

OK_PYTHON = {(3, 12), (3, 13)}


def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _read_reqs(path: Path) -> dict[str, str]:
    """读 `name==version` 形式的依赖清单，注释与空行跳过。"""

    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        if "==" in line:
            name, _, ver = line.partition("==")
            out[name.strip().lower().replace("_", "-")] = ver.strip()
        else:
            out[line.lower().replace("_", "-")] = ""
    return out


def _installed(py: Path) -> dict[str, str]:
    """问目标 venv 自己装了哪些包。用 venv 的解释器，不用当前这个。"""

    try:
        r = subprocess.run([str(py), "-m", "pip", "list", "--format=freeze"],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return {}
    if r.returncode != 0:
        return {}
    out: dict[str, str] = {}
    for line in r.stdout.splitlines():
        name, _, ver = line.partition("==")
        if name:
            out[name.strip().lower().replace("_", "-")] = ver.strip()
    return out


def check() -> int:
    """只体检。返回码：0 可用，1 有问题。"""

    print(f"Python  {sys.version.split()[0]}  ({'符合' if sys.version_info[:2] in OK_PYTHON else '未经验证'})"
          f"  期望 {'/'.join(f'{a}.{b}' for a, b in sorted(OK_PYTHON))}")
    if not VENV.exists():
        print("虚拟环境  缺失（.venv 不存在）→ 运行本脚本生成")
        return 1

    py = _venv_python()
    if not py.exists():
        print(f"虚拟环境  损坏（找不到 {py.relative_to(ROOT)}）→ 删掉 .venv 重跑本脚本")
        return 1
    print(f"虚拟环境  存在  {VENV.relative_to(ROOT)}")

    want = _read_reqs(LOCK if LOCK.exists() else REQ)
    have = _installed(py)
    missing = [n for n in want if n not in have]
    wrong = [f"{n}（要 {want[n]}，装的是 {have[n]}）" for n in want
             if n in have and want[n] and have[n] != want[n]]
    if missing:
        print(f"依赖  缺 {len(missing)} 个：{', '.join(missing[:6])}"
              + ("…" if len(missing) > 6 else ""))
    if wrong:
        print(f"依赖  版本不符 {len(wrong)} 个：{'; '.join(wrong[:4])}"
              + ("…" if len(wrong) > 4 else ""))
    if not missing and not wrong:
        print(f"依赖  {len(want)} 个全部对上（{LOCK.name if LOCK.exists() else REQ.name}）")

    if not ENV.exists():
        print("密钥  .env 不存在 → 从 .env.example 复制后填 DASHSCOPE_API_KEY / DEEPSEEK_API_KEY")
    else:
        keys = [k for k in ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY")
                if re.search(rf"^{k}\s*=\s*\S", ENV.read_text(encoding="utf-8", errors="ignore"), re.M)]
        print(f"密钥  .env 存在，已填 {len(keys)}/2（{', '.join(keys) or '都没填'}）")

    return 0 if not missing and not wrong else 1


def build(loose: bool, index: str | None) -> int:
    if sys.version_info[:2] not in OK_PYTHON:
        print(f"警告：当前 Python {sys.version.split()[0]} 不在验证过的 3.12/3.13 之内。"
              "依赖里锁的是这两个版本上跑通过的组合，继续装可能遇到轮子不匹配。")

    req = REQ if loose else (LOCK if LOCK.exists() else REQ)
    if not req.exists():
        print(f"找不到 {req.name}")
        return 1

    if not VENV.exists():
        print(f"建虚拟环境  {VENV.relative_to(ROOT)}")
        r = subprocess.run([sys.executable, "-m", "venv", str(VENV)])
        if r.returncode != 0:
            print("建环境失败。检查当前 Python 是否带 venv 模块。")
            return r.returncode
    else:
        print(f"虚拟环境已存在  {VENV.relative_to(ROOT)}（跳过创建）")

    cmd = [str(_venv_python()), "-m", "pip", "install", "-r", str(req)]
    if index:
        cmd += ["-i", index]
    print(f"装依赖  {req.name}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("\n安装失败。若是某个包 403 / 超时，换镜像再试：")
        print("  python scripts/bootstrap.py --index https://mirrors.aliyun.com/pypi/simple/")
        return r.returncode

    if not ENV.exists() and ENV_SAMPLE.exists():
        shutil.copyfile(ENV_SAMPLE, ENV)
        print("已从 .env.example 复制出 .env —— 打开它填两个密钥")

    print("\n下一步：")
    print("  1. 编辑 .env，填 DASHSCOPE_API_KEY 与 DEEPSEEK_API_KEY")
    print("  2. 体检： python scripts/bootstrap.py --check")
    print("  3. 启动： .venv/Scripts/python.exe -m streamlit run app.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="建虚拟环境并装依赖（不传 .venv 进仓库）")
    ap.add_argument("--check", action="store_true", help="只体检，不做任何修改")
    ap.add_argument("--loose", action="store_true",
                    help="用 requirements.txt 而不是 requirements.lock.txt")
    ap.add_argument("--index", default=None, help="pip 镜像地址")
    a = ap.parse_args()
    return check() if a.check else build(a.loose, a.index)


if __name__ == "__main__":
    raise SystemExit(main())
