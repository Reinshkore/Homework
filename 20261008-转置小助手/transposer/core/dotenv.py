"""极简 .env 加载器：只做 KEY=VALUE 读取，不引入额外依赖。"""
import os
import pathlib


def load_env(path=None):
    p = pathlib.Path(path) if path else pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val
