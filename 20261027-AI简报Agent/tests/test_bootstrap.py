"""环境配方的自洽性检查。

`.venv` 不进仓库（468 MB、写死了本机绝对路径），所以「能不能装上」全靠两份 requirements
加一个 bootstrap 脚本。这三样东西一旦互相打架，别人克隆下来就是装不上——而这种错
本地永远发现不了，因为本地早就有 venv 了。

这里只查一件有客观答案的事：`requirements.txt` 里每个锁定的版本，必须和
`requirements.lock.txt` 快照一致。两份文件漂移，等于「声明的依赖」和「实际跑通的依赖」不是一回事。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "scripts" / "bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestEnvRecipe(unittest.TestCase):
    def setUp(self):
        self.mod = _load_bootstrap()
        self.req = self.mod._read_reqs(ROOT / "requirements.txt")
        self.lock = self.mod._read_reqs(ROOT / "requirements.lock.txt")

    def test_direct_deps_match_lock(self):
        """requirements.txt 是给人看的声明，lock 是机器读的快照，两者必须一致。"""

        self.assertTrue(self.req, "requirements.txt 没解析出任何依赖")
        self.assertTrue(self.lock, "requirements.lock.txt 没解析出任何依赖")
        drift = {n: (v, self.lock.get(n)) for n, v in self.req.items()
                 if v and self.lock.get(n) != v}
        self.assertEqual(drift, {}, f"两份文件版本不一致（名字：声明版, 快照版）：{drift}")

    def test_all_direct_deps_are_pinned(self):
        """不许留 >= 或空版本：范围依赖等于把「装出什么」交给时间。"""

        loose = [n for n, v in self.req.items() if not v]
        self.assertEqual(loose, [], f"这些依赖没锁版本：{loose}")

    def test_check_runs_and_reports(self):
        """`--check` 必须能独立跑通并报出环境状态。

        这里不断言具体结论（本地有 venv 就 0、没有就 1，取决于跑测试的机器），
        只断言它没崩、且真的去看了虚拟环境——一个永远返回 0 的检查脚本比没有更糟。
        """

        import subprocess

        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "bootstrap.py"), "--check"],
                           capture_output=True, text=True, timeout=300)
        self.assertIn(r.returncode, (0, 1), f"--check 异常退出：{r.returncode}\n{r.stderr}")
        self.assertIn("虚拟环境", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
