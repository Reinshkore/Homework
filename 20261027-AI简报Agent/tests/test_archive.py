"""档案系统的离线测试。

这些断言都指向同一类失败：**串档**。档案是给人用的便利功能，
但实现上它只是几条路径；路径漏传不会报错，只会让数据静默写进 default，
表现为"切了档案但内容没变"。所以每条都直接查磁盘，不看返回值。

**用临时根目录跑，不碰真实的 runtime/archives/**（setUpClass 里重定向 config 路径）。
两个原因：
  1. 上限是 5，测试要造到上限，跟真实数据挤在一起会互相撞；
  2. 某些运行环境的删除保护会把 rmtree 改成"移入回收站"，失败时目录会延迟恢复，
     留在真实目录里就成了幽灵档案，下一次跑直接"已达上限"。
"""

from __future__ import annotations

import runpy
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.builder import load_mcp_config  # noqa: E402
from core import archive, config, history, store  # noqa: E402

ITEM = {
    "title": "某公司开源推理框架",
    "url": "https://example.com/a",
    "date": "2026-10-20",
    "subject": "某公司",
    "event_type": "T4",
}

_PATH_KEYS = ("ARCHIVES_DIR", "BRIEFS_DIR", "DOCS_DIR", "MEMORY_DIR")


class TestArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = Path(tempfile.mkdtemp(prefix="brief-archives-"))
        cls._saved = {k: getattr(config, k) for k in _PATH_KEYS}
        config.ARCHIVES_DIR = cls._tmp / "archives"
        config.BRIEFS_DIR = cls._tmp / "briefs"
        config.DOCS_DIR = cls._tmp / "docs"
        config.MEMORY_DIR = cls._tmp / "memory"
        config.ensure_dirs(config.DEFAULT_ARCHIVE)

    @classmethod
    def tearDownClass(cls):
        for key, value in cls._saved.items():
            setattr(config, key, value)
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def setUp(self):
        self.made: list[str] = []

    def tearDown(self):
        # 清理要验证：删不掉的目录会让"档案数不对"这种失败指向错的地方（踩过）。
        for aid in self.made:
            if archive.exists(aid):
                try:
                    archive.delete(aid)
                except RuntimeError:
                    pass
            for path in (config.archive_dir(aid), config.briefs_dir(aid), config.ledger_dir(aid)):
                shutil.rmtree(path, ignore_errors=True)
            mem = config.archive_memory(aid)
            if mem.exists():
                mem.unlink()
        self.made.clear()

    def _new(self, name: str) -> str:
        aid = archive.create(name)
        self.made.append(aid)
        return aid

    # ---- 隔离
    def test_evidence_is_isolated(self):
        a, b = self._new("甲"), self._new("乙")
        pid_a = store.ensure_period(config.store_db(a), "2026-10-17", "2026-10-24", "学习者", "标准")
        pid_b = store.ensure_period(config.store_db(b), "2026-10-17", "2026-10-24", "学习者", "标准")

        store.add_evidence(config.store_db(a), pid_a, [ITEM], "2026-10-17", "2026-10-24")

        self.assertEqual(len(store.query_rows(config.store_db(a), pid_a)), 1)
        self.assertEqual(len(store.query_rows(config.store_db(b), pid_b)), 0,
                         "乙档案不该看到甲的证据")

    def test_sessions_are_isolated(self):
        a, b = self._new("甲"), self._new("乙")
        tid = history.new_thread_id()
        history.save_message(tid, "user", "只写给甲", db_path=config.sessions_db(a))

        self.assertEqual(len(history.list_sessions(config.sessions_db(a))), 1)
        self.assertEqual(len(history.list_sessions(config.sessions_db(b))), 0)
        self.assertEqual(history.load_messages(tid, config.sessions_db(b)), [],
                         "乙档案里同 id 的会话不该有消息")

    def test_threads_do_not_collide_across_archives(self):
        """同一个 thread_id 在两个档案里是两条不同的会话，互不覆盖。"""

        a, b = self._new("甲"), self._new("乙")
        tid = history.new_thread_id()
        history.save_message(tid, "user", "甲的说法", db_path=config.sessions_db(a))
        history.save_message(tid, "user", "乙的说法", db_path=config.sessions_db(b))

        ma = history.load_messages(tid, config.sessions_db(a))
        mb = history.load_messages(tid, config.sessions_db(b))
        self.assertEqual([m["content"] for m in ma], ["甲的说法"])
        self.assertEqual([m["content"] for m in mb], ["乙的说法"])

    # ---- 增删改
    def test_rename(self):
        aid = self._new("旧名")
        archive.rename(aid, "新名")
        self.assertEqual(archive.name_of(aid), "新名")

    def test_cap(self):
        while archive.count() < archive.MAX_ARCHIVES:
            self._new("占位")
        with self.assertRaises(RuntimeError):
            archive.create("超出上限")

    def test_default_cannot_be_deleted(self):
        with self.assertRaises(RuntimeError):
            archive.delete(config.DEFAULT_ARCHIVE)

    def test_delete_keeps_briefs(self):
        """删档案删运行态，但简报是交付物，要留下。"""

        aid = self._new("要删的")
        brief = config.briefs_dir(aid) / "ai-brief-学习者-标准-2026-10-24.md"
        brief.parent.mkdir(parents=True, exist_ok=True)
        brief.write_text("# 某期简报\n", encoding="utf-8")
        ledger = config.ledger_file(aid)
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text("# 账本\n", encoding="utf-8")
        mem = config.archive_memory(aid)
        self.assertTrue(mem.exists())

        info = archive.delete(aid)
        self.made.remove(aid)

        self.assertFalse(config.archive_dir(aid).exists(), "运行态目录应被删除")
        self.assertFalse(config.ledger_dir(aid).exists(), "账本应被删除")
        self.assertFalse(mem.exists(), "补充记忆应被删除")
        self.assertTrue(brief.exists(), "简报应保留")
        self.assertEqual(info["kept_briefs"], 1)
        brief.unlink()

    def test_delete_removes_empty_briefs_dir(self):
        """没产出的档案，删除时不该在 outputs/briefs/ 下留一个空目录。"""

        aid = self._new("没产出")
        self.assertTrue(config.briefs_dir(aid).exists())
        archive.delete(aid)
        self.made.remove(aid)
        self.assertFalse(config.briefs_dir(aid).exists())

    def test_active_falls_back_after_delete(self):
        aid = self._new("临时当前")
        archive.set_active(aid)
        self.assertEqual(archive.active(), aid)
        archive.delete(aid)
        self.made.remove(aid)
        self.assertEqual(archive.active(), config.DEFAULT_ARCHIVE)

    @unittest.skipUnless(sys.platform == "win32", "文件锁阻止删除只在 Windows 上成立")
    def test_delete_fails_loudly_when_locked(self):
        """删不掉必须报错，不能静默通过。

        实测过一次：界面删档案后目录还在，因为 checkpoints.db 被 Agent 打开着，
        而 rmtree 的失败被 ignore_errors 吞了。界面上看着删了，重启又冒出来。
        """

        aid = self._new("占用中的档案")
        db = config.store_db(aid)
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS t(x)")
            conn.commit()
            with self.assertRaises(RuntimeError):
                archive.delete(aid)
            self.assertTrue(config.archive_dir(aid).exists())
        finally:
            conn.close()

        archive.delete(aid)
        self.made.remove(aid)
        self.assertFalse(config.archive_dir(aid).exists())

    # ---- 装配
    def test_mcp_gets_archive_argument(self):
        """证据库是独立进程，档案只能通过命令行参数传进去。"""

        args = load_mcp_config("abc12345")["brief-store"]["args"]
        self.assertIn("--archive", args)
        self.assertEqual(args[args.index("--archive") + 1], "abc12345")

    def test_brief_store_resolves_archive_paths(self):
        saved = sys.argv
        sys.argv = ["brief_store.py", "--archive", "zz9"]
        try:
            mod = runpy.run_path(str(ROOT / "mcp" / "brief_store.py"),
                                 run_name="brief_store_probe")
        finally:
            sys.argv = saved
        self.assertEqual(mod["ARCHIVE"], "zz9")
        self.assertEqual(mod["DB"], config.store_db("zz9"))
        self.assertEqual(mod["BRIEFS"], config.briefs_dir("zz9"))
        self.assertEqual(mod["LEDGER"], config.ledger_file("zz9"))

    def test_ledger_virtual_path_matches_permission_root(self):
        """权限放行 /docs/ledger/**，所以账本必须真在 docs/ledger/<档案>/ 下。"""

        vpath = config.ledger_virtual_path("zz9")
        self.assertEqual(vpath, "/docs/ledger/zz9/.ai-brief-ledger.md")
        real = config.ledger_file("zz9")
        self.assertEqual(real.parent.name, "zz9")
        self.assertEqual(real.parent.parent.name, "ledger")
        self.assertEqual(real.name, ".ai-brief-ledger.md")

    def test_period_records_model(self):
        """期记录要写是哪个模型跑的；换模型重跑同期时更新，不传时保留旧值。"""

        aid = self._new("模型记录")
        db = config.store_db(aid)
        args = ("2026-10-17", "2026-10-24", "学习者", "标准")
        pid = store.ensure_period(db, *args, model="qwen3.7-plus")
        self.assertEqual(store.query_stats(db, pid)["model"], "qwen3.7-plus")

        store.ensure_period(db, *args, model="deepseek-flash")
        self.assertEqual(store.query_stats(db, pid)["model"], "deepseek-flash")

        store.ensure_period(db, *args)  # 不传模型
        self.assertEqual(store.query_stats(db, pid)["model"], "deepseek-flash",
                         "空模型不该把已有值冲掉")

    def test_old_db_gets_model_column(self):
        """老库补列：CREATE TABLE IF NOT EXISTS 不会加字段，得靠 ALTER。"""

        aid = self._new("老库")
        db = config.store_db(aid)
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE period (id TEXT PRIMARY KEY, window_start TEXT NOT NULL,"
            " window_end TEXT NOT NULL, audience TEXT NOT NULL DEFAULT '',"
            " length TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);"
        )
        conn.execute("INSERT INTO period VALUES ('p1','2026-10-17','2026-10-24','学习者','标准','x')")
        conn.commit()
        conn.close()

        pid = store.latest_period(db)  # 触发 _connect 里的迁移
        self.assertEqual(pid, "p1")
        self.assertEqual(store.query_stats(db, pid)["model"], "")

    def test_archive_db_is_readable_by_sqlite(self):
        """档案统计要能直接读库；库还没建时必须返回 0 而不是抛异常。"""

        aid = self._new("统计")
        self.assertEqual(archive.read_meta(aid)["evidence"], 0)
        pid = store.ensure_period(config.store_db(aid), "2026-10-17", "2026-10-24", "学习者", "标准")
        store.add_evidence(config.store_db(aid), pid, [ITEM], "2026-10-17", "2026-10-24")
        self.assertEqual(archive.read_meta(aid)["evidence"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
