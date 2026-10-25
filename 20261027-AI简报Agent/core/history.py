"""会话记录（给人看的），与 Agent 检查点（给模型看的）分开。

runtime/archives/<id>/checkpoints.db  框架写，存完整消息与状态，让模型接着上下文回答
runtime/archives/<id>/sessions.db     本模块写，存标题、问答、执行步骤，用于列表与回放

两者用同一个 thread_id 关联。分两层的理由：检查点结构跟着框架版本变，
界面不该依赖它；而会话库字段稳定，可以随意查询。

**每个函数都带 db_path 参数**：档案之间互不可见，默认值是 default 档案，
但界面必须显式传，否则切了档案还会往里写。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from core import config


@contextmanager
def _conn(db_path: Path | None = None):
    """必须是上下文管理器：`with sqlite3.connect(...) as conn` 只提交事务，不关闭连接。
    界面每次重跑都要查会话，泄漏的连接会一直攒着，直到触发 ResourceWarning 或文件锁。
    """

    path = Path(db_path) if db_path else config.sessions_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            thread_id  TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id  TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            steps      TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, id);
        """
    )
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def new_thread_id() -> str:
    """纯 uuid，不碰库——新建会话可以先拿 id 再决定落到哪个档案。"""

    return str(uuid.uuid4())


def save_message(thread_id: str, role: str, content: str, steps: list | None = None,
                 db_path: Path | None = None) -> None:
    """追加一条消息；会话第一条消息决定初始标题（之后可用 set_title 改）。"""

    with _conn(db_path) as conn:
        now = _now()
        title = content.strip().replace("\n", " ")[:30] or "新会话"
        conn.execute(
            "INSERT INTO sessions(thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at",
            (thread_id, title, now, now),
        )
        conn.execute(
            "INSERT INTO messages(thread_id, role, content, steps, created_at) VALUES (?, ?, ?, ?, ?)",
            (thread_id, role, content, json.dumps(steps or [], ensure_ascii=False), now),
        )


def list_sessions(db_path: Path | None = None, limit: int = 60) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT s.thread_id, s.title, s.updated_at, COUNT(m.id) AS n "
            "FROM sessions s LEFT JOIN messages m ON m.thread_id = s.thread_id "
            "GROUP BY s.thread_id ORDER BY s.updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_messages(thread_id: str, db_path: Path | None = None) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content, steps FROM messages WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "steps": json.loads(r["steps"])} for r in rows]


def set_title(thread_id: str, title: str, db_path: Path | None = None) -> None:
    """简报场景的标题应是「档位 × 日期」，由一期结束后回写，而不是首句前 30 字。"""

    with _conn(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE thread_id = ?",
            (title.strip() or "未命名", thread_id),
        )


def delete_session(thread_id: str, db_path: Path | None = None) -> None:
    with _conn(db_path) as conn:
        conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
