"""Streamlit 界面。

启动：
    .venv/Scripts/python.exe -m streamlit run app.py

结构：
    顶栏   对话 / 证据 / 账本 / 简报（st.navigation 的 top 位置，与 Deploy 同一行）
    标题 + 档位徽章（四页共用，徽章首格是当前档案）
    侧栏   档案切换（弹出菜单）+ 新建会话 + 按档案折叠的会话列表 + 能力面板
    对话页 输入框固定在窗口底部，翻看历史时也能直接输入

**档案**：一个档案一套证据库、会话、检查点、账本、简报目录与补充记忆。
切档案时整台 Agent 换掉（旧的就地关闭，不留后台进程）——因为 MCP 子进程
启动时用 `--archive` 固定了库路径，换档案必须换进程。

界面层不含任何 Agent 知识：它只消费 build_agent()/stream_events()，
以及证据库与会话库的只读查询。要改 Agent 的能力改 agent/，不要改这里。

证据库由 MCP 进程持有，这里用只读连接看它——WAL 模式下多读者不影响写入。
"""

from __future__ import annotations

import html
import os
import re
import time
from pathlib import Path

import streamlit as st

from agent import runner
from agent.runtime import get_runtime
from core import archive, config, history, settings, store
from tools.search import reset_quota

st.set_page_config(page_title="AI 行业简报", layout="wide")

# 工具 → 阶段。取「最靠后」的那个，用来画进度条。
PHASES = [
    ("定档", {"get_current_time", "resolve_window", "write_todos"}),
    ("检索", {"web_search", "fetch_page"}),
    ("入库", {"evidence_add"}),
    ("判定", {"evidence_query", "evidence_update"}),
    ("交付", {"brief_lint", "publish_brief", "write_file"}),
]

CARD = ("flex:{flex};min-width:112px;padding:7px 12px;border:1px solid rgba(49,51,63,.14);"
        "border-radius:9px;background:rgba(250,250,251,.9);")
LABEL = "font-size:.72rem;color:#6b7280;line-height:1.5;white-space:nowrap;"
VALUE = "font-size:1.02rem;font-weight:600;color:#111827;line-height:1.4;white-space:nowrap;"


# ------------------------------------------------------------------ 档案与运行时
def _activate(archive_id: str, thread_id: str | None = None) -> None:
    """切档案（可选同时切会话）。

    只改 session_state 与查询参数，真正的 Agent 重建交给下一轮的 get_runtime：
    它会发现 `_rt.archive` 与当前档案不符，先关旧的再建新的。这样切换动作本身
    不阻塞（关进程可能要一秒），界面先刷新再等初始化。
    """

    archive.set_active(archive_id)
    st.query_params["archive"] = archive_id

    threads = st.session_state.setdefault("threads", {})
    tid = thread_id or threads.get(archive_id) or history.new_thread_id()
    threads[archive_id] = tid
    st.session_state.thread_id = tid
    st.query_params["thread"] = tid
    reset_quota()


def _runtime():
    return get_runtime(st.session_state, ARCHIVE)


def _drop_runtime() -> None:
    """关掉当前运行时，下一轮 rerun 会按新配置重建 Agent。

    换模型、换密钥之后必须调：Agent 里握着的是旧客户端，只改环境变量不会让它跟着变。
    """

    rt = st.session_state.get("_rt")
    if rt is not None:
        try:
            rt.close()
        except Exception:  # noqa: BLE001  关不掉也要让它从缓存里消失
            pass
        st.session_state.pop("_rt", None)


# ------------------------------------------------------------------ 只读数据
def load_period() -> dict | None:
    db = config.store_db(ARCHIVE)
    if not db.exists():
        return None
    try:
        pid = store.latest_period(db)
        if not pid:
            return None
        return {"id": pid, "stats": store.query_stats(db, pid)}
    except Exception:  # noqa: BLE001
        return None


def list_briefs() -> list[dict]:
    """跨档案扫描往期简报。

    档案删了简报还在（简报是交付物，不是运行态），所以这里不能只扫现存档案的目录，
    要扫磁盘。archive 字段为空的表示旧版直接放在根目录的文件。
    """

    root = config.BRIEFS_DIR
    if not root.exists():
        return []
    out: list[dict] = []
    for p in root.glob("*.md"):
        if not p.name.startswith("_"):
            out.append({"archive": "", "name": p.name, "path": p})
    for d in root.iterdir():
        if not d.is_dir():
            continue
        for p in d.glob("*.md"):
            if not p.name.startswith("_"):
                out.append({"archive": d.name, "name": p.name, "path": p})
    out.sort(key=lambda x: (x["path"].stat().st_mtime, x["name"]), reverse=True)
    return out


def parse_ledger() -> list[dict]:
    """把账本按「期 / 主线」拆成行。只做展示，不做校验。"""

    path = config.ledger_file(ARCHIVE)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        head, _, body = block.partition("\n")
        parts = head.split("｜")
        date = parts[0].strip()
        tier = parts[1].strip() if len(parts) > 1 else ""
        for ml in re.split(r"^### 主线", body, flags=re.M)[1:]:
            verdict = re.search(r"判断：(.+)", ml)
            refute = re.search(r"推翻条件：(.+)", ml)
            rows.append({
                "期": date,
                "档位": tier,
                "判断": (verdict.group(1).strip() if verdict else "")[:70],
                "推翻条件": (refute.group(1).strip() if refute else "")[:70],
            })
    return rows


def phase_state(steps: list[dict]) -> tuple[int, str]:
    """从工具调用序列反推阶段。不要求模型配合，也不依赖它自述。"""

    idx = -1
    for s in steps:
        if s.get("type") != "tool_call":
            continue
        for i, (_, names) in enumerate(PHASES):
            if s.get("name") in names:
                idx = max(idx, i)
    return idx, (PHASES[idx][0] if idx >= 0 else "")


# ------------------------------------------------------------------ 渲染
def trace_markdown(steps: list[dict]) -> str:
    if not steps:
        return "_等待第一步工具调用……_"
    calls = [s for s in steps if s["type"] == "tool_call"]
    lines = [f"**工具调用 {len(calls)} 次**", ""]
    for s in steps:
        indent = "　　" if s.get("depth") else ""
        if s["type"] == "tool_call":
            lines.append(f"{indent}- `{s['name']}`")
        else:
            lines.append(f"{indent}- `{s['name']}` → {s.get('chars') or 0} 字")
    return "\n".join(lines)


def render_steps(steps: list[dict]) -> None:
    for s in steps:
        if s["type"] == "tool_call":
            prefix = "　　↳ 子代理 · " if s.get("depth") else ""
            st.markdown(f"{prefix}`{s['name']}`")
            st.code(str(s.get("args"))[:600], language="json")
        elif s["type"] == "tool_result":
            st.caption(f"↩ `{s['name']}` 返回 {s.get('chars') or 0} 字")


def card(flex: str, label: str, value: str) -> str:
    return (f'<div style="{CARD.format(flex=flex)}">'
            f'<div style="{LABEL}">{html.escape(label)}</div>'
            f'<div style="{VALUE}">{html.escape(value)}</div></div>')


def render_badge(period: dict | None) -> None:
    """档位徽章。不用 st.metric——它的值字号不受控，五项并排时会挤掉长文本。"""

    archive_card = card("1.1", "档案", archive.name_of(ARCHIVE))
    if not period:
        st.markdown(f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:2px 0 10px;">'
                    f'{archive_card}</div>', unsafe_allow_html=True)
        st.caption("本档案还没有简报数据。发一句话开始。")
        return
    s = period["stats"]
    win = s.get("window") or {}
    tier = period["id"].split("|")
    aud, length = (tier[1], tier[2]) if len(tier) >= 3 else ("?", "?")
    span = f"{win.get('start', '?')} → {win.get('end', '?')}"
    html_row = (
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin:2px 0 10px;">'
        + archive_card
        + card("1", "受众 × 篇幅", f"{aud} × {length}")
        + card("1.7", "覆盖期", span)
        + card("0.9", "类型覆盖", f"{s['coverage']}/6")
        + card("0.9", "一手率", f"{int(s['primary_rate'] * 100)}%")
        + card("1", "入选 / 候选", f"{s['in_report']} / {s['total']}")
        + card("1.2", "模型", s.get("model") or "未记录")
        + "</div>"
    )
    st.markdown(html_row, unsafe_allow_html=True)


def short_title(title: str, n: int = 16) -> str:
    t = title.strip().replace("\n", " ")
    return t if len(t) <= n else t[:n] + "…"


def archive_label(archive_id: str) -> str:
    if not archive_id:
        return "未归档"
    return archive.name_of(archive_id) if archive.exists(archive_id) else f"{archive_id}（已删除）"


# ------------------------------------------------------------------ 对话框
@st.dialog("删除会话")
def dialog_delete_session(thread_id: str, title: str, aid: str) -> None:
    st.markdown(f"确定删除会话「**{short_title(title, 40)}**」？")
    st.caption("聊天记录与 Agent 的对话上下文会一并删除，不可恢复。已落盘的简报与账本不受影响。")
    c1, c2 = st.columns(2)
    if c1.button("确认删除", type="primary", width="stretch"):
        history.delete_session(thread_id, config.sessions_db(aid))
        runner.RUNS.pop(thread_id, None)
        try:
            _runtime().call(_runtime().agent.checkpointer.adelete_thread(thread_id))
        except Exception:  # noqa: BLE001  上下文可能本就不存在
            pass
        threads = st.session_state.setdefault("threads", {})
        if threads.get(aid) == thread_id:
            threads[aid] = history.new_thread_id()
        if aid == ARCHIVE and st.session_state.get("thread_id") == thread_id:
            st.session_state.thread_id = threads[aid]
            st.query_params["thread"] = threads[aid]
            reset_quota()
        st.session_state.pop("_pending", None)
        st.rerun()
    if c2.button("取消", width="stretch"):
        st.session_state.pop("_pending", None)
        st.rerun()


@st.dialog("重命名会话")
def dialog_rename_session(thread_id: str, title: str, aid: str) -> None:
    new_title = st.text_input("会话标题", value=title, key=f"rn-{thread_id}")
    c1, c2 = st.columns(2)
    if c1.button("保存", type="primary", width="stretch"):
        history.set_title(thread_id, new_title, config.sessions_db(aid))
        st.session_state.pop("_pending", None)
        st.rerun()
    if c2.button("取消", width="stretch"):
        st.session_state.pop("_pending", None)
        st.rerun()


@st.dialog("新建会话")
def dialog_new_session() -> None:
    ids = [a["id"] for a in archive.list_archives()]
    options = [*ids, "__new__"]
    at_cap = len(ids) >= archive.MAX_ARCHIVES

    pick = st.selectbox(
        "档案", options,
        index=ids.index(ARCHIVE) if ARCHIVE in ids else 0,
        format_func=lambda x: "＋ 新建档案" if x == "__new__" else archive.name_of(x),
    )
    new_name = ""
    if pick == "__new__":
        if at_cap:
            st.warning(f"档案已达上限 {archive.MAX_ARCHIVES} 个，删掉一个不用的再建。")
        new_name = st.text_input("新档案名", placeholder="例如：课程作业 / 给同事的周报")

    c1, c2 = st.columns(2)
    if c1.button("创建", type="primary", width="stretch"):
        try:
            aid = pick
            if pick == "__new__":
                if not new_name.strip():
                    st.error("先给档案起个名字")
                    return
                aid = archive.create(new_name)
            # 新建会话 = 一定用新 thread，不复用该档案上次的位置
            _activate(aid, thread_id=history.new_thread_id())
            st.session_state.pop("_pending", None)
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
    if c2.button("取消", width="stretch"):
        st.session_state.pop("_pending", None)
        st.rerun()


@st.dialog("新建档案")
def dialog_new_archive() -> None:
    name = st.text_input("档案名", placeholder="例如：课程作业 / 给同事的周报")
    st.caption(f"每个档案是一套独立的证据库、会话与账本，最多 {archive.MAX_ARCHIVES} 个。")
    c1, c2 = st.columns(2)
    if c1.button("创建", type="primary", width="stretch"):
        try:
            if not name.strip():
                st.error("先给档案起个名字")
                return
            aid = archive.create(name)
            _activate(aid, thread_id=history.new_thread_id())
            st.session_state.pop("_pending", None)
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
    if c2.button("取消", width="stretch"):
        st.session_state.pop("_pending", None)
        st.rerun()


@st.dialog("重命名档案")
def dialog_rename_archive(aid: str) -> None:
    name = st.text_input("档案名", value=archive.name_of(aid), key=f"arn-{aid}")
    c1, c2 = st.columns(2)
    if c1.button("保存", type="primary", width="stretch"):
        archive.rename(aid, name)
        st.session_state.pop("_pending", None)
        st.rerun()
    if c2.button("取消", width="stretch"):
        st.session_state.pop("_pending", None)
        st.rerun()


@st.dialog("删除档案")
def dialog_delete_archive(aid: str) -> None:
    m = archive.read_meta(aid)
    st.markdown(f"确定删除档案「**{m['name']}**」？")
    st.caption(
        f"会删除：会话 {m['sessions']} 条、证据 {m['evidence']} 条、检查点、账本、补充记忆。"
        "**已生成的简报会保留**，往期简报页仍能看到（档案名显示为 id）。"
    )
    st.warning("此操作不可撤销。")
    c1, c2 = st.columns(2)
    if c1.button("确认删除", type="primary", width="stretch"):
        # 先关运行时再删：checkpoints.db 被 Agent 持有，Windows 上开着就删不掉，
        # 而 rmtree 的失败会被静默吞掉——表现为"删了但下次启动又回来了"。
        if aid == ARCHIVE:
            _drop_runtime()
        try:
            info = archive.delete(aid)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        if aid == ARCHIVE:
            _activate(config.DEFAULT_ARCHIVE, thread_id=history.new_thread_id())
        st.session_state.pop("_pending", None)
        st.session_state["_flash"] = (
            f"已删除档案「{m['name']}」：{'、'.join(info['removed']) or '无'}；"
            f"保留简报 {info['kept_briefs']} 份。"
        )
        st.rerun()
    if c2.button("取消", width="stretch"):
        st.session_state.pop("_pending", None)
        st.rerun()


# ------------------------------------------------------------------ 四个页面
def _render_running(run, left, trace_ph) -> None:
    """渲染一次进行中的运行。会阻塞到跑完为止。

    阻塞是故意的：Streamlit 要更新界面就得重跑脚本。但它不再承担"记录"的责任——
    记录由 Run 自己写，所以这里被中断（切页面、关标签）不会丢东西。
    再次进来时读的是 Run 里已经累积的 steps/text，接得上前一次。
    """

    with left:
        with st.chat_message("user"):
            st.markdown(run.question)

        with st.chat_message("assistant"):
            status = st.status("思考中……", expanded=True)
            placeholder = st.empty()
            phase_ph = st.empty()
            shown = 0

            while True:
                steps = list(run.steps)
                for ev in steps[shown:]:
                    with status:
                        render_steps([ev])
                shown = len(steps)

                calls = run.calls()
                if calls:
                    i, ph = phase_state(steps)
                    elapsed = int(time.monotonic() - run.started_at)
                    # 带上时长：卡住时用户能自己判断"是这个慢，还是死了"
                    status.update(label=f"工作中……{ph}（{calls} 次工具｜{elapsed // 60:02d}:{elapsed % 60:02d}）",
                                  state="running")
                    phase_ph.progress(min((i + 1) / len(PHASES), 1.0), text=f"阶段：{ph}")
                if run.text:
                    placeholder.markdown(run.text + "▌")
                trace_ph.markdown(trace_markdown(steps))

                if run.done.is_set():
                    break
                time.sleep(0.4)

            calls = run.calls()
            status.update(label=f"完成（{calls} 次工具调用）" if calls else "完成（未调用工具）",
                          state="complete", expanded=False)
            placeholder.markdown(run.answer or run.text or "（没有生成回答）")

    # 跑完重新加载一次，让这条消息走"历史消息"的渲染路径（带步骤折叠）
    st.rerun()


def page_chat() -> None:
    run = runner.get(THREAD_ID)
    running = run is not None and run.running

    left, right = st.columns([3.1, 1], gap="large")

    with right:
        st.caption("执行轨迹")
        trace_area = st.container(height=500, border=True)
        trace_ph = trace_area.empty()
        if run is not None:
            trace_ph.markdown(trace_markdown(run.steps))
        else:
            last_steps = next((m["steps"] for m in reversed(MESSAGES) if m.get("steps")), [])
            trace_ph.markdown(trace_markdown(last_steps))

    # 进行中的那条提问由 _render_running 自己画，这里跳过，避免重复
    history_msgs = MESSAGES
    if running and MESSAGES and MESSAGES[-1]["role"] == "user" \
            and MESSAGES[-1]["content"] == run.question:
        history_msgs = MESSAGES[:-1]

    with left:
        for msg in history_msgs:
            with st.chat_message(msg["role"]):
                if msg.get("steps"):
                    calls = sum(1 for e in msg["steps"] if e["type"] == "tool_call")
                    with st.expander(f"执行过程（{calls} 次工具调用）", expanded=False):
                        render_steps(msg["steps"])
                st.markdown(msg["content"])

    # 放在页面根部（不在 column / container 里），Streamlit 才会把它钉在窗口底部
    question = st.chat_input("说一句你要什么，例如：做一份最近一周的 AI 行业简报",
                             disabled=running)
    if question:
        history.save_message(THREAD_ID, "user", question, db_path=config.sessions_db(ARCHIVE))
        # 运行一旦开始就自己管自己：落库、报错、收尾都不依赖这个页面还活着
        runner.start(_runtime(), question, THREAD_ID, config.sessions_db(ARCHIVE), ARCHIVE)
        st.rerun()

    run = runner.get(THREAD_ID)
    if run is not None and run.running:
        _render_running(run, left, trace_ph)


def page_evidence() -> None:
    if not PERIOD:
        st.caption(f"档案「{archive.name_of(ARCHIVE)}」还没有证据。跑一期后这里会显示候选、簇与一手率。")
        return
    s = PERIOD["stats"]
    c = st.columns(5)
    c[0].metric("候选", s["total"])
    c[1].metric("已判定", s["judged"])
    c[2].metric("回声簇", s["echo"])
    c[3].metric("48h 内", s["within_48h"])
    c[4].metric("边界条目", s["boundary"])
    st.caption("事件类型分布：" + "　".join(f"{k}:{v}" for k, v in s["by_type"].items()))

    db = config.store_db(ARCHIVE)
    clusters = store.query_clusters(db, PERIOD["id"])
    if clusters:
        st.markdown("**簇**（信源数决定它能不能独立支撑一条判断）")
        st.dataframe(
            [{"簇": x["cluster_id"], "类型": x["event_type"], "首日": x["first_date"],
              "信源数": x["source_count"], "形态": x["form"],
              "入选": "是" if x["in_report"] else "否",
              "标题": (x["titles"][0] if x["titles"] else "")} for x in clusters],
            width="stretch", hide_index=True,
        )
    rows = store.query_rows(db, PERIOD["id"], limit=60)
    if rows:
        with st.expander(f"全部候选（{len(rows)} 条）", expanded=False):
            st.dataframe(
                [{"id": r["id"], "日期": r["d_public"], "类型": r["event_type"],
                  "主体": r["subject"], "位移": r["displacement"] or "—",
                  "半径": r["radius"] or "—", "一手": "是" if r["is_primary"] else "否",
                  "入选": "是" if r["in_report"] else "否",
                  "信源": r["site"], "标题": r["title"][:40]} for r in rows],
                width="stretch", hide_index=True,
            )


def page_ledger() -> None:
    rows = parse_ledger()
    if not rows:
        st.caption(f"档案「{archive.name_of(ARCHIVE)}」的账本还是空的。每期交付后自动追加一节。")
        return
    st.caption(f"共 {len({r['期'] for r in rows})} 期。账本是跨期回检的依据，只追加不回改。")
    st.dataframe(rows, width="stretch", hide_index=True)


def page_briefs() -> None:
    briefs = list_briefs()
    if not briefs:
        st.caption("还没有落盘的简报。")
        return
    labels = [f"{archive_label(b['archive'])} · {b['name']}" for b in briefs]
    st.caption(f"共 {len(briefs)} 份，跨 {len({b['archive'] for b in briefs})} 个档案（含已删除档案留下的报告）。")
    pick = st.selectbox("选择一期", labels, index=0)
    chosen = briefs[labels.index(pick)]
    st.download_button("下载 Markdown", chosen["path"].read_text(encoding="utf-8"),
                       file_name=chosen["name"], key="dl-brief")
    st.markdown(chosen["path"].read_text(encoding="utf-8"))


def page_settings() -> None:
    """模型与环境变量。

    两条安全约定：
      1. 密钥永远不回显。输入框留空 + placeholder 显示 `sk-…a1b2`，保存时只提交改过的项；
      2. 有任务在跑时禁止保存。改 key 要重建 Agent，而重建会关掉正在跑的运行时。
    """

    if BUSY:
        st.warning("有任务在跑。等这一期结束再改配置——保存会重建 Agent，正在跑的会被中断。")

    # ---- 模型
    st.markdown("**模型**")
    cur = settings.current_model()
    ids = list(settings.MODELS)
    labels = {k: f"{v['label']}　·　{v['vendor']}" for k, v in settings.MODELS.items()}
    pick = st.radio("选择模型", ids, index=ids.index(cur),
                    format_func=lambda x: labels[x], horizontal=True,
                    label_visibility="collapsed", disabled=BUSY)
    s = settings.MODELS[pick]
    ready, why = settings.model_ready(pick)
    st.caption(f"{s['note']}　｜　接口 {s['base_url']}　｜　密钥 "
               + (f"`{s['key_env']}` 已配置" if ready else f"**{why}**"))
    col_a, col_b = st.columns([1, 3])
    if col_a.button("测试连通", disabled=not ready, width="stretch"):
        st.session_state["_probe"] = {**settings.probe(pick), "model": pick}
    res = st.session_state.get("_probe") or {}
    # 只在结果属于当前选中的模型时显示：换个模型还挂着上一个的"连通正常"会误导
    if res.get("model") == pick:
        if res.get("ok"):
            col_b.success(f"连通正常，模型回复：{res['detail'] or '（空）'}")
        else:
            col_b.error(f"不通（{res.get('status', '-')}）：{res['detail'][:200]}")

    if pick != cur:
        if st.button("切换到该模型", type="primary", disabled=BUSY or not ready):
            settings.set_model(pick)
            _drop_runtime()
            st.session_state["_flash"] = f"已切换到 {s['label']}，Agent 已重建"
            st.rerun()
        if not ready:
            st.caption("该模型缺密钥，先在下面填好再切。")

    st.divider()

    # ---- 密钥
    st.markdown("**密钥**")
    st.caption("留空表示不修改。已配置的只显示前后各几位，页面不回显完整值。")
    env = settings.read_env()
    inputs: dict[str, str] = {}
    for key, desc in settings.EDITABLE_ENV.items():
        if key in settings.SECRET_KEYS:
            inputs[key] = st.text_input(
                f"{key}　{desc}", value="", type="password",
                placeholder=f"当前：{settings.mask(env.get(key, ''))}",
                key=f"env-{key}", disabled=BUSY,
            )

    st.markdown("**检索**")
    providers = ["deepseek", "bocha", "mock"]
    cur_provider = env.get("SEARCH_PROVIDER") or config.SEARCH_PROVIDER
    prov = st.selectbox("主检索源", providers,
                        index=providers.index(cur_provider) if cur_provider in providers else 0,
                        key="env-provider", disabled=BUSY)
    fallbacks = ["", "bocha"]
    cur_fb = env.get("SEARCH_FALLBACK") or ""
    fb = st.selectbox("中文降级源（英文查询不降级）", fallbacks,
                      index=fallbacks.index(cur_fb) if cur_fb in fallbacks else 0,
                      format_func=lambda x: x or "不降级",
                      key="env-fallback", disabled=BUSY)
    quota = st.number_input("每期检索次数上限", min_value=4, max_value=80,
                            value=int(env.get("SEARCH_QUOTA") or config.SEARCH_QUOTA),
                            step=2, key="env-quota", disabled=BUSY)

    if st.button("保存配置", type="primary", disabled=BUSY):
        changes = {k: v.strip() for k, v in inputs.items() if v.strip()}
        changes["SEARCH_PROVIDER"] = prov
        changes["SEARCH_FALLBACK"] = fb
        changes["SEARCH_QUOTA"] = str(int(quota))
        changed = settings.update_env(changes)
        settings.reload_env()
        _drop_runtime()  # key 变了，旧的客户端不能继续用
        st.session_state["_flash"] = f"已保存：{'、'.join(changed) or '无变化'}（Agent 已重建）"
        st.rerun()

    st.divider()
    st.markdown("**当前生效值**")
    rows = [{"项": "模型", "值": settings.current_model()},
            {"项": "主检索源", "值": config.SEARCH_PROVIDER},
            {"项": "降级源", "值": config.SEARCH_FALLBACK or "不降级"},
            {"项": "检索配额", "值": str(config.SEARCH_QUOTA)}]
    for key, desc in settings.EDITABLE_ENV.items():
        if key in settings.SECRET_KEYS:
            rows.append({"项": key, "值": settings.mask(settings.read_env().get(key, ""))})
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(f".env：{settings.env_path().relative_to(config.ROOT)}　"
               f"档案：{config.archive_dir(ARCHIVE).relative_to(config.ROOT)}")


# ------------------------------------------------------------------ 初始化
ARCHIVES = archive.list_archives()
_ids = [a["id"] for a in ARCHIVES]

# 当前档案的取值顺序：URL 参数 → 上次使用（runtime/archives/.active）→ default。
# URL 参数优先，是因为刷新与新开标签页时它还在，而 .active 是全局单值，
# 同一个浏览器多开两个标签页会互相覆盖。
_want = st.query_params.get("archive")
if _want not in _ids:
    _want = archive.active()
ARCHIVE: str = _want if _want in _ids else config.DEFAULT_ARCHIVE
st.query_params["archive"] = ARCHIVE

# 会话按档案各记一个位置：切走再切回来，还在原来那条会话上
_THREADS = st.session_state.setdefault("threads", {})
THREAD_ID: str = _THREADS.get(ARCHIVE) or st.query_params.get("thread") or history.new_thread_id()
_THREADS[ARCHIVE] = THREAD_ID
st.query_params["thread"] = THREAD_ID

try:
    INFO = _runtime().info
except Exception as exc:  # noqa: BLE001
    st.error(f"Agent 初始化失败：{exc}")
    st.info("检查 .env 里的 DASHSCOPE_API_KEY 与 LLM_BASE_URL（模板见 .env.example）。")
    st.stop()

# 有任务在跑时，切档案 / 换会话 / 删除一律禁用：切档案会关闭当前运行时，
# 连带停掉它的事件循环，正在跑的那一期会被腰斩。宁可挡住，也不要让用户
# 在"看着正常"的情况下把一次十几分钟的运行弄坏。
BUSY = runner.busy()

SESSIONS_BY_ARCHIVE = {a["id"]: history.list_sessions(config.sessions_db(a["id"]), limit=12)
                       for a in ARCHIVES}
MESSAGES = history.load_messages(THREAD_ID, config.sessions_db(ARCHIVE))
PERIOD = load_period()

_flash = st.session_state.pop("_flash", None)


# ------------------------------------------------------------------ 侧栏（四页共用）
def _session_row(aid: str, s: dict, current: bool) -> None:
    col_a, col_b = st.columns([0.78, 0.22], gap="small")
    if col_a.button(
        ("▶ " if current else "") + short_title(s["title"]),
        key=f"s-{aid}-{s['thread_id']}",
        help=f"{s['title']}｜{s['updated_at']}｜{s['n']} 条",
        width="stretch", disabled=current or BUSY,
    ):
        if aid != ARCHIVE:
            _activate(aid, thread_id=s["thread_id"])
        else:
            st.session_state.thread_id = s["thread_id"]
            st.query_params["thread"] = s["thread_id"]
        st.rerun()
    with col_b.popover("⋯", width="stretch"):
        st.caption(f"{s['updated_at']}｜{s['n']} 条")
        if st.button("重命名", key=f"rnb-{aid}-{s['thread_id']}", width="stretch",
                     disabled=BUSY):
            st.session_state["_pending"] = {"kind": "rename_session",
                                            "thread": s["thread_id"], "archive": aid}
            st.rerun()
        if st.button("删除", key=f"del-{aid}-{s['thread_id']}", width="stretch",
                     disabled=BUSY):
            st.session_state["_pending"] = {"kind": "delete_session",
                                            "thread": s["thread_id"], "archive": aid}
            st.rerun()


with st.sidebar:
    st.markdown("**AI 行业简报**")

    # 档案切换用弹出菜单而不是下拉框：下拉框在"档案被别处的操作改掉"时
    # 会与控件自身状态打架（它保留上一次选中值），表现为切了又弹回去。
    with st.popover(f"档案 · {archive.name_of(ARCHIVE)}", width="stretch"):
        st.caption(f"共 {len(ARCHIVES)} / {archive.MAX_ARCHIVES} 个。每个档案独立证据库、会话与账本。")
        for _a in ARCHIVES:
            _cur = _a["id"] == ARCHIVE
            if st.button(("▶ " if _cur else "") + _a["name"],
                         key=f"arch-{_a['id']}", width="stretch", disabled=_cur or BUSY,
                         help=f"会话 {_a['sessions']}｜证据 {_a['evidence']}"):
                _activate(_a["id"])
                st.rerun()
        st.divider()
        if st.button("＋ 新建档案", key="arch-new", width="stretch", disabled=BUSY):
            st.session_state["_pending"] = {"kind": "new_archive"}
            st.rerun()
        if st.button("重命名当前档案", key="arch-rename", width="stretch", disabled=BUSY):
            st.session_state["_pending"] = {"kind": "rename_archive", "archive": ARCHIVE}
            st.rerun()
        if st.button("删除当前档案", key="arch-delete", width="stretch",
                     disabled=BUSY or ARCHIVE == config.DEFAULT_ARCHIVE,
                     help="默认档案不可删除" if ARCHIVE == config.DEFAULT_ARCHIVE else None):
            st.session_state["_pending"] = {"kind": "delete_archive", "archive": ARCHIVE}
            st.rerun()

    if BUSY:
        st.caption("⏳ 有任务在跑，结束后才能切换档案或会话")
    if st.button("＋ 新会话", type="primary", width="stretch", disabled=BUSY):
        st.session_state["_pending"] = {"kind": "new_session"}
        st.rerun()

    st.divider()
    st.caption("历史会话")
    # 会话按档案折叠。当前档案默认展开，其余收起——列表会越用越长，
    # 不折叠的话五个档案各十条会把侧栏撑成一整页。
    for _a in ARCHIVES:
        _rows = SESSIONS_BY_ARCHIVE.get(_a["id"], [])
        _label = f"{_a['name']}（{len(_rows)}）"
        with st.expander(_label, expanded=(_a["id"] == ARCHIVE)):
            if not _rows:
                st.caption("还没有会话")
            for _s in _rows:
                _session_row(_a["id"], _s, _a["id"] == ARCHIVE and _s["thread_id"] == THREAD_ID)

    st.divider()
    with st.expander("能力与配置", expanded=False):
        st.markdown(f"**模型**　`{INFO['model']}`")
        st.markdown(f"**检索**　`{INFO['search_provider']}`"
                    + (f"（降级 `{INFO['search_fallback']}`）" if INFO["search_fallback"] else ""))
        st.markdown("**工具**　" + "、".join(f"`{t}`" for t in INFO["custom_tools"]))
        st.markdown("**证据库**　" + "、".join(f"`{t}`" for t in INFO["mcp_tools"]))
        st.markdown("**技能**　" + ("、".join(f"`{s}`" for s in INFO["skills"]) or "未接入"))
        st.markdown("**子代理**　" + "、".join(f"`{s}`" for s in INFO["subagents"]))
        st.markdown("**可写目录**　" + "、".join(f"`{p}`" for p in INFO["write_roots"]))
        for _w in INFO["warnings"]:
            st.warning(_w)

    st.caption(f"简报：outputs/briefs/{ARCHIVE}/　账本：docs/ledger/{ARCHIVE}/")
    st.caption("本简报基于公开信息，重要事实以官方发布为准；不含投资建议。")


# ------------------------------------------------------------------ 待处理操作
# 弹窗不能直接在 popover 里调用：一旦在弹窗内点任何控件，脚本会整体重跑，
# 而 popover 里的分支已经不再成立，弹窗会自己消失。
# 所以用 session_state 记下待办，在主流程里稳定地渲染它，直到用户确认或取消。
_pending = st.session_state.get("_pending")
if _pending:
    _kind = _pending.get("kind")
    _aid = _pending.get("archive", "")
    if _kind == "delete_session":
        _t = next((s["title"] for s in SESSIONS_BY_ARCHIVE.get(_aid, [])
                   if s["thread_id"] == _pending["thread"]), "")
        dialog_delete_session(_pending["thread"], _t, _aid)
    elif _kind == "rename_session":
        _t = next((s["title"] for s in SESSIONS_BY_ARCHIVE.get(_aid, [])
                   if s["thread_id"] == _pending["thread"]), "")
        dialog_rename_session(_pending["thread"], _t, _aid)
    elif _kind == "new_session":
        dialog_new_session()
    elif _kind == "new_archive":
        dialog_new_archive()
    elif _kind == "rename_archive":
        dialog_rename_archive(_aid)
    elif _kind == "delete_archive":
        dialog_delete_archive(_aid)


# ------------------------------------------------------------------ 共用头部
st.markdown("### AI 行业简报")
if _flash:
    st.success(_flash)
render_badge(PERIOD)


# ------------------------------------------------------------------ 顶部导航
# position="top" 把页面切换放进顶栏（与 Deploy、设置同一行），不占正文空间。
#
# APP_PAGE 只供无头验收使用：AppTest 无法切换函数式页面（switch_page 只认文件路径），
# 所以用环境变量指定默认页，逐个页面跑一遍。正常运行时不需要设置。
_PAGE_DEFS = [
    ("chat", "对话", page_chat),
    ("evidence", "证据", page_evidence),
    ("ledger", "账本", page_ledger),
    ("briefs", "简报", page_briefs),
    ("settings", "设置", page_settings),
]
_default_page = os.getenv("APP_PAGE", "chat")
if _default_page not in {k for k, _, _ in _PAGE_DEFS}:
    _default_page = "chat"

nav = st.navigation(
    [st.Page(fn, title=title, url_path=key, default=(key == _default_page))
     for key, title, fn in _PAGE_DEFS],
    position="top",
)
nav.run()
