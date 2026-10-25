"""检索与抓取工具（进程内，同步实现）。

为什么同步：Agent 循环本身是异步的，同步工具会短暂阻塞事件循环。但检索是单用户、
串行发生的，阻塞 0.5–3 秒对界面无感；换成 async 工具则要赌框架对协程工具的支持，
多一处会坏的地方。这里选择确定性。

三个必须写进返回值的字段，每个都对应一次实测事故：
  date          缺失就填 None。模型看不到日期就会自己编一个
  provider_used 降级时告诉模型"这不是主源"，否则它会把降级结果当成正常证据
  quota         硬性检索次数，防止模型无限重检把一次简报跑成几十次调用
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import httpx

from core import config
from core.sources import classify, is_publish_page, title_key, url_norm
from tools.timebox import resolve_window

_TIMEOUT_SEARCH = 25
_TIMEOUT_FETCH = 30

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

_state = {"used": 0}


# ------------------------------------------------------------------ 配额
def reset_quota() -> None:
    """每期开始时清零。由 app.py 在切换会话时调用。"""

    _state["used"] = 0


def _quota() -> dict:
    limit = config.SEARCH_QUOTA
    return {"used": _state["used"], "limit": limit, "remaining": max(0, limit - _state["used"])}


# ------------------------------------------------------------------ 小工具
def _date_or_none(value) -> str | None:
    if not value:
        return None
    s = str(value).strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _hit(url, title, date, site, snippet, authority=None) -> dict:
    url = (url or "").strip()
    return {
        "title": (title or "").strip()[:100],
        "url": url,
        "site": (site or "").strip()[:40],
        "date": _date_or_none(date),
        "authority": authority,
        # 240 字是这个字段的平衡点：够模型判断主体与事件类型，又不至于一轮检索就吃掉 3k 上下文。
        # 细节留给 fetch_page，或者留给证据库。
        "snippet": (snippet or "").strip()[:240],
        "kind": classify(url),
        "publish_page": is_publish_page(url),
    }


def _dedupe(hits: list[dict]) -> tuple[list[dict], int]:
    """同一页面的不同 URL、同一篇稿子的跨站转载、以及技能明确排除的域名，都只留一条。"""

    seen_url, seen_title, out = set(), set(), []
    dropped = 0
    for h in hits:
        if not h["url"] or h["kind"] == "excluded":
            # 技能 industries/ai.md 第六节把 arXiv、Papers with Code 列为排除项：
            # 简报看「已可用的能力变化」，不看论文。在工具层拦掉，省上下文也省得模型误用。
            dropped += 1
            continue
        u, t = url_norm(h["url"]), title_key(h["title"])
        if u in seen_url or (t and t in seen_title):
            dropped += 1
            continue
        seen_url.add(u)
        if t:
            seen_title.add(t)
        out.append(h)
    return out, dropped


# ------------------------------------------------------------------ provider
def _deepseek(query: str, start: str, end: str, count: int, site: str, lang: str) -> list[dict]:
    key = config.search_key("deepseek")
    if not key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    base = (config.get("DEEPSEEK_BASE", "https://api.deepseek.com") or "").rstrip("/")

    parts = [query]
    if start and end:
        parts.append(f"after:{start} before:{end}")
    if site:
        parts.append(f"site:{site}")
    if lang == "en":
        parts.append("lang:en")

    resp = httpx.post(
        f"{base}/v1/tools/search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"query": " ".join(parts), "count": count, "need_content": False},
        timeout=_TIMEOUT_SEARCH,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    out = []
    for r in resp.json().get("results", []):
        out.append(
            _hit(
                r.get("url"),
                r.get("title"),
                r.get("date"),
                r.get("site_name"),
                r.get("summary") or r.get("snippet"),
                r.get("authority"),
            )
        )
    return out


def _bocha(query: str, start: str, end: str, count: int, site: str, lang: str) -> list[dict]:
    key = config.search_key("bocha")
    if not key:
        raise RuntimeError("未配置 BOCHA_API_KEY（中文降级源）")
    base = (config.get("BOCHA_BASE", "https://api.bochaai.com") or "").rstrip("/")

    body = {
        "query": f"{query} site:{site}" if site else query,
        "freshness": f"{start}..{end}" if start and end else "noLimit",
        "summary": True,
        "count": count,
    }
    resp = httpx.post(
        f"{base}/v1/web-search",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        timeout=_TIMEOUT_SEARCH,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    if data.get("code") not in (200, None):
        raise RuntimeError(f"code={data.get('code')} msg={data.get('msg')}")

    values = (((data.get("data") or {}).get("webPages") or {}).get("value")) or []
    return [
        _hit(
            r.get("url"),
            r.get("name"),
            r.get("datePublished"),
            r.get("siteName"),
            r.get("summary") or r.get("snippet"),
        )
        for r in values
    ]


def _mock(query: str, start: str, end: str, count: int, site: str, lang: str) -> list[dict]:
    """离线快照。演示兜底用，返回结构与真实检索一致。"""

    path = config.ROOT / "fixtures" / "mock.json"
    if not path.exists():
        raise RuntimeError("mock 模式需要 fixtures/mock.json，文件不存在")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("hits", [])
    if not rows:
        raise RuntimeError("fixtures/mock.json 还是空的，先跑一次真实检索把结果存进去（M6 做）")
    return [_hit(r.get("url"), r.get("title"), r.get("date"), r.get("site"), r.get("snippet"))
            for r in rows][:count]


_PROVIDERS = {"deepseek": _deepseek, "bocha": _bocha, "mock": _mock}


def _call(provider: str, **kwargs) -> dict:
    fn = _PROVIDERS.get(provider)
    if not fn:
        return {"hits": [], "error": f"未知 provider: {provider}"}
    try:
        hits = fn(**kwargs)
        hits, dropped = _dedupe(hits)
        return {"hits": hits, "error": None, "dropped": dropped}
    except Exception as exc:  # noqa: BLE001
        return {"hits": [], "error": f"{type(exc).__name__}: {exc}", "dropped": 0}


# ------------------------------------------------------------------ 对外工具
def web_search(
    query: str,
    lang: str = "zh",
    start: str = "",
    end: str = "",
    count: int = 10,
    site: str = "",
) -> dict:
    """检索网页，返回带日期与来源的事件候选。检索近期事件时必须用它，不要凭记忆回答。

    按事件类型检索，不要堆关键词。六类事件各检一轮：
    T1 模型发布 / T2 能力解锁 / T3 成本与可及性 / T4 推理部署 / T5 算力硬件 / T6 治理生态。

    Args:
        query: 检索词，建议 4–10 个字，一句话描述一个事件类型。最多 128 字。
        lang: "zh" 中文检索（默认）或 "en" 英文检索。国外主体（OpenAI、Anthropic、NVIDIA 等）
            的一手载体只有英文页，T1/T3/T5/T6 应各加一轮 en。
        start: 窗口起始日 YYYY-MM-DD。留空则自动取最近 7 天。
        end: 窗口结束日 YYYY-MM-DD。留空则取今天。
        count: 返回条数 1–10，默认 10。不要为了"多拿点"调大——多出来的几乎都是同事件的转载。
        site: 只检索该域名，例如 "openai.com"。**只在补一手证据时使用**，
            常规检索留空——加了 site 会丢掉大部分结果。
    """

    q = (query or "").strip()
    if not q:
        return {"error": "query 不能为空", "hits": [], "quota": _quota()}
    if len(q) > 128:
        return {"error": "query 超过 128 字，请缩短", "hits": [], "quota": _quota()}
    if lang not in ("zh", "en"):
        return {"error": f'lang 只能是 "zh" 或 "en"，收到 {lang!r}', "hits": [], "quota": _quota()}

    count = max(1, min(int(count or 10), 10))

    if not (start and end):
        win = resolve_window(7)
        start, end = start or win["start"], end or win["end"]
    for label, value in (("start", start), ("end", end)):
        if not _DATE_RE.match(value or ""):
            return {"error": f"{label} 必须是 YYYY-MM-DD，收到 {value!r}", "hits": [], "quota": _quota()}

    q_ = _quota()
    if q_["remaining"] <= 0:
        return {
            "error": f'本期检索配额已用完（{q_["limit"]} 次）。停止检索，按已获得的证据写，'
                     f'并在报告「检索缺口」里如实列出没覆盖的事件类型。',
            "hits": [],
            "quota": q_,
        }

    primary = config.SEARCH_PROVIDER
    fallback = config.SEARCH_FALLBACK
    kwargs = dict(query=q, start=start, end=end, count=count, site=site.strip(), lang=lang)

    _state["used"] += 1
    result = _call(primary, **kwargs)
    used, degraded, note = primary, False, ""

    if result["error"] or not result["hits"]:
        reason = result["error"] or "结果为空"
        # 英文维度不允许降级到中文源：博查的英文查询实测返回的全是中文媒体，
        # 降级会让「已做双语检索」变成假的，比报错更糟。
        if lang == "en":
            return {
                "error": f"英文检索失败（{reason}）。不要用中文结果顶替，把这一类型记为空白并在「检索缺口」写明。",
                "hits": [], "provider_used": primary, "degraded": False, "quota": _quota(),
            }
        if not fallback or fallback == primary:
            return {"error": reason, "hits": [], "provider_used": primary,
                    "degraded": False, "quota": _quota()}

        _state["used"] += 1
        r2 = _call(fallback, **kwargs)
        if r2["error"] or not r2["hits"]:
            return {
                "error": f"主源失败（{reason}）；降级源 {fallback} 也失败（{r2['error'] or '结果为空'}）",
                "hits": [], "provider_used": primary, "degraded": True, "quota": _quota(),
            }
        result, used, degraded = r2, fallback, True
        note = f"主源 {primary} 不可用，已降级到 {fallback}。结果可用，但请在方法论备注里注明本期降级。"

    return {
        "query": q,
        "lang": lang,
        "window": {"start": start, "end": end},
        "provider_used": used,
        "degraded": degraded,
        "hits": result["hits"],
        "dropped_duplicates": result.get("dropped", 0),
        "quota": _quota(),
        "error": None,
        "note": note or "date 为 None 的条目不得进入正文；authority 是站点权威度，不是一手。",
    }


def fetch_page(url: str, max_chars: int = 6000) -> dict:
    """抓取单个网页的正文，用于认定「一手」或核对细节。

    只在需要确认原始载体时使用（官方博客、产品页、定价页、代码仓库 release），
    不要对每条检索结果都抓一遍——会把一期简报的耗时翻三倍。

    Args:
        url: 完整 URL，来自 web_search 返回的 hits。
        max_chars: 正文截断长度，默认 6000 字。
    """

    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return {"error": f"url 必须以 http(s):// 开头，收到 {url!r}", "url": url}

    key = config.search_key("deepseek")
    if not key:
        return {"error": "抓取需要 DEEPSEEK_API_KEY（使用 DeepSeek Fetch 接口）", "url": url}
    base = (config.get("DEEPSEEK_BASE", "https://api.deepseek.com") or "").rstrip("/")

    try:
        resp = httpx.post(
            f"{base}/v1/tools/fetch",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"url": url},
            timeout=_TIMEOUT_FETCH,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "url": url}

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "url": url}

    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    text = data.get("content") or data.get("markdown") or data.get("text") or ""
    if isinstance(text, str):
        # DeepSeek 的 Markdown 导出会把 # - * 之类转义成 \# \-，
        # 不清掉的话模型照抄进正文，读者看到的就是一堆反斜杠
        text = re.sub(r"\\([#*_`\-\[\]()>+.!|])", r"\1", text)
    if not text and isinstance(data, dict):
        text = json.dumps(data, ensure_ascii=False)[:max_chars]

    limit = max(500, min(int(max_chars or 6000), 20000))
    return {
        "url": url,
        "title": data.get("title"),
        "published_at": _date_or_none(data.get("date") or data.get("published_at")),
        "text": text[:limit],
        "chars": len(text),
        "truncated": len(text) > limit,
        "kind": classify(url),
        "publish_page": is_publish_page(url),
        "error": None if text else "响应里没有正文",
    }
