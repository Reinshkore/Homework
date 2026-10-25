#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索连通性与质量探针（M0 决策用）

目的：用同一个查询、同一个时间窗，横向比较各检索源，再决定 SEARCH_PROVIDER。

特点：
- 只用标准库，不需要装任何依赖，也不依赖项目其它代码
- 未填 key 的 provider 自动跳过
- 同时输出两个关键指标：**日期完整率**与**一手候选率**
  （前者决定时间闸门能不能执行，后者决定证据强度上限）

用法：
    python scripts/probe_search.py
    python scripts/probe_search.py --days 3
    python scripts/probe_search.py --queries "开源 模型 权重" "推理 芯片 发布"
    python scripts/probe_search.py --provider deepseek
    python scripts/probe_search.py --json out.json

判读标准（写进 docs/项目计划 的 D1 里）：
- 日期完整率 < 60%：该源无法支撑 48h 新近配额，直接淘汰
- 一手候选率 = 0：能出简报，但一手率注定为 0，报告要注明"全部结论依赖二手转述"
- 任一 provider 报 401/403：先查 key，不要急着换源
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

# ---------------------------------------------------------------- 域名分级
# 规则来源：ai-brief/references/industries/ai.md 第六节「信源」
PRIMARY_DOMAINS = {
    # 模型层
    "openai.com", "anthropic.com", "deepmind.google", "blog.google", "ai.google",
    "meta.com", "ai.meta.com", "x.ai", "deepseek.com", "qwen.ai",
    "tongyi.aliyun.com", "aliyun.com", "moonshot.cn", "zhipuai.cn", "bigmodel.cn",
    "minimax.io", "minimaxi.com", "stepfun.com", "baichuan-ai.com", "01.ai",
    "mistral.ai", "cohere.com", "ai21.com",
    # 算力层
    "nvidia.com", "amd.com", "tsmc.com", "huawei.com", "hisilicon.com", "cambricon.com",
    # 云与应用层
    "microsoft.com", "azure.com", "aws.amazon.com", "cloud.google.com",
    "volcengine.com", "cursor.com", "perplexity.ai", "bytedance.com", "tencent.com",
    # 代码与权重
    "github.com", "huggingface.co", "hf.co", "modelscope.cn", "ollama.com",
    # 研究机构自有发布页（模型卡/技术报告首发处）
    "baai.ac.cn", "shlab.org.cn",
}
SECONDARY_DOMAINS = {
    "jiqizhixin.com", "qbitai.com", "theinformation.com", "venturebeat.com",
    "techcrunch.com", "36kr.com", "infoq.cn", "sspai.com", "ithome.com",
}
EXCLUDED_DOMAINS = {"arxiv.org", "paperswithcode.com", "semanticscholar.org"}


def classify(url: str) -> str:
    """返回 primary / secondary / excluded / other。只按域名判，不做内容判断。"""
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return "other"
    host = host[4:] if host.startswith("www.") else host
    for d in EXCLUDED_DOMAINS:
        if host == d or host.endswith("." + d):
            return "excluded"
    for d in PRIMARY_DOMAINS:
        if host == d or host.endswith("." + d):
            return "primary"
    for d in SECONDARY_DOMAINS:
        if host == d or host.endswith("." + d):
            return "secondary"
    return "other"


# ---------------------------------------------------------------- 基础工具
def load_env(path: Path) -> None:
    """极简 .env 解析：KEY=VALUE，忽略 # 注释与空行。已存在的环境变量优先。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def post_json(url: str, headers: dict, payload: dict, timeout: int = 25):
    """返回 (status, data_or_text, 耗时毫秒)。不抛异常，错误交给调用方展示。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            cost = int((time.time() - t0) * 1000)
            try:
                return resp.status, json.loads(raw), cost
            except json.JSONDecodeError:
                return resp.status, raw[:800], cost
    except urllib.error.HTTPError as e:
        cost = int((time.time() - t0) * 1000)
        detail = e.read().decode("utf-8", errors="replace")[:800]
        return e.code, detail, cost
    except Exception as e:  # 网络、DNS、超时
        return 0, f"{type(e).__name__}: {e}", int((time.time() - t0) * 1000)


def parse_date(value) -> str | None:
    """把各家的日期格式统一成 YYYY-MM-DD，解析不出来就返回 None。

    必须返回 None 而不是截断原文：Tavily 返回的是 RFC1123（"Sat, 24 Oct 2026 ..."），
    直接切片会得到 "Sat, 24 Oc" 这种伪日期，把「日期完整率」算成 1.00。
    度量的口径错了，决策就跟着错。
    """
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    # ISO 系：2026-10-24 / 2026-10-24T05:41:50Z
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # 常见中文格式：2026/10/24、2026年10月24日
    m = re.match(r"^(\d{4})[/年.](\d{1,2})[/月.](\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    # RFC1123 及类似
    try:
        return parsedate_to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        return None


def norm(title, url, date, site, snippet) -> dict:
    return {
        "title": (title or "").strip()[:90],
        "url": (url or "").strip(),
        "date": parse_date(date),
        "site": (site or "").strip(),
        "snippet": (snippet or "").strip()[:160],
        "kind": classify(url or ""),
    }


def title_key(title: str) -> str:
    """标题指纹：去掉空白与标点后取前 24 字，用于识别跨站转载。"""
    t = re.sub(r"[\s\-_|·,，。:：!！?？\"'“”‘’()（）\[\]【】<>《》]", "", title or "")
    return t[:24].lower()


# ---------------------------------------------------------------- 各 provider
def search_deepseek(query: str, start: str, end: str, count: int):
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None
    base = (os.getenv("DEEPSEEK_BASE") or "https://api.deepseek.com").rstrip("/")
    # DeepSeek 用查询语法做时间过滤，运算符拼在 query 里
    q = f"{query} after:{start} before:{end}"
    status, data, cost = post_json(
        f"{base}/v1/tools/search",
        {"Authorization": f"Bearer {key}"},
        {"query": q, "count": count, "need_content": False},
    )
    if status != 200 or not isinstance(data, dict):
        return {"error": f"HTTP {status}: {str(data)[:300]}", "ms": cost}
    rows = [
        norm(r.get("title"), r.get("url"), r.get("date"),
             r.get("site_name"), r.get("summary") or r.get("snippet"))
        for r in data.get("results", [])
    ]
    # authority 0~4，>=3 为极权威；附在 site 后面便于人工看
    for row, r in zip(rows, data.get("results", [])):
        row["authority"] = r.get("authority")
    return {"rows": rows, "ms": cost}


def search_bocha(query: str, start: str, end: str, count: int):
    key = os.getenv("BOCHA_API_KEY", "").strip()
    if not key:
        return None
    base = (os.getenv("BOCHA_BASE") or "https://api.bochaai.com").rstrip("/")
    status, data, cost = post_json(
        f"{base}/v1/web-search",
        {"Authorization": f"Bearer {key}"},
        {"query": query, "freshness": f"{start}..{end}", "summary": True, "count": count},
    )
    if status != 200 or not isinstance(data, dict):
        return {"error": f"HTTP {status}: {str(data)[:300]}", "ms": cost}
    if data.get("code") not in (200, None):
        return {"error": f"code={data.get('code')} msg={data.get('msg')}", "ms": cost}
    values = (((data.get("data") or {}).get("webPages") or {}).get("value")) or []
    rows = [
        norm(r.get("name"), r.get("url"), r.get("datePublished"),
             r.get("siteName"), r.get("summary") or r.get("snippet"))
        for r in values
    ]
    return {"rows": rows, "ms": cost}


def search_zhipu(query: str, start: str, end: str, count: int):
    key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not key:
        return None
    # 智谱只有相对窗口，没有任意区间 —— 这本身就是一项决策依据
    days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days + 1
    recency = ("oneDay" if days <= 1 else "oneWeek" if days <= 7
               else "oneMonth" if days <= 31 else "oneYear" if days <= 365 else "noLimit")
    status, data, cost = post_json(
        "https://open.bigmodel.cn/api/paas/v4/web_search",
        {"Authorization": f"Bearer {key}"},
        {"search_query": query, "search_engine": "search_std", "search_intent": False,
         "count": count, "search_recency_filter": recency},
    )
    if status != 200 or not isinstance(data, dict):
        return {"error": f"HTTP {status}: {str(data)[:300]}", "ms": cost}
    rows = [
        norm(r.get("title"), r.get("link"), r.get("publish_date"),
             r.get("media"), r.get("content"))
        for r in data.get("search_result", [])
    ]
    return {"rows": rows, "ms": cost, "note": f"recency={recency}（不支持任意区间）"}


def search_tavily(query: str, start: str, end: str, count: int):
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return None
    status, data, cost = post_json(
        "https://api.tavily.com/search",
        {},
        {"api_key": key, "query": query, "max_results": count,
         "topic": "news", "start_date": start, "end_date": end},
    )
    if status != 200 or not isinstance(data, dict):
        return {"error": f"HTTP {status}: {str(data)[:300]}", "ms": cost}
    rows = [
        norm(r.get("title"), r.get("url"), r.get("published_date"),
             urllib.parse.urlparse(r.get("url") or "").hostname, r.get("content"))
        for r in data.get("results", [])
    ]
    return {"rows": rows, "ms": cost}


PROVIDERS = {
    "deepseek": search_deepseek,
    "bocha": search_bocha,
    "zhipu": search_zhipu,
    "tavily": search_tavily,
}


# ---------------------------------------------------------------- 输出
def summarize(name: str, res: dict, start: str, end: str) -> dict:
    rows = res.get("rows") or []
    n = len(rows)
    dated = sum(1 for r in rows if r.get("date"))
    primary = sum(1 for r in rows if r.get("kind") == "primary")
    secondary = sum(1 for r in rows if r.get("kind") == "secondary")
    # 窗口外条数：只有日期可解析时才判，无法判的不算违规
    outside = sum(1 for r in rows if r.get("date") and not (start <= r["date"] <= end))
    # 唯一标题数：同一事件被多站转载会重复，这是「同源只算一票」要处理的对象
    uniq = len({title_key(r.get("title", "")) for r in rows if r.get("title")})
    return {
        "provider": name,
        "n": n,
        "ms": res.get("ms"),
        "date_rate": round(dated / n, 2) if n else 0.0,
        "primary": primary,
        "secondary": secondary,
        "primary_rate": round(primary / n, 2) if n else 0.0,
        "outside": outside,
        "uniq": uniq,
        "dup_rate": round(1 - uniq / n, 2) if n else 0.0,
        "error": res.get("error"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="检索连通性与质量探针")
    ap.add_argument("--queries", nargs="*", default=["大模型 开源 权重 发布", "推理 芯片 降价"],
                    help="一个或多个查询（默认两条，覆盖中英文习惯）")
    ap.add_argument("--days", type=int, default=7, help="时间窗天数，默认 7")
    ap.add_argument("--count", type=int, default=10, help="每个查询返回条数，默认 10")
    ap.add_argument("--provider", nargs="*", default=None, help="只测指定 provider")
    ap.add_argument("--json", default=None, help="把原始结果写入该文件")
    args = ap.parse_args()

    load_env(ENV_PATH)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=args.days)
    start, end = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    print(f"时间窗：{start} .. {end}（{args.days} 天）")
    print(f"查询：{args.queries}")
    print(f".env ：{ENV_PATH}" + ("" if ENV_PATH.exists() else "  ← 不存在，只用系统环境变量"))
    print("=" * 78)

    targets = args.provider or list(PROVIDERS)
    all_raw, summary = {}, []

    for name in targets:
        fn = PROVIDERS.get(name)
        if not fn:
            print(f"[跳过] 未知 provider：{name}")
            continue
        for q in args.queries:
            res = fn(q, start, end, args.count)
            if res is None:
                print(f"[跳过] {name}：未配置 key")
                break
            if res.get("error"):
                print(f"[失败] {name}：{res['error']}（{res.get('ms')}ms）")
                summary.append({"provider": name, "error": res["error"], "n": 0,
                                "date_rate": 0.0, "primary_rate": 0.0})
                break
            rows = res.get("rows") or []
            if res.get("note"):
                print(f"[提示] {name}：{res['note']}")
            print(f"\n── {name} ｜ “{q}” ｜ {len(rows)} 条 ｜ {res.get('ms')}ms")
            for i, r in enumerate(rows[:5], 1):
                mark = {"primary": "★", "secondary": "·", "excluded": "×"}.get(r["kind"], " ")
                auth = f" a={r['authority']}" if r.get("authority") is not None else ""
                print(f"  {i}. {mark} {r['date'] or '无日期'}  {r['site'] or '-'}{auth}")
                print(f"     {r['title']}")
                print(f"     {r['url'][:100]}")
            all_raw.setdefault(name, []).append({"query": q, "rows": rows, "ms": res.get("ms")})
        else:
            merged = {"rows": [r for item in all_raw.get(name, []) for r in item["rows"]],
                      "ms": max((item["ms"] or 0) for item in all_raw.get(name, [])) if all_raw.get(name) else None}
            summary.append(summarize(name, merged, start, end))

    print("\n" + "=" * 78)
    print("对比（★=官方域名 ·=次选媒体）：")
    print(f"{'provider':<10}{'条数':>5}{'延迟ms':>8}{'日期率':>7}{'一手':>5}{'次选':>5}"
          f"{'窗口外':>7}{'唯一题':>7}{'重复率':>7}  状态")
    for s in summary:
        if s.get("error"):
            print(f"{s['provider']:<10}{'-':>5}{'-':>8}{'-':>7}{'-':>5}{'-':>5}"
                  f"{'-':>7}{'-':>7}{'-':>7}  {str(s['error'])[:36]}")
        else:
            print(f"{s['provider']:<10}{s['n']:>5}{s['ms'] or 0:>8}{s['date_rate']:>7.2f}"
                  f"{s['primary']:>5}{s['secondary']:>5}{s['outside']:>7}{s['uniq']:>7}"
                  f"{s['dup_rate']:>7.2f}  ok")

    print("\n判读：")
    print("  · 日期率 < 0.6      → 撑不起 48h 新近配额，淘汰")
    print("  · 窗口外 > 0        → 该源没有真正执行时间过滤（或日期字段不可靠）")
    print("  · 重复率 > 0.3      → 同一事件被多站转载，证据库必须做标题级去重")
    print("  · 一手 = 0          → 能出简报，但报告须注明「全部结论依赖二手转述」；")
    print("                        补救办法是用 site: 运算符定向检索官方域名")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"window": {"start": start, "end": end},
                        "queries": args.queries, "raw": all_raw},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n原始结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
