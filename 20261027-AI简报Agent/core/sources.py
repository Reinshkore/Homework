"""信源规则：域名分级、URL 规范化、标题指纹、页面性质判定。

这些是「证据强度」的全部判据，M2 的检索去重与 M5 的 brief_lint 都依赖它，
所以单独成模块——两边各写一份必然分叉。

规则来源：ai-brief/references/industries/ai.md 第六节（优先 / 次选 / 排除）
与 quality.md 四之一（一手怎么认定）。
"""

from __future__ import annotations

import re
import urllib.parse

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
    # 研究机构自有发布页
    "baai.ac.cn", "shlab.org.cn",
}

SECONDARY_DOMAINS = {
    "jiqizhixin.com", "qbitai.com", "theinformation.com", "venturebeat.com",
    "techcrunch.com", "36kr.com", "infoq.cn", "sspai.com", "ithome.com",
    "bloomberg.com", "scmp.com", "qz.com", "the-decoder.com", "interconnects.ai",
}

EXCLUDED_DOMAINS = {"arxiv.org", "paperswithcode.com", "semanticscholar.org"}

# 这些路径段说明页面是「发布载体」而不是文档、教程或营销页
_PUBLISH_PATH = re.compile(
    r"/(blog|news|press|announcements?|releases?|pricing|models?|research|newsroom)(/|$)",
    re.IGNORECASE,
)
_DOC_PATH = re.compile(r"/(docs?|documentation|guide|tutorial|manual)(/|$)", re.IGNORECASE)

# 归一化时丢掉的跟踪参数（不去掉会导致同一页面被判成不同 URL）
_TRACKING = re.compile(r"^(utm_|spm|scm|from|share|wid|ref|source|fbclid|gclid)", re.IGNORECASE)

_TITLE_NOISE = re.compile(r"[\s\-_|·,，。:：!！?？\"'“”‘’()（）\[\]【】<>《》]")


def host_of(url: str) -> str:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def classify(url: str) -> str:
    """primary / secondary / excluded / other。只按域名判，不做内容判断。"""

    host = host_of(url)
    if not host:
        return "other"
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


def url_norm(url: str) -> str:
    """去重键：小写域名、去 www、去 fragment、去跟踪参数、去尾斜杠。"""

    try:
        p = urllib.parse.urlparse(url.strip())
    except Exception:
        return url.strip()
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    query_pairs = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True)
        if not _TRACKING.match(k)
    ]
    query = urllib.parse.urlencode(sorted(query_pairs))
    path = p.path.rstrip("/") or "/"
    return f"{p.scheme or 'https'}://{host}{path}" + (f"?{query}" if query else "")


def title_key(title: str) -> str:
    """标题指纹，用于识别跨站转载。

    URL 去重拦不住「同一篇稿子发在网易和新浪」，标题指纹可以。
    取前 24 字是因为转载时标题常被加前后缀（"| xxx网"、"- 财经频道"）。
    """

    return _TITLE_NOISE.sub("", title or "")[:24].lower()


def is_publish_page(url: str) -> bool:
    """页面性质是否像「发布载体」。`/docs/` 一律不算，哪怕域名是官方的。

    实测教训：`site:minimax.io` 返回的全是 `platform.minimax.io/docs/*` 的 API 文档页，
    不是模型发布页；`site:github.com` 返回的是第三方仓库里的接入 PR。
    域名对不等于载体对。
    """

    try:
        path = urllib.parse.urlparse(url).path or "/"
    except Exception:
        return False
    if _DOC_PATH.search(path):
        return False
    return bool(_PUBLISH_PATH.search(path))


def is_primary(url: str, fetched: bool = False) -> bool:
    """技能的「一手」：官方域名 + 页面性质像发布页 + 真的读到过原始载体。

    authority（站点权威度）不参与判断——新浪财经能拿最高档，但它只是转载。
    """

    return fetched and classify(url) == "primary" and is_publish_page(url)
