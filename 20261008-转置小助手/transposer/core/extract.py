"""抽取编排：规则抽取（确定性）+ LLM 抽取（live / cache / offline 三层）。"""
import datetime
import json
import pathlib

from .llm import llm_available, llm_extract, LLMError

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "cache"


def get_llm_records(sample_id, text, template, force=False):
    """返回 (llm_records, mode)。

    mode:
      "llm_live"   —— 本次实时调用 LLM
      "llm_cache"  —— 命中缓存（此前某次 live 调用的结果）
      "offline"    —— 无 key 或调用失败，规则兜底
    llm_records 为 None 时表示 offline。
    """
    cache_path = CACHE_DIR / f"{sample_id}.json"
    if not force and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data.get("llm_records"), "llm_cache"

    if llm_available():
        try:
            records = llm_extract(text, template)
            CACHE_DIR.mkdir(exist_ok=True)
            cache_path.write_text(json.dumps({
                "sample_id": sample_id,
                "mode": "llm_live",
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "llm_records": records,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return records, "llm_live"
        except Exception:  # 调用失败统一降级为离线规则模式
            return None, "offline"
    return None, "offline"
