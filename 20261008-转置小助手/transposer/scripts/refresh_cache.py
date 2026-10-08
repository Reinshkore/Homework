"""填写 .env 中的 TRANSPOSER_API_KEY 后运行一次，生成 LLM 抽取缓存。

    之后演示默认走缓存回放，不依赖网络与 key。
    运行：python scripts/refresh_cache.py
    """
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.mapping import get_template
from core.pipeline import transpose, list_samples
from core.llm import llm_available, llm_config

if not llm_available():
    print("未检测到 TRANSPOSER_API_KEY（transposer/.env），请先填写后再运行本脚本。")
    sys.exit(1)

print(f"使用模型：{llm_config()['model']} @ {llm_config()['base']}")
ok, fail = 0, 0
for s in list_samples():
    try:
        r = transpose(s["template_id"], s["sample_id"], force_llm=True)
        n = len(r["records"])
        assert r["mode"] in ("llm_live", "llm_cache"), f"模式异常：{r['mode']}"
        print(f"  [OK] {s['sample_id']}  mode={r['mode']}  records={n}")
        ok += 1
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {s['sample_id']}: {e}")
        fail += 1
print(f"完成：成功 {ok} / 失败 {fail}")
sys.exit(0 if fail == 0 else 1)
