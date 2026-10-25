"""时间类工具。

为什么日期必须走工具：技能的规则里全是日期运算——7 天窗口起止、48 小时新近、
12 小时确定性封顶、半衰期 7/14/21/30/90 天、悬置事项 7/14 天过期、
边界条目"窗口起点前 48 小时"。模型算"7 天前是几号"在跨月、跨年会错，而且错得无声。
一处错，报告头的覆盖期、账本的起止、新近度判定会连带全错。
"""

from __future__ import annotations

from datetime import datetime, timedelta

_WEEKDAYS = "一二三四五六日"


def get_current_time() -> str:
    """获取当前本地日期、时间和星期。回答「今天」「现在」「几号」「星期几」时必须用它，不要凭记忆。"""

    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f" 星期{_WEEKDAYS[now.weekday()]}"


def resolve_window(days: int = 7) -> dict:
    """按天数算出一个时间窗（含首含尾），返回起止日期与星期。

    凡是涉及「最近 N 天」「覆盖期」这类日期计算都用它，不要自己算。
    返回的 start / end 同时用于：检索参数、报告头的「覆盖期」、账本的「起始/截止」。

    Args:
        days: 窗口天数。7 表示从今天往前 7 天，1 表示只含今天。
    """

    if days < 1:
        raise ValueError("days 必须 >= 1")

    now = datetime.now()
    start_dt = now - timedelta(days=days)
    # 48 小时新近配额按"当前时刻往前 48 小时"算，不是按自然日
    cutoff = now - timedelta(hours=48)

    return {
        "start": start_dt.strftime("%Y-%m-%d"),
        "end": now.strftime("%Y-%m-%d"),
        "days": days,
        "hours": days * 24,
        "weekday_start": f"星期{_WEEKDAYS[start_dt.weekday()]}",
        "weekday_end": f"星期{_WEEKDAYS[now.weekday()]}",
        "within_48h_from": cutoff.strftime("%Y-%m-%d %H:%M"),
        "note": "start 与 end 均含当日；within_48h_from 用于判断是否满足新近配额",
    }
