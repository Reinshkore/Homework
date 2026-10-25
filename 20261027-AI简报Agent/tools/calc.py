"""数值计算工具（从 demo5 移植）。

简报里用得到的场景：降价幅度换算、单位成本对比、参数量与激活量换算、
以及"一份简报用了多少次检索"这类自检。凡是要精确的算术，不要让模型心算。
"""

from __future__ import annotations

import ast
import operator

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    """只允许数字与算术运算，避免直接 eval 带来的代码执行风险。"""

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("只支持数字与 + - * / // % ** 运算")


def calculator(expression: str) -> str:
    """精确计算数学表达式，例如 "(3.5 - 2.0) / 3.5 * 100"。

    涉及数值计算时务必使用，不要心算——心算的百分比与倍数错误无法被读者发现。

    Args:
        expression: 只包含数字、括号与 + - * / // % ** 的表达式
    """

    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:  # noqa: BLE001
        return f"计算失败：{exc}"
