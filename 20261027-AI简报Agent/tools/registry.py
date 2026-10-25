"""进程内工具登记处。

只放「无状态或只读外部」的工具。需要独占写库、且要绕过 Agent 虚拟文件系统权限的，
放 MCP 服务器（mcp/brief_store.py），不要放这里。
"""

from tools.calc import calculator
from tools.search import fetch_page, web_search
from tools.timebox import get_current_time, resolve_window

PY_TOOLS = [
    get_current_time,
    resolve_window,
    calculator,
    web_search,
    fetch_page,
]
