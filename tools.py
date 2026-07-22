"""
LangGraph 工具节点 —— 订单查询 + 工单生成 + 联网搜索
"""
import os
import json
from datetime import datetime
from langchain_core.tools import tool
from config import MOCK_ORDERS, SESSION_DIR

@tool
def query_order(order_id: str) -> str:
    """查询订单物流状态。输入订单号(如ORDER001)，返回订单详情。"""
    order = MOCK_ORDERS.get(order_id.upper().strip())
    if not order:
        return f"未找到订单 {order_id}，请确认订单号是否正确。可用订单号: {', '.join(MOCK_ORDERS.keys())}"
    return json.dumps(order, ensure_ascii=False, indent=2)

@tool
def generate_ticket(user_query: str, chat_context: str, reason: str) -> str:
    """生成售后工单。当用户情绪激烈或明确要求人工时调用。"""
    ticket = {
        "ticket_id": f"TK{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "created_at": datetime.now().isoformat(),
        "reason": reason,
        "user_query": user_query[:200],
        "context": chat_context[:500],
        "priority": "高" if "投诉" in user_query or "退款" in user_query else "中",
        "status": "待处理",
    }
    # 持久化工单
    ticket_file = SESSION_DIR / f"ticket_{ticket['ticket_id']}.json"
    ticket_file.write_text(json.dumps(ticket, ensure_ascii=False, indent=2), encoding="utf-8")

    return json.dumps(ticket, ensure_ascii=False, indent=2)

# 工具映射
TOOLS = [query_order, generate_ticket]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
