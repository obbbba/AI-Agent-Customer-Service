"""
LangGraph 多 Agent 电商智能客服 —— 核心图结构

Graph:
  START → intent_agent → rag_agent → sentiment_agent
                                              ↓
                          ┌─ sent > 0.7 → ticket_tool → END(人工)
                          └─ sent ≤ 0.7 → response_agent → END(AI回复)
"""

import operator
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from config import LLM_CONFIG, SENTIMENT_THRESHOLD
from knowledge_base import search_similar, build_index
from tools import query_order, generate_ticket

# ======= LLM（可动态刷新） =======
def get_llm():
    return ChatOpenAI(
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"],
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
        max_tokens=LLM_CONFIG["max_tokens"],
    )

# ======= 全局 State =======
class AgentState(TypedDict):
    user_query: str
    chat_history: Annotated[list[dict], operator.add]
    intent: str                              # 物流/退换货/商品咨询/投诉/其他
    sentiment_score: float                   # 0~1, 越高越负面
    rag_docs: list[dict]                     # 检索到的知识库文档
    need_human: bool                         # 是否转人工
    order_info: str                          # 订单查询结果
    ticket_content: str                      # 工单内容
    final_response: str                      # 最终回复
    agent_logs: Annotated[list[str], operator.add]  # 调试日志

# ======= Agent 1: 意图解析 =======
INTENT_PROMPT = """你是电商客服意图识别专家。根据用户问题，只输出一个意图标签：
- 物流查询：查快递、物流状态、配送进度
- 退换货：退货、换货、退款、申请售后
- 商品咨询：产品参数、价格、库存、使用说明
- 投诉：投诉商家、投诉骑手、食品安全、质量问题
- 订单查询：查订单、订单状态、修改订单
- 其他：以上都不是

只回复标签，不要解释。"""

def intent_agent(state: AgentState) -> dict:
    query = state["user_query"]
    msgs = [
        {"role": "system", "content": INTENT_PROMPT},
        {"role": "user", "content": query},
    ]
    resp = get_llm().invoke([(m["role"], m["content"]) for m in msgs])
    intent = resp.content.strip()
    return {
        "intent": intent,
        "agent_logs": [f"🎯 意图解析: {intent}"],
    }

# ======= Agent 2: RAG 检索 =======
def rag_agent(state: AgentState) -> dict:
    query = state["user_query"]
    intent = state["intent"]
    docs = search_similar(query, intent)
    return {
        "rag_docs": docs,
        "agent_logs": [f"📚 RAG检索: 找到 {len(docs)} 篇相关文档 (意图:{intent})"],
    }

# ======= Agent 3: 情绪风险判别 =======
SENTIMENT_PROMPT = """分析用户情绪，只输出 0~1 之间的数字:
- 0.0~0.3: 平和咨询
- 0.3~0.5: 轻微不满
- 0.5~0.7: 明显不满
- 0.7~0.9: 激烈投诉
- 0.9~1.0: 极端愤怒

只输出数字，如 0.35"""

def sentiment_agent(state: AgentState) -> dict:
    query = state["user_query"]
    msgs = [
        {"role": "system", "content": SENTIMENT_PROMPT},
        {"role": "user", "content": query},
    ]
    resp = get_llm().invoke([(m["role"], m["content"]) for m in msgs])
    try:
        score = float(resp.content.strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = 0.3

    need_human = score > SENTIMENT_THRESHOLD or any(
        kw in query for kw in ["投诉", "退款", "赔偿", "举报", "投诉你", "找领导", "315"]
    )

    return {
        "sentiment_score": score,
        "need_human": need_human,
        "agent_logs": [f"😊 情绪分析: {score:.2f} {'⚠️转人工' if need_human else '✅正常'} 阈值:{SENTIMENT_THRESHOLD}"],
    }

# ======= Agent 4: 应答生成 =======
RESPONSE_PROMPT = """你是美团电商客服。规则:
- 结合知识库和对话历史回答
- 语气亲切专业
- 需要转人工时输出安抚话术
- 如有订单信息，引用具体物流状态

知识库参考:
{rag_context}

用户意图: {intent}
用户问题: {query}

请回复:"""

def response_agent(state: AgentState) -> dict:
    query = state["user_query"]
    intent = state["intent"]
    docs = state.get("rag_docs", [])
    need_human = state.get("need_human", False)

    rag_text = "\n".join([d["content"][:500] for d in docs[:3]]) if docs else "无相关文档"

    if need_human:
        prompt = RESPONSE_PROMPT.format(rag_context=rag_text, intent=intent, query=query)
        prompt += "\n注意: 用户情绪激烈，请输出安抚话术并告知正在转接人工客服处理。"
    else:
        prompt = RESPONSE_PROMPT.format(rag_context=rag_text, intent=intent, query=query)

    msgs = [{"role": "user", "content": prompt}]
    resp = get_llm().invoke([(m["role"], m["content"]) for m in msgs])

    return {
        "final_response": resp.content,
        "agent_logs": [f"✍️ 生成回复 {'(安抚+转人工)' if need_human else '(正常应答)'}"],
    }

# ======= 工具节点: 订单查询 =======
def order_tool_node(state: AgentState) -> dict:
    query = state["user_query"]
    # 提取订单号
    import re
    order_ids = re.findall(r'ORDER\d+', query.upper())
    if order_ids:
        result = query_order.invoke({"order_id": order_ids[0]})
    else:
        result = "未检测到订单号，请提供订单号(如ORDER001)"

    return {
        "order_info": str(result),
        "agent_logs": [f"📦 订单查询: {str(result)[:100]}"],
    }

# ======= 工具节点: 工单生成 =======
def ticket_tool_node(state: AgentState) -> dict:
    query = state["user_query"]
    context = "\n".join([d["content"][:200] for d in state.get("rag_docs", [])])
    result = generate_ticket.invoke({
        "user_query": query,
        "chat_context": context,
        "reason": f"情绪分{state['sentiment_score']:.2f}超标, 意图:{state['intent']}",
    })
    return {
        "ticket_content": str(result),
        "agent_logs": [f"🎫 工单已生成: {str(result)[:100]}"],
    }

# ======= 路由: 意图分发 =======
def route_after_intent(state: AgentState) -> Literal["order_tool", "rag_agent"]:
    intent = state.get("intent", "")
    query = state.get("user_query", "").upper()
    if "物流" in intent or "订单" in intent or "ORDER" in query:
        return "order_tool"
    return "rag_agent"

# ======= 路由: 情绪判别 =======
def route_after_sentiment(state: AgentState) -> Literal["ticket_tool", "response_agent"]:
    if state.get("need_human"):
        return "ticket_tool"
    return "response_agent"

# ======= 构建 Graph =======
def build_graph() -> StateGraph:
    w = StateGraph(AgentState)

    w.add_node("intent_agent", intent_agent)
    w.add_node("rag_agent", rag_agent)
    w.add_node("sentiment_agent", sentiment_agent)
    w.add_node("order_tool", order_tool_node)
    w.add_node("ticket_tool", ticket_tool_node)
    w.add_node("response_agent", response_agent)

    w.set_entry_point("intent_agent")

    # 意图分发: 订单查询走工具，其他走RAG
    w.add_conditional_edges(
        "intent_agent",
        route_after_intent,
        {"order_tool": "order_tool", "rag_agent": "rag_agent"},
    )

    w.add_edge("rag_agent", "sentiment_agent")
    w.add_edge("order_tool", "sentiment_agent")

    # 情绪路由: 激烈投诉走工单，正常走回复
    w.add_conditional_edges(
        "sentiment_agent",
        route_after_sentiment,
        {"ticket_tool": "ticket_tool", "response_agent": "response_agent"},
    )

    w.add_edge("ticket_tool", "response_agent")
    w.add_edge("response_agent", END)

    return w.compile()

agent_graph = build_graph()
