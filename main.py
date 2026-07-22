"""
AI 智能客服 × LangGraph — Gradio 前端
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, set_key
import gradio as gr
from graph import agent_graph, AgentState
from knowledge_base import build_index, search_similar
from doc_parser import parse_file
from config import SESSION_DIR, DATA_DIR, LLM_CONFIG

load_dotenv()
ENV_FILE = Path(__file__).parent / ".env"
SESSIONS_FILE = SESSION_DIR / "chat_sessions.json"


# ======= 会话管理 =======
def load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    default = {"default": {"title": "默认对话", "history": [], "created": datetime.now().isoformat()}}
    save_sessions(default)
    return default

def save_sessions(sessions: dict):
    SESSIONS_FILE.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")

def get_session_list():
    sessions = load_sessions()
    return list(sessions.keys())

def new_session(title: str = ""):
    sessions = load_sessions()
    sid = f"chat_{datetime.now().strftime('%m%d%H%M%S')}"
    sessions[sid] = {"title": title or f"对话 {len(sessions)+1}", "history": [], "created": datetime.now().isoformat()}
    save_sessions(sessions)
    return sid, get_session_choices(), []

def delete_session(sid: str):
    sessions = load_sessions()
    if sid not in sessions:
        return get_session_choices(), []
    if len(sessions) <= 1:
        return get_session_choices(), sessions.get("default", {}).get("history", [])
    del sessions[sid]
    save_sessions(sessions)
    first = list(sessions.keys())[0]
    return get_session_choices(), sessions[first].get("history", [])

def get_session_choices():
    sessions = load_sessions()
    return [(extract_text(sessions[s]["title"]), s) for s in sessions]

def get_session_history(sid: str):
    sessions = load_sessions()
    if sid in sessions:
        return sessions[sid]["history"]
    return []

def extract_text(content) -> str:
    """统一提取文本：兼容字符串和 Gradio 结构化格式"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)

def normalize_history(history: list) -> list:
    """统一历史格式为 {role, content} 纯字符串"""
    result = []
    for m in history:
        if isinstance(m, dict):
            result.append({"role": m.get("role", "user"), "content": extract_text(m.get("content", ""))})
    return result

def save_session_history(sid: str, history: list):
    sessions = load_sessions()
    if sid in sessions:
        history = normalize_history(history)
        sessions[sid]["history"] = history
        # 标题取第一条用户消息
        for m in history:
            if m["role"] == "user":
                sessions[sid]["title"] = m["content"][:30]
                break
        save_sessions(sessions)


# ======= 设置 =======
def save_api_key(ds_key: str, tv_key: str):
    if ds_key.strip():
        set_key(str(ENV_FILE), "DEEPSEEK_API_KEY", ds_key)
        os.environ["DEEPSEEK_API_KEY"] = ds_key
        LLM_CONFIG["api_key"] = ds_key
    if tv_key.strip():
        set_key(str(ENV_FILE), "TAVILY_API_KEY", tv_key)
        os.environ["TAVILY_API_KEY"] = tv_key
    return "✅ 设置已保存，立即生效"

def upload_kb_file(file):
    if file is None: return "未选择文件"
    try:
        src = Path(file.name)
        dst = DATA_DIR / src.name
        dst.write_bytes(src.read_bytes())
        name, text = parse_file(str(dst))
        (DATA_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
        build_index(force=True)
        return f"✅ {src.name} 已上传 ({len(text)} 字)"
    except Exception as e:
        return f"❌ 上传失败: {e}"

def reload_kb():
    build_index(force=True)
    files = list(DATA_DIR.glob("*"))
    return "\n".join(f"📄 {f.name}" for f in sorted(files)) if files else "知识库为空"

def list_kb_files():
    files = list(DATA_DIR.glob("*"))
    return "\n".join(f"📄 {f.name} ({f.stat().st_size}B)" for f in sorted(files)) if files else "知识库为空"


# ======= 对话 =======
def chat_flow(message: str, history: list):
    history = history or []
    if not message.strip():
        yield history, "", "", 0, False, ""
        return

    state: AgentState = {
        "user_query": message, "chat_history": history,
        "intent": "", "sentiment_score": 0.0, "rag_docs": [],
        "need_human": False, "order_info": "", "ticket_content": "",
        "final_response": "", "agent_logs": [],
    }

    result = agent_graph.invoke(state)
    intent = result.get("intent", "")
    score = result.get("sentiment_score", 0.0)
    need_human = result.get("need_human", False)
    response = result.get("final_response", "")
    logs = result.get("agent_logs", [])
    docs = result.get("rag_docs", [])
    ticket = result.get("ticket_content", "")

    debug = ["=" * 45, "📋 全局 State", "=" * 45]
    debug.append(f"🎯 意图: {intent}")
    debug.append(f"😊 情绪分: {score:.2f} {'⚠️超标' if need_human else '✅正常'}")
    debug.append(f"📚 检索文档: {len(docs)} 篇")
    for i, d in enumerate(docs[:5]):
        debug.append(f"  [{i+1}] {d['title']} ({d.get('score',0):.3f})")
    debug.append(f"🔄 转人工: {'是' if need_human else '否'}")
    if ticket: debug.append(f"🎫 工单: 已生成")
    debug.append("\n" + "=" * 45 + "\n🔍 Agent 执行日志\n" + "=" * 45)
    for log in logs: debug.append(f"  {log}")

    display = response + ("\n\n🎫 工单已生成，人工客服将尽快处理。" if need_human and ticket else "")
    partial = ""
    for ch in display:
        partial += ch
        if len(partial) % 25 == 0:
            yield list(history) + [{"role": "user", "content": message}, {"role": "assistant", "content": partial}], "\n".join(debug), intent, score, need_human, ticket[:200] if ticket else ""

    yield list(history) + [{"role": "user", "content": message}, {"role": "assistant", "content": display}], "\n".join(debug), intent, score, need_human, ticket[:200] if ticket else ""

def clear_all():
    return [], "", "", 0, False, ""


# ======= Gradio UI =======
with gr.Blocks(title="AI 智能客服 × LangGraph") as demo:
    gr.Markdown("# AI智能客服")
    gr.Markdown("*多 Agent 协同 | 意图 → RAG → 情绪 → 应答 | 条件路由 + 工具调用*")

    with gr.Tabs():
        # ===== Tab 1: 对话 =====
        with gr.Tab("💬 对话"):
            with gr.Row():
                # 左侧：会话列表 + 聊天
                with gr.Column(scale=1):
                    with gr.Row():
                        session_dd = gr.Dropdown(
                            label="📂 会话", choices=get_session_choices(), value="default", scale=3,
                            interactive=True, allow_custom_value=True,
                        )
                        new_btn = gr.Button("＋新建", size="sm", scale=1)
                        del_btn = gr.Button("🗑删除", size="sm", scale=1)

                    chatbot = gr.Chatbot(height=420, label="", value=get_session_history("default"))
                    with gr.Row():
                        msg = gr.Textbox(placeholder="输入问题...", container=False, scale=4)
                        send_btn = gr.Button("发送", variant="primary", scale=1)

                    gr.Examples([
                        ["我的订单ORDER001到哪了"],
                        ["外卖超时了怎么办"],
                        ["如何申请退款"],
                        ["你们这破外卖！迟了一小时还洒了！投诉！退款！"],
                        ["食品安全问题怎么处理"],
                    ], inputs=msg, label="📝 示例")

                # 右侧：调试面板
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Agent 调试面板")
                    debug_output = gr.Textbox(label="全局 State + 日志", lines=22, max_lines=30, interactive=False)
                    with gr.Row():
                        intent_disp = gr.Textbox(label="意图", scale=1, interactive=False)
                        score_disp = gr.Number(label="情绪分", scale=1, interactive=False)
                    with gr.Row():
                        human_disp = gr.Checkbox(label="转人工", interactive=False)
                        ticket_disp = gr.Textbox(label="工单", scale=2, interactive=False)

        # ===== Tab 2: 设置 =====
        with gr.Tab("⚙️ 设置"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔑 API Key")
                    api_key_input = gr.Textbox(
                        label="DeepSeek API Key", value=os.getenv("DEEPSEEK_API_KEY", ""),
                        type="password", placeholder="sk-...",
                    )
                    api_key_input2 = gr.Textbox(
                        label="Tavily API Key (联网搜索)", value=os.getenv("TAVILY_API_KEY", ""),
                        type="password", placeholder="tvly-...",
                    )
                    api_save_btn = gr.Button("💾 保存全部", variant="primary")
                    api_status = gr.Textbox(label="状态", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### 📚 知识库")
                    kb_file_list = gr.Textbox(label="已有文件", value=list_kb_files(), lines=5, interactive=False)
                    kb_reload_btn = gr.Button("🔄 重载索引", size="sm")

            gr.Markdown("---")
            gr.Markdown("### 📤 上传文档 (PDF / Word / TXT)")
            with gr.Row():
                kb_upload = gr.File(label="选择文件", file_types=[".pdf", ".docx", ".doc", ".txt", ".md"])
                kb_upload_btn = gr.Button("上传", variant="primary")
            kb_upload_status = gr.Textbox(label="状态", interactive=False)


    # ===== 事件 =====
    def switch_session(sid):
        return get_session_history(sid)

    def do_new_session():
        s = load_sessions()
        sid = f"chat_{datetime.now().strftime('%m%d%H%M%S')}"
        s[sid] = {"title": f"对话 {len(s)+1}", "history": [], "created": datetime.now().isoformat()}
        save_sessions(s)
        return gr.update(choices=get_session_choices(), value=sid), []

    def do_del_session(sid):
        s = load_sessions()
        if len(s) > 1 and sid in s:
            del s[sid]
            save_sessions(s)
        s = load_sessions()
        first = list(s.keys())[0]
        return gr.update(choices=get_session_choices(), value=first), s[first].get("history", [])

    demo.load(lambda: (gr.update(choices=get_session_choices()), get_session_history("default")), None, [session_dd, chatbot])

    session_dd.change(switch_session, session_dd, chatbot)
    new_btn.click(do_new_session, None, [session_dd, chatbot])
    del_btn.click(do_del_session, session_dd, [session_dd, chatbot])

    # 对话
    def respond_and_save(message, history, sid):
        history = list(history) if history else []
        final = history
        for h, debug, intent, score, human, ticket in chat_flow(message, history):
            final = h
            yield h, debug, intent, score, human, ticket
        if final:
            save_session_history(sid, final)

    msg.submit(respond_and_save, [msg, chatbot, session_dd], [chatbot, debug_output, intent_disp, score_disp, human_disp, ticket_disp]).then(lambda: "", None, msg)
    send_btn.click(respond_and_save, [msg, chatbot, session_dd], [chatbot, debug_output, intent_disp, score_disp, human_disp, ticket_disp]).then(lambda: "", None, msg)

    # 设置
    api_save_btn.click(save_api_key, [api_key_input, api_key_input2], api_status)
    kb_reload_btn.click(reload_kb, None, kb_file_list)
    kb_upload_btn.click(upload_kb_file, kb_upload, kb_upload_status)


if __name__ == "__main__":
    build_index()
    demo.launch(server_name="0.0.0.0", server_port=8000)
