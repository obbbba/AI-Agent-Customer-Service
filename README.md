# AI智能客服 × LangGraph

> **多 Agent 协同电商客服系统** | 基于 LangGraph 的有向图 Agent 架构

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)

## 📖 项目简介

一个**生产级 AI 智能客服系统**，基于 LangGraph 多 Agent 协同架构。与传统的单轮 RAG 问答不同，本系统将客服流程拆分为多个独立 Agent 节点，通过**有向图（StateGraph）编排协作**，实现意图识别、知识库检索、情绪分析、工单生成、条件路由的完整闭环。

## 🔥 与传统 RAG 客服的核心差异

| 对比维度 | 传统 RAG 客服 | 本项目 |
|---------|-------------|--------|
| 架构 | 单次检索→生成 | **多 Agent 有向图协作** |
| 流程控制 | if/else 硬编码 | **LangGraph 条件边路由** |
| 情绪感知 | 无 | **LLM 打分 + 阈值自动转人工** |
| 工具调用 | 无 | **订单查询 + 工单自动生成** |
| 可观测性 | 黑盒 | **Agent State 实时调试面板** |
| 会话管理 | 无/简单 | **多会话 + JSON 持久化** |

## 🏗️ 系统架构

```
                    ┌─────────────────┐
                    │   用户输入消息    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  ① 意图解析Agent │  → 物流/退换货/商品咨询/投诉
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
     ┌────────▼────────┐           ┌───────▼───────┐
     │  ② RAG检索Agent │           │  📦 订单查询   │
     │  FAISS向量搜索   │           │  (LangChain工具) │
     └────────┬────────┘           └───────┬───────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  ③ 情绪判别Agent │  → 情绪分值 0~1
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
     ┌────────▼────────┐           ┌───────▼───────┐
     │   情绪正常 <0.7  │           │  情绪超标 ≥0.7  │
     └────────┬────────┘           └───────┬───────┘
              │                             │
     ┌────────▼────────┐           ┌───────▼───────┐
     │ ④ 应答生成Agent  │           │  🎫 工单生成   │
     │  正常客服回复     │           │  自动转人工     │
     └────────┬────────┘           └───────┬───────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │      结束        │
                    └─────────────────┘
```

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | **LangGraph** (StateGraph 有向图) |
| LLM | **DeepSeek V4 Pro** (兼容 OpenAI) |
| 向量检索 | **FAISS** + text2vec-base-chinese |
| 文档解析 | PyPDF2 + python-docx (PDF/Word/TXT) |
| 工具系统 | **LangChain @tool** 装饰器 |
| 前端 | **Gradio** 4.0+ (双栏可视化) |
| 会话存储 | JSON 本地持久化 |
| 环境管理 | python-dotenv |

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/obbbba/AI-Agent-Customer-Service.git
cd AI-Agent-Customer-Service
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

创建 `.env` 文件（已加入 .gitignore，不会上传）：

```env
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
TAVILY_API_KEY=tvly-your-key-here  # 可选：联网搜索
```

### 4. 启动

```bash
python main.py
```

浏览器打开 `http://localhost:8000`

## 📂 项目结构

```
AI-Agent-Customer-Service/
├── main.py                # Gradio 前端入口
├── graph.py               # LangGraph 图定义（5节点 + 条件路由）
├── knowledge_base.py      # FAISS 向量检索 + 文本搜索
├── tools.py               # LangChain 工具（订单查询、工单生成）
├── doc_parser.py          # PDF/Word/TXT 文档解析
├── config.py              # 统一配置管理
├── requirements.txt       # 依赖清单
├── data/
│   └── knowledge_sample.txt  # 示例知识库
└── sessions/              # 会话持久化（自动创建）
```

## 🎯 核心功能

- ✅ **意图识别** — LLM 自动分类用户意图（物流/退换货/咨询/投诉）
- ✅ **RAG 检索** — FAISS 向量搜索 + 文本关键词混合检索
- ✅ **情绪分析** — LLM 打分（0~1），超阈值自动触发工单
- ✅ **订单查询** — 识别订单号自动查询物流状态
- ✅ **工单生成** — 高危投诉自动汇总上下文生成标准化工单
- ✅ **多会话管理** — 多对话切换 + JSON 持久化，刷新不丢失
- ✅ **Agent 调试面板** — 右侧实时展示 State 状态 + 执行日志
- ✅ **文档上传** — 支持 PDF/Word/TXT 拖拽上传自动解析

## 📸 界面预览

界面分为两个 Tab：

**💬 对话测试** — 左列聊天窗口 + 右列 Agent State 调试面板
- 左侧：会话切换、聊天输入、示例问题
- 右侧：实时显示意图标签、情绪分值、检索文档、是否转人工

**⚙️ 设置** — API Key 配置 + 知识库上传管理
- DeepSeek / Tavily API Key 在线配置
- PDF / Word / TXT 文档上传并自动解析

## 📝 License

MIT © 2026

---

# AI Customer Service × LangGraph

> **Multi-Agent E-commerce Customer Service** | Graph-based Agent Architecture with LangGraph

## Overview

A production-grade AI customer service system built on **LangGraph multi-agent collaboration**. Unlike traditional single-turn RAG systems, this project decomposes the customer service workflow into multiple independent Agent nodes orchestrated through a **StateGraph**, enabling intent recognition, knowledge retrieval, sentiment analysis, ticket generation, and conditional routing in a complete closed loop.

## Key Differences from Traditional RAG

| Aspect | Traditional RAG | This Project |
|--------|----------------|--------------|
| Architecture | Single retrieval→generation | **Multi-Agent StateGraph** |
| Flow Control | Hardcoded if/else | **LangGraph Conditional Edges** |
| Sentiment Detection | None | **LLM Scoring + Auto-Escalation** |
| Tool Use | None | **Order Query + Ticket Generation** |
| Observability | Black box | **Real-time Agent State Panel** |
| Session Management | None/Simple | **Multi-Session + JSON Persistence** |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | **LangGraph** (StateGraph) |
| LLM | **DeepSeek V4 Pro** (OpenAI-compatible) |
| Vector Search | **FAISS** + text2vec-base-chinese |
| Document Parsing | PyPDF2 + python-docx |
| Tool System | **LangChain @tool** decorators |
| Frontend | **Gradio** 4.0+ (dual-panel) |
| Storage | JSON local persistence |

## Quick Start

```bash
git clone https://github.com/obbbba/AI-Agent-Customer-Service.git
cd AI-Agent-Customer-Service
pip install -r requirements.txt
# Create .env with DEEPSEEK_API_KEY
python main.py
# Open http://localhost:8000
```

## Architecture

The system uses a 5-node StateGraph with conditional routing:

```
User Input → ① Intent Agent → ② RAG Agent → ③ Sentiment Agent
                                              ├─ Normal → ④ Response Agent
                                              └─ High Risk → 🎫 Ticket Tool → ④ Response Agent
```

## License

MIT © 2026
