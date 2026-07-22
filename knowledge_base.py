"""
RAG 知识库 —— DeepSeek Embedding API + 文本搜索 Fallback
零本地模型依赖，无需下载，即开即用
"""

import pickle
import re
from pathlib import Path
from openai import OpenAI
from config import LLM_CONFIG, VECTOR_TOP_K, VECTOR_DIR, DATA_DIR

# 全局缓存
_chunks: list[dict] = []
_vectors: list[list[float]] = []

def get_embedding_client():
    return OpenAI(
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"],
    )

def chunk_text(text: str, title: str = "") -> list[dict]:
    """文本分块"""
    paragraphs = text.replace("\r\n", "\n").split("\n\n")
    chunks = []
    for p in paragraphs:
        p = p.strip()
        if not p: continue
        # 按句号进一步拆分长段落
        if len(p) > 500:
            sentences = re.split(r'(?<=[。！？])', p)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) < 500:
                    buf += s
                else:
                    if buf.strip(): chunks.append({"title": title, "content": buf.strip()})
                    buf = s
            if buf.strip(): chunks.append({"title": title, "content": buf.strip()})
        else:
            chunks.append({"title": title, "content": p})
    return chunks

def get_embedding(text: str) -> list[float]:
    """调用 DeepSeek API 获取向量"""
    try:
        client = get_embedding_client()
        r = client.embeddings.create(model="deepseek-chat", input=text[:1000])
        return r.data[0].embedding
    except Exception:
        # Fallback: 返回零向量，走文本搜索
        return []

def cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b: return 0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0

def keyword_score(query: str, text: str) -> float:
    """文本关键词匹配分数"""
    q_words = set(query.lower().split())
    t_lower = text.lower()
    score = sum(1 for w in q_words if w in t_lower)
    # 精确匹配加分
    if query.lower() in t_lower: score += 3
    return score

def build_index(force=False):
    """构建索引（文本 + 可选向量）"""
    global _chunks, _vectors

    # 加载所有知识库
    all_chunks = []
    for f in DATA_DIR.glob("*.txt"):
        text = f.read_text(encoding="utf-8")
        all_chunks.extend(chunk_text(text, f.stem))

    if not all_chunks:
        _chunks = []
        _vectors = []
        return

    _chunks = all_chunks

    # 尝试向量化
    _vectors = []
    try:
        for c in all_chunks[:50]:  # 限制前50个chunk避免超量
            _vectors.append(get_embedding(c["content"][:500]))
    except Exception:
        _vectors = []

    # 持久化 chunks
    pkl_path = VECTOR_DIR / "chunks.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_chunks, f)

def search_similar(query: str, intent: str = "", top_k: int = VECTOR_TOP_K) -> list[dict]:
    """混合搜索：向量 + 文本"""
    if not _chunks:
        build_index()
    if not _chunks:
        return []

    # 1. 向量搜索
    if _vectors:
        try:
            q_vec = get_embedding(query)
            if q_vec:
                scored = [(cosine_sim(q_vec, v), i) for i, v in enumerate(_vectors)]
                scored.sort(key=lambda x: x[0], reverse=True)
                results = []
                for score, idx in scored[:top_k]:
                    if idx < len(_chunks) and score > 0.5:
                        results.append({**_chunks[idx], "score": round(score, 3)})
                if results:
                    return results
        except Exception:
            pass

    # 2. 文本兜底
    scored = [(keyword_score(query, c["content"]), i) for i, c in enumerate(_chunks)]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, idx in scored[:top_k]:
        if score > 0:
            results.append({**_chunks[idx], "score": score / 10})
    return results
