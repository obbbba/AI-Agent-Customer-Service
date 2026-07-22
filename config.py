"""
统一配置文件
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ======= 项目路径 =======
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SESSION_DIR = ROOT / "sessions"
VECTOR_DIR = ROOT / "vector_store"

for d in [DATA_DIR, SESSION_DIR, VECTOR_DIR]:
    d.mkdir(exist_ok=True)

# ======= LLM 配置 =======
LLM_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "model": os.getenv("LLM_MODEL", "deepseek-chat"),
    "temperature": 0.3,
    "max_tokens": 1024,
}

# ======= 向量库配置 =======
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"  # 中文向量模型
VECTOR_TOP_K = 5
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ======= 情绪阈值 =======
SENTIMENT_THRESHOLD = 0.7  # >0.7 自动转人工

# ======= 订单模拟数据 =======
MOCK_ORDERS = {
    "ORDER001": {"status": "已签收", "物流": "顺丰快递 SF123456", "商品": "iPhone 15 Pro", "金额": 8999},
    "ORDER002": {"status": "运输中", "物流": "中通快递 ZT789012", "商品": " Nike Air Max", "金额": 799},
    "ORDER003": {"status": "待发货", "物流": "待分配", "商品": "华为MateBook", "金额": 5999},
    "ORDER004": {"status": "已退货", "物流": "圆通快递 YT345678", "商品": "Sony耳机", "金额": 299},
}
