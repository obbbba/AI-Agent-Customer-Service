"""
文档解析 —— 支持 PDF / Word / TXT
"""

from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document


def parse_pdf(file_path: str) -> str:
    """解析 PDF 文件"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"[PDF解析失败: {e}]"


def parse_docx(file_path: str) -> str:
    """解析 Word 文件"""
    try:
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text.strip()
    except Exception as e:
        return f"[DOCX解析失败: {e}]"


def parse_txt(file_path: str) -> str:
    """解析 TXT 文件"""
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return Path(file_path).read_text(encoding="gbk").strip()


def parse_file(file_path: str) -> tuple[str, str]:
    """解析任意支持的文件，返回 (文件名, 内容)"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = parse_pdf(str(path))
    elif suffix in (".docx", ".doc"):
        text = parse_docx(str(path))
    elif suffix in (".txt", ".md", ".csv", ".json"):
        text = parse_txt(str(path))
    else:
        return (path.name, f"[不支持的文件格式: {suffix}]")

    return (path.stem, text)
