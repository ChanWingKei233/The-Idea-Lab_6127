# test CICD
"""
app/data_process.py

核心模块1：文本预处理（供训练模块调用）

- 功能1：处理 NLTK 数据下载
  自动检测 stopwords 与 punkt 是否已安装，缺失则静默下载；
  捕获并处理 LookupError，避免调用时弹出窗口或报错。

- 功能2：文本清洗函数 preprocess_text
  流程：转小写 → 去标点（str.maketrans） → 分词（word_tokenize） → 去英文停用词（stopwords.words('english')）
  输入：原始文本字符串；输出：清洗后的字符串（用空格连接）。
"""

from __future__ import annotations

import string
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ------------------------------
# 功能1：NLTK 数据下载与异常处理
# ------------------------------
def _has_resource(resource_path: str) -> bool:
    """检查 NLTK 资源是否已存在。"""
    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        return False


def ensure_nltk_data() -> None:
    """确保使用到的 NLTK 数据集已就绪（stopwords 与 punkt）。

    - 首选静默检测，不存在则静默下载（quiet=True，避免弹窗）
    - 兼容性说明：tokenizer 资源路径使用 'tokenizers/punkt'
    - 显式捕获 LookupError，防止调用侧出现未处理异常
    """
    # 1) stopwords
    if not _has_resource('corpora/stopwords'):
        nltk.download('stopwords', quiet=True)
    if not _has_resource('corpora/stopwords'):
        raise LookupError("Failed to obtain NLTK resource: 'stopwords'. Please check network or NLTK data path.")

    # 2) punkt 分词模型
    if not _has_resource('tokenizers/punkt'):
        nltk.download('punkt', quiet=True)
    if not _has_resource('tokenizers/punkt'):
        raise LookupError("Failed to obtain NLTK resource: 'punkt'. Please check network or NLTK data path.")


# ------------------------------
# 功能2：文本清洗
# ------------------------------
def preprocess_text(text: str) -> str:
    """按照既定流程清洗文本，返回清洗后的字符串。

    步骤（严格按顺序执行）：
    1. 转小写
    2. 去标点：使用 str.maketrans('', '', string.punctuation)
    3. 分词：nltk.tokenize.word_tokenize
    4. 去英文停用词：nltk.corpus.stopwords.words('english')
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # —— 预检查并准备 NLTK 资源 ——
    ensure_nltk_data()

    # 1) 转小写
    text = text.lower()

    # 2) 去标点
    punctuation_to_remove = string.punctuation.replace('!', '').replace('?', '')
    text = text.translate(str.maketrans('', '', punctuation_to_remove + string.digits))
    
    # 3) 分词
    words = word_tokenize(text)

    # 补充：处理否定词与后续词连接（如 "not good" → "not_good"）
    processed_words = []
    i = 0
    while i < len(words):
        if words[i] in {'dont', 'not', 'no', 'never'} and i + 1 < len(words):
            processed_words.append(f"{words[i]}_{words[i+1]}")
            i += 2
        else:
            processed_words.append(words[i])
            i += 1

    # 4) 去英文停用词
    stop_set = set(stopwords.words('english'))
    filtered = [w for w in processed_words if w not in stop_set and len(w) >= 2]
    # 输出为以空格连接的字符串
    return ' '.join(filtered)


__all__ = ["ensure_nltk_data", "preprocess_text"]


if __name__ == '__main__':
    # 简单自检：演示流程（不会弹出下载窗口）
    ensure_nltk_data()
    demo = "Hello, World! This is a simple DEMO of, say, text Preprocessing."
    print("Raw:", demo)
    print("Processed:", preprocess_text(demo))
