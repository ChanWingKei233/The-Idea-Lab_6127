"""
app/data_process.py

核心模块1：文本预处理
基于《内容自动审核系统.docx》实现（供训练模块调用）

- 功能1（对应文档“单元格2”）：处理 NLTK 数据下载
  自动检测 stopwords 与 punkt 是否已安装，缺失则静默下载；
  捕获并处理 LookupError，避免组员调用时弹出窗口或报错。

- 功能2（对应文档“单元格4”）：文本清洗函数 preprocess_text
  严格按照流程：转小写 → 去标点（str.maketrans） → 分词（word_tokenize） → 去英文停用词（stopwords.words('english')）
  输入：原始文本字符串；输出：清洗后的字符串（用空格连接）。
"""

from __future__ import annotations

import string
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



# ------------------------------
# 功能1：NLTK 数据下载与异常处理（文档单元格2）
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
    - 显式捕获 LookupError，防止组员侧出现未处理异常
    """
    # 1) stopwords
    if not _has_resource('corpora/stopwords'):
        # 静默下载，避免 GUI 弹窗
        nltk.download('stopwords', quiet=True)
    # 再次验证，若仍缺失则抛出可读性更好的异常说明
    if not _has_resource('corpora/stopwords'):
        raise LookupError("Failed to obtain NLTK resource: 'stopwords'. Please check network or NLTK data path.")

    # 2) punkt 分词模型
    if not _has_resource('tokenizers/punkt'):
        nltk.download('punkt', quiet=True)
    if not _has_resource('tokenizers/punkt'):
        raise LookupError("Failed to obtain NLTK resource: 'punkt'. Please check network or NLTK data path.")


# ------------------------------
# 功能2：文本清洗（文档单元格4）
# ------------------------------
def preprocess_text(text: str) -> str:
    """按照文档流程清洗文本，返回清洗后的字符串。

    步骤（严格按顺序执行）：
    1. 转小写
    2. 去标点：使用 str.maketrans('', '', string.punctuation)
    3. 分词：nltk.tokenize.word_tokenize
    4. 去英文停用词：nltk.corpus.stopwords.words('english')
    """
    if not isinstance(text, str):
        # 允许调用方传入非字符串（如 None/数字），统一转为字符串处理
        text = "" if text is None else str(text)

    # —— 预检查并准备 NLTK 资源（单元格2 要求）——
    try:
        ensure_nltk_data()
    except LookupError:
        # 尽力而为仍失败时，转为友好错误信息，避免训练管道崩溃
        # 调用方可选择捕获此异常并进行降级处理
        raise

    # 1) 转小写（文档单元格4 - 步骤1）
    lowered = text.lower()

    # 2) 去标点（文档单元格4 - 步骤2）
    table = str.maketrans('', '', string.punctuation)
    no_punc = lowered.translate(table)

    # 3) 分词（文档单元格4 - 步骤3）
    tokens = word_tokenize(no_punc)

    # 4) 去英文停用词（文档单元格4 - 步骤4）
    stop_set = set(stopwords.words('english'))
    filtered = [tok for tok in tokens if tok and tok not in stop_set]

    # 输出为以空格连接的字符串
    return ' '.join(filtered)


__all__ = ["ensure_nltk_data", "preprocess_text"]


if __name__ == '__main__':
    # 简单自检：演示流程（不会弹出下载窗口）
    ensure_nltk_data()
    demo = "Hello, World! This is a simple DEMO of, say, text Preprocessing."
    print("Raw:", demo)
    print("Processed:", preprocess_text(demo))
