# -*- coding: utf-8 -*-
"""
核心模块3：预测功能（对应《内容自动审核系统.docx》单元格6）
- 依赖：app/data_process.py 的 preprocess_text（wenjiang）
- 加载：项目根目录 audit_model.pkl / tfidf_vectorizer.pkl（xiaotian 训练生成）
- 提供：
  1) load_trained_model()  —— 加载模型与向量器（缺失时抛“请先训练”的异常）
  2) predict_content(text) —— 清洗→向量化→预测→返回中文标签（0/1/2）
  3) test_predict()        —— 5 个示例用例打印结果（供联调与测试）
"""

import os
import sys
import joblib
from app.services.text_processors.model_train import train_model
# import model_train
from app.services.text_processors.data_process import preprocess_text
# import data_process

# --- 路径：当前文件位于 app_xiaotian/，模型文件位于项目根目录 -------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))     # .../The-Idea-Lab_6127/app_xiaotian
PROJECT_ROOT = os.path.dirname(CUR_DIR)                  # .../The-Idea-Lab_6127
MODEL_PATH = os.path.join(PROJECT_ROOT, "audit_model.pkl")
VEC_PATH = os.path.join(PROJECT_ROOT, "tfidf_vectorizer.pkl")

# --- 功能1：模型加载（提示词第3点 / 文档单元格6前置） -------------------------------
def load_trained_model():
    """
    加载训练好的模型与 TF-IDF 向量器。
    若文件缺失，抛异常并提示：请先运行训练命令。
    """
    model_path, vec_path=train_model()
    if model_path is None and vec_path is not None:
        print("模型缺失")
    if vec_path is None and model_path is not None:
        print("向量器缺失")
    if vec_path is None and model_path is None:
        print("未找到训练产物")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

# --- 功能2：预测函数（提示词第4点 / 文档单元格6标签 0/1/2） -------------------------
_LABEL_MAP = {0: "正常文本", 1: "冒犯性文本", 2: "仇恨言论"}

def predict_content(text: str) -> str:
    """
    输入文本 → preprocess_text 清洗 → 向量化 → 模型预测 → 返回中文标签。
    """
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("预测文本不能为空字符串，例如：predict_content('I hate you!')")
    model, vectorizer = load_trained_model()
    clean = preprocess_text(text)
    X = vectorizer.transform([clean])
    pred = int(model.predict(X)[0])
    return _LABEL_MAP.get(pred, f"未知标签({pred})")

# --- 功能3：测试用例（提示词第5点 / 文档单元格6示例风格） ----------------------------
def test_predict(input_content: str) -> str:
    # 单元格6：5个固定样例（保持一字不差）
    # "I hate you! You are terrible.",           # 期望：仇恨/冒犯倾向（通常→“仇恨言论”）
    # "Have a great day! I hope you're well.",   # 期望：正常文本
    # "Go away, you idiot!",                     # 期望：冒犯性文本
    # "I love learning Python.",                 # 期望：正常文本
    # "You are a stupid person."                 # 期望：冒犯性文本
    
    test_texts = []
    result=""
    if not input_content:
        return "输入内容为空"
    else:
        test_texts.append(input_content)   
        print("\n===== 预测结果 =====")
        for text in test_texts:
            try:
                result = predict_content(text)
            except Exception as e:
                result = f"[ERROR] {e}"
            print(f"- {text}\n  -> {result}")
    return result



# 直接运行：python app_xiaotian/predict.py
if __name__ == "__main__":
    test_predict()
