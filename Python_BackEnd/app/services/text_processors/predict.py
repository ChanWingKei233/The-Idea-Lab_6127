# test CICD
# -*- coding: utf-8 -*-
"""
核心模块3：预测功能
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
# 从 model_train 模块中导入特定函数
from app.services.text_processors.model_train import create_offensive_binary_features 
import scipy.sparse as sp  # 用于合并特征矩阵

CUR_DIR = os.path.dirname(os.path.abspath(__file__))          
DATASHEET_PATH = os.path.join(CUR_DIR, "data")     
MODEL_PATH = os.path.join(DATASHEET_PATH, "audit_model.pkl")
VEC_PATH = os.path.join(DATASHEET_PATH, "tfidf_vectorizer.pkl")

# --- 功能1：模型加载-------------------------------
def load_trained_model():
    """
    加载训练好的模型与 TF-IDF 向量器。
    若文件缺失，抛异常并提示：请先运行训练命令。
    """
    if not os.path.isfile(MODEL_PATH):
        raise Exception("请先运行 train 命令")
    
    if not os.path.isfile(VEC_PATH):
        raise Exception("请先运行 train 命令")
        
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
        return model, vectorizer
    except Exception as e:
        # 如果加载过程中出现错误（如文件损坏），也抛出异常
        raise Exception(f"模型加载失败: {e}")

# --- 功能2：预测函数------------------------
_LABEL_MAP = {0: "仇恨言论", 1: "冒犯性言论", 2: "正常言论"}


def predict_content(text: str) -> str:
    """
    输入文本 → preprocess_text 清洗 → 向量化 → 模型预测 → 返回中文标签。
    """
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("预测文本不能为空字符串，例如：predict_content('I hate you!')")
    
    model, vectorizer = load_trained_model()
    clean = preprocess_text(text)
    # 对测试文本提取并合并特征（与训练时一致）
    tfidf_test = vectorizer.transform([clean])
    binary_test = create_offensive_binary_features([clean])
    X_test_combined = sp.hstack([tfidf_test, binary_test])
    pred = model.predict(X_test_combined)[0]
     
    print(f"文本：{text}")
    print(f"清洗后：{clean}")
    print(f"预测结果：{_LABEL_MAP[pred]}\n")
    return _LABEL_MAP.get(pred, f"未知标签({pred})")

# --- 功能3：测试用例 ----------------------------
def test_predict() -> None:
    # 5个测试样例
    test_texts = [
        "I hate you! You are a nigger.",           # 期望：仇恨/冒犯倾向（通常→“仇恨言论”）
        "Have a great day! I hope you're well.",   # 期望：正常文本
        "Go away, you idiot!",                     # 期望：冒犯性文本
        "I love learning Python.",                 # 期望：正常文本
        "You are a stupid person."                 # 期望：冒犯性文本
    ]
    print("== 预测测试==")
    for s in test_texts:
        try:
            label = predict_content(s)
        except Exception as e:
            label = f"[ERROR] {e}"
        #print(f"- {s}\n  -> {label}")


# 直接运行：python app_xiaotian/predict.py
if __name__ == "__main__":
    test_predict()
