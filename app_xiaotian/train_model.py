# %%
"""
train_model.py

功能：
1. 检查本地是否存在数据集 data/tweets.csv
2. 若无，自动从 GitHub 下载 labeled_data.csv
3. 清洗文本并训练随机森林模型
4. 输出准确率与预测分布
"""

import os
import pandas as pd
import numpy as np
import joblib
import urllib.request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data_process import preprocess_text


# ------------------------------
# 自动下载数据集（标准 Python 实现）
# ------------------------------
def ensure_dataset(data_path: str):
    """检查本地是否存在 tweets.csv，否则自动下载"""
    if os.path.exists(data_path):
        print(f"✅ 已找到数据文件：{data_path}")
        return

    print("⚠️ 未找到数据文件，正在下载公开数据集 (labeled_data.csv)...")
    url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
    tmp_path = "data/labeled_data.csv"
    os.makedirs("data", exist_ok=True)

    urllib.request.urlretrieve(url, tmp_path)
    print("✅ 下载完成：labeled_data.csv")

    # 只保留 tweet 和 class 两列，保存为 tweets.csv
    df = pd.read_csv(tmp_path)[["tweet", "class"]]
    df.to_csv(data_path, index=False)
    print(f"✅ 数据集已保存为：{data_path}")


# ------------------------------
# 类别分布可视化
# ------------------------------
def evaluate_distribution(y_true, y_pred):
    label_map = {0: "正常", 1: "冒犯性", 2: "仇恨言论"}
    print("\n📊 预测分布：")
    pred_dist = pd.Series(y_pred).value_counts(normalize=True).sort_index() * 100
    for i, p in pred_dist.items():
        print(f"  类别 {i} ({label_map.get(i)}): {p:.1f}%")

    print("\n📊 真实分布：")
    true_dist = pd.Series(y_true).value_counts(normalize=True).sort_index() * 100
    for i, p in true_dist.items():
        print(f"  类别 {i} ({label_map.get(i)}): {p:.1f}%")


# ------------------------------
# 主流程：训练与评估
# ------------------------------
def train_and_evaluate(data_path="data/tweets.csv", save_dir="models"):
    ensure_dataset(data_path)

    print("\n🔍 读取数据...")
    df = pd.read_csv(data_path)
    print("数据样本数：", len(df))

    print("🧹 开始清洗文本...")
    df["processed_text"] = df["tweet"].apply(preprocess_text)
    print("✅ 文本清洗完成。")

    print("🧠 向量化中...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["processed_text"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🌲 训练随机森林模型...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        min_samples_split=8,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("📈 模型评估中...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 训练完成！模型准确率：{acc:.3f}")
    print("\n📊 分类报告：\n", classification_report(y_test, y_pred))

    # 类别分布
    evaluate_distribution(y_test, y_pred)

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "rf_model.joblib"))
    joblib.dump(vectorizer, os.path.join(save_dir, "tfidf_vectorizer.joblib"))
    print(f"\n💾 模型与向量器已保存至：{save_dir}/")


# ------------------------------
# 脚本入口
# ------------------------------
if __name__ == "__main__":
    train_and_evaluate()


# %%
"""
predict_text.py
功能：加载模型并对新文本进行预测
"""

import os
import joblib
import pandas as pd
from typing import List, Union
from data_process import preprocess_text


def load_model(model_dir: str = "models"):
    """加载模型与向量器"""
    model = joblib.load(os.path.join(model_dir, "rf_model.joblib"))
    vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    print("✅ 模型与向量器加载完成。")
    return model, vectorizer


def predict_texts(texts: Union[str, List[str]], model, vectorizer):
    """预测输入文本"""
    if isinstance(texts, str):
        texts = [texts]

    cleaned = [preprocess_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)

    label_map = {0: "正常", 1: "冒犯性", 2: "仇恨言论"}
    result = pd.DataFrame({
        "原始文本": texts,
        "清洗后文本": cleaned,
        "预测类别": [label_map.get(p, "未知") for p in preds]
    })

    return result


if __name__ == "__main__":
    model, vectorizer = load_model("models")

    demo_texts = [
        "I love your work!",
        "You're so stupid.",
        "I hate those people, they should be banned."
    ]
    results = predict_texts(demo_texts, model, vectorizer)
    print("\n📊 预测结果：")
    print(results)



