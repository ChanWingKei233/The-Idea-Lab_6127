# -*- coding: utf-8 -*-
"""
核心模块2：模型训练（对应《内容自动审核系统.docx》单元格3 + 单元格5）
- 功能1（单元格3）：自动下载数据集、读取并校验列名
- 功能2（单元格5）：预处理 -> TF-IDF -> 8:2 划分 -> 随机森林(n_estimators=50, random_state=42)
- 训练后打印准确率（预计 75%~85%），并保存模型与向量器（audit_model.pkl / tfidf_vectorizer.pkl）
"""

import os
import sys
import subprocess
import shutil
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
# import data_process
from app.services.text_processors.data_process import preprocess_text

# --- 路径与常量 ------------------------------------------------------------------
DATA_URL = ("https://raw.githubusercontent.com/t-davidson/"
            "hate-speech-and-offensive-language/master/data/labeled_data.csv")

# 当前文件位于 app_xiaotian/ 目录；模型与数据统一保存在项目根目录
CUR_DIR = os.path.dirname(os.path.abspath(__file__))           # .../The-Idea-Lab_6127/app_xiaotian
PROJECT_ROOT = os.path.dirname(CUR_DIR)                        # .../The-Idea-Lab_6127
DATA_PATH = os.path.join(PROJECT_ROOT, "labeled_data.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "audit_model.pkl")
VEC_PATH = os.path.join(PROJECT_ROOT, "tfidf_vectorizer.pkl")

# --- 依赖：导入预处理函数（来自 wenjiang 的 app/data_process.py） --------------------
# 优先包方式导入：from app.data_process import preprocess_text
# 若包导入失败，则把 PROJECT_ROOT 加到 sys.path 后再导入，做兜底。

# try:
#     from app.data_process import preprocess_text
# except Exception:
#     if PROJECT_ROOT not in sys.path:
#         sys.path.append(PROJECT_ROOT)
#     from app.data_process import preprocess_text


# --- 工具函数：数据集下载（curl/wget/urllib 三重兜底） ------------------------------
def download_dataset(output_path: str = DATA_PATH, url: str = DATA_URL) -> None:
    """下载数据集到项目根目录；若已存在则跳过。"""
    if os.path.exists(output_path):
        print(f"[info] 数据集已存在：{output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if shutil.which("curl"):
        print("[info] 使用 curl 下载数据集...")
        code = subprocess.call(["curl", "-L", "-o", output_path, url])
        if code == 0 and os.path.exists(output_path):
            print("[info] curl 下载完成")
            return
        print("[warn] curl 下载失败，尝试 wget/urllib")

    if shutil.which("wget"):
        print("[info] 使用 wget 下载数据集...")
        code = subprocess.call(["wget", "-O", output_path, url])
        if code == 0 and os.path.exists(output_path):
            print("[info] wget 下载完成")
            return
        print("[warn] wget 下载失败，尝试 urllib")

    print("[info] 使用 urllib 下载数据集（兜底方案）...")
    urllib.request.urlretrieve(url, output_path)
    if not os.path.exists(output_path):
        raise RuntimeError("数据集下载失败：未能在任何方式下成功下载。")


# --- 读取并校验数据（必须含 tweet / class 列） --------------------------------------
def load_and_validate_dataset(csv_path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"tweet", "class"}
    cols_lower = [c.lower() for c in df.columns]
    df.columns = cols_lower
    if not required.issubset(set(cols_lower)):
        raise ValueError(
            f"数据列缺失：需要包含 {required}；实际列为 {set(df.columns)}。"
            "（对应文档单元格3校验）"
        )
    return df[["tweet", "class"]].dropna()


# --- 主函数：训练模型（单元格5） ----------------------------------------------------
def train_model(
    n_estimators: int = 50,
    random_state: int = 42,
    save_model_path: str = MODEL_PATH,
    save_vec_path: str = VEC_PATH,
) -> float:
    """按文档单元格5训练模型并保存，返回测试集准确率。"""
    # 单元格3：下载与读取数据
    download_dataset(DATA_PATH, DATA_URL)
    df = load_and_validate_dataset(DATA_PATH)

    # 预处理 -> 向量化 -> 划分
    print("[info] 开始文本清洗（调用 wenjiang 的 preprocess_text）...")
    X_clean = df["tweet"].astype(str).apply(preprocess_text)
    y = df["class"].astype(int)

    print("[info] TF-IDF 向量化...")
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X_clean)

    print("[info] 划分训练/测试集（8:2，random_state=42）...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=random_state, stratify=y
    )

    print(f"[info] 训练随机森林（n_estimators={n_estimators}, random_state={random_state}）...")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[result] 测试集准确率：{acc * 100:.2f}%  （预期区间 75% ~ 85%）")

    joblib.dump(clf, save_model_path)
    joblib.dump(vectorizer, save_vec_path)
    print(f"[save] 模型已保存：{save_model_path}")
    print(f"[save] 向量器已保存：{save_vec_path}")
    print(f"[save] 数据集位置：{DATA_PATH}")

    if not (0.75 <= acc <= 0.85):
        print("[warn] 准确率不在 75%-85% 区间内：建议复查预处理或重训（样本划分有随机性）。")

    # return acc
    return save_model_path, save_vec_path


if __name__ == "__main__":
    # 直接运行：python app_xiaotian/model_train.py
    train_model()
