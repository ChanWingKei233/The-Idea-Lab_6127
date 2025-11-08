# -*- coding: utf-8 -*-
"""
核心模块2：模型训练
- 功能1：自动下载数据集、读取并校验列名
- 功能2：预处理 -> TF-IDF -> 8:2 划分 -> 随机森林(n_estimators=50, random_state=42)
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
import scipy.sparse as sp  # 用于合并特征矩阵
from imblearn.over_sampling import SMOTE



# --- 路径与常量 ------------------------------------------------------------------
DATA_URL = ("https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))          
DATASHEET_PATH = os.path.join(CUR_DIR, "data")                     
os.makedirs(DATASHEET_PATH, exist_ok=True)
DATA_PATH = os.path.join(DATASHEET_PATH, "labeled_data.csv")
MODEL_PATH = os.path.join(DATASHEET_PATH, "audit_model.pkl")
VEC_PATH = os.path.join(DATASHEET_PATH, "tfidf_vectorizer.pkl")

# --- 依赖：导入预处理函数（来自 wenjiang 的 app/data_process.py） --------------------
# 优先包方式导入：from app.data_process import preprocess_text
# 若包导入失败，则把 PROJECT_ROOT 加到 sys.path 后再导入，做兜底。

# try:
#     from app.data_process import preprocess_text
# except Exception:
#     if PROJECT_ROOT not in sys.path:
#         sys.path.append(PROJECT_ROOT)
#     from app.data_process import preprocess_text

# ---------------- 关键优化1：定义明确中性词（切断错误关联） ----------------
neutral_words = {'great', 'day', 'hope', 'well', 'good', 'nice', 'happy', 'love', 'learn', 'python'}  # 覆盖误判案例中的词

# ---------------- 关键优化2：扩展轻度冒犯词表+创建专属特征 ----------------
mild_offensive_words = {'idiot', 'fool', 'dumb', 'lame', 'stupid', 'silly', 'moron'}  # 扩展词表

# ---------------- 关键优化3：加权TF-IDF（压制中性词，强化冒犯词） ----------------
class WeightedTfidfVectorizer(TfidfVectorizer):
    def _compute_idf(self, doc_counts, total_docs):
        idf = super()._compute_idf(doc_counts, total_docs)
        for idx, word in enumerate(self.get_feature_names_out()):
            # 对中性词增加IDF（降低权重，减少对仇恨类的影响）
            if word in neutral_words:
                idf[idx] *= 2.0  # 权重降低一半
            # 对轻度冒犯词降低IDF（提高权重）
            elif word in mild_offensive_words:
                idf[idx] *= 0.3  # 权重提升约3倍
        return idf

# ---------------- 关键优化4：为轻度冒犯词创建二进制特征（强制模型关注） ----------------
def create_offensive_binary_features(texts):
    """为每个文本创建二进制特征：是否包含轻度冒犯词（1=包含，0=不包含）"""
    binary_features = []
    for text in texts:
        has_offensive = 1 if any(word in text for word in mild_offensive_words) else 0
        binary_features.append([has_offensive])
    return sp.csr_matrix(binary_features)  # 转换为稀疏矩阵，方便与TF-IDF合并


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
        )
    return df[["tweet", "class"]].dropna()


# --- 主函数：训练模型 ----------------------------------------------------
def train_model(
    save_model_path: str = MODEL_PATH,
    save_vec_path: str = VEC_PATH,
) -> float:
    """按训练模型并保存，返回测试集准确率。"""
    # 下载与读取数据
    download_dataset(DATA_PATH, DATA_URL)
    df = load_and_validate_dataset(DATA_PATH)

    # 预处理 -> 向量化 -> 划分
    print("[info] 开始文本清洗（调用 wenjiang 的 preprocess_text）...")
    # X_clean = df["tweet"].astype(str).apply(preprocess_text)
    df['processed_text'] = df['tweet'].apply(preprocess_text)
    y = df["class"].astype(int)

    print("[info] TF-IDF 向量化...")
    # 2. 提取TF-IDF特征
    vectorizer = WeightedTfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=1
    )
    X_tfidf = vectorizer.fit_transform(df['processed_text'])

    # 3. 提取轻度冒犯词二进制特征
    X_binary = create_offensive_binary_features(df['processed_text'])

    # 4. 合并特征（TF-IDF + 二进制特征，强化冒犯词信号）
    X = sp.hstack([X_tfidf, X_binary])  # 横向合并特征矩阵

    # 5. 数据拆分
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. 过采样（调整正常类权重，避免被误判）
    train_class_counts = pd.Series(y_train).value_counts()
    majority_count = train_class_counts[1]
    smote = SMOTE(
        random_state=42,
        sampling_strategy={0: min(majority_count, 15000), 1: majority_count, 2: min(majority_count, 15000)}  # 正常类过采样到与多数类一致
    )
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 7. 模型（提高正常类权重，减少误判）
    model = RandomForestClassifier(
        n_estimators=400,  # 大幅增加树数量，强制学习低频特征
        class_weight={0: 1.2, 1: 1.8, 2: 1.2},  # 冒犯类（1）权重进一步提高
        min_samples_split=2,  # 允许最细分裂，捕捉"idiot"这类低频词
        max_depth=70,  # 更深的树，学习更多细节
        random_state=42,
        bootstrap=False  # 不使用bootstrap抽样，全量数据训练每棵树（增强对低频词的学习）
    )
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[result] 测试集准确率：{acc * 100:.2f}%  （预期区间 75% ~ 85%）")

    joblib.dump(model, save_model_path)
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
