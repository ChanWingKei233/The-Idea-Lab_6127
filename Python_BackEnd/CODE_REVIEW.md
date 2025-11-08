# 内容审核系统代码审查清单
## 一、审查说明
1. **审查目的**：验证代码是否符合逻辑、团队开发规范，确保功能完整、可维护、无潜在问题；
2. **审查范围**：项目全模块（基础模块 / 核心模块 / CLI 入口 / 测试模块 / CI/CD 配置）；
3. **审查标准**：
   - 功能一致性：符合文档描述（如模型参数、预处理步骤）；
   - 代码规范性：命名清晰（如函数用 snake_case）、注释完整（关键步骤标文档对应关系）、无冗余代码；
   - 协作兼容性：模块间依赖正常（如 CLI 可调用核心模块）；
   - 测试完整性：测试用例覆盖正常/异常场景，通过率100%。

## 二、按成员分工分类审查（核心部分）
### YE LANMENG(初始化配置、部署模块：CI/CD 配置 + 项目文档)
| 序号 | 审查模块                | 审查内容 | 审查结果 | 验证方式 |
| -- | --------------------- | -------- | -------- | -------- |
| 1  | 初始化配置（.gitignore/requirements.txt） | 1. .gitignore 是否包含冗余文件（__pycache__/、*.pkl、labeled_data.csv）；2. requirements.txt 是否包含所有依赖（pandas==2.1.4、nltk==3.8.1、scikit-learn==1.3.2），无多余/缺失库。 | \[√] 符合 | 1. 查看根目录.gitignore；2. 本地执行`pip install -r requirements.txt`，无报错。                                                                      |

### wenjiang（基础模块：文本预处理）
| 序号 | 审查模块                | 审查内容 | 审查结果 | 验证方式 |
| -- | --------------------- | -------- | -------- | -------- |
| 1  | 文本预处理（src/data_process.py）        | 1. NLTK数据下载：自动检测stopwords/punkt，处理LookupError；2. 预处理步骤：转小写→去标点→分词→去停用词；3. 非字符串输入（None/空文本）返回合理结果。 | \[√] 符合 | 1. 删除NLTK数据后调用函数，验证自动下载；2. 输入`"I hate you!..."`，输出`"hate terrible"`；3. 测试None输入，无报错。 |
| 2  | 代码规范性   | 1. 函数名preprocess_text（snake_case），变量名清晰；2. 关键步骤加注释（如“NLTK数据下载”）。 | \[√] 符合 | 查看src/data_process.py代码。   |
### 2.2 xiaotian（核心模块：模型训练 + 预测）
| 序号 | 审查模块               | 审查内容 | 审查结果 | 验证方式 |
| -- | -------------------- | -------- | -------- | -------- |
| 1  | 模型训练（src/model_train.py） | 1. 自动下载labeled_data.csv，验证“tweet”“class”列；2. 随机森林参数n_estimators=50、random_state=42；3. 生成audit_model.pkl和tfidf_vectorizer.pkl，准确率75%-85%。 | \[❌] 符合 | 1. 执行`python -c "from src.model_train import train_model; train_model()"`；2. 查看代码参数；3. 日志显示准确率不在75%-85%区间内，建议复查预处理或重训（样本划分有随机性）。  |
| 2  | 文本预测（src/predict.py）      | 1. 模型缺失抛提示“请先运行train命令”；2. 标签0=正常/1=冒犯/2=仇恨；3. test_predict含5个文档案例，结果符合预期。    | \[❌] 符合 | 1. 删除模型后调用predict_content；2. 输入任意测试文本，执行test_predict，均返回“冒犯性文本”，结果有误。 |