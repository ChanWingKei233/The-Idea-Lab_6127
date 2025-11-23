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
| 1  | 初始化配置（.gitignore/requirements.txt） | 1. .gitignore 是否包含冗余文件（__pycache__/、*.pkl、labeled_data.csv）；2. requirements.txt 是否包含所有依赖（pandas==2.1.4、nltk==3.8.1、scikit-learn==1.3.2），无多余/缺失库。 | \[√] 符合 | 1. 查看根目录.gitignore；2. 本地执行`pip install -r requirements.txt`，无报错。                                      |
| 2  | CI/CD配置 |  1. 触发条件：推送代码到dev或main分支 / 发起 PR 到dev或main分支（符合持续集成逻辑）；2. 流程步骤完整：   - 拉代码 → 安装 Python 3.13 → 安装依赖 → 提前下载 NLTK 数据（解决下载慢问题）；   - 运行 unittest 测试（调用 yongqi 的 test_audit_unittest.py）；   - 上传测试报告（test_report_unittest.txt）；3. 环境兼容：安装 wget 用于数据集下载，指定 Python 3.13（与本地开发环境一致）。| \[√] 符合 | 1. 查看 ci-cd.yml 代码 → 核对步骤顺序和命令正确性（如 NLTK 下载命令、unittest 运行命令）；2. 推送一个空提交（git commit --allow-empty -m "测试 CI"）→ 查看 GitHub Actions 日志 → 确认所有步骤显示绿色对勾；3. 下载 GitHub Actions Artifacts→ 验证 test_report_unittest.txt 存在且记录完整|

### PANG WENJIANG（基础模块：文本预处理）
| 序号 | 审查模块                | 审查内容 | 审查结果 | 验证方式 |
| -- | --------------------- | -------- | -------- | -------- |
| 1  | 文本预处理（src/data_process.py）        | 1. NLTK数据下载：自动检测stopwords/punkt，处理LookupError；2. 预处理步骤：转小写→去标点→分词→去停用词；3. 非字符串输入（None/空文本）返回合理结果。 | \[√] 符合 | 1. 删除NLTK数据后调用函数，验证自动下载；2. 输入`"I hate you!..."`，输出`"hate terrible"`；3. 测试None输入，无报错。 |
| 2  | 代码规范性   | 1. 函数名preprocess_text（snake_case），变量名清晰；2. 关键步骤加注释（如“NLTK数据下载”）。 | \[√] 符合 | 查看src/data_process.py代码。   |

### HONG XIAOTIAN（核心模块：模型训练 + 预测）
| 序号 | 审查模块               | 审查内容 | 审查结果 | 验证方式 |
| -- | -------------------- | -------- | -------- | -------- |
| 1  | 模型训练（src/model_train.py） | 1. 自动下载labeled_data.csv，验证“tweet”“class”列；2. 随机森林参数n_estimators=50、random_state=42；3. 生成audit_model.pkl和tfidf_vectorizer.pkl，准确率最好在75%-85%。 | \[√]符合 | 1. 执行`python -c "from src.model_train import train_model; train_model()"`；2. 查看代码参数；3. 日志显示准确率略高于85%，不影响后续预测结果。  |
| 2  | 文本预测（src/predict.py）      | 1. 模型缺失抛提示“请先运行train命令”；2. 标签0=正常/1=冒犯/2=仇恨；3. test_predict含5个文档案例，结果符合预期。    | \[√] 符合 | 1. 删除模型后调用predict_content；2. 输入任意测试文本，执行test_predict，返回预期结果。 |

### CHEN YINGQI（CLI 入口、前端界面对接）
| 序号 | 审查模块               | 审查内容 | 审查结果 | 验证方式 |
| -- | -------------------- | -------- | -------- | -------- |
| 1  | 命令实现 | 1. 支持 train（训练模型）、test（运行测试案例）、predict（单文本审核）三个命令；2. predict 命令强制文本用引号包裹（如 predict "test"），未包裹时提示示例 “请用引号包裹文本（示例：python run.py predict "test"）”；3. 命令调用核心模块正确（train 调用train_model，test 调用 test_predict）| \[√] 符合 | 分别执行三条命令，均符合要求 |

| 2  | 前端界面对接 | 1. API 接口实现: 提供 /api/predict (或类似) 接口用于接收前端 POST 请求；2. 请求处理: 正确解析前端发送的 JSON 数据（如 {"text": "You are stupid"}）；3. 响应格式: 返回统一格式的 JSON 响应，包含预测结果；4. 跨域支持: 配置了 CORS，允许前端域名访问后端 API。 | \[√] 符合 ｜前端验证: 打开前端页面，输入文本并提交，观察是否能正确显示预测结果。｜


### LYU YONGQI（unittest测试）
| 序号 | 审查模块               | 审查内容 | 审查结果 | 验证方式 |
| -- | -------------------- | -------- | -------- | -------- |
| 1  | 测试覆盖度 | 1. 覆盖全模块：   - 预处理模块（TestPreprocess：正常文本 / 空文本 / None 输入）；   - 核心模块（TestModelAndPredict：模型加载 / 缺失、预测标签正确性）；   - CLI 模块（TestCLI：train/test/predict 命令）；2. 每个模块覆盖「正常场景 + 异常场景」（如模型缺失测试、空文本测试），无遗漏核心功能。）| \[√] 符合 | 1. 查看 test_audit_unittest.py 代码 → 确认 3 个测试类、8 个用例；2. 核对用例是否覆盖文档关键逻辑（如预处理步骤、预测标签定义）。|
| 2  | 测试有效性 | 1. 断言逻辑准确：   - test_preprocess_normal 断言输出为 “hate terrible”（文档单元格 4）；   - test_predict_cases 断言仇恨言论返回 2、冒犯性返回 1、正常文本返回 0（文档单元格 5）；2. 直接运行测试脚本 → 所有用例显示 OK，无 FAIL/ERROR。| \[√] 符合 | 1. 执行 python -m unittest tests/test_audit_unittest.py -v → 确认输出 Ran 8 tests in XXs. OK；2. 查看用例中 self.assertXXX 断言 → 核对预期结果与文档一致。|
| 3  | 报告生成与可执行性 |1. 支持生成文本报告（test_report_unittest.txt），包含用例名称、执行结果、总统计（用例数 / 失败数 / 错误数）；2. 无需额外依赖（仅用 Python 内置 unittest），CI 环境可直接运行。| \[√] 符合 | 1. 执行 python tests/test_audit_unittest.py → 检查 tests/ 目录生成报告文件；2. 打开报告 → 确认记录完整（如 测试用例总数：8，失败：0，错误：0）。|
