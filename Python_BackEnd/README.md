# 内容审核系统
这是一款基于 Python 的文本审核工具，通过 AI 技术识别正常文本、冒犯性文本、仇恨言论，满足要求：
    - 支持 CLI 命令行界面
    - 集成机器学习模型
    - 包含测试框架（pytest）
    - 搭建 CI/CD 持续集成流程
    - 编制专业项目文档


## 如何克隆此仓库
要克隆此仓库到你的本地计算机，请打开终端或命令提示符，然后输入以下命令：
```bash
git clone https://github.com/ChanWingKei233/The-Idea-Lab_6127.git
```

## 项目正确运行方式
1. 使用虚拟环境及装包：
```bash
# 1. 创建虚拟环境
python3 -m venv .venv
# 2. 激活（macOS / bash / zsh）
source .venv/bin/activate
# 3. 可选：升级 pip
python -m pip install --upgrade pip
# 4. 安装 requirements.txt里所有的包
python -m pip install -r requirements.txt
```

2. 在安装新包时：
```bash
pip install 包名
# 安装时将包记录在requirements.txt
pip freeze > requirements.txt
```

## 项目CLI命令：
```bash
# 1. 训练模型
python run.py train
# 2. 测试文本
python run.py test
# 3. 预测文本
python run.py predict 'You are a nigger.'
```

## Unittest及生成测试报告
```bash
python tests/test_audit_unittest.py
```



## 项目文档索引
1. 开发规范：`DEVELOPMENT_GUIDE.md`（分支命名、PR 提交、AI 辅助规则）
2. 代码审查：`CODE_REVIEW.md`（后续补充，记录代码质量检查结果）