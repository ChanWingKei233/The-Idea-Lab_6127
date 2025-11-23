# run.py
from app import create_app
import argparse
import sys
from pathlib import Path
import joblib

# 创建应用实例
app = create_app()

# 导入各模块功能
sys.path.append(str(Path(__file__).parent / "app"))
from app.services.text_processors.model_train import train_model  # xiaotian 开发的训练模块
from app.services.text_processors.predict import test_predict     # xiaotian 开发的测试模块
from app.services.text_processors.predict import predict_content  # xiaotian 开发的预测模块

def main():
    # 初始化命令行解析器
    parser = argparse.ArgumentParser(
        description="内容自动审核系统 CLI 入口程序",
        epilog="关键说明：train 依赖 wenjiang 预处理模块与 xiaotian 核心模块；predict 需用引号包裹文本"
    )

    # 子命令容器
    subparsers = parser.add_subparsers(dest="command", help="可用命令：train/test/predict")

    # 命令1：train（训练模型）
    subparsers.add_parser("train", help="训练内容审核模型，并保存训练产物")

    # 命令2：test（测试模型）
    subparsers.add_parser("test", help="运行5个测试案例，验证模型效果")

    # 命令3：predict（预测文本）
    predict_parser = subparsers.add_parser("predict", help="预测单个文本的审核结果（文本需用引号包裹）")
    predict_parser.add_argument("text", type=str, help="待审核文本（示例：\"test text\"）")

    # 命令4：服务启动选项（--serve）
    parser.add_argument(
        "--serve", 
        action="store_true", 
        help="启动Flask服务（监听配置文件中指定的端口）"
    )

    # 解析命令行参数
    args = parser.parse_args()


    # 核心修改：如果没有指定任何命令和参数，默认启动服务
    if args.command is None and not args.serve:
        args.serve = True

    # 处理不同命令
    if args.serve:
        # 从配置中读取端口
        app.run(host='0.0.0.0', port=app.config['PORT'])
    elif args.command == "train":
        print("=== 开始训练模型 ===")
        train_model()  # 调用 xiaotian 开发的训练函数
        print("=== 模型训练完成！ ===")

    elif args.command == "test":
        test_predict()  # 调用 xiaotian 开发的测试函数

    elif args.command == "predict":
        text = args.text
        # 调用预测函数并输出结果
        print("=== 审核结果 ===")
        predict_content(text)  # 去除引号后预测
      
    else:
        # 输错命令时提示
        print(f"❌ 未知命令：{args.command}")
        print("可用命令：train（训练）、test（测试）、predict（预测）")
        print("使用 --help 查看详细说明")
        sys.exit(1)


if __name__ == '__main__':
    main()
