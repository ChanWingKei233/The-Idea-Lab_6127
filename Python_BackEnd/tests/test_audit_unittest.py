import unittest
import os
import sys
import subprocess
import shutil

# --- 核心修改：动态添加项目根目录到 sys.path ---
# 获取当前测试文件的绝对路径
CURRENT_FILE_PATH = os.path.abspath(__file__)
# 获取 tests 目录的路径
TESTS_DIR = os.path.dirname(CURRENT_FILE_PATH)
# 获取项目根目录（Python_BackEnd）的路径
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
# 将项目根目录添加到 sys.path 的最前面，确保优先搜索
sys.path.insert(0, PROJECT_ROOT)

from app.services.text_processors.data_process import preprocess_text
from app.services.text_processors.model_train import train_model, MODEL_PATH, VEC_PATH
from app.services.text_processors.predict import load_trained_model, predict_content

# --- 全局配置 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

class Test1Preprocess(unittest.TestCase):
    """
    测试 wenjiang 负责的文本预处理模块
    """
    def test_preprocess_normal(self):
        """测试正常文本预处理流程"""
        input_text = "I hate you! You are terrible."
        expected_output = "hate terrible"
        self.assertEqual(preprocess_text(input_text), expected_output)

    def test_preprocess_empty(self):
        """测试空字符串和 None 输入"""
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text(None), "")

class Test2ModelPredict(unittest.TestCase):
    """
    测试 xiaotian 负责的核心模块
    """
    @classmethod
    def setUpClass(cls):
        """
        在所有测试用例执行前运行一次，用于全局初始化。
        这里提前训练好模型，供所有测试用例使用。
        """
        print("--- 开始训练模型，供测试使用 ---")
        # 切换到项目根目录执行训练，确保文件生成在正确位置
        original_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        try:
            train_model()
        finally:
            os.chdir(original_cwd)
        print("--- 模型训练完毕 ---")

    def test_model_load_exist(self):
        """测试模型文件存在时，加载成功"""
        model, vectorizer = load_trained_model()
        # 简单断言模型和向量器对象被成功创建
        self.assertIsNotNone(model)
        self.assertIsNotNone(vectorizer)

    def test_model_load_missing(self):
        """测试模型文件缺失时，抛出正确异常"""
        # 备份并删除模型文件
        if os.path.exists(MODEL_PATH): shutil.move(MODEL_PATH, f"{MODEL_PATH}.bak")
        if os.path.exists(VEC_PATH): shutil.move(VEC_PATH, f"{VEC_PATH}.bak")

        try:
            with self.assertRaises(Exception) as context:
                load_trained_model()
            self.assertIn("请先运行 train 命令", str(context.exception))
        finally:
            # 恢复模型文件
            if os.path.exists(f"{MODEL_PATH}.bak"): shutil.move(f"{MODEL_PATH}.bak", MODEL_PATH)
            if os.path.exists(f"{VEC_PATH}.bak"): shutil.move(f"{VEC_PATH}.bak", VEC_PATH)

    def test_predict_cases(self):
        """测试3个典型预测案例"""
        test_cases = [
            ("I hate you! You are a nigger.", "仇恨言论"),
            ("You are stupid", "冒犯性言论"),
            ("Have a nice day", "正常言论")
        ]
        for text, expected_label in test_cases:
            with self.subTest(text=text):
                result_label = predict_content(text)
                print(result_label, expected_label)
                self.assertEqual(result_label, expected_label)

class Test3CLI(unittest.TestCase):
    """
    测试 yingqi 负责的 CLI 模块
    """
    def run_cli_command(self, command):
        """辅助函数：执行 CLI 命令并返回结果"""
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True  # 在 Windows 上可能需要，确保能识别 python 命令
        )
        return result.returncode, result.stdout, result.stderr

    def test_cli_train(self):
        # 先清理旧模型
        for file_path in [MODEL_PATH, VEC_PATH]:
            if os.path.exists(file_path):
                os.remove(file_path)

        returncode, stdout, stderr = self.run_cli_command("python run.py train")
        
        self.assertEqual(returncode, 0, f"训练命令执行失败，错误信息: {stderr}")
        self.assertTrue(os.path.exists(MODEL_PATH))
        self.assertTrue(os.path.exists(VEC_PATH))
   

    def test_cli_test(self):
        """测试 'test' 命令，验证输出包含预期关键词"""
        returncode, stdout, stderr = self.run_cli_command("python run.py test")
        
        self.assertEqual(returncode, 0, f"测试命令执行失败，错误信息: {stderr}")
        self.assertIn("仇恨言论", stdout)    # 匹配CLI实际输出的关键词
        self.assertIn("冒犯性言论", stdout)  # 匹配CLI实际输出的关键词
        self.assertIn("正常言论", stdout)    # 匹配CLI实际输出的关键词

    def test_cli_predict(self):
        """测试 'predict' 命令，验证输出正确"""
        test_text = "You are stupid"
        returncode, stdout, stderr = self.run_cli_command(f'python run.py predict "{test_text}"')
        
        self.assertEqual(returncode, 0, f"预测命令执行失败，错误信息: {stderr}")
        self.assertIn("冒犯性言论", stdout)

# 测试报告保存路径
REPORT_DIR = os.path.join(PROJECT_ROOT, "tests")
REPORT_PATH = os.path.join(REPORT_DIR, "test_report_unittest.txt")

import time
class TestResultWithTime(unittest.TextTestResult):
    """扩展测试结果类，记录每个用例的执行时间"""
    def startTest(self, test):
        self._start_time = time.time()
        super().startTest(test)

    def addSuccess(self, test):
        elapsed = time.time() - self._start_time
        self.test_times[test] = elapsed
        super().addSuccess(test)

    def addFailure(self, test, err):
        elapsed = time.time() - self._start_time
        self.test_times[test] = elapsed
        super().addFailure(test, err)

    def addError(self, test, err):
        elapsed = time.time() - self._start_time
        self.test_times[test] = elapsed
        super().addError(test, err)


class TestRunnerWithReport(unittest.TextTestRunner):
    """自定义测试运行器，生成详细测试报告"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_times = {}

    def _makeResult(self):
        result = TestResultWithTime(self.stream, self.descriptions, self.verbosity)
        result.test_times = self.test_times
        return result

    def run(self, test):
        # 确保报告目录存在
        os.makedirs(REPORT_DIR, exist_ok=True)
        
        # 执行测试
        result = super().run(test)
        
        # 生成测试报告
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"测试报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总用例数: {result.testsRun}\n")
            f.write(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}\n")
            f.write(f"失败: {len(result.failures)}\n")
            f.write(f"错误: {len(result.errors)}\n")
            f.write("="*60 + "\n\n")
            
            # 详细记录每个用例
            for test_case in result.test_times:
                case_name = str(test_case).split()[0]  # 提取用例名称
                case_doc = test_case._testMethodDoc or "无描述"  # 用例描述
                elapsed = result.test_times[test_case]  # 耗时
                
                # 判断执行结果
                if test_case in [f[0] for f in result.failures]:
                    status = "失败"
                elif test_case in [e[0] for e in result.errors]:
                    status = "错误"
                else:
                    status = "成功"
                
                f.write(f"用例名称: {case_name}\n")
                f.write(f"用例描述: {case_doc}\n")
                f.write(f"执行结果: {status}\n")
                f.write(f"耗时: {elapsed:.6f}秒\n")
                f.write("-"*60 + "\n")
        
        print(f"\n测试报告已生成: {REPORT_PATH}")
        return result


if __name__ == "__main__":
    # 使用自定义运行器执行测试并生成报告
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Test1Preprocess)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Test2ModelPredict))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Test3CLI))
    
    runner = TestRunnerWithReport(verbosity=2)
    runner.run(suite)


