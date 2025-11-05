from app.services.text_processors.predict import test_predict

test_texts=[]
def process_text(input_content: str) -> str:
    """处理文本的核心逻辑（原 app.py 中的处理代码）"""
    result=""
    if not input_content:
        return "输入内容为空"
    else:
        test_texts.append(input_content)   
        print("\n===== 预测结果 =====")
        for text in test_texts:
            result = test_predict(text)
            print(f"文本：{text}")
            print(f"结果：{result}\n")
        
    # 示例处理：在文本前添加标识
    return f"{result}"
