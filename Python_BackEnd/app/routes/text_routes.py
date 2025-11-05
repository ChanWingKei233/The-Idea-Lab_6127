# app/routes/text_routes.py
from flask import Blueprint, request, jsonify
# from app.services.text_service import process_text  # 导入业务逻辑
from app.services.text_processors.predict import test_predict

# 创建蓝图（相当于路由分组，名称为 'text'）
text_bp = Blueprint('text', __name__)  # 统一前缀 /api

# 原有的文本处理接口
@text_bp.route('/process', methods=['POST'])
def handle_text():
    data = request.get_json()
    input_content = data.get('content', '')
    
    # 调用业务逻辑处理（核心逻辑在 services 中）
    result = test_predict(input_content)
    
    return jsonify({'processed_content': result})