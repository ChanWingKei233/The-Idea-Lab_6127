# app/__init__.py
from flask import Flask
from flask_cors import CORS
from config import Config

def create_app(config_class=Config):
    # 初始化 Flask 应用
    app = Flask(__name__)
    app.config.from_object(config_class)  # 加载配置
    
    # 解决跨域
    CORS(app)
    
    # 注册路由蓝图（Blueprint）
    from app.routes import text_bp  # 导入路由蓝图
    app.register_blueprint(text_bp)  # 注册文本处理路由
    
    return app