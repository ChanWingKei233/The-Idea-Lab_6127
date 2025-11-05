# app/routes/__init__.py
from app.routes.text_routes import text_bp

# 导出蓝图，方便在 app/__init__.py 中注册
__all__ = ['text_bp']