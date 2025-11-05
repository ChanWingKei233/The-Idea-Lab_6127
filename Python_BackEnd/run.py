# run.py
from app import create_app

# 创建应用实例
app = create_app()

if __name__ == '__main__':
    # 从配置中读取端口
    app.run(host='0.0.0.0', port=app.config['PORT'])