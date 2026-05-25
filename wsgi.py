import os

# Render 等平台默认用户目录不可写；须在 import ultralytics 之前设置
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/Ultralytics')
os.makedirs(os.environ['YOLO_CONFIG_DIR'], exist_ok=True)

from app import create_app

app = create_app()
