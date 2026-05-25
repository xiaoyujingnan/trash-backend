import os

# Render 免费实例 512MB：减少线程争用与 matplotlib 写盘失败
os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/Ultralytics')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TORCH_NUM_THREADS', '1')
os.makedirs(os.environ['YOLO_CONFIG_DIR'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

from app import create_app

app = create_app()
