import os
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent

def _database_uri() -> str:
    url = (os.getenv('DATABASE_URL') or os.getenv('SQLALCHEMY_DATABASE_URI') or '').strip()
    if url.startswith('postgres://'):
        url = 'postgresql+psycopg2://' + url[len('postgres://') :]
    elif url.startswith('postgresql://') and '+psycopg2' not in url:
        url = 'postgresql+psycopg2://' + url[len('postgresql://') :]
    if url:
        return url
    if os.getenv('RENDER') or os.getenv('RENDER_SERVICE_ID'):
        raise RuntimeError(
            '未设置 DATABASE_URL。请在 Render：PostgreSQL → Connect → 复制 Internal Database URL，'
            '粘贴到 Web 服务 Environment；或使用 Link Database 自动注入。'
        )
    return 'postgresql+psycopg2://localhost:5432/trashnet'

class Config:
    SQLALCHEMY_DATABASE_URI = _database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-me-in-render-dashboard')

    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', str(256 * 1024 * 1024)))
    MAX_DETECT_IMAGE_BYTES = int(os.getenv('MAX_DETECT_IMAGE_BYTES', str(20 * 1024 * 1024)))
    DETECT_IMAGE_BOX_SIZE = int(os.getenv('DETECT_IMAGE_BOX_SIZE', '640'))

    BACKEND_ROOT = _BACKEND_ROOT
    REPO_ROOT = BACKEND_ROOT.parent
    PROJECT_ROOT = BACKEND_ROOT

    TRAINING_ROOT = os.getenv('TRAINING_ROOT', 'training')

    ENABLE_DATASET_INGEST = os.getenv('ENABLE_DATASET_INGEST', 'false').lower() in (
        '1', 'true', 'yes', 'on',
    )

    DATASET_ROOT = os.getenv('DATASET_ROOT', 'datasets')
    AUTO_INGEST_CONF_THRESHOLD = float(os.getenv('AUTO_INGEST_CONF_THRESHOLD', '0.85'))

    TRAIN_DATA_YAML = os.getenv('TRAIN_DATA_YAML', 'trash4.yaml')
    TRAIN_SCRIPT = os.getenv('TRAIN_SCRIPT', 'train.py')
    TRAIN_LOGS_DIR = os.getenv('TRAIN_LOGS_DIR', 'logs')
    TRAIN_PROJECT_DIR = os.getenv('TRAIN_PROJECT_DIR', 'runs')
