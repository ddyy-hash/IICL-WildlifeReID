import os
from datetime import timedelta
from pathlib import Path


# 使用 pathlib 确保路径解析为绝对路径
BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dog-reid-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///dog_reid.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 上传配置
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or str(BASE_DIR / 'uploads')
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'dav'}
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_UPLOAD_SIZE', 100)) * 1024 * 1024

    # 会话配置
    PERMANENT_SESSION_LIFETIME = timedelta(hours=int(os.environ.get('SESSION_HOURS', 2)))

    # 实时检测配置
    RTSP_TIMEOUT = int(os.environ.get('RTSP_TIMEOUT', 10))
    MAX_RETRY_ATTEMPTS = int(os.environ.get('MAX_RETRY_ATTEMPTS', 5))
    
    # 特征数据库路径配置（使用绝对路径）
    FEATURES_DB_PATH = str(BASE_DIR / 'fea_data' / 'universal_features_h.npy')
    
    # 模型路径配置
    MODEL_DIR = str(BASE_DIR / 'fea_data')
    ILLUMINATION_MODEL_PATH = str(BASE_DIR / 'fea_data' / 'illumination_robust_model.pth')
    YOLO_MODEL_PATH = str(BASE_DIR / 'fea_data' / 'yolov8m-seg.pt')
    EFFICIENT_SAM_PATH = str(BASE_DIR / 'fea_data' / 'efficient_sam_vits.pt')


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_ECHO = True


class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_ECHO = False

