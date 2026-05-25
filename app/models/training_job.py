from app import db
from datetime import datetime


class TrainingJob(db.Model):
    """管理员触发的训练任务（异步记录状态与日志路径）。"""

    __tablename__ = 'training_job'

    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(24), nullable=False, default='queued')  # queued|running|succeeded|failed|terminated
    epochs = db.Column(db.Integer, nullable=False, default=30)
    # 以下为空则执行线程内回退到 config 默认值
    batch_size = db.Column(db.Integer, nullable=True)
    imgsz = db.Column(db.Integer, nullable=True)
    device = db.Column(db.String(32), nullable=True)
    patience = db.Column(db.Integer, nullable=True)
    # 逻辑路径：training/bases/<文件名>（物理目录 backend/training/bases/）
    base_weights_rel = db.Column(db.String(512), nullable=True)

    log_path = db.Column(db.String(500), nullable=True)
    message = db.Column(db.Text, nullable=True)
    output_weights_path = db.Column(db.String(500), nullable=True)
    created_by_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    finished_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'status': self.status,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'imgsz': self.imgsz,
            'device': self.device,
            'patience': self.patience,
            'base_weights_rel': self.base_weights_rel,
            'log_path': self.log_path,
            'message': self.message,
            'output_weights_path': self.output_weights_path,
            'created_by_user_id': self.created_by_user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'finished_at': self.finished_at.isoformat() if self.finished_at else None,
        }
