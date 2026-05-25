from app import db
from datetime import datetime
from sqlalchemy import UniqueConstraint


class PendingSample(db.Model):
    """低置信度检测框：每条对应一次检测中的一框；待管理员审核后写入数据集或丢弃。"""

    __tablename__ = 'pending_sample'
    __table_args__ = (
        UniqueConstraint('detection_id', 'box_index', name='uq_pending_detection_box'),
    )

    id = db.Column(db.Integer, primary_key=True)
    detection_id = db.Column(db.Integer, db.ForeignKey('detection_result.id'), nullable=False)
    box_index = db.Column(db.Integer, nullable=False, default=0)
    predicted_class = db.Column(db.String(64), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    bbox_json = db.Column(db.Text, nullable=True)
    status = db.Column(
        db.String(24),
        nullable=False,
        default='pending',
    )
    corrected_class = db.Column(db.String(64), nullable=True)
    resolver_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    resolved_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    detection = db.relationship(
        'DetectionResult',
        backref=db.backref('pending_samples', lazy='dynamic'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'detection_id': self.detection_id,
            'box_index': self.box_index,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'bbox_json': self.bbox_json,
            'status': self.status,
            'corrected_class': self.corrected_class,
            'resolver_user_id': self.resolver_user_id,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


def delete_pending_samples_for_detection_ids(detection_ids):
    """在删除 detection_result 行之前调用；避免 ORM 将 detection_id 置空违反非空约束。"""
    if not detection_ids:
        return
    ids = [int(x) for x in detection_ids if x is not None]
    if not ids:
        return
    PendingSample.query.filter(PendingSample.detection_id.in_(ids)).delete(
        synchronize_session=False
    )
