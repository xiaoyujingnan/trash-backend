from app import db
from datetime import datetime

class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    result_path = db.Column(db.String(500), nullable=True)
    detected_objects = db.Column(db.Text, nullable=True)

    confidence_scores = db.Column(db.Text, nullable=True)
    model_version = db.Column(db.String(32), nullable=True)
    processing_time = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'file_path': self.file_path,
            'result_path': self.result_path,
            'detected_objects': self.detected_objects,
            'confidence_scores': self.confidence_scores,
            'model_version': self.model_version,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'user_id': self.user_id,
        }
