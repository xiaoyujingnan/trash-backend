from sqlalchemy import func
from app import db

class UserProfile(db.Model):
    __tablename__ = 'user_profile'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    nickname = db.Column(db.String(80))
    phone = db.Column(db.String(30))
    signature = db.Column(db.String(255))
    avatar = db.Column(db.Text)
    updated_at = db.Column(db.DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def to_dict(self):
        return {
            'nickname': self.nickname,
            'phone': self.phone,
            'signature': self.signature,
            'avatar': self.avatar,
        }
