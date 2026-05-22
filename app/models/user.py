from sqlalchemy import func
from app import db

SUPER_ADMIN_USERNAME = 'admin'

class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(120), unique=True)
    email_password_hash = db.Column(db.String(255))
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    is_disabled = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_login_at = db.Column(db.DateTime(timezone=True), nullable=True)

    detection_results = db.relationship('DetectionResult', backref='user', lazy=True)
    profile = db.relationship('UserProfile', backref='user', uselist=False, lazy=True)

    def is_super_admin(self):
        """超级管理员（用户名 admin）：不可由普通管理员降级/删除/禁用等。"""
        return (self.username or '') == SUPER_ADMIN_USERNAME

    @classmethod
    def get_by_username_exact(cls, username: str, **filters):
        """按用户名精确匹配（区分大小写；MySQL 需 binary，PostgreSQL 默认区分大小写）。"""
        if username is None or str(username) == '':
            return None
        name = str(username)
        bind = db.session.get_bind()
        dialect = bind.dialect.name if bind is not None else 'postgresql'
        if dialect == 'mysql':
            q = cls.query.filter(func.binary(cls.username) == name)
        else:
            q = cls.query.filter(cls.username == name)
        for key, value in filters.items():
            q = q.filter(getattr(cls, key) == value)
        return q.first()

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_admin': self.is_admin,
            'is_super_admin': self.is_super_admin(),
            'is_disabled': bool(self.is_disabled),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None,
        }
