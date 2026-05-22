from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import importlib
import logging
from pathlib import Path

db = SQLAlchemy()
logger = logging.getLogger(__name__)

def _pg_quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'

def _ensure_pending_sample_unique_constraint(app):
    from sqlalchemy import inspect, text

    with app.app_context():
        try:
            if db.engine.dialect.name != 'postgresql':
                return
            insp = inspect(db.engine)
            if 'pending_sample' not in insp.get_table_names():
                return

            indexes = insp.get_indexes('pending_sample')
            uniques = insp.get_unique_constraints('pending_sample')

            single_detection_unique = []
            has_composite = False

            for idx in indexes:
                if not idx.get('unique'):
                    continue
                cols = list(idx.get('column_names') or [])
                if cols == ['detection_id']:
                    single_detection_unique.append(idx.get('name'))
                elif cols == ['detection_id', 'box_index']:
                    has_composite = True

            for uq in uniques:
                cols = list(uq.get('column_names') or [])
                if cols == ['detection_id', 'box_index']:
                    has_composite = True
                elif cols == ['detection_id']:
                    name = uq.get('name')
                    if name:
                        single_detection_unique.append(name)

            with db.engine.connect() as conn:
                has_non_unique_detection = any(
                    'detection_id' in (ix.get('column_names') or [])
                    and not ix.get('unique')
                    for ix in indexes
                )
                if single_detection_unique and not has_non_unique_detection:
                    conn.execute(
                        text(
                            'CREATE INDEX IF NOT EXISTS ix_pending_sample_detection_id '
                            'ON pending_sample (detection_id)'
                        )
                    )
                for name in dict.fromkeys(single_detection_unique):
                    if not name:
                        continue
                    conn.execute(text(f'DROP INDEX IF EXISTS {_pg_quote_ident(name)}'))
                if not has_composite:
                    conn.execute(
                        text(
                            'CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_detection_box '
                            'ON pending_sample (detection_id, box_index)'
                        )
                    )
                conn.commit()
        except Exception:
            logger.warning(
                'ensure pending_sample unique (detection_id, box_index) failed',
                exc_info=True,
            )

def _ensure_detection_model_version_column(app):
    from sqlalchemy import inspect, text

    with app.app_context():
        try:
            cols = {c['name'] for c in inspect(db.engine).get_columns('detection_result')}
            if 'model_version' in cols:
                return
            with db.engine.connect() as conn:
                if db.engine.dialect.name == 'postgresql':
                    conn.execute(
                        text(
                            'ALTER TABLE detection_result '
                            'ADD COLUMN IF NOT EXISTS model_version VARCHAR(32) NULL'
                        )
                    )
                else:
                    conn.execute(
                        text(
                            'ALTER TABLE detection_result '
                            'ADD COLUMN model_version VARCHAR(32) NULL'
                        )
                    )
                conn.commit()
        except Exception:
            logger.warning(
                'ensure detection_result.model_version column failed',
                exc_info=True,
            )

def _ensure_default_admin(app):
    from werkzeug.security import generate_password_hash
    from sqlalchemy.exc import IntegrityError, OperationalError

    with app.app_context():
        try:
            from app.models import User
        except Exception:
            return
        from app.models.user import SUPER_ADMIN_USERNAME

        if User.query.filter_by(username=SUPER_ADMIN_USERNAME).first():
            return
        row = User(
            username=SUPER_ADMIN_USERNAME,
            password_hash=generate_password_hash('admin123456'),
            email=None,
            email_password_hash=None,
            is_admin=True,
            is_disabled=False,
        )
        db.session.add(row)
        try:
            db.session.commit()
        except (IntegrityError, OperationalError):
            db.session.rollback()

def create_app(config='config.Config'):
    app = Flask(__name__)
    app.config.from_object(config)
    backend_root = Path(app.config['BACKEND_ROOT'])

    def _resolve_runtime_path(base: Path, value) -> str:
        p = Path(str(value))
        if p.is_absolute():
            return str(p.resolve())
        return str((base / p).resolve())

    app.config['DATASET_ROOT'] = _resolve_runtime_path(backend_root, app.config['DATASET_ROOT'])
    app.config['TRAINING_ROOT'] = _resolve_runtime_path(backend_root, app.config['TRAINING_ROOT'])
    app.config['UPLOAD_FOLDER'] = os.path.abspath(
        os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    )
    app.config['ACTIVE_UPLOADED_MODEL_REL'] = ''
    app.config['CURRENT_MODEL_VERSION'] = ''

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'detect_model'), exist_ok=True)

    from app.services.detection import bootstrap_detection_model

    bootstrap_detection_model(app)

    db.init_app(app)
    importlib.import_module('app.models')
    with app.app_context():
        db.create_all()
    _ensure_default_admin(app)
    _ensure_detection_model_version_column(app)
    _ensure_pending_sample_unique_constraint(app)

    from app.services.training import reclaim_stale_training_jobs

    try:
        reclaim_stale_training_jobs(app)
    except Exception:
        logger.warning('reclaim_stale_training_jobs failed', exc_info=True)

    CORS(app)

    from app.api import api_bp

    app.register_blueprint(api_bp, url_prefix='/api')

    @app.route('/api/health')
    def health():
        return {'status': 'ok'}, 200

    @app.route('/uploads/<path:filename>')
    def uploaded_files(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    return app
