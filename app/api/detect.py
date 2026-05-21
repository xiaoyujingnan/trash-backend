"""检测、模型与全站阈值（HTTP 层；推理与模型逻辑在 services/detection）。"""
import json
import logging
import os
from pathlib import Path

from flask import current_app, jsonify, request
from werkzeug.utils import secure_filename

from app import db
from app.api.auth import get_current_user_required, resolve_bearer_user
from app.models import DetectionResult
from app.services.detection import (
    DetectionService,
    ModelUnavailableError,
    activate_detect_model,
    apply_site_detect_thresholds,
    bootstrap_detection_model,
    clear_yolo_cache,
    get_model_version_info,
    is_usable_weights_file,
    read_active_model_rel,
    read_current_model_label,
    read_detect_settings,
    resolve_detect_model_rel,
    route_detection_sample,
    write_detect_settings,
)
from app.utils import ALLOWED_IMAGE_EXTENSIONS, detect_style_image_basename
from app.utils import (
    ImageUploadError,
    prepare_detect_inference_only,
    prepare_detect_upload,
)
from app.utils import safe_remove_file, to_upload_relative_path
from . import api_bp

logger = logging.getLogger(__name__)

MAX_MODEL_BYTES = 200 * 1024 * 1024
MODEL_UNAVAILABLE_BODY = {'error': '当前检测功能不可用', 'unavailable': True}

def _overwrite_confirmed() -> bool:
    raw = (request.form.get('overwrite') or request.args.get('overwrite') or '').strip().lower()
    return raw in ('1', 'true', 'yes', 'on')

def _model_upload_basename(filename: str) -> str:
    """保留上传文件名（仅做路径安全处理，不改为固定名）。"""
    base = os.path.basename(str(filename or '').replace('\\', '/').strip())
    if not base or base in ('.', '..') or '..' in base:
        return ''
    return secure_filename(base) or base

def _sync_active_from_config() -> str:
    root = current_app.config['UPLOAD_FOLDER']
    rel = read_active_model_rel(root)
    if not rel:
        rel = bootstrap_detection_model(current_app)
    current_app.config['ACTIVE_UPLOADED_MODEL_REL'] = rel or ''
    ver = read_current_model_label(root) if rel else ''
    current_app.config['CURRENT_MODEL_VERSION'] = ver
    return rel or ''

def _upload_root() -> str:
    return current_app.config['UPLOAD_FOLDER']

def _active_inference_model_rel() -> str:
    rel = (current_app.config.get('ACTIVE_UPLOADED_MODEL_REL') or '').strip()
    ok = resolve_detect_model_rel(current_app.config['UPLOAD_FOLDER'], rel, check_usable=True) if rel else None
    if not ok:
        ok = _sync_active_from_config()
    return ok or ''

def _current_version_label() -> str:
    return (current_app.config.get('CURRENT_MODEL_VERSION') or '').strip() or read_current_model_label(
        _upload_root()
    )

def _model_meta_for_detect() -> dict:
    info = get_model_version_info(_upload_root())
    label = str(info.get('current_model') or '').strip()
    return {
        'modelVersion': label,
        'modelPath': info.get('model_path_resolved') or info.get('model_path') or '',
        'modelUpdateTime': info.get('update_time') or '',
    }

def _require_detection_service():
    rel = _active_inference_model_rel()
    if not rel:
        raise ModelUnavailableError('当前检测功能不可用')
    settings = {'modelVersion': _current_version_label()}
    return DetectionService(
        upload_root=current_app.config['UPLOAD_FOLDER'],
        active_uploaded_rel=rel,
    ), settings

def _sanitize_detect_settings(user, settings):
    if not isinstance(settings, dict):
        return {}
    out = dict(settings)
    if not user.is_super_admin():
        out.pop('userModelRelative', None)
        out.pop('user_model_relative', None)
    return out

def _parse_client_settings() -> dict:
    try:
        raw = request.form.get('settings')
        if raw:
            data = json.loads(raw) if isinstance(raw, str) else {}
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _prepare_detect_settings(user) -> dict:
    settings = _sanitize_detect_settings(user, _parse_client_settings())
    settings = apply_site_detect_thresholds(settings, _upload_root())
    settings.update(_model_meta_for_detect())
    return settings

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def _version_status_payload(for_super_admin: bool = False) -> dict:
    info = get_model_version_info(_upload_root())
    rel = info.get('model_path_resolved') or ''
    body = {
        'success': True,
        'available': bool(rel),
        'current_model': info.get('current_model') or '',
        'update_time': info.get('update_time') or '',
    }
    if for_super_admin:
        body['model_relative_path'] = rel
    return body

@api_bp.route('/detect/model', methods=['GET', 'POST'])
def detection_model():
    """GET：当前模型状态；POST：上传并启用模型权重。"""
    if request.method == 'GET':
        try:
            user, err, status = get_current_user_required()
            if not user:
                return err, status
            _sync_active_from_config()
            return jsonify(_version_status_payload(for_super_admin=user.is_super_admin())), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    try:
        user, err, status = get_current_user_required()
        if not user:
            return err, status
        if not user.is_super_admin():
            return jsonify({'error': '该账号权限不够'}), 403

        f = request.files.get('weights')
        if not f or not f.filename:
            return jsonify({'error': '请选择 .pt 或 .pth 模型文件'}), 400

        orig_name = _model_upload_basename(f.filename)
        ext = orig_name.rsplit('.', 1)[-1].lower() if orig_name and '.' in orig_name else ''
        if ext not in ('pt', 'pth'):
            return jsonify({'error': '仅支持 .pt 或 .pth 模型文件'}), 400

        root = current_app.config['UPLOAD_FOLDER']
        model_dir = os.path.join(root, 'detect_model')
        os.makedirs(model_dir, exist_ok=True)
        dest_abs = os.path.join(model_dir, orig_name)
        if os.path.isfile(dest_abs) and not _overwrite_confirmed():
            return jsonify({
                'error': f'已存在同名模型文件「{orig_name}」，是否覆盖？',
                'file_exists': True,
                'require_confirm': True,
                'existing_name': orig_name,
            }), 409
        f.save(dest_abs)
        try:
            sz = os.path.getsize(dest_abs)
        except OSError:
            sz = 0
        if sz < 1:
            safe_remove_file(dest_abs)
            return jsonify({'error': '模型文件为空'}), 400
        if sz > MAX_MODEL_BYTES:
            safe_remove_file(dest_abs)
            return jsonify({
                'error': f'模型文件过大（上限 {MAX_MODEL_BYTES // (1024 * 1024)}MB）',
            }), 400

        model_rel = f'detect_model/{orig_name}'.replace('\\', '/')
        if not is_usable_weights_file(dest_abs):
            safe_remove_file(dest_abs)
            return jsonify({'error': '此模型文件不可用'}), 400

        try:
            applied = activate_detect_model(root, model_rel)
        except ValueError as ve:
            safe_remove_file(dest_abs)
            return jsonify({'error': str(ve)}), 400

        _sync_active_from_config()
        clear_yolo_cache()

        return jsonify({
            'success': True,
            'current_model': applied['current_model'],
            'model_relative_path': applied['model_path'],
            'uploaded_filename': orig_name,
            'update_time': applied['update_time'],
            'message': f'已上传并启用 {applied["current_model"]}',
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/admin/detect/settings', methods=['GET', 'PUT'])
def admin_detect_settings():
    """全站检测阈值：仅超级管理员可查看与修改。"""
    user, err, status = get_current_user_required()
    if not user:
        return err, status
    if not user.is_super_admin():
        return jsonify({'error': '该账号权限不够'}), 403

    root = _upload_root()
    if request.method == 'GET':
        return jsonify({'success': True, 'settings': read_detect_settings(root)}), 200

    data = request.get_json(silent=True) or {}
    saved = write_detect_settings(root, data)
    return jsonify({
        'success': True,
        'settings': saved,
        'message': '全站检测阈值已更新',
    }), 200

@api_bp.route('/detect', methods=['POST'])
def detect_image():
    try:
        user, auth_err = resolve_bearer_user()
        if auth_err:
            return auth_err[0], auth_err[1]
        if not user:
            return jsonify({'error': '请先登录后再进行检测'}), 401
        if 'image' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        if not file or not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400

        settings = _prepare_detect_settings(user)
        model_meta = _model_meta_for_detect()
        model_ver = model_meta.get('modelVersion') or ''

        upload_root = current_app.config['UPLOAD_FOLDER']
        detects_dir = os.path.join(upload_root, 'detects')
        os.makedirs(detects_dir, exist_ok=True)

        stored_name = detect_style_image_basename(
            user.id,
            file.filename,
            seq=1,
            username=getattr(user, 'username', None),
            model_version=model_ver,
        )
        abs_file_path = os.path.join(detects_dir, stored_name)
        stored_file_rel = f'detects/{stored_name}'.replace('\\', '/')

        try:
            detection_service, infer_settings = _require_detection_service()
        except ModelUnavailableError:
            return jsonify(MODEL_UNAVAILABLE_BODY), 503

        try:
            letterbox_meta, letterbox_bgr = prepare_detect_upload(
                file,
                abs_file_path,
                max_bytes=current_app.config.get('MAX_DETECT_IMAGE_BYTES', 20 * 1024 * 1024),
                box_size=int(current_app.config.get('DETECT_IMAGE_BOX_SIZE', 640)),
            )
        except ImageUploadError as e:
            return jsonify({'error': str(e)}), 400

        try:
            result = detection_service.detect_image(
                abs_file_path,
                settings={**settings, **infer_settings},
                letterbox_bgr=letterbox_bgr,
                letterbox_meta=letterbox_meta,
                result_stem=Path(stored_name).stem,
            )
        except ModelUnavailableError:
            safe_remove_file(abs_file_path)
            return jsonify(MODEL_UNAVAILABLE_BODY), 503

        model_display = result.pop('model_display_name', None)
        result_ver = result.pop('model_version', None) or model_ver
        settings_out = dict(settings)
        if model_display:
            settings_out['modelDisplayName'] = model_display
        if result_ver:
            settings_out['modelVersion'] = result_ver

        logger.info(
            'detect saved user_id=%s filename=%s model_version=%s objects=%s',
            user.id,
            stored_name,
            result_ver or '-',
            result.get('total_objects', 0),
        )

        result_rel = to_upload_relative_path(result.get('result_path'), upload_root)

        detection_result = DetectionResult(
            filename=stored_name,
            file_path=stored_file_rel,
            result_path=result_rel,
            detected_objects=json.dumps(result.get('detected_objects', [])),
            confidence_scores=json.dumps({
                'scores': result.get('confidence_scores', []),
                'settings': settings_out,
            }),
            processing_time=result.get('processing_time'),
            model_version=result_ver or None,
            user_id=user.id,
        )
        db.session.add(detection_result)
        db.session.commit()

        routing_info = {}
        try:
            routing_info = route_detection_sample(current_app, detection_result, result)
        except Exception as route_err:
            db.session.rollback()
            logger.warning('route_detection_sample failed detection_id=%s: %s', detection_result.id, route_err)
            routing_info = {'mode': 'routing_error', 'reason': str(route_err)}

        result['settings'] = settings_out
        return jsonify({
            'success': True,
            'result': result,
            'detection_id': detection_result.id,
            'routing': routing_info,
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_bp.route('/detect/preview', methods=['POST'])
def detect_preview():
    try:
        user, auth_err = resolve_bearer_user()
        if auth_err:
            return auth_err[0], auth_err[1]
        if not user:
            return jsonify({'error': '请先登录后再进行检测'}), 401
        if 'image' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400

        file = request.files['image']
        if not file or file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400

        settings = _prepare_detect_settings(user)

        try:
            detection_service, infer_settings = _require_detection_service()
        except ModelUnavailableError:
            return jsonify(MODEL_UNAVAILABLE_BODY), 503

        try:
            letterbox_meta, letterbox_bgr = prepare_detect_inference_only(
                file,
                max_bytes=current_app.config.get('MAX_DETECT_IMAGE_BYTES', 20 * 1024 * 1024),
                box_size=int(current_app.config.get('DETECT_IMAGE_BOX_SIZE', 640)),
            )
        except ImageUploadError as e:
            return jsonify({'error': str(e)}), 400

        try:
            result = detection_service.detect_image(
                None,
                settings={**settings, **infer_settings},
                letterbox_bgr=letterbox_bgr,
                letterbox_meta=letterbox_meta,
                save_result=False,
            )
        except ModelUnavailableError:
            return jsonify(MODEL_UNAVAILABLE_BODY), 503

        model_display = result.pop('model_display_name', None)
        settings_out = dict(settings)
        if model_display:
            settings_out['modelDisplayName'] = model_display
        result['settings'] = settings_out
        return jsonify({'success': True, 'result': result}), 200
    except Exception as e:
        logger.exception('detect_preview failed')
        return jsonify({'error': str(e)}), 500
