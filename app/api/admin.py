"""管理员：用户、样本审核、训练任务。"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

import cv2
from flask import current_app, jsonify, request
from sqlalchemy import func, or_
from sqlalchemy.orm import joinedload
from werkzeug.utils import secure_filename

from app import db
from app.api.auth import get_current_admin
from app.api.profile import DEFAULT_USER_AVATAR_URL, normalize_avatar_path
from app.models import DetectionResult, PendingSample, TrainingJob, User, UserProfile
from app.models.pending_sample import delete_pending_samples_for_detection_ids
from app.services.dataset import (
    set_detection_dataset_paths,
    try_finalize_detection_dataset,
    update_per_box_entry,
)
from app.services.training import (
    build_training_live_payload,
    collect_training_device_info,
    enrich_training_job_dict,
    get_training_root,
    load_training_metrics_best_row,
    resolve_train_project_dir,
    resolve_training_weights_file,
    start_training_job_async,
    stop_training_job,
)
from app.utils import resolve_stored_file_path, safe_remove_file
from . import api_bp

MAX_TRAINING_WEIGHT_BYTES = 200 * 1024 * 1024


def _purge_user_uploaded_files(user_id):
    """
    删除用户关联的检测原图、结果图及自定义头像（不删全站共享的 default_user_avatar.png）。
    在删除 User / DetectionResult / UserProfile 的数据库行之前调用。
    """
    upload_root = current_app.config['UPLOAD_FOLDER']

    detections = DetectionResult.query.filter_by(user_id=user_id).all()
    for d in detections:
        for stored in (d.file_path, d.result_path):
            safe_remove_file(resolve_stored_file_path(stored, upload_root))

    profile = UserProfile.query.filter_by(user_id=user_id).first()
    if not profile or not profile.avatar:
        return
    av = normalize_avatar_path(profile.avatar)
    if not av or av == DEFAULT_USER_AVATAR_URL:
        return
    path = resolve_stored_file_path(av, upload_root)
    if not path:
        return
    if os.path.basename(path).lower() == 'default_user_avatar.png':
        return
    safe_remove_file(path)


def _deny_non_super(operator):
    """删除、改角色等仅超级管理员（admin）可用。"""
    if not operator.is_super_admin():
        return jsonify({'success': False, 'message': '该账号权限不够'}), 403
    return None


def _deny_status_change(operator, target, is_disabled):
    """
    超级管理员：可对除自己外的任意账号禁用/启用。
    普通管理员：仅可查看；封禁仅针对普通用户（非 admin、非其他管理员、非自己）。
    """
    if operator.is_super_admin():
        if target.id == operator.id and is_disabled:
            return jsonify({'success': False, 'message': '不能禁用自己的账号'}), 400
        return None
    if target.is_super_admin():
        return jsonify({'success': False, 'message': '不可操作超级管理员账号'}), 403
    if target.id == operator.id:
        return jsonify({'success': False, 'message': '不能操作自己的账号'}), 403
    if target.is_admin:
        return jsonify({'success': False, 'message': '普通管理员仅可禁用或启用普通用户'}), 403
    return None


def _user_admin_row(u):
    row = u.to_dict()
    p = u.profile
    row['avatar'] = normalize_avatar_path(p.avatar) if p else ''
    return row


def _users_base_query(keyword: str):
    """关键词筛选后的用户查询（未排序、未分组）。"""
    query = User.query
    if keyword:
        query = query.filter(
            or_(
                func.lower(User.username).like(f'%{keyword}%'),
                func.lower(User.email).like(f'%{keyword}%'),
            )
        )
    return query


def _users_stats(query):
    return {
        'total': query.count(),
        'admin': query.filter(User.is_admin.is_(True)).count(),
        'user': query.filter(User.is_admin.is_(False)).count(),
        'disabled': query.filter(User.is_disabled.is_(True)).count(),
    }


_USER_GROUP_FILTERS = {
    'all': None,
    'admin': User.is_admin.is_(True),
    'user': User.is_admin.is_(False),
    'disabled': User.is_disabled.is_(True),
}


@api_bp.route('/admin/users', methods=['GET'])
def list_users():
    admin, error_resp, status = get_current_admin()
    if not admin:
        return error_resp, status

    keyword = str(request.args.get('keyword') or '').strip().lower()
    group = str(request.args.get('group') or 'all').strip().lower()
    if group not in _USER_GROUP_FILTERS:
        group = 'all'
    sort_by = str(request.args.get('sort_by') or 'id').strip().lower()
    sort_order = str(request.args.get('sort_order') or 'desc').strip().lower()

    base_query = _users_base_query(keyword)
    stats = _users_stats(base_query)

    list_query = base_query
    group_filter = _USER_GROUP_FILTERS[group]
    if group_filter is not None:
        list_query = list_query.filter(group_filter)

    list_query = list_query.options(joinedload(User.profile))
    if sort_by == 'created_at':
        sort_col = User.created_at
    elif sort_by == 'last_login_at':
        sort_col = User.last_login_at
    else:
        sort_col = User.id
    if sort_order == 'asc':
        list_query = list_query.order_by(sort_col.asc())
    else:
        list_query = list_query.order_by(sort_col.desc())

    users = list_query.all()
    return jsonify({
        'success': True,
        'users': [_user_admin_row(u) for u in users],
        'stats': stats,
        'group': group,
    }), 200


@api_bp.route('/admin/users/<int:user_id>/role', methods=['PUT'])
def update_user_role(user_id):
    admin, error_resp, status = get_current_admin()
    if not admin:
        return error_resp, status

    deny = _deny_non_super(admin)
    if deny:
        return deny

    target = User.query.get_or_404(user_id)
    data = request.get_json() or {}
    is_admin = bool(data.get('is_admin'))

    if target.id == admin.id and not is_admin:
        return jsonify({'success': False, 'message': '不能取消自己的管理员权限'}), 400

    if target.is_super_admin() and not is_admin:
        return jsonify({'success': False, 'message': '超级管理员账号不可取消管理员权限'}), 400

    target.is_admin = is_admin
    db.session.commit()
    return jsonify({'success': True, 'user': target.to_dict(), 'message': '角色更新成功'}), 200


@api_bp.route('/admin/users/<int:user_id>/status', methods=['PUT'])
def update_user_status(user_id):
    admin, error_resp, status = get_current_admin()
    if not admin:
        return error_resp, status

    target = User.query.get_or_404(user_id)
    data = request.get_json() or {}
    is_disabled = bool(data.get('is_disabled'))

    deny = _deny_status_change(admin, target, is_disabled)
    if deny:
        return deny

    target.is_disabled = is_disabled
    db.session.commit()
    return jsonify({'success': True, 'user': target.to_dict(), 'message': '状态更新成功'}), 200


@api_bp.route('/admin/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    admin, error_resp, status = get_current_admin()
    if not admin:
        return error_resp, status

    deny = _deny_non_super(admin)
    if deny:
        return deny

    target = User.query.get_or_404(user_id)

    if target.id == admin.id:
        return jsonify({'success': False, 'message': '不能删除自己的账号'}), 400

    _purge_user_uploaded_files(target.id)
    dids = [
        row[0]
        for row in db.session.query(DetectionResult.id)
        .filter_by(user_id=target.id)
        .all()
    ]
    delete_pending_samples_for_detection_ids(dids)
    DetectionResult.query.filter_by(user_id=target.id).delete()
    UserProfile.query.filter_by(user_id=target.id).delete()
    db.session.delete(target)
    db.session.commit()

    return jsonify({'success': True, 'message': '用户删除成功'}), 200


def _profile_dict_safe(profile, user=None):
    if not profile:
        return {}
    d = profile.to_dict()
    d['avatar'] = normalize_avatar_path(d.get('avatar'))
    if user:
        d['nickname'] = d.get('nickname') or user.username
    return d


@api_bp.route('/admin/users/<int:user_id>/detail', methods=['GET'])
def get_user_detail(user_id):
    admin, error_resp, status = get_current_admin()
    if not admin:
        return error_resp, status

    user = User.query.get_or_404(user_id)
    profile = UserProfile.query.filter_by(user_id=user.id).first()
    recent = (
        DetectionResult.query.filter_by(user_id=user.id)
        .order_by(DetectionResult.id.desc())
        .limit(8)
        .all()
    )

    return jsonify({
        'success': True,
        'user': user.to_dict(),
        'profile': _profile_dict_safe(profile, user),
        'recent_detections': [d.to_dict() for d in recent],
    }), 200


def _parse_batch_ids(raw):
    if not isinstance(raw, list):
        return []
    out = []
    for x in raw:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


@api_bp.route('/admin/users/batch', methods=['POST'])
def batch_user_actions():
    admin, error_resp, status = get_current_admin()
    if not admin:
        return error_resp, status

    data = request.get_json() or {}
    action = str(data.get('action') or '').strip().lower()
    user_ids = _parse_batch_ids(data.get('user_ids') or [])
    if not user_ids:
        return jsonify({'success': False, 'message': '请选择至少一个用户'}), 400

    results = []

    if action == 'delete':
        deny = _deny_non_super(admin)
        if deny:
            return deny
        for uid in user_ids:
            target = User.query.get(uid)
            if not target:
                results.append({'id': uid, 'ok': False, 'message': '用户不存在'})
                continue
            if target.id == admin.id:
                results.append({'id': uid, 'ok': False, 'message': '不能删除自己的账号'})
                continue
            _purge_user_uploaded_files(target.id)
            dids = [
                row[0]
                for row in db.session.query(DetectionResult.id)
                .filter_by(user_id=target.id)
                .all()
            ]
            delete_pending_samples_for_detection_ids(dids)
            DetectionResult.query.filter_by(user_id=target.id).delete()
            UserProfile.query.filter_by(user_id=target.id).delete()
            db.session.delete(target)
            db.session.commit()
            results.append({'id': uid, 'ok': True, 'message': '已删除'})
        return jsonify({'success': True, 'results': results}), 200

    if action == 'set_status':
        is_disabled = bool(data.get('is_disabled'))
        for uid in user_ids:
            target = User.query.get(uid)
            if not target:
                results.append({'id': uid, 'ok': False, 'message': '用户不存在'})
                continue
            deny = _deny_status_change(admin, target, is_disabled)
            if deny:
                body = deny.get_json(silent=True) or {}
                results.append({'id': uid, 'ok': False, 'message': body.get('message', '无权限')})
                continue
            target.is_disabled = is_disabled
            db.session.commit()
            results.append({'id': uid, 'ok': True, 'message': '状态已更新'})
        return jsonify({'success': True, 'results': results}), 200

    if action == 'set_role':
        deny = _deny_non_super(admin)
        if deny:
            return deny
        is_admin_flag = bool(data.get('is_admin'))
        for uid in user_ids:
            target = User.query.get(uid)
            if not target:
                results.append({'id': uid, 'ok': False, 'message': '用户不存在'})
                continue
            if target.id == admin.id and not is_admin_flag:
                results.append({'id': uid, 'ok': False, 'message': '不能取消自己的管理员权限'})
                continue
            if target.is_super_admin() and not is_admin_flag:
                results.append({'id': uid, 'ok': False, 'message': '超级管理员账号不可降为普通用户'})
                continue
            target.is_admin = is_admin_flag
            db.session.commit()
            results.append({'id': uid, 'ok': True, 'message': '角色已更新'})
        return jsonify({'success': True, 'results': results}), 200

    return jsonify({'success': False, 'message': '不支持的操作类型'}), 400


# --- 待审核样本 ---


def _parse_bbox_xyxy_from_request(data):
    raw = data.get('bbox_xyxy') or data.get('bbox')
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, raw[:4])
    except (TypeError, ValueError):
        return None
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


def _read_image_wh(image_path, upload_folder):
    p = resolve_stored_file_path(image_path, upload_folder)
    if not p or not os.path.isfile(p):
        return None
    im = cv2.imread(p)
    if im is None:
        return None
    h, w = im.shape[:2]
    return w, h


def _clip_xyxy_to_image(bbox, w, h):
    w = max(float(w), 1.0)
    h = max(float(h), 1.0)
    x1, y1, x2, y2 = map(float, bbox)
    x1 = max(0.0, min(x1, w - 1.0))
    x2 = max(0.0, min(x2, w))
    y1 = max(0.0, min(y1, h - 1.0))
    y2 = max(0.0, min(y2, h))
    if x2 <= x1 + 0.5 or y2 <= y1 + 0.5:
        return None
    return [x1, y1, x2, y2]


def _finalize_after_review(det, username=None):
    fin = try_finalize_detection_dataset(
        current_app.config['DATASET_ROOT'],
        det,
        det.file_path,
        username=username,
        upload_folder=current_app.config['UPLOAD_FOLDER'],
    )
    if fin.get('finalized'):
        set_detection_dataset_paths(
            det,
            fin.get('dataset_image'),
            fin.get('dataset_label'),
        )
    return fin


@api_bp.route('/admin/pending-samples', methods=['GET'])
def list_pending_samples():
    admin, err, status = get_current_admin()
    if not admin:
        return err, status

    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    st = request.args.get('status', 'pending')

    q = PendingSample.query
    if st and st != 'all':
        q = q.filter(PendingSample.status == st)

    paginated = q.order_by(PendingSample.id.desc()).paginate(page=page, per_page=per_page, error_out=False)
    rows = []
    for ps in paginated.items:
        row = ps.to_dict()
        det = DetectionResult.query.get(ps.detection_id)
        row['filename'] = det.filename if det else ''
        row['created_at_detection'] = det.created_at.isoformat() if det and det.created_at else None
        rows.append(row)

    return jsonify({
        'items': rows,
        'total': paginated.total,
        'pages': paginated.pages,
        'current_page': page,
    }), 200


def _apply_pending_review(ps: PendingSample, admin, data: dict):
    action = (data.get('action') or '').strip().lower()
    if action not in ('accept', 'correct', 'discard'):
        return {'success': False, 'message': 'action 须为 accept | correct | discard'}, 400

    det = DetectionResult.query.get(ps.detection_id)
    if not det:
        return {'success': False, 'message': '关联检测记录不存在'}, 404

    u_submit = User.query.get(det.user_id) if det.user_id else None
    uname = getattr(u_submit, 'username', None) if u_submit else None
    box_idx = int(ps.box_index or 0)

    if action == 'discard':
        ps.status = 'discarded'
        ps.resolver_user_id = admin.id
        ps.resolved_at = datetime.utcnow()
        update_per_box_entry(det, box_idx, mode='discarded', pending_sample_id=ps.id)
        db.session.commit()
        fin = _finalize_after_review(det, username=uname)
        db.session.commit()
        return {
            'success': True,
            'message': fin.get('message') or '已丢弃',
            'dataset_finalized': bool(fin.get('finalized')),
            'pending_remaining_on_detection': fin.get('pending_remaining', 0),
            'dataset_image': fin.get('dataset_image'),
            'dataset_label': fin.get('dataset_label'),
            'box_count': fin.get('box_count', 0),
        }, 200

    corrected = (data.get('corrected_class') or '').strip()
    if action == 'correct':
        if not corrected:
            return {'success': False, 'message': 'correct 时请提供 corrected_class'}, 400
        label = corrected
        ps.corrected_class = corrected
    else:
        label = ps.predicted_class
        ps.corrected_class = None

    bbox = _parse_bbox_xyxy_from_request(data)
    if bbox is None and ps.bbox_json:
        try:
            bbox = json.loads(ps.bbox_json)
        except Exception:
            bbox = None

    if not bbox or len(bbox) < 4:
        return {'success': False, 'message': '缺少有效 bbox，无法记录标注'}, 400

    upload_root = current_app.config['UPLOAD_FOLDER']
    wh = _read_image_wh(det.file_path, upload_root)
    if wh:
        clipped = _clip_xyxy_to_image(bbox, wh[0], wh[1])
        if clipped is None:
            return {'success': False, 'message': 'bbox 与图像尺寸不匹配或过小'}, 400
        bbox = clipped

    ps.bbox_json = json.dumps(bbox, ensure_ascii=False)
    ps.status = 'ingested'
    ps.resolver_user_id = admin.id
    ps.resolved_at = datetime.utcnow()
    update_per_box_entry(
        det,
        box_idx,
        mode='review_ingested',
        pending_sample_id=ps.id,
        approved_class=label,
        approved_bbox=bbox,
    )
    db.session.commit()

    fin = _finalize_after_review(det, username=uname)
    db.session.commit()

    msg = fin.get('message') or '已记录'
    if not fin.get('finalized') and fin.get('pending_remaining', 0) > 0:
        msg = f'本框已记录；该图尚有 {fin["pending_remaining"]} 个框待审核，完成后将统一写入标注文件'

    return {
        'success': True,
        'message': msg,
        'dataset_finalized': bool(fin.get('finalized')),
        'pending_remaining_on_detection': fin.get('pending_remaining', 0),
        'dataset_image': fin.get('dataset_image'),
        'dataset_label': fin.get('dataset_label'),
        'box_count': fin.get('box_count', 0),
    }, 200


@api_bp.route('/admin/pending-samples/<int:sample_id>/review', methods=['POST'])
def review_pending_sample(sample_id):
    admin, err, status = get_current_admin()
    if not admin:
        return err, status

    ps = PendingSample.query.get(sample_id)
    if not ps:
        return jsonify({'success': False, 'message': '记录不存在'}), 404
    if ps.status != 'pending':
        return jsonify({'success': False, 'message': '该样本已处理'}), 400

    data = request.get_json(silent=True) or {}
    payload, code = _apply_pending_review(ps, admin, data)
    return jsonify(payload), code


# --- 训练任务 ---


def _list_base_weight_files(training_root: Path) -> list[Path]:
    bases = training_root / 'bases'
    if not bases.is_dir():
        return []
    return sorted(
        (
            p
            for p in bases.iterdir()
            if p.is_file() and p.suffix.lower() in ('.pt', '.pth')
        ),
        key=lambda p: p.name.lower(),
    )


def _remove_other_base_weights(training_root: Path, keep: Path) -> None:
    keep_resolved = keep.resolve()
    for p in _list_base_weight_files(training_root):
        try:
            if p.resolve() == keep_resolved:
                continue
        except OSError:
            continue
        try:
            p.unlink()
        except OSError:
            pass


def _base_weight_item(weight_path: Path) -> dict:
    rel = f'training/bases/{weight_path.name}'.replace('\\', '/')
    try:
        sz = weight_path.stat().st_size
    except OSError:
        sz = 0
    return {
        'rel_path': rel,
        'label': weight_path.name,
        'bytes': sz,
        'builtin': False,
    }


def _training_overwrite_confirmed() -> bool:
    raw = (request.form.get('overwrite') or request.args.get('overwrite') or '').strip().lower()
    return raw in ('1', 'true', 'yes', 'on')


def _optional_int(val, lo, hi):
    if val is None:
        return None
    if isinstance(val, str) and not val.strip():
        return None
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return None


def _sanitize_device(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if len(s) > 32:
        return 'cpu'
    if not re.match(r'^[A-Za-z0-9:_.+-]+$', s):
        return 'cpu'
    return s


@api_bp.route('/admin/training/device-info', methods=['GET'])
def training_device_info():
    admin, err, status = get_current_admin()
    if not admin:
        return err, status
    return jsonify(collect_training_device_info()), 200


@api_bp.route('/admin/training/base-weights', methods=['GET'])
def list_training_base_weights():
    admin, err, status = get_current_admin()
    if not admin:
        return err, status

    training_root = get_training_root(current_app.config)
    items = [_base_weight_item(p) for p in _list_base_weight_files(training_root)]
    return jsonify({'items': items}), 200


@api_bp.route('/admin/training/upload-base-weights', methods=['POST'])
def upload_training_base_weights():
    admin, err, status = get_current_admin()
    if not admin:
        return err, status

    if 'weights' not in request.files:
        return jsonify({'success': False, 'message': '请使用表单字段 weights'}), 400
    f = request.files['weights']
    if not f or not f.filename:
        return jsonify({'success': False, 'message': '文件为空'}), 400
    lower = f.filename.lower()
    if not (lower.endswith('.pt') or lower.endswith('.pth')):
        return jsonify({'success': False, 'message': '仅支持 .pt / .pth'}), 400

    training_root = get_training_root(current_app.config)
    dest_dir = training_root / 'bases'
    dest_dir.mkdir(parents=True, exist_ok=True)

    safe = secure_filename(f.filename) or 'weights.pt'
    dest_path = dest_dir / safe
    if dest_path.is_file() and not _training_overwrite_confirmed():
        return jsonify({
            'success': False,
            'message': f'已存在同名权重文件「{safe}」，是否覆盖？',
            'file_exists': True,
            'require_confirm': True,
            'existing_name': safe,
        }), 409

    f.save(str(dest_path))
    try:
        sz = dest_path.stat().st_size
    except OSError:
        sz = 0
    if sz > MAX_TRAINING_WEIGHT_BYTES:
        try:
            dest_path.unlink(missing_ok=True)
        except OSError:
            pass
        return jsonify({'success': False, 'message': f'权重文件过大（>{MAX_TRAINING_WEIGHT_BYTES // (1024 * 1024)}MB）'}), 400

    _remove_other_base_weights(training_root, dest_path)
    rel = f'training/bases/{safe}'.replace('\\', '/')
    return jsonify({
        'success': True,
        'rel_path': rel,
        'bytes': sz,
        'message': f'已上传训练权重 {safe}',
    }), 200


@api_bp.route('/admin/training/jobs', methods=['GET'])
def list_training_jobs():
    admin, err, status = get_current_admin()
    if not admin:
        return err, status
    limit = min(request.args.get('limit', 20, type=int), 50)
    jobs = TrainingJob.query.order_by(TrainingJob.id.desc()).limit(limit).all()
    cfg = current_app.config
    return jsonify({'jobs': [enrich_training_job_dict(j, cfg) for j in jobs]}), 200


@api_bp.route('/admin/training/jobs', methods=['POST'])
def create_training_job():
    admin, err, status = get_current_admin()
    if not admin:
        return err, status

    data = request.get_json(silent=True) or {}
    cfg = current_app.config

    try:
        epochs = int(data.get('epochs') or cfg.get('TRAIN_DEFAULT_EPOCHS', 30))
    except (TypeError, ValueError):
        epochs = cfg.get('TRAIN_DEFAULT_EPOCHS', 30)
    epochs = max(1, min(epochs, 500))

    base_rel = str(data.get('base_weights_rel') or '').strip().replace('\\', '/')
    if not base_rel:
        return jsonify({'success': False, 'message': '请选择训练初始权重'}), 400

    upload_root = Path(cfg['UPLOAD_FOLDER']).resolve()
    training_root = get_training_root(cfg)

    try:
        resolve_training_weights_file(upload_root, base_rel, training_root)
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400

    running = TrainingJob.query.filter(TrainingJob.status.in_(['queued', 'running'])).first()
    if running:
        return jsonify({'success': False, 'message': '已有训练任务进行中', 'job': running.to_dict()}), 409

    job = TrainingJob(
        status='queued',
        epochs=epochs,
        batch_size=_optional_int(data.get('batch_size'), 1, 128),
        imgsz=_optional_int(data.get('imgsz'), 320, 1280),
        patience=_optional_int(data.get('patience'), 1, 200),
        device=_sanitize_device(data.get('device')),
        base_weights_rel=base_rel,
        created_by_user_id=admin.id,
    )
    db.session.add(job)
    db.session.commit()
    start_training_job_async(current_app._get_current_object(), job.id)
    return jsonify({'success': True, 'job': enrich_training_job_dict(job, cfg)}), 201


@api_bp.route('/admin/training/jobs/<int:job_id>/live', methods=['GET'])
def training_job_live(job_id):
    admin, err, status = get_current_admin()
    if not admin:
        return err, status
    job = TrainingJob.query.get_or_404(job_id)
    project_root = Path(current_app.config['PROJECT_ROOT']).resolve()
    payload = build_training_live_payload(project_root, current_app.config, job)
    db.session.refresh(job)
    return jsonify({'job': enrich_training_job_dict(job, current_app.config), **payload}), 200


@api_bp.route('/admin/training/jobs/<int:job_id>/metrics', methods=['GET'])
def training_job_metrics(job_id):
    admin, err, status = get_current_admin()
    if not admin:
        return err, status
    job = TrainingJob.query.get_or_404(job_id)
    if job.status != 'succeeded':
        return jsonify({'success': False, 'message': '仅训练成功后可读取指标', 'metrics': None}), 200
    project_root = Path(current_app.config['PROJECT_ROOT']).resolve()
    train_dir = resolve_train_project_dir(current_app.config)
    metrics = load_training_metrics_best_row(project_root, train_dir, job)
    if not metrics:
        run = enrich_training_job_dict(job, current_app.config).get('run_name') or ''
        return jsonify({
            'success': False,
            'message': '未找到 results.csv',
            'metrics': None,
            'run_name': run,
            'has_metrics': False,
        }), 200
    return jsonify({
        'success': True,
        'metrics': metrics,
        'run_name': metrics.get('run_name') or '',
        'has_metrics': True,
    }), 200


@api_bp.route('/admin/training/jobs/<int:job_id>/stop', methods=['POST'])
def training_job_stop(job_id):
    admin, err, status = get_current_admin()
    if not admin:
        return err, status
    job = TrainingJob.query.get_or_404(job_id)
    if job.status != 'running':
        return jsonify({'success': False, 'message': '任务未在训练中'}), 400
    ok, msg = stop_training_job(current_app._get_current_object(), job_id)
    if not ok:
        return jsonify({'success': False, 'message': msg}), 400
    return jsonify({'success': True, 'message': '正在终止训练进程…'}), 200
