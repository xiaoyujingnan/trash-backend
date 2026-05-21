"""识别历史：列表、详情、图片、删除。"""
import json
import os
from datetime import datetime, timedelta

from flask import current_app, jsonify, request, send_file
from sqlalchemy import or_
from sqlalchemy.orm import joinedload

from app import db
from app.api.auth import get_current_user_required
from app.api.profile import normalize_avatar_path
from app.models import DetectionResult, User
from app.models.pending_sample import delete_pending_samples_for_detection_ids
from app.utils import resolve_stored_file_path, safe_remove_file
from . import api_bp

_MACRO_CLASS_JSON_SNIPPETS = {
    '可回收物': ['"class": "recyclable"'],
    '有害垃圾': ['"class": "hazardous"'],
    '厨余垃圾': ['"class": "kitchen"'],
    '其他垃圾': ['"class": "other"'],
}

_CLASS_TO_MACRO = {
    'recyclable': '可回收物',
    'other': '其他垃圾',
    'hazardous': '有害垃圾',
    'kitchen': '厨余垃圾',
    'wet': '厨余垃圾',
    'dry': '其他垃圾',
    'cardboard': '可回收物',
    'glass': '可回收物',
    'metal': '可回收物',
    'paper': '可回收物',
    'plastic': '可回收物',
    'battery': '有害垃圾',
    'medicine': '有害垃圾',
    'paint': '有害垃圾',
    'food': '厨余垃圾',
    'leftovers': '厨余垃圾',
    'vegetable': '厨余垃圾',
    'fruit': '厨余垃圾',
    'trash': '其他垃圾',
    'tissue': '其他垃圾',
    'dust': '其他垃圾',
    '纸箱': '可回收物',
    '玻璃瓶': '可回收物',
    '易拉罐': '可回收物',
    '塑料瓶': '可回收物',
    '废电池': '有害垃圾',
    '过期药品': '有害垃圾',
    '菜叶': '厨余垃圾',
    '果皮': '厨余垃圾',
    '纸巾': '其他垃圾',
    '其他垃圾': '其他垃圾',
}

def _macro_from_top_detection(detected_objects_json):
    """取置信度最高的一条映射到大类；空或异常归为其他垃圾。"""
    if not detected_objects_json:
        return '其他垃圾'
    try:
        arr = json.loads(detected_objects_json)
        if not isinstance(arr, list) or not len(arr):
            return '其他垃圾'
        top = max(arr, key=lambda x: float(x.get('confidence') or 0))
        raw = str(top.get('class') or '').strip()
        key = raw.lower()
        if raw in ('湿垃圾', '干垃圾'):
            return '厨余垃圾' if raw == '湿垃圾' else '其他垃圾'
        return _CLASS_TO_MACRO.get(key, _CLASS_TO_MACRO.get(raw, '其他垃圾'))
    except Exception:
        return '其他垃圾'

def scoped_detection_query(user):
    q = DetectionResult.query
    if not getattr(user, 'is_admin', False):
        q = q.filter(DetectionResult.user_id == user.id)
    return q

def _compute_filtered_overview_stats(filtered_query):
    order = ['可回收物', '有害垃圾', '厨余垃圾', '其他垃圾']
    rows = filtered_query.with_entities(
        DetectionResult.detected_objects,
        DetectionResult.processing_time,
    ).all()
    counts = {m: 0 for m in order}
    time_sum = 0.0
    time_n = 0
    for blob, pt in rows:
        macro = _macro_from_top_detection(blob)
        if macro not in counts:
            macro = '其他垃圾'
        counts[macro] += 1
        if pt is not None:
            try:
                time_sum += float(pt)
                time_n += 1
            except (TypeError, ValueError):
                pass
    total = sum(counts.values())
    macro_distribution = []
    for macro in order:
        c = int(counts.get(macro, 0))
        pct = round(100.0 * c / total, 1) if total else 0.0
        macro_distribution.append({'macro': macro, 'count': c, 'percent': pct})
    avg_processing_time = round(time_sum / time_n, 2) if time_n else 0.0
    return {
        'macro_distribution': macro_distribution,
        'avg_processing_time': avg_processing_time,
        'overview_total': total,
    }

def _apply_detection_history_filters(query):
    date_from = (request.args.get('date_from') or '').strip()
    date_to = (request.args.get('date_to') or '').strip()
    if date_from:
        try:
            start = datetime.strptime(date_from, '%Y-%m-%d')
            query = query.filter(DetectionResult.created_at >= start)
        except ValueError:
            pass
    if date_to:
        try:
            end_day = datetime.strptime(date_to, '%Y-%m-%d')
            query = query.filter(DetectionResult.created_at < end_day + timedelta(days=1))
        except ValueError:
            pass

    macros_raw = (request.args.get('macros') or '').strip()
    parts = [p.strip() for p in macros_raw.split(',') if p.strip()]
    allowed = set(_MACRO_CLASS_JSON_SNIPPETS.keys())
    parts = [p for p in parts if p in allowed]
    if parts:
        conds = []
        for m in parts:
            for sub in _MACRO_CLASS_JSON_SNIPPETS[m]:
                conds.append(DetectionResult.detected_objects.like(f'%{sub}%'))
        if conds:
            query = query.filter(or_(*conds))
    return query

def _remove_detection_files(detection):
    upload_root = current_app.config['UPLOAD_FOLDER']
    for stored in (detection.file_path, detection.result_path):
        path = resolve_stored_file_path(stored, upload_root)
        safe_remove_file(path)

@api_bp.route('/detections', methods=['GET'])
def get_detections():
    try:
        user, err, status = get_current_user_required()
        if not user:
            return err, status

        page = max(1, request.args.get('page', 1, type=int) or 1)
        per_page = max(1, min(request.args.get('per_page', 10, type=int) or 10, 200))

        query = scoped_detection_query(user)
        query = _apply_detection_history_filters(query)
        base_count = scoped_detection_query(user)
        now = datetime.utcnow()
        utc_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        next_midnight = utc_midnight + timedelta(days=1)
        today_count = base_count.filter(
            DetectionResult.created_at >= utc_midnight,
            DetectionResult.created_at < next_midnight,
        ).count()

        overview = _compute_filtered_overview_stats(query)

        detections = query.order_by(DetectionResult.id.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        items = detections.items

        if getattr(user, 'is_admin', False):
            uids = list({d.user_id for d in items if d.user_id})
            user_map = {}
            if uids:
                for u in User.query.filter(User.id.in_(uids)).options(joinedload(User.profile)):
                    prof = u.profile
                    user_map[u.id] = {
                        'user_username': u.username,
                        'user_avatar': normalize_avatar_path(prof.avatar) if prof else '',
                    }
            payload = []
            for d in items:
                row = d.to_dict()
                ext = user_map.get(d.user_id) if d.user_id else None
                if ext:
                    row.update(ext)
                else:
                    row['user_username'] = ''
                    row['user_avatar'] = ''
                payload.append(row)
        else:
            payload = [d.to_dict() for d in items]

        return jsonify({
            'detections': payload,
            'total': detections.total,
            'pages': detections.pages,
            'current_page': page,
            'today_count': today_count,
            'macro_distribution': overview['macro_distribution'],
            'avg_processing_time': overview['avg_processing_time'],
            'overview_total': overview['overview_total'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/detections/<int:detection_id>', methods=['GET'])
def get_detection(detection_id):
    try:
        user, err, status = get_current_user_required()
        if not user:
            return err, status

        detection = DetectionResult.query.get_or_404(detection_id)
        if not getattr(user, 'is_admin', False) and detection.user_id != user.id:
            return jsonify({'error': '无权限访问该记录'}), 403
        return jsonify(detection.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/detections/<int:detection_id>/file', methods=['GET'])
def get_detection_file(detection_id):
    try:
        user, err, status = get_current_user_required()
        if not user:
            return err, status

        kind = (request.args.get('kind') or 'result').lower()
        detection = DetectionResult.query.get_or_404(detection_id)
        if not getattr(user, 'is_admin', False) and detection.user_id != user.id:
            return jsonify({'error': '无权限访问该记录'}), 403
        stored = detection.result_path if kind != 'original' else detection.file_path
        if not stored:
            return jsonify({'error': '文件不存在'}), 404
        path = resolve_stored_file_path(stored, current_app.config['UPLOAD_FOLDER'])
        if not path or not os.path.exists(path):
            return jsonify({'error': '文件不存在'}), 404
        return send_file(path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/detections/<int:detection_id>', methods=['DELETE'])
def delete_detection(detection_id):
    try:
        user, err, status = get_current_user_required()
        if not user:
            return err, status

        detection = DetectionResult.query.get_or_404(detection_id)
        if not getattr(user, 'is_admin', False) and detection.user_id != user.id:
            return jsonify({'error': '无权限删除该记录'}), 403

        _remove_detection_files(detection)
        delete_pending_samples_for_detection_ids([detection.id])
        db.session.delete(detection)
        db.session.commit()
        return jsonify({'success': True, 'message': '删除成功'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/detections/batch_delete', methods=['POST'])
def batch_delete_detections():
    try:
        user, err, status = get_current_user_required()
        if not user:
            return err, status

        if not getattr(user, 'is_admin', False):
            return jsonify({'error': '仅管理员可批量删除识别记录'}), 403

        data = request.get_json() or {}
        ids = data.get('ids') or []
        if not isinstance(ids, list) or not ids:
            return jsonify({'error': 'ids 参数无效'}), 400

        targets = scoped_detection_query(user).filter(DetectionResult.id.in_(ids)).all()
        for detection in targets:
            _remove_detection_files(detection)
        det_ids = [d.id for d in targets]
        delete_pending_samples_for_detection_ids(det_ids)
        for detection in targets:
            db.session.delete(detection)
        db.session.commit()

        return jsonify({
            'success': True,
            'deleted': len(targets),
            'message': f'批量删除完成：{len(targets)} 条',
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
