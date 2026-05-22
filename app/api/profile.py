from flask import request, jsonify, current_app, send_from_directory
import os
from werkzeug.security import check_password_hash, generate_password_hash
from app.api.auth import get_current_user
from app.utils import (
    ALLOWED_IMAGE_EXTENSIONS,
    resolve_stored_file_path,
    safe_remove_file,
    sanitize_username_for_filename,
)
from app import db
from app.models import User, UserProfile
from . import api_bp

DEFAULT_USER_AVATAR_URL = '/api/files/avatars/default_user_avatar.png'

def get_or_create_profile(user_id):
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    if profile:
        return profile
    profile = UserProfile(user_id=user_id, avatar=DEFAULT_USER_AVATAR_URL)
    db.session.add(profile)
    db.session.commit()
    return profile

def merge_user_profile(user, profile):
    profile.avatar = normalize_avatar_path(profile.avatar)
    user_dict = user.to_dict()
    user_dict.update(profile.to_dict())
    user_dict['nickname'] = user_dict.get('nickname') or user_dict.get('username')
    return user_dict

def is_allowed_image(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ALLOWED_IMAGE_EXTENSIONS

def normalize_avatar_path(path):
    value = str(path or '').strip()
    if not value:
        return ''
    if value.startswith('/api/files/'):
        return value
    if value.startswith('/uploads/'):
        return '/api/files/' + value.replace('/uploads/', '', 1)
    if value.startswith('http://') or value.startswith('https://'):
        marker = '/uploads/'
        if marker in value:
            return '/api/files/' + value.split(marker, 1)[1]
    return value

@api_bp.route('/files/<path:filename>', methods=['GET'])
def profile_files(filename):
    upload_root = current_app.config['UPLOAD_FOLDER']
    target_path = os.path.join(upload_root, filename)
    if os.path.exists(target_path):
        return send_from_directory(upload_root, filename)

    return jsonify({'success': False, 'message': '文件不存在'}), 404

@api_bp.route('/profile', methods=['GET'])
def get_profile():
    user, error_response, status = get_current_user()
    if not user:
        return error_response, status

    profile = get_or_create_profile(user.id)
    merged = merge_user_profile(user, profile)
    return jsonify({'success': True, 'user': merged, 'message': '获取个人资料成功'}), 200

@api_bp.route('/profile', methods=['PUT'])
def update_profile():
    user, error_response, status = get_current_user()
    if not user:
        return error_response, status

    data = request.get_json() or {}
    profile = get_or_create_profile(user.id)

    allowed_profile_fields = {
        'nickname': 80,
        'phone': 30,
        'signature': 255,
        'avatar': 1024 * 1024,
    }
    for field, max_len in allowed_profile_fields.items():
        if field in data:
            value = data.get(field)
            if value is None:
                setattr(profile, field, None)
                continue
            text = str(value).strip()
            if len(text) > max_len:
                return jsonify({'success': False, 'message': f'{field} 超出长度限制'}), 400
            setattr(profile, field, text)

    if 'email' in data:
        email = str(data.get('email') or '').strip()
        if email and '@' not in email:
            return jsonify({'success': False, 'message': '邮箱格式不正确'}), 400
        if email and email != user.email:
            duplicate = User.query.filter(User.email == email, User.id != user.id).first()
            if duplicate:
                return jsonify({'success': False, 'message': '该邮箱已被绑定！'}), 400
        user.email = email or None

    if 'email_password' in data:
        ep = str(data.get('email_password') or '').strip()
        if ep:
            if len(ep) < 6 or len(ep) > 32:
                return jsonify({'success': False, 'message': '邮箱登录密码长度需为 6~32 位'}), 400
            user.email_password_hash = generate_password_hash(ep)

    db.session.commit()
    merged = merge_user_profile(user, profile)
    return jsonify({'success': True, 'user': merged, 'message': '更新个人资料成功'}), 200

@api_bp.route('/profile/password', methods=['PUT'])
def change_password():
    user, error_response, status = get_current_user()
    if not user:
        return error_response, status

    data = request.get_json() or {}
    old_password = str(data.get('old_password') or '').strip()
    new_password = str(data.get('new_password') or '').strip()
    confirm_password = str(data.get('confirm_password') or '').strip()

    if not old_password or not new_password or not confirm_password:
        return jsonify({'success': False, 'message': '请完整填写旧密码和新密码'}), 400
    if new_password != confirm_password:
        return jsonify({'success': False, 'message': '两次输入的新密码不一致'}), 400
    if len(new_password) < 6 or len(new_password) > 32:
        return jsonify({'success': False, 'message': '新密码长度需为 6~32 位'}), 400

    old_ok = False
    if user.password_hash and check_password_hash(user.password_hash, old_password):
        old_ok = True
    if user.email_password_hash and check_password_hash(user.email_password_hash, old_password):
        old_ok = True
    if not old_ok:
        return jsonify({'success': False, 'message': '旧密码不正确'}), 400
    if user.password_hash and check_password_hash(user.password_hash, new_password):
        return jsonify({'success': False, 'message': '新密码不能与旧密码相同'}), 400

    new_hash = generate_password_hash(new_password)
    user.password_hash = new_hash
    user.email_password_hash = new_hash
    db.session.commit()
    return jsonify({'success': True, 'message': '密码修改成功，请重新登录'}), 200

@api_bp.route('/profile/avatar', methods=['POST'])
def upload_avatar():
    user, error_response, status = get_current_user()
    if not user:
        return error_response, status

    if 'avatar' not in request.files:
        return jsonify({'success': False, 'message': '未检测到上传文件'}), 400

    file = request.files['avatar']
    if not file or not file.filename:
        return jsonify({'success': False, 'message': '文件为空'}), 400
    if not is_allowed_image(file.filename):
        return jsonify({'success': False, 'message': '仅支持 png/jpg/jpeg/webp/gif 图片'}), 400

    profile = get_or_create_profile(user.id)
    upload_root = current_app.config['UPLOAD_FOLDER']
    old_av = normalize_avatar_path(profile.avatar)
    if old_av and old_av != DEFAULT_USER_AVATAR_URL:
        old_path = resolve_stored_file_path(old_av, upload_root)
        if old_path and os.path.basename(old_path).lower() != 'default_user_avatar.png':
            safe_remove_file(old_path)

    fname = file.filename or ''
    ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else 'jpg'
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        ext = 'jpg'
    if ext == 'jpeg':
        ext = 'jpg'
    slug = sanitize_username_for_filename(getattr(user, 'username', None))
    unique_name = f'{int(user.id)}_{slug}.{ext}' if slug else f'{int(user.id)}.{ext}'

    avatar_dir = os.path.join(upload_root, 'avatars')
    os.makedirs(avatar_dir, exist_ok=True)
    save_path = os.path.join(avatar_dir, unique_name)
    file.save(save_path)

    avatar_url = f'/api/files/avatars/{unique_name}'
    profile.avatar = avatar_url
    db.session.commit()

    merged = merge_user_profile(user, profile)
    return jsonify({'success': True, 'user': merged, 'avatar_url': avatar_url, 'message': '头像上传成功'}), 200
