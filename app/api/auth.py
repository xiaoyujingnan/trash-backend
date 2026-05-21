"""登录注册与 JWT 鉴权（原 deps 已并入）。"""
import datetime
import jwt
from datetime import timezone

from flask import current_app, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

from app import db
from app.models import User
from app.models.user import SUPER_ADMIN_USERNAME
from . import api_bp

def resolve_bearer_user():
    """无 Authorization：匿名 (None, None)；有 Bearer 则解析用户。"""
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None, None
    token = auth_header.split(' ', 1)[1].strip()
    if not token:
        return None, (jsonify({'error': '未登录或登录已过期'}), 401)
    try:
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = payload.get('user_id')
        if not user_id:
            return None, (jsonify({'error': '登录信息无效'}), 401)
        user = User.query.get(user_id)
        if not user:
            return None, (jsonify({'error': '用户不存在'}), 401)
        if getattr(user, 'is_disabled', False):
            return None, (jsonify({'error': '账号已被禁用'}), 403)
        return user, None
    except jwt.ExpiredSignatureError:
        return None, (jsonify({'error': '登录已过期'}), 401)
    except jwt.InvalidTokenError:
        return None, (jsonify({'error': '登录信息无效'}), 401)

def get_current_user_required():
    user, err = resolve_bearer_user()
    if err:
        return None, err[0], err[1]
    if not user:
        return None, jsonify({'error': '未登录或登录已过期'}), 401
    return user, None, None

def get_current_user():
    """个人资料等接口：success/message 风格鉴权。"""
    user, err = resolve_bearer_user()
    if err:
        body = err[0].get_json(silent=True) or {}
        msg = body.get('message') or body.get('error') or '未登录或登录已过期'
        return None, jsonify({'success': False, 'message': msg}), err[1]
    if not user:
        return None, jsonify({'success': False, 'message': '未提供有效授权信息'}), 401
    return user, None, None

def get_current_admin():
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None, jsonify({'success': False, 'message': '未登录或登录已过期'}), 401
    token = auth_header.split(' ', 1)[1].strip()
    if not token:
        return None, jsonify({'success': False, 'message': '未登录或登录已过期'}), 401
    try:
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = payload.get('user_id')
        user = User.query.get(user_id) if user_id else None
        if not user:
            return None, jsonify({'success': False, 'message': '用户不存在'}), 404
        if getattr(user, 'is_disabled', False):
            return None, jsonify({'success': False, 'message': '账号已被禁用'}), 403
        if not user.is_admin:
            return None, jsonify({'success': False, 'message': '无管理员权限'}), 403
        return user, None, None
    except jwt.ExpiredSignatureError:
        return None, jsonify({'success': False, 'message': '登录状态已过期，请重新登录'}), 401
    except jwt.InvalidTokenError:
        return None, jsonify({'success': False, 'message': '登录信息无效'}), 401

@api_bp.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()

        if not data or not data.get('username') or not data.get('email') or not data.get('password') or not data.get('email_password'):
            return jsonify({'success':False,'message':'缺少用户信息'}),400

        if User.query.filter_by(username=data['username']).first():
            return jsonify({'success':False,'message':'该用户已存在！'}),400

        if str(data.get('username') or '').strip().lower() == SUPER_ADMIN_USERNAME.lower():
            return jsonify({'success': False, 'message': '该用户名为系统保留，不可注册'}), 400

        if User.query.filter_by(email=data['email']).first():
            return jsonify({'success':False,'message': '该邮箱已被绑定！'}), 400

        user = User(
            username=data['username'],
            password_hash=generate_password_hash(data['password']),
            email=data['email'],
            email_password_hash=generate_password_hash(data['email_password']),
        )

        db.session.add(user)
        db.session.commit()

        return jsonify({'success':True,'message': '注册成功', 'user': user.to_dict()}), 201

    except Exception as e:
        return jsonify({'success':False,'message':'服务器异常，请稍后再试'}),500

@api_bp.route('/user_login', methods=['POST'])
def user_login():
    try:
        data = request.get_json()

        if not data or not data.get('account') or not data.get('password'):
            return jsonify({'success':False,'message':'缺少账号/邮箱或密码'}),400

        account = data['account']
        is_email_login = False
        user = User.get_by_username_exact(account)
        if not user:
            user = User.query.filter_by(email=account).first()
            if user:
                is_email_login = True

        if not user:
             return jsonify({'success':False,'message':'该账户不存在！'}),404

        if getattr(user, 'is_admin', False):
            return jsonify({
                'success': False,
                'message': '管理员账号请使用「管理员登录」入口登录',
            }), 403

        if getattr(user, 'is_disabled', False):
            return jsonify({'success': False, 'message': '该账号已被禁用，请联系管理员'}), 403

        if is_email_login:
            password_ok = user.email_password_hash and check_password_hash(user.email_password_hash,data['password'])
        else:
            password_ok = check_password_hash(user.password_hash,data['password'])

        if password_ok:
            user.last_login_at = datetime.datetime.now(timezone.utc)
            db.session.commit()
            token = jwt.encode({
                'user_id':user.id,
                'username':user.username,
                'exp':datetime.datetime.now(timezone.utc)+datetime.timedelta(hours=24)},current_app.config['SECRET_KEY'],algorithm='HS256'
            )
        else:
            return jsonify({'success':False,'message':'密码错误！'}),401
        return jsonify({'success':True,'token':token,'user':user.to_dict(),'message':'登陆成功！'}),200

    except Exception as e:
        return jsonify({'success':False,'message':'服务器异常，请稍后再试'}),500

@api_bp.route('/admin_login', methods=['POST'])
def admin_login():
    try:
        data = request.get_json() or {}

        account = str(data.get('account') or data.get('username') or '').strip()
        password = data.get('password')
        password = str(password or '')

        if not account or not password:
            return jsonify({'success': False, 'message': '缺少账号/邮箱或密码'}), 400

        is_email_login = False
        user = User.get_by_username_exact(account, is_admin=True)
        if not user:
            user = User.query.filter_by(email=account, is_admin=True).first()
            if user:
                is_email_login = True

        if not user:
            return jsonify({'success': False, 'message': '该账户不存在！'}), 404

        if getattr(user, 'is_disabled', False):
            return jsonify({'success': False, 'message': '该账号已被禁用，请联系管理员'}), 403

        if is_email_login:
            password_ok = user.email_password_hash and check_password_hash(
                user.email_password_hash, password
            )
        else:
            password_ok = check_password_hash(user.password_hash, password)

        if not password_ok:
            return jsonify({'success': False, 'message': '密码错误！'}), 401

        user.last_login_at = datetime.datetime.now(timezone.utc)
        db.session.commit()
        token = jwt.encode(
            {
                'user_id': user.id,
                'username': user.username,
                'is_admin': user.is_admin,
                'exp': datetime.datetime.now(timezone.utc) + datetime.timedelta(hours=2),
            },
            current_app.config['SECRET_KEY'],
            algorithm='HS256',
        )
        return jsonify({'success': True, 'token': token, 'user': user.to_dict(), 'message': '登陆成功！'}), 200

    except Exception:
        return jsonify({'success': False, 'message': '服务器异常，请稍后再试'}), 500
