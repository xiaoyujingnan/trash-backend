from flask import Blueprint, jsonify

api_bp = Blueprint('api', __name__)


@api_bp.route('/health', methods=['GET'])
def health():
    from flask import current_app

    rel = (current_app.config.get('ACTIVE_UPLOADED_MODEL_REL') or '').strip()
    return jsonify({
        'status': 'ok',
        'model_ready': bool(rel),
        'model': rel,
    }), 200


from . import auth  # noqa: F401,E402
from . import profile  # noqa: F401,E402
from . import detect  # noqa: F401,E402
from . import history  # noqa: F401,E402
from . import admin  # noqa: F401,E402
