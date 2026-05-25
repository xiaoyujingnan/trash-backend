from flask import Blueprint, jsonify

api_bp = Blueprint('api', __name__)


@api_bp.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


from . import auth  # noqa: F401,E402
from . import profile  # noqa: F401,E402
from . import detect  # noqa: F401,E402
from . import history  # noqa: F401,E402
from . import admin  # noqa: F401,E402
