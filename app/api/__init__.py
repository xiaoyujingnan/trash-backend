from flask import Blueprint

api_bp = Blueprint('api', __name__)

from . import auth  # noqa: F401,E402
from . import profile  # noqa: F401,E402
from . import detect  # noqa: F401,E402
from . import history  # noqa: F401,E402
from . import admin  # noqa: F401,E402
