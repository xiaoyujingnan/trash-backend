from flask import Blueprint

api_bp = Blueprint('api', __name__)

from . import auth
from . import profile
from . import detect
from . import history
from . import admin
