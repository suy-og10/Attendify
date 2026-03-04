from flask import Blueprint

student_bp = Blueprint('student', __name__, template_folder='../../frontend/templates')

from . import routes
