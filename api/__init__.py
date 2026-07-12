import os

from dotenv import load_dotenv
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from .auth import require_api_key
from .routes.chat_generation import chat_generation_bp
from .routes.code_generation import code_generation_bp
from .routes.health import health_bp
from .routes.models import models_bp
from .routes.video_generation import video_generation_bp
from .routes.video_rendering import video_rendering_bp

_DEFAULT_LIMITS = ["60 per minute", "500 per hour"]


def create_app():
    app = Flask(__name__, static_folder="public", static_url_path="/public")

    load_dotenv()

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=_DEFAULT_LIMITS,
        storage_uri=os.getenv("RATELIMIT_STORAGE_URI", "memory://"),
    )

    app.register_blueprint(video_rendering_bp)
    app.register_blueprint(code_generation_bp)
    app.register_blueprint(chat_generation_bp)
    app.register_blueprint(video_generation_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(models_bp)

    # Tighter per-IP limits on compute-heavy endpoints
    limiter.limit("20 per minute")(video_rendering_bp)
    limiter.limit("20 per minute")(video_generation_bp)
    limiter.limit("30 per minute")(chat_generation_bp)
    limiter.limit("60 per minute")(code_generation_bp)

    CORS(app)

    app.before_request(require_api_key)

    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({"error": f"Rate limit exceeded: {e.description}"}), 429

    @app.route("/")
    def hello_world():
        return "Generative Manim Processor"

    @app.route("/openapi.yaml")
    def openapi():
        return send_from_directory(app.static_folder, "openapi.yaml")

    return app
