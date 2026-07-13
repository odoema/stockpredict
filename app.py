"""
Flask application factory for StockPredict.
"""
import logging
import os

from flask import Flask
from flask_cors import CORS

from config import config_by_name


def create_app(env: str | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        env: "development" or "production". Falls back to the FLASK_ENV
            environment variable, defaulting to "production".
    """
    env = env or os.environ.get("FLASK_ENV", "production")
    app = Flask(__name__)
    app.config.from_object(config_by_name.get(env, config_by_name["production"]))

    CORS(app)

    logging.basicConfig(
        level=logging.DEBUG if app.config["DEBUG"] else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app.config["DATA_DIR"].mkdir(parents=True, exist_ok=True)

    # Register blueprints
    from api.routes import api_bp
    from views import views_bp

    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.errorhandler(404)
    def not_found(_error):
        return {"error": "Not found"}, 404

    @app.errorhandler(500)
    def server_error(_error):
        app.logger.exception("Unhandled server error")
        return {"error": "Internal server error"}, 500

    return app
