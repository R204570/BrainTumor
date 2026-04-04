"""
app.py - NeuroAssist Flask application factory
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask
from psycopg2 import OperationalError

from config import Config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_runtime_dirs(app: Flask) -> None:
    """Create runtime directories used by uploads and static artifacts."""
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["ANNOTATED_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.static_folder or "static").mkdir(parents=True, exist_ok=True)


def create_app(config_class=Config) -> Flask:
    """Application factory that builds and configures the Flask app."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config_class)
    app.logger.warning(
        "[app.py] config loaded LLM_PROVIDER=%s GROQ_MODEL=%s GROQ_BASE_URL=%s LLM_TIMEOUT_SECONDS=%s",
        app.config.get("LLM_PROVIDER"),
        app.config.get("GROQ_MODEL"),
        app.config.get("GROQ_BASE_URL"),
        app.config.get("LLM_TIMEOUT_SECONDS"),
    )

    _ensure_runtime_dirs(app)

    # Initialize DB pool at startup so DB errors surface early.
    from db.connection import bootstrap_database, ensure_schema_compatibility, init_pool

    try:
        init_pool()
    except OperationalError:
        if app.config.get("AUTO_BOOTSTRAP_DB", True):
            bootstrap_database()
        else:
            raise
    else:
        ensure_schema_compatibility()

    # Register blueprints.
    from routes.chat import chat_bp
    from routes.report import report_bp
    from routes.session import session_bp
    from routes.upload import upload_bp

    app.register_blueprint(session_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(report_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=app.config.get("DEBUG", True),
    )
