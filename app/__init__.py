from pathlib import Path

from flask import Flask

from .config import load_config
from .extensions import cache, limiter
from .services.llm import configure_gemini_client
from .services.nlp import NLPProcessor


def create_app():
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )

    app.config.update(load_config())
    app.config["JSON_SORT_KEYS"] = False
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

    cache.init_app(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})
    limiter.init_app(app)

    app.extensions["nlp_processor"] = NLPProcessor()
    configure_gemini_client(app.config["GOOGLE_API_KEY"])

    from .routes import bp

    app.register_blueprint(bp)
    return app
