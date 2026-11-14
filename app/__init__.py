import logging
import uuid
from pathlib import Path

from flask import Flask, jsonify, request

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
    app.json.sort_keys = False

    # Configure structured logging
    log_level = getattr(logging, app.config.get("LOG_LEVEL", "INFO"), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )
    logger = logging.getLogger("app")

    cache.init_app(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})
    limiter.init_app(app)

    # Initialize NLP processor singleton
    app.extensions["nlp_processor"] = NLPProcessor()

    # Initialize Provider-Agnostic LLM Chain (Groq -> OpenRouter -> Gemini -> Offline Template)
    from .llm.chain import LLMChain
    raw_providers = [p.strip().lower() for p in app.config.get("LLM_PROVIDERS", "groq,openrouter,gemini").split(",") if p.strip()]
    app.extensions["llm_chain"] = LLMChain(providers=raw_providers)
    logger.info(f"LLMChain initialized with provider priority: {raw_providers}")

    # 413 Payload Too Large error handler
    @app.errorhandler(413)
    def file_too_large(error):
        return jsonify({"error": "File too large (max 5MB)."}), 413

    # Global 500 error handler that never leaks raw str(e) to clients
    @app.errorhandler(500)
    def internal_server_error(error):
        req_id = getattr(request, "request_id", str(uuid.uuid4()))
        logger.error(f"[RequestID: {req_id}] Internal server error: {error}", exc_info=True)
        return jsonify({
            "error": "An unexpected server error occurred. Please try again.",
            "request_id": req_id
        }), 500

    # Add security headers and request-id tracing
    @app.after_request
    def apply_security_headers(response):
        req_id = getattr(request, "request_id", str(uuid.uuid4()))
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    from .routes import bp
    app.register_blueprint(bp)

    return app
