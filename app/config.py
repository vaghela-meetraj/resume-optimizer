import os
from dotenv import load_dotenv


def load_config():
    """Load configuration from environment variables with safe defaults."""
    load_dotenv()

    return {
        "MAX_CONTENT_LENGTH": int(os.getenv("MAX_CONTENT_LENGTH", 5 * 1024 * 1024)),  # 5MB strict limit
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", "").strip(),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", "").strip(),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", "").strip(),
        "LLM_PROVIDERS": os.getenv("LLM_PROVIDERS", "groq,openrouter,gemini").strip(),
        "GROQ_MODEL": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip(),
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        "OPENROUTER_MODEL": os.getenv("OPENROUTER_MODEL", "openrouter/free").strip(),
        "LLM_TEMPERATURE": float(os.getenv("LLM_TEMPERATURE", 0.3)),
        "LLM_MAX_TOKENS": int(os.getenv("LLM_MAX_TOKENS", 4000)),
        "LLM_TIMEOUT": float(os.getenv("LLM_TIMEOUT", 15.0)),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO").upper().strip(),
        "SECRET_KEY": os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod"),
    }
