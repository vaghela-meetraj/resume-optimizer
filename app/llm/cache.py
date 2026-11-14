import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days


def compute_llm_cache_key(missing_skills: List[str], level: str = "mid", prompt_version: str = "v3") -> str:
    """Generate deterministic SHA256 cache key based on missing skills set, level, and prompt version."""
    sorted_skills = ",".join(sorted(s.lower().strip() for s in missing_skills))
    raw_str = f"{sorted_skills}:{level.lower().strip()}:{prompt_version}"
    return f"llm_proj_{hashlib.sha256(raw_str.encode('utf-8')).hexdigest()}"


def clear_project_cache(cache_instance) -> bool:
    """Clear all cached entries from the cache instance."""
    if not cache_instance:
        return False
    try:
        if hasattr(cache_instance, "clear"):
            cache_instance.clear()
            logger.info("Project cache cleared completely.")
            return True
    except Exception as e:
        logger.warning(f"Error clearing cache: {e}")
    return False


def get_cached_projects(cache_instance, cache_key: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached projects if present."""
    if not cache_instance:
        return None
    try:
        cached = cache_instance.get(cache_key)
        if cached:
            logger.info(f"Cache HIT for key: {cache_key[:16]}... (serving from 7-day cache)")
            return cached
    except Exception as e:
        logger.warning(f"Cache lookup error: {e}")
    return None


def set_cached_projects(cache_instance, cache_key: str, data: Dict[str, Any], timeout: int = CACHE_TTL_SECONDS):
    """Store generated projects in cache for 7 days."""
    if not cache_instance:
        return
    try:
        cache_instance.set(cache_key, data, timeout=timeout)
        logger.info(f"Cached projects stored for 7 days under key: {cache_key[:16]}...")
    except Exception as e:
        logger.warning(f"Cache storage error: {e}")

