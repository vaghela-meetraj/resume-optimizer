import json
import logging
import os
import time
from typing import Any, Dict

from .base import LLMClient, LLMError, ProjectIdeas, RateLimited
from .parser import parse_llm_projects

logger = logging.getLogger(__name__)


class GroqAdapter(LLMClient):
    """Adapter for Groq LLM API using OpenAI-compatible SDK."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
        self.model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
        self.temperature = float(os.getenv("LLM_TEMPERATURE", 0.3))

    def generate_json(self, prompt: str, schema_class: type = ProjectIdeas, max_tokens: int = 4000, timeout: float = 15.0) -> Dict[str, Any]:
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not configured.")

        from openai import OpenAI, RateLimitError, APIError

        timeout = timeout or float(os.getenv("LLM_TIMEOUT", 15.0))
        max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", 4000))

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key,
            timeout=timeout,
            max_retries=0
        )

        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert technical career coach and software engineering lead. Always return strict, valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            content = response.choices[0].message.content or ""
            tokens_used = getattr(response.usage, "total_tokens", 0)

            # Robust parse supporting {"Projects": [...]}, {"ProjectIdeas": [...]}, or bare list
            parsed_list = parse_llm_projects(content)
            data_out = {"Projects": parsed_list, "ProjectIdeas": parsed_list}
            if schema_class == ProjectIdeas:
                try:
                    data_out = schema_class.model_validate({"ProjectIdeas": parsed_list}).model_dump()
                    data_out["Projects"] = parsed_list
                except Exception:
                    pass

            return {
                "data": data_out,
                "raw_projects": parsed_list,
                "tokens_used": tokens_used,
                "latency_ms": round(elapsed_ms, 2),
                "provider": "groq",
                "model": self.model
            }

        except RateLimitError as e:
            retry_after = 60
            if hasattr(e, "response") and e.response is not None:
                retry_header = e.response.headers.get("retry-after")
                if retry_header and retry_header.isdigit():
                    retry_after = int(retry_header)
            raise RateLimited(f"Groq rate limit exceeded: {e}", retry_after=retry_after, provider="groq")
        except Exception as e:
            raise LLMError(f"Groq API error ({self.model}): {e}")


class GeminiAdapter(LLMClient):
    """Adapter for Google Gemini API."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY", "")).strip()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self.temperature = float(os.getenv("LLM_TEMPERATURE", 0.3))

    def generate_json(self, prompt: str, schema_class: type = ProjectIdeas, max_tokens: int = 8192, timeout: float = 15.0) -> Dict[str, Any]:
        if not self.api_key:
            raise LLMError("GOOGLE_API_KEY is not configured.")

        from google import genai
        from google.genai import types

        timeout = timeout or float(os.getenv("LLM_TIMEOUT", 15.0))
        max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", 8192))

        client = genai.Client(api_key=self.api_key)
        t0 = time.perf_counter()
        try:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=max_tokens,
                temperature=self.temperature
            )
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            tokens_used = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_used = getattr(response.usage_metadata, "total_token_count", 0)

            parsed_list = parse_llm_projects(response.text or "")
            data_out = {"Projects": parsed_list, "ProjectIdeas": parsed_list}
            if schema_class == ProjectIdeas:
                try:
                    data_out = schema_class.model_validate({"ProjectIdeas": parsed_list}).model_dump()
                    data_out["Projects"] = parsed_list
                except Exception:
                    pass

            return {
                "data": data_out,
                "raw_projects": parsed_list,
                "tokens_used": tokens_used,
                "latency_ms": round(elapsed_ms, 2),
                "provider": "gemini",
                "model": self.model
            }

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                raise RateLimited(f"Gemini rate limit exceeded: {e}", retry_after=60, provider="gemini")
            raise LLMError(f"Gemini API error ({self.model}): {e}")
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                raise RateLimited(f"Gemini rate limit exceeded: {e}", retry_after=60, provider="gemini")
            raise LLMError(f"Gemini API error ({self.model}): {e}")


class OpenRouterAdapter(LLMClient):
    """Adapter for OpenRouter API."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
        self.model = model or os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
        self.temperature = float(os.getenv("LLM_TEMPERATURE", 0.3))

    def generate_json(self, prompt: str, schema_class: type = ProjectIdeas, max_tokens: int = 4000, timeout: float = 15.0) -> Dict[str, Any]:
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not configured.")

        from openai import OpenAI, RateLimitError

        timeout = timeout or float(os.getenv("LLM_TIMEOUT", 15.0))
        max_tokens = max_tokens or int(os.getenv("LLM_MAX_TOKENS", 4000))

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            timeout=timeout,
            max_retries=0,
            default_headers={
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "Resume Optimizer"
            }
        )

        t0 = time.perf_counter()
        try:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert technical career coach. Output strictly valid JSON with top-level key 'Projects'."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    extra_body={"reasoning": {"max_tokens": 0}}
                )
            except Exception:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert technical career coach. Output strictly valid JSON with top-level key 'Projects'."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=self.temperature
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000
            content = response.choices[0].message.content or ""
            tokens_used = getattr(response.usage, "total_tokens", 0)

            parsed_list = parse_llm_projects(content)
            data_out = {"Projects": parsed_list, "ProjectIdeas": parsed_list}
            if schema_class == ProjectIdeas:
                try:
                    data_out = schema_class.model_validate({"ProjectIdeas": parsed_list}).model_dump()
                    data_out["Projects"] = parsed_list
                except Exception:
                    pass

            return {
                "data": data_out,
                "raw_projects": parsed_list,
                "tokens_used": tokens_used,
                "latency_ms": round(elapsed_ms, 2),
                "provider": "openrouter",
                "model": self.model
            }

        except RateLimitError as e:
            raise RateLimited(f"OpenRouter rate limit: {e}", retry_after=60, provider="openrouter")
        except Exception as e:
            raise LLMError(f"OpenRouter API error ({self.model}): {e}")


class TemplateFallbackAdapter(LLMClient):
    """Deterministic offline fallback generator ensuring 100% availability."""

    def generate_json(self, prompt: str, schema_class: type = ProjectIdeas, max_tokens: int = 1500, timeout: float = 15.0) -> Dict[str, Any]:
        # High quality offline projects upgraded to new validator specifications
        sample_projects = [
            {
                "Domain": "healthcare/clinic ops",
                "TargetUser": "a 3-person dental clinic receptionist",
                "Problem": "Patient appointment check-ins and intake forms are manually tracked on paper leading to lost medical histories and long waiting times.",
                "ProjectTitle": "Clinic Patient Intake & Appointment Portal",
                "ProjectName": "Clinic Patient Intake & Appointment Portal",
                "Description": "A 3-person dental clinic receptionist struggles with manual paper intake forms that cause lost medical histories and waiting room delays. The architecture connects a FastAPI backend with PostgreSQL and Redis caching for instant patient lookups. The finished demo showcases an interactive receptionist dashboard with automated patient check-in and digital form verification.",
                "KeySkills": ["Docker", "PostgreSQL", "Redis"],
                "SkillsPracticed": ["Docker", "PostgreSQL", "Redis"],
                "BuiltOn": ["Python"],
                "TechStack": ["FastAPI", "PostgreSQL", "Redis", "Docker", "Python"],
                "Impact": "Demonstrates relational data architecture and containerized microservice workflows critical for enterprise healthcare integrations.",
                "ResumeBullets": [
                    "Engineered FastAPI patient intake microservice leveraging Redis caching, reducing patient record retrieval latency by [X]% under heavy clinic traffic.",
                    "Implemented PostgreSQL relational database schema with indexed patient records, cutting receptionist appointment lookup times by [X]% without data loss.",
                    "Containerized microservices stack utilizing Docker Compose and automated testing, eliminating deployment setup overhead by [X]% so that clinic staff work uninterrupted."
                ]
            },
            {
                "Domain": "personal finance",
                "TargetUser": "a freelance graphic designer",
                "Problem": "Tracking fluctuating client invoices and quarterly estimated taxes across spreadsheets is manual and leads to costly missed payment deadlines.",
                "ProjectTitle": "Freelance Invoice Tracker & Tax Reminder Service",
                "ProjectName": "Freelance Invoice Tracker & Tax Reminder Service",
                "Description": "A freelance designer frequently misses payment deadlines and miscalculates quarterly taxes due to chaotic spreadsheet tracking. The architecture streams webhook invoice updates into a Python worker that calculates tax liabilities and sends scheduled alerts. The finished demo shows an interactive dashboard with automatic invoice status updates and quarterly tax countdowns.",
                "KeySkills": ["Kafka", "SQL", "WebSockets"],
                "SkillsPracticed": ["Kafka", "SQL", "WebSockets"],
                "BuiltOn": ["Python"],
                "TechStack": ["Python", "Kafka", "SQL", "WebSockets"],
                "Impact": "Exhibits event-driven asynchronous streaming and real-time dashboard notifications required for financial data products.",
                "ResumeBullets": [
                    "Engineered real-time freelance invoice streaming service using Kafka message brokers, cutting invoice payment tracking turnaround times by [X]%.",
                    "Optimized client billing analytics utilizing PostgreSQL SQL query indexing, reducing invoice calculation processing time by [X]% without database timeouts.",
                    "Implemented interactive client alert notifications utilizing WebSockets connections, enabling instant payment notifications to [X] freelancers without manual page refreshes."
                ]
            }
        ]
        data_out = {"Projects": sample_projects, "ProjectIdeas": sample_projects}
        if schema_class == ProjectIdeas:
            try:
                data_out = schema_class.model_validate({"ProjectIdeas": sample_projects}).model_dump()
                data_out["Projects"] = sample_projects
            except Exception:
                pass

        return {
            "data": data_out,
            "raw_projects": sample_projects,
            "tokens_used": 0,
            "latency_ms": 1.0,
            "provider": "template",
            "model": "offline-curated"
        }

