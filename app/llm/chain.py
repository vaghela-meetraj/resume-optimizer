import logging
import os
import time
from typing import Any, Dict, List, Optional

from .adapters import GeminiAdapter, GroqAdapter, OpenRouterAdapter, TemplateFallbackAdapter
from .base import LLMClient, LLMError, ProjectIdeas, RateLimited
from .parser import parse_llm_projects
from .validator import check_candidate_four_keys, validate_projects

logger = logging.getLogger(__name__)


class LLMChain:
    """
    Provider-agnostic LLM fallback chain with strict validation and retry flow.
    Sequentially attempts configured providers, performs corrective retry on validation failure,
    recovers best 4-key candidates, and falls back to deterministic offline templates.
    """

    def __init__(self, providers: List[str] = None, daily_budget: int = 1000):
        if providers is None:
            raw = os.getenv("LLM_PROVIDERS", "groq,openrouter,gemini")
            providers = [p.strip().lower() for p in raw.split(",") if p.strip()]

        self.provider_names = providers
        self.daily_budget = daily_budget
        self.daily_calls = 0
        self.budget_date = time.strftime("%Y-%m-%d")

        # Provider cooldown timestamps: { "groq": 1720000000.0 }
        self.cooldowns: Dict[str, float] = {}

        # Instantiate adapters lazily
        self._adapters: Dict[str, LLMClient] = {
            "groq": GroqAdapter(),
            "gemini": GeminiAdapter(),
            "openrouter": OpenRouterAdapter(),
            "template": TemplateFallbackAdapter()
        }

    def _reset_budget_if_new_day(self):
        today = time.strftime("%Y-%m-%d")
        if today != self.budget_date:
            self.budget_date = today
            self.daily_calls = 0
            logger.info("Daily LLM budget counter reset for new date.")

    def is_in_cooldown(self, provider: str) -> bool:
        cooldown_until = self.cooldowns.get(provider, 0)
        return time.time() < cooldown_until

    def set_cooldown(self, provider: str, retry_after: int = 60):
        self.cooldowns[provider] = time.time() + max(10, retry_after)
        logger.warning(f"Provider '{provider}' set to cooldown for {retry_after}s.")

    def _extract_projects(self, res: Any) -> List[Dict[str, Any]]:
        """Extract projects list from diverse provider output shapes."""
        if not isinstance(res, dict):
            return []
        if "raw_projects" in res and isinstance(res["raw_projects"], list):
            return res["raw_projects"]
        data = res.get("data")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "Projects" in data and isinstance(data["Projects"], list):
                return data["Projects"]
            if "ProjectIdeas" in data and isinstance(data["ProjectIdeas"], list):
                return data["ProjectIdeas"]
        return []

    def _format_problem_prompt(self, base_prompt: str, problems: List[str]) -> str:
        """Append validator feedback to prompt for corrective retry."""
        if not problems:
            return base_prompt
        bullet_points = "\n".join(f"- {p}" for p in problems)
        return (
            f"{base_prompt}\n\n"
            f"Your previous output had these problems:\n"
            f"{bullet_points}\n\n"
            f"Fix them and return the full JSON again."
        )

    def execute(
        self,
        prompt: str,
        schema_class: type = ProjectIdeas,
        max_tokens: int = 8192,
        timeout: float = 15.0,
        missing_skills: Optional[List[str]] = None,
        matched_skills: Optional[List[str]] = None,
        validate: bool = True,
        providers: Optional[List[str]] = None,
        jd_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute request through sequential fallback chain with strict validation and retry.
        Uses `providers` if explicitly provided (e.g. ['openrouter', 'gemini']),
        or falls back to configured `self.provider_names`.
        """
        self._reset_budget_if_new_day()

        # Check daily budget limit
        if self.daily_calls >= self.daily_budget:
            logger.warning(f"Daily LLM budget exhausted ({self.daily_calls}/{self.daily_budget}). Using template mode.")
            logger.info("Project generation completed using provider: template, retries: 0")
            return self._adapters["template"].generate_json(prompt, schema_class)

        active_providers = [p.strip().lower() for p in providers] if providers is not None else list(self.provider_names)

        retries_used = 0
        best_candidate_projects: Optional[List[Dict[str, Any]]] = None
        best_candidate_provider: Optional[str] = None
        last_problems: List[str] = []

        for name in active_providers:
            if name in ("template", "offline"):
                continue
            if name not in self._adapters:
                continue
            if self.is_in_cooldown(name):
                remaining = int(self.cooldowns[name] - time.time())
                logger.info(f"Skipping '{name}' (in cooldown for {remaining}s remaining).")
                continue

            adapter = self._adapters[name]
            provider_failed = False
            problems = []

            # Format prompt: attach previous problem feedback if available
            current_prompt = self._format_problem_prompt(prompt, last_problems) if last_problems else prompt

            # Attempt 1: Standard generation
            try:
                logger.info(f"Attempting LLM generation with provider: '{name}'...")
                res = adapter.generate_json(current_prompt, schema_class, max_tokens=max_tokens, timeout=timeout)
                self.daily_calls += 1
                projects = self._extract_projects(res)

                # Track potential 4-key candidates
                four_keys = [p for p in projects if isinstance(p, dict) and check_candidate_four_keys(p)]
                if len(four_keys) >= 2 and best_candidate_projects is None:
                    best_candidate_projects = four_keys[:2]
                    best_candidate_provider = name

                if validate:
                    problems = validate_projects(projects, missing_skills, matched_skills, jd_keywords=jd_keywords)
                else:
                    problems = []

                if not problems:
                    logger.info(f"Project generation completed using provider: {name}, retries: {retries_used}")
                    return res
                else:
                    logger.warning(f"Provider '{name}' output had {len(problems)} validation problem(s): {problems}")
                    last_problems = problems

            except RateLimited as e:
                logger.warning(f"Provider '{name}' rate limited: {e}. Applying cooldown.")
                self.set_cooldown(name, e.retry_after)
                provider_failed = True
            except Exception as e:
                logger.warning(f"Provider '{name}' call failed: {e}.")
                provider_failed = True
                last_problems = [f"API Error: {str(e)[:120]}"]

            # Attempt 2: Single retry with error feedback if validation problems exist
            if not provider_failed and problems and not self.is_in_cooldown(name):
                retries_used += 1
                retry_prompt = self._format_problem_prompt(prompt, problems)
                try:
                    logger.info(f"Retrying provider '{name}' with error feedback (retry #{retries_used})...")
                    res = adapter.generate_json(retry_prompt, schema_class, max_tokens=max_tokens, timeout=timeout)
                    self.daily_calls += 1
                    projects = self._extract_projects(res)

                    four_keys = [p for p in projects if isinstance(p, dict) and check_candidate_four_keys(p)]
                    if len(four_keys) >= 2 and best_candidate_projects is None:
                        best_candidate_projects = four_keys[:2]
                        best_candidate_provider = name

                    if validate:
                        retry_problems = validate_projects(projects, missing_skills, matched_skills, jd_keywords=jd_keywords)
                    else:
                        retry_problems = []

                    if not retry_problems:
                        logger.info(f"Project generation completed using provider: {name}, retries: {retries_used}")
                        return res
                    else:
                        logger.warning(f"Provider '{name}' retry output still had {len(retry_problems)} validation problem(s).")
                        last_problems = retry_problems
                except RateLimited as e:
                    logger.warning(f"Provider '{name}' retry rate limited: {e}. Applying cooldown.")
                    self.set_cooldown(name, e.retry_after)
                except Exception as e:
                    logger.warning(f"Provider '{name}' retry failed: {e}.")
                    last_problems = [f"API Error: {str(e)[:120]}"]

        # Fallback to best 4-key candidate or offline template
        if best_candidate_projects and len(best_candidate_projects) >= 2:
            provider_used = best_candidate_provider or "candidate"
            logger.info(f"Using best 4-key candidate output from '{provider_used}' (retries: {retries_used}).")
            logger.info(f"Project generation completed using provider: {provider_used}, retries: {retries_used}")
            return {
                "data": {
                    "Projects": best_candidate_projects,
                    "ProjectIdeas": best_candidate_projects
                },
                "raw_projects": best_candidate_projects,
                "tokens_used": 0,
                "latency_ms": 0.0,
                "provider": provider_used,
                "model": f"{provider_used}-candidate"
            }

        logger.info(f"All remote providers failed or invalid. Falling back to offline static templates (retries: {retries_used}).")
        logger.info(f"Project generation completed using provider: template, retries: {retries_used}")
        template_adapter = self._adapters.get("template", TemplateFallbackAdapter())
        return template_adapter.generate_json(prompt, schema_class)
