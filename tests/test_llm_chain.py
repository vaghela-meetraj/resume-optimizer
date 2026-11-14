import time
import pytest
from app.llm.base import LLMClient, LLMError, ProjectIdeas, RateLimited
from app.llm.chain import LLMChain
from app.llm.cache import compute_llm_cache_key, get_cached_projects, set_cached_projects


class FakeSuccessClient(LLMClient):
    def __init__(self, name="fake_groq"):
        self.name = name
        self.call_count = 0

    def generate_json(self, prompt: str, schema_class: type = ProjectIdeas, max_tokens: int = 1500, timeout: float = 15.0):
        self.call_count += 1
        projects = [
            {
                "ProjectTitle": f"Custom {self.name} Portfolio Project",
                "ProjectName": f"Custom {self.name} Portfolio Project",
                "Description": "Build an enterprise application. This project practices high scale backend architecture.",
                "KeySkills": ["Docker", "Python", "Kubernetes"],
                "SkillsPracticed": ["Docker", "Python", "Kubernetes"],
                "TechStack": ["Docker", "Python", "Kubernetes"],
                "BuiltOn": ["Python"],
                "Difficulty": "Intermediate",
                "EstimatedDays": 7,
                "Milestones": ["Setup environment", "Core logic", "Integration", "Demo and document"],
                "SuggestedStack": "Python, Docker",
                "DefinitionOfDone": "Tests passing.",
                "ResumeBullets": [
                    "Built containerized services utilizing Docker Compose, reducing local deployment times by [X]% under peak loads.",
                    "Implemented asynchronous background tasks utilizing Python, eliminating manual processing delays for [X] users without errors.",
                    "Deployed scalable clusters utilizing Kubernetes manifests, cutting service failure recovery time by [X]% so that systems stay online."
                ]
            },
            {
                "ProjectTitle": f"Distributed {self.name} Stream Engine",
                "ProjectName": f"Distributed {self.name} Stream Engine",
                "Description": "Build real time analytics platform. Ingests high throughput event pipelines.",
                "KeySkills": ["Kafka", "PostgreSQL", "Redis"],
                "SkillsPracticed": ["Kafka", "PostgreSQL", "Redis"],
                "TechStack": ["Kafka", "PostgreSQL", "Redis", "Python"],
                "BuiltOn": ["Python"],
                "Difficulty": "Intermediate",
                "EstimatedDays": 7,
                "Milestones": ["Setup environment", "Core logic", "Integration", "Demo and document"],
                "SuggestedStack": "Kafka, PostgreSQL",
                "DefinitionOfDone": "Tests passing.",
                "ResumeBullets": [
                    "Engineered streaming event consumers utilizing Kafka, reducing data pipeline processing latency by [X]% under heavy loads.",
                    "Optimized relational query indexing utilizing PostgreSQL, eliminating database lookup bottlenecks for [X] concurrent requests without timeouts.",
                    "Configured distributed caching layers utilizing Redis clusters, cutting roundtrip query response times by [X]% so that dashboards load instantly."
                ]
            }
        ]
        return {
            "data": {
                "ProjectIdeas": projects,
                "Projects": projects
            },
            "raw_projects": projects,
            "tokens_used": 150,
            "latency_ms": 45.0,
            "provider": self.name,
            "model": "fake-model"
        }


class FakeRateLimitedClient(LLMClient):
    def __init__(self, name="fake_groq", retry_after=30):
        self.name = name
        self.retry_after = retry_after
        self.call_count = 0

    def generate_json(self, prompt: str, schema_class: type = ProjectIdeas, max_tokens: int = 1500, timeout: float = 15.0):
        self.call_count += 1
        raise RateLimited(f"{self.name} 429 quota exhausted", retry_after=self.retry_after, provider=self.name)


class FakeFailingClient(LLMClient):
    def __init__(self, name="fake_groq"):
        self.name = name
        self.call_count = 0

    def generate_json(self, prompt: str, schema_class: type = ProjectIdeas, max_tokens: int = 1500, timeout: float = 15.0):
        self.call_count += 1
        raise LLMError(f"{self.name} network socket timeout")


def test_chain_primary_provider_success():
    """Primary provider succeeds on first attempt."""
    chain = LLMChain(providers=["groq", "gemini"])
    chain._adapters["groq"] = FakeSuccessClient("groq")
    chain._adapters["gemini"] = FakeSuccessClient("gemini")

    res = chain.execute("Build projects")
    assert res["provider"] == "groq"
    assert chain._adapters["groq"].call_count == 1
    assert chain._adapters["gemini"].call_count == 0


def test_chain_falls_through_on_429_rate_limit():
    """When primary provider is 429 rate limited, chain sets cooldown and falls through to next provider."""
    chain = LLMChain(providers=["groq", "gemini"])
    chain._adapters["groq"] = FakeRateLimitedClient("groq", retry_after=60)
    chain._adapters["gemini"] = FakeSuccessClient("gemini")

    res = chain.execute("Build projects")
    assert res["provider"] == "gemini"
    assert chain.is_in_cooldown("groq") is True

    # Second call should skip groq entirely because it is in cooldown
    res2 = chain.execute("Build projects again")
    assert res2["provider"] == "gemini"
    assert chain._adapters["groq"].call_count == 1  # Not called again


def test_chain_falls_through_to_template_when_all_fail():
    """When all remote providers fail, chain falls back to offline template generator."""
    chain = LLMChain(providers=["groq", "gemini", "openrouter"])
    chain._adapters["groq"] = FakeFailingClient("groq")
    chain._adapters["gemini"] = FakeFailingClient("gemini")
    chain._adapters["openrouter"] = FakeFailingClient("openrouter")

    res = chain.execute("Build projects")
    assert res["provider"] == "template"
    assert len(res["data"]["ProjectIdeas"]) >= 2


def test_chain_daily_budget_limit():
    """When daily call budget is exhausted, chain skips directly to template mode."""
    chain = LLMChain(providers=["groq"], daily_budget=2)
    chain._adapters["groq"] = FakeSuccessClient("groq")

    res1 = chain.execute("Call 1")
    assert res1["provider"] == "groq"
    res2 = chain.execute("Call 2")
    assert res2["provider"] == "groq"

    # Call 3 exceeds daily budget of 2
    res3 = chain.execute("Call 3")
    assert res3["provider"] == "template"


class SimpleMockCache:
    def __init__(self):
        self.store = {}
    def get(self, k):
        return self.store.get(k)
    def set(self, k, v, timeout=None):
        self.store[k] = v


def test_caching_by_sha256_missing_skills():
    """Submitting the same missing skills set produces identical cache key and retrieves cached data."""
    mock_cache = SimpleMockCache()
    key1 = compute_llm_cache_key(["Docker", "Kafka", "AWS"], level="Senior", prompt_version="v2")
    # Order of skills should not alter hash
    key2 = compute_llm_cache_key(["AWS", "Docker", "Kafka"], level="Senior", prompt_version="v2")
    assert key1 == key2

    set_cached_projects(mock_cache, key1, {"projects": [{"name": "Cached Project"}]})
    cached = get_cached_projects(mock_cache, key2)
    assert cached is not None
    assert cached["projects"][0]["name"] == "Cached Project"

