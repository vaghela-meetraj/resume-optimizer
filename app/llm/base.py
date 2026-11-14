from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LLMError(Exception):
    """Base exception for all LLM errors."""
    pass


class RateLimited(LLMError):
    """Raised when an LLM provider returns a 429 Too Many Requests."""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60, provider: str = "unknown"):
        super().__init__(message)
        self.retry_after = retry_after
        self.provider = provider


class ProjectIdea(BaseModel):
    ProjectTitle: str = Field(description="Actionable title for the portfolio project")
    Description: str = Field(description="2-3 sentence overview explaining how this project solves real business problems")
    KeySkills: List[str] = Field(description="List of target missing skills practiced in this project")
    ResumeBullets: List[str] = Field(default_factory=list, description="3 high-impact resume bullet points with [X] metrics")
    
    # Real-world problem context
    Domain: Optional[str] = None
    TargetUser: Optional[str] = None
    Problem: Optional[str] = None
    Impact: Optional[str] = None
    BuiltOn: Optional[List[str]] = None
    JDKeywordsCovered: Optional[List[str]] = None
    
    # Backward compatibility fields
    ProjectName: Optional[str] = None
    SkillsPracticed: Optional[List[str]] = None
    ResumeBullet: Optional[str] = None
    Milestones: Optional[List[str]] = None
    Difficulty: Optional[str] = None
    EstimatedDays: Optional[int] = None
    SuggestedStack: Optional[str] = None
    DefinitionOfDone: Optional[str] = None


class ProjectIdeas(BaseModel):
    ProjectIdeas: List[ProjectIdea] = Field(description="List of tailored project ideas")


class LLMClient(ABC):
    """Abstract interface for LLM provider adapters."""

    @abstractmethod
    def generate_json(self, prompt: str, schema_class: type = ProjectIdeas, max_tokens: int = 1500, timeout: float = 15.0) -> Dict[str, Any]:
        """
        Execute completion and return a dictionary containing:
        {
            "data": dict (validated JSON),
            "tokens_used": int,
            "latency_ms": float,
            "provider": str,
            "model": str
        }
        """
        pass

