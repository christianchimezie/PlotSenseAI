from abc import ABC, abstractmethod
from typing import List

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    LINK: str

    @abstractmethod
    def __init__(self, api_key: str):
        """Initialize the provider with an API key."""
        pass

    @abstractmethod
    def query(self, prompt: str, model: str, **kwargs) -> str:
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        pass

    @abstractmethod
    def validate_key(self) -> bool:
        pass

