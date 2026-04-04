from typing import List
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from .base import LLMProvider
from ..registry_loader import get_registry_loader


class OllamaProvider(LLMProvider):
    """
    Provider for Ollama models using the OpenAI-compatible API.
    This allows querying a locally running Ollama instance.
    """

    LINK = "👉 https://ollama.ai/library 👈"

    def __init__(self, api_key: str = ""):
        # Ollama typically doesn't require an API key (local service)
        self.api_key = api_key
        self.client = None

    def _init_client(self):
        """Initialize the OpenAI-compatible client for local Ollama."""
        if not self.client:
            # Default local Ollama endpoint
            self.client = OpenAI(
                base_url="http://localhost:11434/v1", # Ollama’s OpenAI-compatible API
                api_key=self.api_key or "ollama", # Dummy key for OpenAI client compatibility
            )

    def query(self, prompt: str, model: str, **kwargs) -> str:
        """
        Query the Ollama model using OpenAI-compatible endpoint.
        """
        self._init_client()
        if not self.client:
            raise ValueError("Ollama client not initialized. Call validate_key() first.")

        messages: list[ChatCompletionUserMessageParam] = kwargs.pop(
            "messages", None
        )
        if not messages and prompt:
            messages = [{"role": "user", "content": prompt}]
        elif not messages and not prompt:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        try:
            response = self.client.chat.completions.create(
                model=model,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Ollama query failed: {e}")

    def list_models(self) -> List[str]:
        """Return list of supported Ollama models from registry."""
        registry = get_registry_loader()
        return registry.get_provider_models("ollama", "default")

    def validate_key(self) -> bool:
        """
        Validate connection to local Ollama instance.
        """
        try:
            self._init_client()
            if not self.client:
                raise ValueError("Ollama OpenAI client not initialized.")
            response = self.client.chat.completions.create(
                model="llama3",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5
            )
            return bool(response)
        except Exception:
            return False

