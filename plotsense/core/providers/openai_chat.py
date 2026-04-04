from typing import List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from .base import LLMProvider
from ..registry_loader import get_registry_loader


class OpenAIChatProvider(LLMProvider):
    """Provider integration for OpenAI Chat models."""

    LINK = "👉 https://platform.openai.com/api-keys 👈"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None

    def _init_client(self):
        """Initialize OpenAI client if not already created."""
        if not self.client:
            self.client = OpenAI(api_key=self.api_key)

    def query(
        self,
        prompt: Optional[str],
        model: str,
        **kwargs,
    ) -> str:
        """
        Send a prompt or messages to OpenAI Chat Completion API.
        Supports both chat-style input and single text prompts.
        """
        self._init_client()
        if not self.client:
            raise ValueError("OpenAI client not initialized. Call validate_key() first.")

        # Handle either prompt or messages
        messages: list[ChatCompletionUserMessageParam] = kwargs.pop("messages", None)
        if not messages and prompt:
            messages = [{"role": "user", "content": prompt}]
        elif not messages and not prompt:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI chat query failed: {e}")

    def list_models(self) -> List[str]:
        """Return list of supported OpenAI chat models from registry."""
        registry = get_registry_loader()
        return registry.get_provider_models("openai", "chat")

    def validate_key(self) -> bool:
        """
        Validate the provided OpenAI API key by performing a lightweight test query.
        """
        try:
            self._init_client()
            if not self.client:
                raise ValueError("OpenAI client not initialized.")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except Exception:
            return False
