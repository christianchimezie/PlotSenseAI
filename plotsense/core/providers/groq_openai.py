from typing import List
from openai.types.chat import ChatCompletionUserMessageParam
from openai import OpenAI
from .base import LLMProvider


class GroqProvider(LLMProvider):
    """Provider for Groq models using the unified OpenAI SDK interface."""

    LINK = "👉 https://console.groq.com/keys 👈"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None

    def _init_client(self):
        """Initialize Groq client via OpenAI-compatible endpoint."""
        if not self.client:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"  # Key difference
            )

    def query(self, prompt: str, model: str, **kwargs) -> str:
        """
        Send a chat completion query to Groq via OpenAI SDK.
        """
        self._init_client()
        if not self.client:
            raise ValueError("Groq client not initialized. Call validate_key() first.")

        # Ensure messages are present
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
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq query failed: {e}")

    def list_models(self) -> List[str]:
        """
        Available Groq Llama models (you can update this dynamically later).
        """
        return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

    def validate_key(self) -> bool:
        """
        Simple ping to check API validity.
        """
        try:
            self._init_client()
            if not self.client:
                raise ValueError("Groq OpenAI client not initialized.")
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5
            )
            return bool(response)
        except Exception:
            return False

