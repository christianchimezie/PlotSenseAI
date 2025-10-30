from typing import List
from groq import Groq
from groq.types.chat import ChatCompletionUserMessageParam
from .base import LLMProvider


class GroqProvider(LLMProvider):
    """Provider integration for Groq's fast inference API."""
    
    LINK = "👉 https://console.groq.com/keys 👈"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None

    def _init_client(self):
        """Initialize Groq client if not already created."""
        if not self.client:
            self.client = Groq(api_key=self.api_key)

    def query(
        self,
        prompt: str,
        model: str,
        **kwargs,
    ) -> str:
        """
        Send a text prompt to Groq (Llama models) and return the response text.
        Supports OpenAI-style chat completions.
        """
        self._init_client()
        if not self.client:
            raise ValueError("Groq client not initialized. Call validate_key() first.")

        # Build messages dynamically (fallback if only prompt is given)
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
            raise RuntimeError(f"Groq query failed: {e}")

    def list_models(self) -> List[str]:
        """Return a curated list of supported Groq models."""
        return [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
        ]

    def validate_key(self) -> bool:
        """
        Validate the provided Groq API key by making a lightweight request.
        """
        try:
            self._init_client()
            if not self.client:
                raise ValueError("Groq client not initialized.")
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except Exception:
            return False
