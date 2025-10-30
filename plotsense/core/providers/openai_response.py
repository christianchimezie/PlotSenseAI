from typing import List, Optional
from openai import OpenAI
from .base import LLMProvider


class OpenAIResponseProvider(LLMProvider):
    """Provider integration for OpenAI's Responses API."""

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
        Send a prompt to OpenAI's Responses API and return the generated text.
        The Responses endpoint supports text, image, and JSON outputs.
        """
        self._init_client()
        if not self.client:
            raise ValueError("OpenAI client not initialized. Call validate_key() first.")

        if not prompt:
            raise ValueError("'prompt' must be provided for Responses API queries.")

        try:
            if "max_tokens" in kwargs:
                kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
            # The Responses API expects `input`, not `messages`
            response = self.client.responses.create(
                model=model,
                input=prompt,
                **kwargs,
            )

            # The unified field for plain text output is `output_text`
            return getattr(response, "output_text", "") or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI response query failed: {e}")

    def list_models(self) -> List[str]:
        """Return a curated list of supported OpenAI Response models."""
        return [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
        ]

    def validate_key(self) -> bool:
        """
        Validate the provided OpenAI API key by performing a lightweight test query.
        """
        try:
            self._init_client()
            if not self.client:
                raise ValueError("OpenAI client not initialized.")
            response = self.client.responses.create(
                model="gpt-4.1-mini",
                input="ping",
                max_output_tokens=16,
            )
            return bool(getattr(response, "output_text", ""))
        except Exception as e:
            print(f"OpenAI Responses API key validation failed: {e}")
            return False

