from typing import List
from anthropic import Anthropic
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Provider integration for Anthropic's Claude models."""
    
    LINK = "👉 https://console.anthropic.com/account/keys 👈"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None

    def _init_client(self):
        """Initialize Anthropic client if not already created."""
        if not self.client:
            self.client = Anthropic(api_key=self.api_key)

    def query(self, prompt: str, model: str, **kwargs) -> str:
        """Send a message to Anthropic Claude model and return its response text."""
        if not self.client:
            raise ValueError(
                "Anthropic client not initialized. Call validate_key() first."
            )

        messages = kwargs.pop("messages", None)
        if not messages and prompt:
            # Default to a single user message
            messages = [{"role": "user", "content": prompt}]
        elif not messages and not prompt:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        try:
            response = self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.content[0].text if response and response.content else ""
        except Exception as e:
            raise RuntimeError(f"Anthropic query failed: {e}")

    def list_models(self) -> List[str]:
        """
        Return a list of supported Anthropic models.
        This list can be expanded as new Claude versions are released.
        """
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ]

    def validate_key(self) -> bool:
        """
        Validate the provided API key by performing a lightweight test.
        Returns True if successful, False otherwise.
        """
        try:
            self._init_client()
            if not self.client:
                raise ValueError(
                    "Anthropic client not initialized. Call validate_key() first."
                )
            # Perform a trivial, cheap call to verify authentication
            self.client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False

