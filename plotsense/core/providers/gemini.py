from typing import List, Optional
from google import genai
from google.genai.types import GenerateContentConfig
from .base import LLMProvider


class GeminiProvider(LLMProvider):
    """Provider integration for Google's Gemini models (v2 SDK)."""

    LINK = "👉 https://aistudio.google.com/app/apikey 👈"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None

    def _init_client(self):
        """Initialize Anthropic client if not already created."""
        if not self.client:
            self.client = genai.Client(api_key=self.api_key)

    def query(
        self,
        prompt: str,
        model: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Send a text (or multimodal) prompt to Gemini and return the response text.
        Supports text-only and text+image queries.

        Uses the new google-genai v2 API.
        """
        try:
            # Build input depending on image presence
            if base64_image:
                # Multimodal: send both text and image
                contents = [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image,
                        }
                    },
                ]
            else:
                # Text-only
                contents = prompt 

            self._init_client()
            if not self.client:
                raise ValueError("Gemini client initialization failed.")

            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                **kwargs,
            )

            # Return clean text or empty string if missing
            return getattr(response, "text", "") or ""

        except Exception as e:
            raise RuntimeError(f"Gemini query failed: {e}")

    def list_models(self) -> List[str]:
        """
        Return a curated list of Gemini models.
        """
        return [
            "gemini-2.5-flash",
            "gemini-2.0-pro",
            "gemini-1.5-flash",
        ]

    def validate_key(self) -> bool:
        """
        Validate the provided Gemini API key by attempting a trivial generation.
        """
        try:
            self._init_client()
            if not self.client:
                raise ValueError("Gemini client initialization failed.")
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents="ping",
                config=GenerateContentConfig(max_output_tokens=5),
            )
            return bool(response.text)
        except Exception:
            return False
