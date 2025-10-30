from typing import List
from openai import OpenAI
# AzureOpenAI,
from openai.types.chat import ChatCompletionUserMessageParam
from .base import LLMProvider


class AzureOpenAIProvider(LLMProvider):
    """Provider integration for Azure-hosted OpenAI models."""

    LINK = "👉 https://portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub/feature/OpenAI 👈"

    def __init__(
        self, api_key: str,
        endpoint: str = "https://models.github.ai/inference",
        api_version: str = "2024-02-15-preview"
    ):
        """
        Args:
            api_key: Azure OpenAI API key
            endpoint: Full Azure endpoint (e.g. https://<your-resource>.openai.azure.com/)
            api_version: Azure OpenAI API version
        """
        self.api_key = api_key
        self.endpoint = endpoint
        # self.api_version = api_version
        self.client = None

    def _init_client(self):
        """Initialize the Azure OpenAI client."""
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint not provided.")
        if not self.client:
            self.client = OpenAI(
                api_key=self.api_key,
                # api_version=self.api_version,
                # azure_endpoint=self.endpoint,
                base_url=self.endpoint,
            )

    def query(self, prompt: str, model: str, **kwargs) -> str:
        """
        Send a prompt to Azure OpenAI Chat Completion API.
        """
        self._init_client()
        if not self.client:
            raise ValueError("AzureOpenAI client not initialized. Call validate_key() first.")

        # Ensure messages format exists in kwargs
        messages: list[ChatCompletionUserMessageParam] = kwargs.pop("messages", None)
        if not messages and prompt:
            messages = [{"role": "user", "content": prompt}]
        elif not messages and not prompt:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        try:
            if "max_tokens" in kwargs:
                kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI query failed: {e}")

    def list_models(self) -> List[str]:
        """
        Return a suggested list of Azure OpenAI deployable model names.
        (These must match your deployment names in Azure.)
        """
        return [
            "openai/gpt-5",
            # "gpt-4o",
            # "gpt-4-turbo",
            # "gpt-35-turbo",
            # "gpt-4",
        ]

    def validate_key(self) -> bool:
        """
        Attempt a lightweight ping to validate Azure OpenAI credentials.
        """
        try:
            self._init_client()
            if not self.client:
                raise ValueError("AzureOpenAI client not initialized. Call validate_key() first.")
            response = self.client.chat.completions.create(
                model="openai/gpt-5",
                messages=[{"role": "user", "content": "ping"}],
                max_completion_tokens=5
            )
            return bool(response)
        except Exception as e:
            print(f"⚠️ Azure OpenAI API key validation failed: {e}")
            return False

