from typing import Dict, List, Optional, Type

from plotsense.core.providers.anthropic import AnthropicProvider
from plotsense.core.providers.azure_openai import AzureOpenAIProvider
from plotsense.core.providers.base import LLMProvider
from plotsense.core.providers.gemini import GeminiProvider
from plotsense.core.providers.ollama_openai import OllamaProvider
from plotsense.core.providers.openai_chat import OpenAIChatProvider
from plotsense.core.utils import prompt_for_api_key
from .groq import GroqProvider
from .openai_response import OpenAIResponseProvider


class ProviderManager:
    """Manages multiple LLM providers, their API keys, and interactions."""

    SUPPORTED_PROVIDERS: Dict[str, Dict[str, Type[LLMProvider]]] = {
        "groq": {
            "default": GroqProvider,
        },
        "openai": {
            "chat": OpenAIChatProvider,
            "response": OpenAIResponseProvider,
        },
        "anthropic": {
            "default": AnthropicProvider,
        },
        "gemini": {
            "default": GeminiProvider,
        },
        "azure": {
            "default": AzureOpenAIProvider,
        },
        "ollama": {
            "default": OllamaProvider,
        },
    }

    def __init__(
        self, api_keys: Dict[str, str], interactive: bool = True,
        restrict_to: Optional[List[str]] = None
    ):
        self.api_keys = api_keys or {}
        self.interactive = interactive
        self.providers = {}
        self.restrict_to = set(restrict_to) if restrict_to else None

        # Normalize restrict_to list
        if restrict_to:
            invalid = [p for p in restrict_to if p not in self.SUPPORTED_PROVIDERS]
            if invalid:
                raise ValueError(
                    f"Unsupported provider(s): {invalid}. "
                    f"Supported providers: {list(self.SUPPORTED_PROVIDERS.keys())}"
                )
            self.restrict_to = set(restrict_to)
        else:
            self.restrict_to = None

        self._init_providers()

    def _init_providers(self):
        """Initialize all registered providers and validate their API keys."""
        for vendor_name, variants in self.SUPPORTED_PROVIDERS.items():
            # Skip if restrict_to is provided and this vendor isn’t included
            if self.restrict_to and vendor_name not in self.restrict_to:
                continue

            for variant_name, provider_cls in variants.items():
                full_name = f"{vendor_name}_{variant_name}"
                link = getattr(provider_cls, "LINK", f"https://{vendor_name}.com")

                api_key: Optional[str] = self.api_keys.get(vendor_name)
                if not api_key:
                    # Try to prompt only if interactive and not restricted
                    api_key = prompt_for_api_key(
                        vendor_name,
                        link,
                        self.interactive,
                        skip_if_missing=bool(self.restrict_to),
                    )
                    if not api_key:
                        # Skip this provider if key is still missing
                        print(f"⏩ Skipping {full_name.upper()} (no API key provided).")
                        continue

                    self.api_keys[vendor_name] = api_key

                if not isinstance(api_key, str) or not api_key.strip():
                    print(f"⚠️ Skipping {full_name.upper()} due to invalid API key format.")
                    continue

                provider = provider_cls(api_key=api_key)

                try:
                    if provider.validate_key():
                        print(f"✅ {full_name.upper()} API key validated successfully.")
                        self.providers[full_name] = provider
                    else:
                        print(f"❌ {full_name.upper()} API key invalid or unverified.")
                except Exception as e:
                    print(f"⚠️  Error validating {full_name.upper()} API key: {e}")

    def get_provider(self, vendor_name: str, variant_name: str = ""):
        """
        Get or initialize a provider (with optional variant) on demand.

        Args:
            vendor_name: Name of the AI provider (e.g., "openai", "groq")
            variant_name: Optional variant name (e.g., "chat", "completion")

        Returns:
            Initialized provider instance
        """
        # Compose a unique key for storage
        full_name = f"{vendor_name}_{variant_name}" if variant_name else vendor_name

        if vendor_name not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unknown provider: {vendor_name}")

        if full_name not in self.providers:
            variants = self.SUPPORTED_PROVIDERS[vendor_name]

            # Determine class safely
            provider_cls = None
            if variant_name:
                provider_cls = variants.get(variant_name)
                if not provider_cls:
                    raise ValueError(
                        f"Unknown variant '{variant_name}' for provider '{vendor_name}'"
                    )
            else:
                variant_name, provider_cls = next(iter(variants.items()))
                full_name = f"{vendor_name}_{variant_name}"

            link = getattr(provider_cls, "LINK", f"https://{vendor_name}.com")

            api_key: Optional[str] = self.api_keys.get(vendor_name)
            if not api_key:
                api_key = prompt_for_api_key(vendor_name, link, self.interactive)
                if not api_key:
                    raise ValueError(f"Missing API key for {vendor_name}")
                self.api_keys[vendor_name] = api_key

            # if not isinstance(api_key, str):
            #     raise TypeError(f"API key for {vendor_name} must be a string")

            provider = provider_cls(api_key=api_key)

            try:
                if provider.validate_key():
                    print(f"✅ {full_name.upper()} API key validated successfully.")
                else:
                    print(f"❌ {full_name.upper()} API key invalid or unverified.")
            except Exception as e:
                print(f"⚠️ Error validating {full_name.upper()} API key: {e}")

            self.providers[full_name] = provider

        return self.providers[full_name]

    def list_all_models(self):
        all_models = {}
        for name, provider in self.providers.items():
            try:
                all_models[name] = provider.list_models()
            except Exception as e:
                print(f"⚠️  Failed to list models for {name}: {str(e)}")
        return all_models

    def query(self, provider_name: str, model: str, prompt: str, **kwargs):
        """Query a specific provider with a prompt and model."""
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not initialized.")
        return provider.query(prompt, model, **kwargs)

    def get_model_costs(self) -> Dict[str, float]:
        """
        Return a global map of model names to approximate per-request cost multipliers.
        This helps CostOptimizedStrategy prioritize cheaper models.
        """
        # In a real system, this could come from provider-specific metadata
        return {
            # OpenAI
            "gpt-4o-mini": 0.01,
            "gpt-4o": 0.03,
            "gpt-4-turbo": 0.025,
            "gpt-3.5-turbo": 0.008,
            # Groq (Llama)
            "llama-3.1-8b-instant": 0.005,
            "llama-3.3-70b-versatile": 0.02,
            # Anthropic
            "claude-3-haiku": 0.009,
            "claude-3-sonnet": 0.02,
            "claude-3-opus": 0.05,
            # Gemini
            "gemini-1.5-flash": 0.006,
            "gemini-1.5-pro": 0.02,
            # Azure (proxy to GPT costs)
            "azure-gpt-4o-mini": 0.011,
            "azure-gpt-4o": 0.031,
            # Ollama (local = near-zero cost)
            "llama3": 0.001,
            "mistral": 0.002,
        }

    def get_model_performance(self) -> Dict[str, float]:
        """
        Return approximate relative performance scores for each model.
        Higher means better performance (accuracy, reasoning ability, etc.).
        """
        return {
            # OpenAI
            "gpt-4o": 10.0,
            "gpt-4o-mini": 8.5,
            "gpt-4-turbo": 9.5,
            "gpt-3.5-turbo": 7.5,

            # Anthropic
            "claude-3-opus": 9.8,
            "claude-3-sonnet": 9.0,
            "claude-3-haiku": 7.0,

            # Groq
            "llama-3.3-70b-versatile": 8.8,
            "llama-3.1-8b-instant": 6.5,

            # Gemini
            "gemini-1.5-pro": 9.3,
            "gemini-1.5-flash": 7.8,

            # Azure (maps to OpenAI)
            "azure-gpt-4o": 9.8,
            "azure-gpt-4o-mini": 8.3,

            # Ollama (local models)
            "mistral": 6.0,
            "llama3": 6.8,
        }

