from typing import Dict, List, Optional, Type

from plotsense.core.providers.anthropic import AnthropicProvider
from plotsense.core.providers.azure_openai import AzureOpenAIProvider
from plotsense.core.providers.base import LLMProvider
from plotsense.core.providers.gemini import GeminiProvider
from plotsense.core.providers.ollama_openai import OllamaProvider
from plotsense.core.providers.openai_chat import OpenAIChatProvider
from plotsense.core.utils import prompt_for_api_key
from plotsense.core.registry_loader import get_registry_loader
from .groq import GroqProvider
from .groq_openai import GroqOpenAIProvider
from .openai_response import OpenAIResponseProvider


class ProviderManager:
    """Manages multiple LLM providers, their API keys, and interactions."""

    SUPPORTED_PROVIDERS: Dict[str, Dict[str, Type[LLMProvider]]] = {
        "groq": {
            "default": GroqProvider,
            "openai": GroqOpenAIProvider,
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
        selected_providers: Optional[List[str]] = None
    ):
        """Initialize ProviderManager.

        Args:
            api_keys: Dict of available API keys (intent: supply credentials only, not selection)
            interactive: Whether to prompt for missing keys of selected providers
            selected_providers: List of provider names to initialize (intent: determines which providers to use)

        Design principle:
        - selected_providers determines WHICH providers to initialize
        - api_keys only provides credentials (if available)
        - Unselected providers are never initialized, validated, or prompted for
        """
        # Filter out None/empty values from api_keys to avoid treating them as "no credentials"
        self.api_keys = {k: v for k, v in (api_keys or {}).items() if v} if api_keys else {}
        self.interactive = interactive
        self.providers = {}
        self.selected_providers = set(selected_providers) if selected_providers else None

        # Validate selected_providers list
        if self.selected_providers:
            invalid = [p for p in self.selected_providers if p not in self.SUPPORTED_PROVIDERS]
            if invalid:
                raise ValueError(
                    f"Unsupported provider(s): {invalid}. "
                    f"Supported providers: {list(self.SUPPORTED_PROVIDERS.keys())}"
                )

        self._init_providers()

    def _init_provider_variants(self, vendor_name: str, api_key: str):
        """Initialize all variants of a provider."""
        variants = self.SUPPORTED_PROVIDERS[vendor_name]
        for variant_name, provider_cls in variants.items():
            full_name = f"{vendor_name}_{variant_name}"
            provider = provider_cls(api_key=api_key)

            try:
                if provider.validate_key():
                    print(f"✅ {full_name.upper()} API key validated successfully.")
                    self.providers[full_name] = provider
                else:
                    print(f"❌ {full_name.upper()} API key invalid or unverified.")
            except Exception as e:
                print(f"⚠️  Error validating {full_name.upper()} API key: {e}")

    def _init_providers(self):
        """Initialize selected providers.

        DESIGN PRINCIPLE: selected_providers determines WHAT to initialize.
        api_keys only provides credentials. Interactive prompting happens for selected
        providers with missing keys.
        """
        # Determine which providers to initialize
        if self.selected_providers:
            providers_to_init = self.selected_providers
        else:
            # No selection: initialize all providers with available keys (fallback)
            providers_to_init = set(self.api_keys.keys())
            if not providers_to_init:
                raise ValueError(
                    "No providers selected and no API keys provided. "
                    "Pass selected_providers or provide api_keys."
                )

        # Initialize each selected provider
        for vendor_name in providers_to_init:
            if vendor_name not in self.SUPPORTED_PROVIDERS:
                raise ValueError(f"Unknown provider: {vendor_name}")

            api_key = self.api_keys.get(vendor_name)
            if not api_key:
                api_key = self._get_api_key_for_provider(vendor_name)

            if not api_key or not isinstance(api_key, str) or not api_key.strip():
                raise ValueError(
                    f"API key required for selected provider '{vendor_name}'. "
                    f"Pass it via api_keys dict or provide it interactively."
                )

            self._init_provider_variants(vendor_name, api_key)

    def _get_api_key_for_provider(self, vendor_name: str) -> Optional[str]:
        """Get API key for a provider: from dict, or prompt if interactive.

        Args:
            vendor_name: Name of the provider

        Returns:
            API key string, or None if user skips in interactive mode

        Raises:
            ValueError if key missing and not interactive
        """
        # Already have it
        if vendor_name in self.api_keys:
            return self.api_keys[vendor_name]

        if not self.interactive:
            raise ValueError(
                f"API key required for provider '{vendor_name}' but not provided. "
                f"Pass it via api_keys dict or enable interactive mode."
            )

        # Prompt interactively
        provider_cls = self.SUPPORTED_PROVIDERS[vendor_name].get("default")
        if not provider_cls:
            # Fallback to first variant if no default
            provider_cls = next(iter(self.SUPPORTED_PROVIDERS[vendor_name].values()))

        link = getattr(provider_cls, "LINK", f"https://{vendor_name}.com")

        return prompt_for_api_key(
            vendor_name, link,
            interactive=self.interactive,
            skip_if_missing=False  # For selected providers, skip is not allowed
        )

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

        Loads from registry, so updates don't require package release.
        This helps CostOptimizedStrategy prioritize cheaper models.
        """
        registry = get_registry_loader()
        return registry.get_model_costs()

    def get_model_performance(self) -> Dict[str, float]:
        """
        Return approximate relative performance scores for each model.

        Loads from registry, so updates don't require package release.
        Higher means better performance (accuracy, reasoning ability, etc.).
        """
        registry = get_registry_loader()
        return registry.get_model_performance()
