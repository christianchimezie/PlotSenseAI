from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from typing import Dict, List, Optional, Tuple

from plotsense.core.enums.strategy import StrategyName
from plotsense.core.strategies.round_robin import RoundRobinStrategy
from plotsense.core.strategies.cost_optimized import CostOptimizedStrategy
from plotsense.core.strategies.performance_optimized import PerformanceOptimizedStrategy
from plotsense.core.strategies.fallback_chain import FallbackChainStrategy


class AIModelInterface:
    """
    Handles all low-level interactions with LLM providers.
    Acts as a bridge between PlotExplainer (or any client)
    and ProviderManager.
    """

    def __init__(self, provider_manager, timeout: int = 30):
        self.manager = provider_manager
        self.timeout = timeout

    def _init_strategy(
        self, strategy_name: StrategyName,
        available_models: List[Tuple[str, str]]
    ):
        try:
            strategy_name = StrategyName(strategy_name)
        except ValueError:
            raise ValueError(f"Invalid strategy name: {strategy_name}")

        if strategy_name == StrategyName.ROUND_ROBIN:
            return RoundRobinStrategy(available_models)
        elif strategy_name == StrategyName.COST_OPTIMIZED:
            return CostOptimizedStrategy(available_models, self.manager)
        elif strategy_name == StrategyName.PERFORMANCE:
            return PerformanceOptimizedStrategy(available_models, self.manager)
        elif strategy_name == StrategyName.FALLBACK_CHAIN:
            return FallbackChainStrategy(available_models)

    def query_all_models(
        self,
        prompt: str,
        debug: bool = False,
        base64_image: Optional[str] = None,
        custom_parameters: Optional[Dict] = None,
        strategy: StrategyName = StrategyName.ROUND_ROBIN,
        max_workers: int = 6,
    ) -> Dict[str, str]:
        """
        Query all available models (across all providers) in parallel.
        Uses ThreadPoolExecutor for concurrency.
        Returns a mapping of "provider:model" -> response_text.

        Notes:
        - Keeps strategy initialization for compatibility (strategy instance
          can be used later to reorder or filter models).
        - Each model is queried independently; failures don't stop the rest.
        """
        # Get available models as list of tuples: [(provider, model_name), ...]
        all_models = self.manager.list_all_models()
        self.available_models = [
            (provider, model)
            for provider, models in all_models.items()
            for model in models
        ]
        if not self.available_models:
            raise ValueError("No available models found from provider manager.")

        # Initialize strategy instance (keeps previous behavior)
        strategy_instance = self._init_strategy(
            strategy, self.available_models
        )

        results: Dict[str, str] = {}

        # Get models to query from strategy
        models_to_query = self._get_models_to_query(strategy_instance, debug)

        if not models_to_query:
            raise ValueError("Strategy did not return any models to query.")

        # Query models based on strategy type
        if isinstance(strategy_instance, FallbackChainStrategy):
            results = self._query_sequential(models_to_query, prompt, base64_image, custom_parameters)
        else:
            results = self._query_parallel(models_to_query, prompt, base64_image, custom_parameters, max_workers)

        return results

    def _get_models_to_query(self, strategy_instance, debug: bool):
        """Get models to query from strategy, with fallback."""
        try:
            models_to_query = strategy_instance.select_model(len(self.available_models))
        except AttributeError:
            # Fallback: strategy not yet implementing selection
            models_to_query = self.available_models

        if debug:
            print(f"\n[DEBUG] Strategy '{strategy_instance.__class__.__name__}' selected models:")
            for prov, mod in models_to_query:
                print(f"  - {prov}:{mod}")

        return models_to_query

    def _query_one(self, provider: str, model_name: str, prompt: str, base64_image: Optional[str], custom_parameters: Optional[Dict]):
        """Query a single model and handle errors."""
        model_key = f"{provider}:{model_name}"
        try:
            resp = self.query_model(
                provider=provider,
                model=model_name,
                prompt=prompt,
                base64_image=base64_image,
                custom_parameters=custom_parameters,
            )
            return model_key, resp
        except Exception as e:
            warnings.warn(f"[AIModelInterface] Query failed for {model_key} -> {e}")
            return model_key, f"Error: {e}"

    def _query_sequential(self, models_to_query, prompt: str, base64_image: Optional[str], custom_parameters: Optional[Dict]) -> Dict[str, str]:
        """Query models sequentially (for FallbackChainStrategy)."""
        results = {}
        for provider, model_name in models_to_query:
            key, resp = self._query_one(provider, model_name, prompt, base64_image, custom_parameters)
            results[key] = resp
            if not resp.lower().startswith("error"):
                # Stop at first success (fallback semantics)
                break
        return results

    def _query_parallel(
        self,
        models_to_query,
        prompt: str,
        base64_image: Optional[str],
        custom_parameters: Optional[Dict],
        max_workers: int,
    ) -> Dict[str, str]:
        """Query models in parallel using ThreadPoolExecutor."""
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(self._query_one, provider, model_name, prompt, base64_image, custom_parameters): (provider, model_name)
                for provider, model_name in self.available_models
            }

            for future in as_completed(future_to_key):
                key, resp = future.result()
                results[key] = resp

        return results

    def _route_vendor_query(
        self,
        vendor: str,
        provider: str,
        model: str,
        messages: List,
        prompt: str,
        base64_image: Optional[str],
        generation_params: Dict
    ) -> str:
        """Route query to appropriate vendor handler."""
        if vendor == "openai":
            if "chat" in provider.lower():
                return self.manager.query(
                    provider,
                    model=model,
                    messages=messages,
                    prompt=prompt,
                    **generation_params,
                )
            elif "response" in provider.lower():
                return self.manager.query(
                    provider,
                    model=model,
                    prompt=prompt,
                    **generation_params,
                )
        elif vendor == "azure":
            return self.manager.query(
                provider,
                model=model,
                messages=messages,
                prompt=prompt,
                **generation_params,
            )
        elif vendor == "groq":
            return self.manager.query(
                provider,
                model=model,
                messages=messages,
                prompt=prompt,
                **generation_params,
            )
        elif vendor == "anthropic":
            return self.manager.query(
                provider,
                model=model,
                messages=messages,
                prompt=prompt,
                **generation_params,
            )
        elif vendor == "gemini":
            return self.manager.query(
                provider,
                model=model,
                messages=messages,
                prompt=prompt,
                image=base64_image,
                **generation_params,
            )
        elif vendor == "ollama":
            return self.manager.query(
                provider,
                model=model,
                prompt=prompt,
                image=base64_image,
                **generation_params,
            )
        else:
            print(f"[AIModelInterface] Warning: Using default query handling for {provider}:{model}")
            return self.manager.query(
                provider,
                model=model,
                messages=messages,
                prompt=prompt,
                **generation_params,
            )

    def query_model(
        self,
        provider: str,
        model: str,
        prompt: str,
        base64_image: Optional[str] = None,
        custom_parameters: Optional[Dict] = None
    ) -> str:
        """
        Query a model via the provider manager.
        Handles provider-specific formatting and error management.

        Provider names follow vendor_variant format (e.g., groq_default, openai_chat).
        We extract the vendor to determine message formatting.
        """
        if provider not in self.manager.providers:
            raise ValueError(f"Unknown provider: {provider}")

        try:
            vendor = provider.split("_")[0].lower()
            messages = self._build_messages(vendor, model, prompt, base64_image)
            generation_params = {"temperature": 0.7, "max_tokens": 1000, **(custom_parameters or {})}
            return self._route_vendor_query(
                vendor, provider, model, messages, prompt, base64_image, generation_params
            )
        except Exception as e:
            warnings.warn(f"[AIModelInterface] Querying error for {provider}:{model} -> {str(e)}")
            return f"Error: {e}"

    def _calculate_base_weight(self, model_name: str, provider: str) -> float:
        """Calculate base weight for a model based on name and provider."""
        lname = model_name.lower()
        lprov = provider.lower()

        if "gpt-4" in lname or "gpt4" in lname or "gpt-4o" in lname:
            return 2.0
        elif "claude" in lname:
            return 1.8
        elif "gemini" in lname or "gemini-pro" in lname:
            return 1.6
        elif "llama" in lname or "groq" in lprov:
            return 1.2
        elif "azure" in lprov:
            return 1.8 if "gpt-4" in lname or "gpt4" in lname else 1.1
        elif "ollama" in lprov:
            return 1.0
        return 1.0

    def get_model_weights(self) -> Dict[str, float]:
        """
        Return model weighting for ensemble scoring.

        Weighting strategy (default heuristics):
          - OpenAI GPT-4 variants       -> higher weight (2.0)
          - Anthropic Claude family     -> high weight (1.8)
          - Google Gemini               -> high weight (1.6)
          - Azure (OpenAI in Azure)     -> treated similar to openai (1.8 for gpt-4)
          - Groq (Llama variants)       -> moderate weight (1.2)
          - Ollama / local models       -> lower/moderate weight (1.0)
          - Other / unknown             -> base weight (1.0)

        Returns:
            dict of "provider:model" -> normalized_weight
        """
        all_models = self.manager.list_all_models()
        self.available_models = [
            (provider, model)
            for provider, models in all_models.items()
            for model in models
        ]

        raw_weights: Dict[str, float] = {}

        for provider, model_name in self.available_models:
            key = f"{provider}:{model_name}"
            base = self._calculate_base_weight(model_name, provider)
            raw_weights[key] = base

        # Normalize to sum to 1
        total = sum(raw_weights.values()) or 1.0
        normalized = {k: (v / total) for k, v in raw_weights.items()}
        return normalized

    def _messages_openai_azure(self, model: str, prompt: str, base64_image: Optional[str]) -> List:
        """Build messages for OpenAI/Azure models."""
        model_lower = model.lower()
        if base64_image and any(tag in model_lower for tag in ["gpt-4o", "gpt-4-turbo", "gpt-4-vision"]):
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ]
        return [
            {"role": "system", "content": "You are a helpful data visualization assistant."},
            {"role": "user", "content": prompt},
        ]

    def _messages_anthropic(self, prompt: str, base64_image: Optional[str]) -> List:
        """Build messages for Anthropic Claude models."""
        if base64_image:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                    ],
                }
            ]
        return [{"role": "user", "content": prompt}]

    def _messages_gemini(self, prompt: str, base64_image: Optional[str]) -> List:
        """Build messages for Gemini models."""
        if base64_image:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "data": base64_image, "mime_type": "image/jpeg"},
                    ],
                }
            ]
        return [{"role": "user", "content": prompt}]

    def _messages_ollama(self, prompt: str, base64_image: Optional[str]) -> List:
        """Build messages for Ollama models."""
        if base64_image:
            return [{"role": "user", "content": f"{prompt}\n\n[Image attached as base64 input]"}]
        return [{"role": "user", "content": prompt}]

    def _messages_default(self, prompt: str, base64_image: Optional[str]) -> List:
        """Build messages for unknown vendors."""
        if base64_image:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ]
        return [{"role": "user", "content": prompt}]

    def _build_messages(
        self, vendor: str, model: str, prompt: str,
        base64_image: Optional[str] = None
    ):
        """
        Build messages dynamically based on vendor capabilities.
        Supports multimodal input where possible (OpenAI GPT-4o, Gemini, Anthropic, etc.).
        Falls back to text-only prompt for vendors without image support.

        Args:
            vendor: The vendor name (e.g., "groq", "openai", "anthropic")
                   Extracted from full provider names like "groq_default", "openai_chat"
            model: The model name
            prompt: The text prompt
            base64_image: Optional base64-encoded image
        """
        vendor_lower = vendor.lower()

        if vendor_lower in {"openai", "azure"}:
            return self._messages_openai_azure(model, prompt, base64_image)
        elif vendor_lower == "anthropic":
            return self._messages_anthropic(prompt, base64_image)
        elif vendor_lower == "gemini":
            return self._messages_gemini(prompt, base64_image)
        elif vendor_lower == "groq":
            return [{"role": "user", "content": prompt}]
        elif vendor_lower == "ollama":
            return self._messages_ollama(prompt, base64_image)
        else:
            return self._messages_default(prompt, base64_image)
