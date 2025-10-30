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

        # --- 1️⃣ Let strategy select or order models ---
        # Most strategies (RoundRobin, CostOptimized, etc.) will implement a method
        # like `.select_models(n: int)` or `.get_next_batch()`.
        # If not, we simply use all available models.
        try:
            # Example interface: select_models returns a prioritized list
            models_to_query = strategy_instance.select_model(len(self.available_models))
        except AttributeError:
            # Fallback: strategy not yet implementing selection
            models_to_query = self.available_models

        if not models_to_query:
            raise ValueError("Strategy did not return any models to query.")

        if debug:
            print(f"\n[DEBUG] Strategy '{strategy_instance.__class__.__name__}' selected models:")
            for prov, mod in models_to_query:
                print(f"  - {prov}:{mod}")

        def _query_one(provider: str, model_name: str):
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

        # FallbackChainStrategy -> sequential queries until one succeeds
        if isinstance(strategy_instance, FallbackChainStrategy):
            for provider, model_name in models_to_query:
                key, resp = _query_one(provider, model_name)
                results[key] = resp
                if not resp.lower().startswith("error"):
                    # Stop at first success (fallback semantics)
                    break
        else:
            # Run queries concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {
                    executor.submit(_query_one, provider, model_name): (provider, model_name)
                    for provider, model_name in self.available_models
                }

                for future in as_completed(future_to_key):
                    key, resp = future.result()
                    results[key] = resp

        return results

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
        """
        if provider not in self.manager.providers:
            raise ValueError(f"Unknown provider: {provider}")

        try:
            # Build messages depending on provider/model
            messages = self._build_messages(
                provider, model, prompt, base64_image
            )
            generation_params = {"temperature": 0.7, "max_tokens": 1000, **(custom_parameters or {})}

            provider_lower = provider.lower()
            # model_lower = model.lower()

            # -------------------- OPENAI (Chat + Response) --------------------
            if "openai" in provider_lower:
                # if "gpt" in model_lower or "chat" in model_lower:
                if "chat" in provider_lower:
                    # Chat-based models (GPT-4, GPT-3.5, etc.)
                    return self.manager.query(
                        provider,
                        model=model,
                        messages=messages,
                        prompt=prompt,
                        **generation_params,
                    )
                elif "response" in provider_lower:
                    # Response-based models (completion endpoints)
                    return self.manager.query(
                        provider,
                        model=model,
                        prompt=prompt,
                        **generation_params,
                    )

            # -------------------- AZURE OPENAI --------------------
            elif "azure" in provider_lower:
                # Azure follows OpenAI's API style but requires deployment-specific name
                return self.manager.query(
                    provider,
                    model=model,
                    messages=messages,
                    prompt=prompt,
                    **generation_params,
                )

            # -------------------- GROQ --------------------
            elif "groq" in provider_lower:
                # Typically text-only Llama-style models
                return self.manager.query(
                    provider,
                    model=model,
                    messages=messages,
                    prompt=prompt,
                    **generation_params,
                )

            # -------------------- ANTHROPIC --------------------
            elif "anthropic" in provider_lower:
                # Claude models (text + multimodal optional)
                return self.manager.query(
                    provider,
                    model=model,
                    messages=messages,
                    prompt=prompt,
                    **generation_params,
                )

            # -------------------- GEMINI --------------------
            elif "gemini" in provider_lower:
                # Supports text + images
                return self.manager.query(
                    provider,
                    model=model,
                    messages=messages,
                    prompt=prompt,
                    image=base64_image,
                    **generation_params,
                )

            # -------------------- OLLAMA --------------------
            elif "ollama" in provider_lower:
                # Local models; prompt only, may support images if model allows
                return self.manager.query(
                    provider,
                    model=model,
                    prompt=prompt,
                    image=base64_image,
                    **generation_params,
                )

            # -------------------- DEFAULT / UNKNOWN --------------------
            else:
                print(f"[AIModelInterface] Warning: Using default query handling for {provider}:{model}")
                # Fallback for new or custom providers
                return self.manager.query(
                    provider,
                    model=model,
                    messages=messages,
                    prompt=prompt,
                    **generation_params,
                )

        except Exception as e:
            warnings.warn(f"[AIModelInterface] Querying error for {provider}:{model} -> {str(e)}")
            return f"Error: {e}"
        finally:
            return f"Error: No valid query handler found for provider '{provider}'."

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
            lname = model_name.lower()
            lprov = provider.lower()

            # Base preference by model name
            if "gpt-4" in lname or "gpt4" in lname or "gpt-4o" in lname:
                base = 2.0
            elif "claude" in lname:
                base = 1.8
            elif "gemini" in lname or "gemini-pro" in lname:
                base = 1.6
            elif "llama" in lname or "groq" in lprov:
                # groq's Llama-based models - decent but not highest
                base = 1.2
            elif "azure" in lprov:
                # Azure OpenAI often runs OpenAI models; favor if contains gpt-4
                base = 1.8 if "gpt-4" in lname or "gpt4" in lname else 1.1
            elif "ollama" in lprov:
                base = 1.0
            else:
                base = 1.0

            # Provider-level adjustments (optional)
            if lprov == "anthropic":
                base *= 1.0  # already accounted by 'claude' checks
            if lprov == "openai":
                base *= 1.0
            if lprov == "groq":
                base *= 1.0
            if lprov == "azure":
                base *= 1.0

            raw_weights[key] = base

        # Normalize to sum to 1
        total = sum(raw_weights.values()) or 1.0
        normalized = {k: (v / total) for k, v in raw_weights.items()}
        return normalized

    def _build_messages(
        self, provider: str, model: str, prompt: str,
        base64_image: Optional[str] = None
    ):
        """
        Build messages dynamically based on provider capabilities.
        Supports multimodal input where possible (OpenAI GPT-4o, Gemini, Anthropic, etc.).
        Falls back to text-only prompt for providers without image support.
        """
        provider_lower = provider.lower()
        model_lower = model.lower()

        # --- 1️⃣ OpenAI / Azure (GPT-4, GPT-4o, GPT-3.5 etc.) ---
        if provider_lower in {"openai", "azure"}:
            if base64_image and any(tag in model_lower for tag in ["gpt-4o", "gpt-4-turbo", "gpt-4-vision"]):
                # Chat message with multimodal support
                return [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ]
            else:
                # Standard chat completion format
                return [
                    {"role": "system", "content": "You are a helpful data visualization assistant."},
                    {"role": "user", "content": prompt},
                ]

        # --- 2️⃣ Anthropic (Claude) ---
        elif provider_lower == "anthropic":
            # Claude supports multimodal via text + image blocks in messages
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
            else:
                return [
                    {"role": "user", "content": prompt}
                ]

        # --- 3️⃣ Gemini (Google) ---
        elif provider_lower == "gemini":
            # Gemini API supports multimodal via a combined structure
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
            else:
                return [
                    {"role": "user", "content": prompt}
                ]

        # --- 4️⃣ Groq (LLaMA / Mistral etc. – text-only) ---
        elif provider_lower == "groq":
            return [
                {"role": "user", "content": prompt}
            ]

        # --- 5️⃣ Ollama (local models; may support image, but prompt-based) ---
        elif provider_lower == "ollama":
            if base64_image:
                # Send inline text prompt mentioning image context
                return [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n[Image attached as base64 input]"
                    }
                ]
            else:
                return [
                    {"role": "user", "content": prompt}
                ]

        # --- 6️⃣ Default / Unknown provider fallback ---
        else:
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
            else:
                return [
                    {"role": "user", "content": prompt}
                ]

