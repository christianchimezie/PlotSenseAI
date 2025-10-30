from typing import Dict, List, Optional, Tuple
from plotsense.core.strategies.strategy import Strategy

MODEL_PERFORMANCE_MAP = {
    "gpt-4o": 10,
    "gpt-4o-mini": 8,
    "llama-3.3-70b-versatile": 9,
    "llama-3.1-8b-instant": 6,
}

class PerformanceOptimizedStrategy(Strategy):
    """Prefer highest-performance models first."""

    def __init__(
        self, provider_models: List[Tuple[str, str]], provider_manager
    ):
        super().__init__(provider_models)

        # Get dynamic performance scores from ProviderManager
        self.performance_map: Dict[str, float] = provider_manager.get_model_performance()

        # Sort models descending by performance score
        self.model_list = sorted(
            provider_models,
            key=lambda p_m: self.performance_map.get(p_m[1], 0),
            reverse=True,
        )

    def select_models(self, n: int) -> List[Tuple[str, str]]:
        """Return the top `n` highest-performing models."""
        return self.model_list[:n]

    def select_model(
        self, iteration: int, current_explanation: Optional[str] = None
    ) -> Tuple[str, str]:
        """Start from best model; fallback to lower-tier ones if needed."""
        if not self.model_list:
            raise ValueError("No models available in strategy.")
        index = min(iteration, len(self.model_list) - 1)
        return self.model_list[index]
