from typing import Dict, List, Optional, Tuple
from plotsense.core.strategies.strategy import Strategy


class CostOptimizedStrategy(Strategy):
    """Prioritize cheaper models first, fallback to pricier if needed."""

    def __init__(self, provider_models: List[Tuple[str, str]], provider_manager):
        super().__init__(provider_models)

        self.cost_map: Dict[str, float] = provider_manager.get_model_costs()

        # Sort models by ascending cost (lowest first)
        self.model_list = sorted(
            provider_models, 
            key=lambda p_m: self.cost_map.get(p_m[1], float("inf"))
        )

    def select_models(self, n: int) -> List[Tuple[str, str]]:
        """
        Return the top `n` cheapest models.
        """
        return self.model_list[:n]

    def select_model(
        self, iteration: int, current_explanation: Optional[str] = None
    ) -> Tuple[str, str]:
        """Use the cheapest available model, escalate if iteration increases."""
        if not self.model_list:
            raise ValueError("No models available in strategy.")
        index = min(iteration, len(self.model_list) - 1)
        return self.model_list[index]

