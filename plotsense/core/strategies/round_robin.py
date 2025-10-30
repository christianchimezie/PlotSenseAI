from typing import List, Optional, Tuple
from plotsense.core.strategies.strategy import Strategy

class RoundRobinStrategy(Strategy):
    """Cycle through all models evenly."""

    def __init__(self, provider_models: List[Tuple[str, str]]):
        super().__init__(provider_models)
        self.model_list = provider_models
        self._last_index = -1

    def select_model(
        self, iteration: int, current_explanation: Optional[str] = None
    ) -> Tuple[str, str]:
        if not self.model_list:
            raise ValueError("No models available in strategy.")
        # Pick model based directly on iteration count
        index = iteration % len(self.model_list)
        return self.model_list[index]
