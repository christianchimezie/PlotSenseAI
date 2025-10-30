from typing import List, Optional, Tuple
from plotsense.core.strategies.strategy import Strategy

class FallbackChainStrategy(Strategy):
    """Try providers/models in fixed order until one succeeds."""

    def __init__(self, provider_models: List[Tuple[str, str]]):
        super().__init__(provider_models)
        # Deterministic order; could later be made configurable
        self.model_list = provider_models
        self._last_success_index = 0

    def select_model(
        self, iteration: int, current_explanation: Optional[str] = None
    ) -> Tuple[str, str]:
        """If previous success exists, keep using it; otherwise go to next."""
        if not self.model_list:
            raise ValueError("No models available in strategy.")
        index = min(iteration, len(self.model_list) - 1)
        return self.model_list[index]

    def report_success(self, index: int):
        """Optionally record which model last succeeded."""
        self._last_success_index = index
