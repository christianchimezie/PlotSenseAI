from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

class Strategy(ABC):
    """
    Base Strategy interface for selecting provider/model pairs.
    Each strategy returns a tuple: (provider_name, model_name)
    """

    def __init__(self, provider_models: List[Tuple[str, str]]):
        """
        Args:
            provider_models: dict mapping provider_name -> list of models
        """
        self.provider_models = provider_models

    @abstractmethod
    def select_model(self, iteration: int, current_explanation: Optional[str] = None) -> Tuple[str, str]:
        """
        Return a (provider, model) tuple for the given iteration.

        Args:
            iteration: current iteration index (0-based)
            current_explanation: optionally, the current explanation for refinement

        Returns:
            (provider_name, model_name)
        """
        pass

