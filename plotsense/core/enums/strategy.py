from enum import Enum

class StrategyName(str, Enum):
    ROUND_ROBIN = "round_robin"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE = "performance"
    FALLBACK_CHAIN = "fallback"

