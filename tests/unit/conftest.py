"""
Unit test specific fixtures and configuration.

Unit tests are fast, isolated, and don't require external resources.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, PropertyMock, patch


@pytest.fixture
def simple_numeric_dataframe():
    """Small DataFrame with simple numeric data for unit tests."""
    n = 10
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.arange(n),
        'y': np.random.rand(n),
        'value': np.random.uniform(1, 100, n),
        'count': np.random.randint(1, 10, n),
    })
