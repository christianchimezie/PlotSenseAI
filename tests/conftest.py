"""
Shared pytest fixtures and configuration for PlotSense tests.

This file contains fixtures that are used across multiple test modules,
reducing code duplication and ensuring consistency.
"""

import os
import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, PropertyMock, patch
from PIL import Image
from dotenv import load_dotenv

# Use non-interactive backend for all tests
matplotlib.use('Agg')

# Load environment variables
load_dotenv()

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# ============================================================================
# Session-level fixtures (shared across all tests)
# ============================================================================


@pytest.fixture(scope="session")
def test_api_key():
    """Provide a test API key for testing."""
    return os.getenv('GROQ_API_KEY', 'test_key_placeholder')


# ============================================================================
# Data fixtures
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """
    Standard sample DataFrame for testing.
    Contains multiple data types: datetime, categorical, numeric, boolean.
    """
    n = 100
    np.random.seed(SEED)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n),
        "category": np.random.choice(list("ABCDE"), n),
        "value": np.random.uniform(1, 100, n),
        "count": np.random.randint(1, 100, n),
        "flag": np.random.choice([0, 1], n),
        "x": np.arange(n),
        "y": np.random.rand(n),
        "z": np.random.rand(n),
    })


@pytest.fixture
def small_dataframe():
    """Small DataFrame for quick tests."""
    n = 20
    np.random.seed(SEED)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n),
        "category": np.random.choice(list("ABCDE"), n),
        "value": np.random.uniform(1, 100, n),
        "count": np.random.randint(1, 100, n),
        "flag": np.random.choice([0, 1], n),
        "x": np.arange(n),
        "y": np.random.rand(n),
        "z": np.random.rand(n),
    })


@pytest.fixture
def large_dataframe():
    """Large DataFrame for performance/stress tests."""
    n = 1000
    np.random.seed(SEED)
    return pd.DataFrame({
        "x": np.arange(n),
        "y": np.random.rand(n),
        "value": np.random.normal(0, 1, n),
        "category": np.random.choice(list("ABCDE"), n),
        "count": np.random.randint(0, 100, n)
    })


@pytest.fixture
def sample_suggestions():
    """Sample visualization suggestions DataFrame."""
    return pd.DataFrame({
        'plot_type': ['scatter', 'bar', 'barh', 'hist', 'boxplot', 'violinplot', 'pie', 'hexbin'],
        'variables': ['x,y', 'value,category', 'value,category', 'value', 'value', 'value,category', 'category', 'x,y'],
        'ensemble_score': np.random.rand(8)
    })


# ============================================================================
# Plot fixtures
# ============================================================================

@pytest.fixture
def simple_plot():
    """Create a simple line plot for testing."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    yield ax
    plt.close(fig)


@pytest.fixture
def sample_plot(sample_dataframe):
    """Create a sample scatter plot with categorical coloring."""
    fig, ax = plt.subplots()
    sample_dataframe.plot.scatter(x='x', y='y', c='category', cmap='viridis', ax=ax)
    yield ax
    plt.close(fig)


@pytest.fixture
def scatter_plot(small_dataframe):
    """Create a scatter plot for testing."""
    fig, ax = plt.subplots()
    ax.scatter(small_dataframe['x'], small_dataframe['y'])
    yield ax
    plt.close(fig)


@pytest.fixture
def bar_plot():
    """Create a bar plot for testing."""
    fig, ax = plt.subplots()
    ax.bar(['A', 'B', 'C'], [3, 7, 2])
    yield ax
    plt.close(fig)


# ============================================================================
# Mock fixtures for external APIs
# ============================================================================

@pytest.fixture
def mock_groq_client():
    """Mock Groq client for testing without API calls."""
    with patch('groq.Groq') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_groq_completion():
    """Mock Groq API completion response."""
    mock_message = MagicMock()
    type(mock_message).content = PropertyMock(return_value="Mock explanation")

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def llm_dummy_response():
    """Sample LLM response for testing recommendation parsing."""
    return """
Plot Type: scatter
Variables: value, count
ensemble_score: 1.0
model_agreement: 2
source_models: llama-3.3-70b-versatile, llama-3.1-8b-instant

---
Plot Type: bar
Variables: category, count
ensemble_score: 0.5
model_agreement: 1
source_models: llama-3.3-70b-versatile

"""


# ============================================================================
# Temporary file fixtures
# ============================================================================

@pytest.fixture
def temp_image_path(simple_plot, tmp_path):
    """Create a temporary image file for testing."""
    output_path = tmp_path / "test_plot.jpg"
    simple_plot.figure.savefig(output_path)
    yield output_path
    # Cleanup happens automatically with tmp_path


# ============================================================================
# Auto-use fixtures (applied to all tests)
# ============================================================================

@pytest.fixture(autouse=True)
def reset_matplotlib():
    """Reset matplotlib state before each test."""
    yield
    plt.close('all')


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(SEED)
    yield


# ============================================================================
# Markers helpers
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (test component interaction)"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests (test full workflows)"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time to run"
    )
    config.addinivalue_line(
        "markers", "api: Tests that make external API calls"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: Tests requiring valid API credentials"
    )
    config.addinivalue_line(
        "markers", "plotting: Tests that generate visualizations"
    )


# ============================================================================
# Test collection hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Automatically add markers to tests based on their characteristics.
    """
    for item in items:
        # Add 'plotting' marker to tests that use plot fixtures
        if any(fixture in item.fixturenames for fixture in
               ['sample_plot', 'simple_plot', 'scatter_plot', 'bar_plot']):
            item.add_marker(pytest.mark.plotting)

        # Add 'api' marker to tests that use API mocks
        if any(fixture in item.fixturenames for fixture in
               ['mock_groq_client', 'mock_groq_completion']):
            item.add_marker(pytest.mark.api)

        # Add 'slow' marker to tests with 'large' in the name
        if 'large' in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
