import pandas as pd
import pytest
import matplotlib.pyplot as plt
from plotsense.plot_generator.altoplot_v2 import generate_plot
from unittest.mock import patch

# Mock Titanic dataset
@pytest.fixture
def titanic_df():
    return pd.DataFrame({
        'survived': [0, 1, 1, 0, 1],
        'sex': ['male', 'female', 'female', 'male', 'female'],
        'age': [22, 38, 26, 35, 28]
    })

# Mock suggestion for bar chart
@pytest.fixture
def bar_suggestion():
    return pd.Series({
        'plot_type': 'bar chart',
        'variables': 'survived, sex',
        'rationale': 'This visual can reveal the relationship...',
        'ensemble_score': 1.0
    })

# Mock suggestion for histogram
@pytest.fixture
def hist_suggestion():
    return pd.Series({
        'plot_type': 'histogram',
        'variables': 'age,',
        'rationale': 'This visual can show the distribution of age...',
        'ensemble_score': 0.9
    })

# Mock suggestion for 2D histogram
@pytest.fixture
def hist2d_suggestion():
    return pd.Series({
        'plot_type': '2d histogram',
        'variables': 'age, survived',
        'rationale': 'This visual can show the 2D distribution...',
        'ensemble_score': 0.85
    })


@patch('matplotlib.pyplot.show')
def test_generate_plot_bar_default(mock_show, titanic_df, bar_suggestion):
    try:
        fig = generate_plot(titanic_df, bar_suggestion, color='blue')
        assert isinstance(fig, plt.Figure)
        print("Bar plot with default suggestion generated successfully")
    except Exception as e:
        print(f"Bar plot with default suggestion failed: {e}")
        raise
    finally:
        plt.close(fig)
