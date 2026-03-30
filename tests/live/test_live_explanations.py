"""
Live tests for PlotExplainer with real API calls.

These tests require actual API keys and make real requests.
Skip with: pytest -m "not live"
"""

import os
import pytest
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from plotsense.explanations.explanations import explainer

load_dotenv()


@pytest.mark.live
class TestLiveExplanationGeneration:
    """Live tests that require real API calls to Groq/OpenAI."""

    @pytest.fixture
    def simple_plot(self):
        """Create a simple plot for explanation."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_title("Simple Line Plot")
        yield ax
        plt.close(fig)

    def test_groq_explanation_generation(self, simple_plot, groq_api_key):
        """Test explanation generation with real Groq API."""
        explanation = explainer(
            plot_object=simple_plot,
            prompt="Explain what this plot shows",
            api_keys={'groq': groq_api_key},
            max_iterations=1
        )

        assert explanation is not None
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_openai_explanation_generation(self, simple_plot, openai_api_key):
        """Test explanation generation with real OpenAI API."""
        explanation = explainer(
            plot_object=simple_plot,
            prompt="What does this plot visualize?",
            api_keys={'openai': openai_api_key},
            selected_models=[("openai", "gpt-4.1")],
            max_iterations=1
        )

        assert explanation is not None
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_groq_explanation_with_refinement(self, simple_plot, groq_api_key):
        """Test explanation with multiple refinement iterations."""
        explanation = explainer(
            plot_object=simple_plot,
            prompt="Analyze this plot in detail",
            api_keys={'groq': groq_api_key},
            max_iterations=2
        )

        assert explanation is not None
        assert isinstance(explanation, str)
        assert len(explanation) > 0
