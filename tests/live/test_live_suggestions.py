"""
Live tests for VisualizationRecommender with real API calls.

These tests require actual API keys and make real requests.
Skip with: pytest -m "not live"
"""

import os
import pytest
import pandas as pd
from dotenv import load_dotenv

from plotsense.visual_suggestion.suggestions import VisualizationRecommender

load_dotenv()


@pytest.mark.live
class TestLiveRecommendationGeneration:
    """Live tests that require real API calls to Groq/OpenAI."""

    @pytest.fixture
    def sample_dataframe(self):
        """Sample data for recommendations."""
        return pd.DataFrame({
            "Year": [2020, 2021, 2022, 2023],
            "Sales": [150, 200, 250, 300],
            "Profit": [40, 50, 65, 80],
            "Category": ["A", "B", "A", "B"],
        })

    def test_groq_recommendation_generation(self, sample_dataframe, groq_api_key):
        """Test recommendation generation with real Groq API."""
        recommender = VisualizationRecommender(api_keys={"groq": groq_api_key})
        recommender.set_dataframe(sample_dataframe)

        recs = recommender.recommend_visualizations(n=3)

        assert recs is not None
        assert isinstance(recs, pd.DataFrame)
        assert len(recs) > 0
        assert 'plot_type' in recs.columns

    def test_openai_recommendation_generation(self, sample_dataframe, openai_api_key):
        """Test recommendation generation with real OpenAI API."""
        recommender = VisualizationRecommender(
            api_keys={
                "openai": openai_api_key,
            }
        )
        recommender.set_dataframe(sample_dataframe)

        recs = recommender.recommend_visualizations(
            n=2,
            selected_models=[("openai", "gpt-4.1")]
        )

        assert recs is not None
        assert isinstance(recs, pd.DataFrame)
        assert len(recs) > 0
        assert 'plot_type' in recs.columns
