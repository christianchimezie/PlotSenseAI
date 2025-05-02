# tests/test_visual_suggestion.py
import os
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

import warnings


# SUT
from plotsense.visual_suggestion.suggestions import VisualizationRecommender

load_dotenv()  # make .env vars visible for tests
SEED = 42
rng = np.random.default_rng(SEED)

warnings.filterwarnings("ignore", category=UserWarning, module="plotsense.visual_suggestion")
# ---------- fixtures ---------------------------------------------------------
@pytest.fixture
def sample_dataframe():
    """Deterministic sample frame for every test."""
    n = 100
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "category": rng.choice(list("ABC"), n),
            "value": rng.normal(0, 1, n),
            "count": rng.integers(0, 100, n),
            "flag": rng.choice([True, False], n),
        }
    )

@pytest.fixture
def mock_recommender(sample_dataframe):
    """Light‑weight stub for the heavy recommender object."""
    recommender = MagicMock(spec=VisualizationRecommender)
    recommender.debug = False
    recommender.df = sample_dataframe
    recommender.timeout = 30
    recommender.api_keys = {"groq": "test_key"}
    # expose real (static) attrs so tests that inspect them still work
    recommender.DEFAULT_MODELS = VisualizationRecommender.DEFAULT_MODELS
    return recommender


@pytest.fixture
def llm_dummy_response():
    return """
Plot Type: scatter 
Variables: value, count
Rationale: Shows relationship between two continuous variables
---
Plot Type: bar
Variables: category, count
Rationale: Compares discrete categories with their counts
"""


# ---------- unit tests -------------------------------------------------------
class TestInitialization:
    def test_init_without_keys(self, monkeypatch):
        """Test initialization without API keys"""
        monkeypatch.setenv("GROQ_API_KEY", "env_key")
        r = VisualizationRecommender()
        assert r.api_keys["groq"] == "env_key"

    def test_init_with_keys(self):
        """Test initialization with provided API keys"""
        r = VisualizationRecommender(api_keys={"groq": "provided"})
        assert r.api_keys["groq"] == "provided"

    def test_missing_key_noninteractive(self):
        """Test initialization with missing API key in non-interactive mode"""
        with pytest.raises(ValueError, match="GROQ API key is required"):
            VisualizationRecommender(api_keys={"groq": None}, interactive=False)

    def test_missing_key_interactive(self, monkeypatch):
        """Test interactive key input"""
        monkeypatch.setattr("builtins.input", lambda _: "typed_key")
        r = VisualizationRecommender(api_keys={"groq": None}, interactive=True)
        assert r.api_keys["groq"] == "typed_key"

    def test_default_models(self):
        """Test default model configuration"""
        r = VisualizationRecommender(api_keys={'groq': 'test_key'})
        assert 'llama3-70b-8192' in r.DEFAULT_MODELS['groq'][0]
        assert isinstance(r.DEFAULT_MODELS['groq'], list)

    def test_model_weights_sum_to_one(self):
        """Test model weights are properly initialized"""
        r = VisualizationRecommender(api_keys={"groq": "x"})
        assert pytest.approx(sum(r.model_weights.values())) == 1.0


class TestDataFrameHandling:
    def test_set_dataframe(self, sample_dataframe):
        """Test setting dataframe"""
        r = VisualizationRecommender(api_keys={"groq": "x"})
        r.set_dataframe(sample_dataframe)
        assert r.df.equals(sample_dataframe)

    def test_describe_dataframe_contains_columns(self, sample_dataframe):
        """Test DataFrame description generation"""
        r = VisualizationRecommender(api_keys={"groq": "x"})
        r.set_dataframe(sample_dataframe)
        desc = r._describe_dataframe()
        for col in sample_dataframe.columns:
            assert col in desc


class TestPromptGeneration:
    def test_create_prompt_mentions_examples(self, sample_dataframe):
        r = VisualizationRecommender(api_keys={"groq": "x"})
        r.set_dataframe(sample_dataframe)
        r.n_to_request = 5
        prompt = r._create_prompt("demo")
        assert "matplotlib function name" in prompt
        assert "Example correct responses" in prompt


class TestResponseParsing:
    def test_parse_valid_response(self, mock_recommender):
        resp = """
        Plot Type: line
        Variables: date, value
        Rationale: Trend
        ---
        Plot Type: histogram
        Variables: value
        Rationale: Distribution
        """
        parsed = VisualizationRecommender._parse_recommendations(
            mock_recommender, resp, "test-model"
        )
        assert len(parsed) == 2
        assert parsed[0]["plot_type"] == "line"
        assert parsed[0]['variables'] == 'date, value'
        assert parsed[0]['rationale'] == 'Trend'
        assert parsed[1]['plot_type'] == 'histogram'
        assert parsed[1]['variables'] == 'value'
        assert parsed[1]['rationale'] == 'Distribution'

    def test_parse_ignores_empty_or_malformed(self, mock_recommender):
        malformed = "nonsense"
        assert (
            VisualizationRecommender._parse_recommendations(
                mock_recommender, malformed, "test"
            )
            == []
        )


class TestLLMIntegration:

    @patch('plotsense.visual_suggestion.suggestions.Groq')
    def test_query_llm(self, mock_groq):
        """Test LLM query method"""
        # Setup
        r = VisualizationRecommender(api_keys={"groq": "test_key"})
        
        mock_client = MagicMock()
        r.clients['groq'] = mock_client  # Directly set the mocked client

        # Create separate mock for chat.completions.create
        mock_chat_completions = MagicMock()
        mock_client.chat.completions = mock_chat_completions

        # Setup response
        mock_message = MagicMock()
        mock_message.content = "test response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # Assign the return value to the create method
        mock_chat_completions.create.return_value = mock_response

        # Execute
        response = r._query_llm("test prompt", "llama3-70b-8192")

        # Verify
        mock_chat_completions.create.assert_called_once_with(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0.4,
            max_tokens=1000,
            timeout=r.timeout  # ensure consistency
        )

        assert response == "test response"


class TestRecommendationGeneration:
    
    @patch('plotsense.visual_suggestion.suggestions.VisualizationRecommender._query_llm')
    def test_get_recommendations(self, mock_query, llm_dummy_response):
        """Test recommendation generation"""
        
        mock_query.return_value = llm_dummy_response

        recommender = VisualizationRecommender(api_keys={"groq": "dummy"})
        recommender.df = pd.DataFrame(columns=["value", "count", "category", "time"])  # ✅ Important fix

        model_name = "llama3-70b-8192"
        prompt = "describe the best five charts for this data"

        recs = recommender._get_model_recommendations(
            model=model_name,
            prompt=prompt,
            query_func=mock_query
        )

        assert len(recs) == 2
        for rec in recs:
            assert {'plot_type', 'variables', 'rationale'} <= rec.keys()
            assert rec['source_model'] == model_name


    
    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_get_all_recommendations(self, mock_executor, sample_dataframe):
        r = VisualizationRecommender(api_keys={"groq": "x"})
        r.set_dataframe(sample_dataframe)

        # Fake future
        fake_future = Mock()
        fake_future.result.return_value = [
            {"plot_type": "scatter", "variables": "value,count", "rationale": "demo"}
        ]

        # Configure the mock executor
        mock_exec_instance = mock_executor.return_value
        mock_exec_instance.__enter__.return_value.submit.return_value = fake_future

        # Call the method
        recs = r._get_all_recommendations()

        # Assert the outcome
        assert isinstance(recs, dict)
        assert recs  # Ensure it's not empty


class TestErrorHandling:
    def test_no_dataframe_error(self, mock_recommender):
        """Test error when no DataFrame is set"""
        r= VisualizationRecommender(api_keys={"groq": "x"})
        with pytest.raises(ValueError, match="No DataFrame set"):
            r.recommend_visualizations()
    
    @patch('plotsense.visual_suggestion.suggestions.VisualizationRecommender._query_llm')
    def test_model_failure_handling(self, mock_query):
        """Test handling of model failures"""

        # Simulate the model failure by raising an exception in the mock
        mock_query.side_effect = Exception("API error")

        # Create an instance and set a valid DataFrame
        r = VisualizationRecommender(api_keys={"groq": "x"})
        r.df = pd.DataFrame()

        # Expect multiple warnings, one per model
        with pytest.warns(UserWarning, match="Error processing model"):
            all_recs = r._get_all_recommendations()

        # Validate all returned recommendations are empty lists (i.e., no successful model response)
        assert all(isinstance(val, list) and len(val) == 0 for val in all_recs.values())



    # ---------- edge / error‑handling tests -------------------------------------
    def test_recommend_without_dataframe_raises(tmp_path):
        r = VisualizationRecommender(api_keys={"groq": "x"})
        with pytest.raises(ValueError, match="No DataFrame"):
            r.recommend_visualizations()