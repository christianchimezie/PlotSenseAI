import pytest
import base64
import os
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
import warnings
from plotsense.explanations.explanations import PlotExplainer, _explainer_instance
import  builtins
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend suitable for testing
# Load environment variables for testing
load_dotenv()

@pytest.fixture
def sample_plot():
    """Fixture that creates a simple matplotlib plot"""
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title("Sine Wave")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return fig

@pytest.fixture
def mock_groq_client():
    """Fixture that mocks the Groq client"""
    with patch('groq.Groq') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def plot_explainer():
    """Fixture that creates a PlotExplainer instance with test config"""
      # Replace with your actual module name
    
    # Use a test API key or None to test environment variable fallback
    return PlotExplainer(api_keys={'groq': 'test-key'})

def test_initialization_with_api_keys(plot_explainer):
    """Test initialization with API keys"""
    assert plot_explainer.api_keys['groq'] == 'test-key'
    assert plot_explainer.timeout == 30

def test_initialization_without_api_keys():
    """Test initialization without API keys (using environment variables)"""
    
    
    # Temporarily set environment variable
    os.environ['GROQ_API_KEY'] = 'env-test-key'
    
    try:
        explainer = PlotExplainer(api_keys={})
        assert explainer.api_keys['groq'] == 'env-test-key'
    finally:
        # Clean up
        del os.environ['GROQ_API_KEY']

def test_initialization_missing_keys():
    """Test initialization when required keys are missing"""
    
    
    with pytest.raises(ValueError, match="GROQ API key is required"):
        PlotExplainer(api_keys={})

def test_plot_to_bytes_figure(sample_plot, plot_explainer):
    """Test converting Figure to bytes"""
    bytes_data = plot_explainer._plot_to_bytes(sample_plot)
    assert isinstance(bytes_data, bytes)
    assert len(bytes_data) > 0

def test_plot_to_bytes_axes(sample_plot, plot_explainer):
    """Test converting Axes to bytes"""
    bytes_data = plot_explainer._plot_to_bytes(sample_plot.axes[0])
    assert isinstance(bytes_data, bytes)
    assert len(bytes_data) > 0

def test_select_model_rotation(plot_explainer):
    """Test model rotation selection"""
    models = plot_explainer._select_model_rotation()
    assert isinstance(models, list)
    assert len(models) > 0

@patch('plotsense.explanations.explanations.PlotExplainer._query_llama3')
def test_generate_explanation(mock_query, sample_plot, plot_explainer):
    """Test explanation generation"""
    mock_query.return_value = "Test explanation"
    img_bytes = plot_explainer._plot_to_bytes(sample_plot)
    explanation = plot_explainer._generate_explanation(img_bytes, "test prompt", "llama-3.2-90b-vision-preview")
    assert explanation == "Test explanation"
    mock_query.assert_called_once()

@patch('plotsense.explanations.explanations.PlotExplainer._query_llama3')
def test_generate_critique(mock_query, sample_plot, plot_explainer):
    """Test critique generation"""
    mock_query.return_value = "Test critique"
    img_bytes = plot_explainer._plot_to_bytes(sample_plot)
    critique = plot_explainer._generate_critique(img_bytes, "test explanation", "test prompt", "llama-3.2-90b-vision-preview")
    assert critique == "Test critique"
    mock_query.assert_called_once()

@patch('plotsense.explanations.explanations.PlotExplainer._query_llama3')
def test_generate_refinement(mock_query, sample_plot, plot_explainer):
    """Test refinement generation"""
    mock_query.return_value = "Test refinement"
    img_bytes = plot_explainer._plot_to_bytes(sample_plot)
    refinement = plot_explainer._generate_refinement(
        img_bytes, "test explanation", "test critique", "test prompt", "llama-3.2-90b-vision-preview"
    )
    assert refinement == "Test refinement"
    mock_query.assert_called_once()

@patch('plotsense.explanations.explanations.PlotExplainer._generate_explanation')
@patch('plotsense.explanations.explanations.PlotExplainer._generate_critique')
@patch('plotsense.explanations.explanations.PlotExplainer._generate_refinement')
def test_refine_plot_explanation(mock_refine, mock_critique, mock_explain, sample_plot, plot_explainer):
    """Test the full refinement process"""
    # Setup mock return values
    mock_explain.return_value = "Initial explanation"
    mock_critique.side_effect = ["Critique 1", "Critique 2"]
    mock_refine.side_effect = ["Refined 1", "Refined 2"]
    
    # Test with 3 iterations (1 initial + 2 refinements)
    result = plot_explainer.refine_plot_explanation(sample_plot, iterations=3)
    
    assert result == "Refined 2"
    assert mock_explain.call_count == 1
    assert mock_critique.call_count == 2
    assert mock_refine.call_count == 2

def test_query_llm_success(plot_explainer, sample_plot, mock_groq_client):
    """Test successful LLM query with proper mocking"""
    
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Insight: Trend A dominates."

    mock_completion.choices = [mock_choice]
    mock_groq_client.chat.completions.create.return_value = mock_completion

    plot_explainer.clients["groq"] = mock_groq_client

    img_bytes = plot_explainer._plot_to_bytes(sample_plot)
    result = plot_explainer._query_llama3(img_bytes, prompt="What's the trend?")
    assert result == "Insight: Trend A dominates."
    mock_groq_client.chat.completions.create.assert_called_once()


@patch('builtins.input', return_value='test-key')
def test_interactive_key_input(mock_input):
    """Test interactive API key input"""
    
    
    # Ensure no environment variable is set
    if 'GROQ_API_KEY' in os.environ:
        del os.environ['GROQ_API_KEY']
    
    explainer = PlotExplainer(api_keys={})
    assert explainer.api_keys['groq'] == 'test-key'

def test_invalid_iterations(sample_plot, plot_explainer):
    """Test invalid iteration values"""
    with pytest.raises(ValueError, match="Iterations must be between 1 and 5"):
        plot_explainer.refine_plot_explanation(sample_plot, iterations=0)
    
    with pytest.raises(ValueError, match="Iterations must be between 1 and 5"):
        plot_explainer.refine_plot_explanation(sample_plot, iterations=6)

def test_module_level_explainer(sample_plot):
    """Test the package-level convenience function"""
    
    # Reset the global instance
    global _explainer_instance
    _explainer_instance = None
    
    # Mock the instance methods
    with patch('plotsense.explanations.explanations.PlotExplainer.refine_plot_explanation') as mock_method:
        mock_method.return_value = "Test explanation"

        r= PlotExplainer(api_keys={"groq": "x"})
        _explainer_instance = r  # Manually assign to global

        
        result = r.refine_plot_explanation(sample_plot)
        
        assert result == "Test explanation"
        assert _explainer_instance is not None
        mock_method.assert_called_once()

def test_model_rotation_override(sample_plot, plot_explainer):
    """Test custom model rotation"""
    custom_models = ["model1", "model2"]
    with patch.object(plot_explainer, '_generate_explanation') as mock_explain, \
         patch.object(plot_explainer, '_generate_critique') as mock_critique, \
         patch.object(plot_explainer, '_generate_refinement') as mock_refine:

        mock_explain.return_value = "Initial"
        mock_critique.return_value = "Critique"
        mock_refine.return_value = "Refined"

        plot_explainer.refine_plot_explanation(
            sample_plot,
            iterations=2,
            model_rotation=custom_models
        )

        # Check that the methods were called with the correct models
        # For explanation - first model
        args, kwargs = mock_explain.call_args
        assert len(args) >= 3  # img_bytes, prompt, model
        assert args[2] == "model1"  # model is 3rd positional argument

        # For critique - first model
        args, kwargs = mock_critique.call_args
        assert len(args) >= 4  # img_bytes, explanation, prompt, model
        assert args[3] == "model1"

        # For refinement - second model
        args, kwargs = mock_refine.call_args
        assert len(args) >= 5  # img_bytes, explanation, critique, prompt, model
        assert args[4] == "model2"

def test_warning_on_missing_client():
    """Test warning when client library is not available"""
    with patch.dict('sys.modules', {'groq': None}):
        with warnings.catch_warnings(record=True) as warning_list:
            # Ensure warnings are shown
            warnings.simplefilter("always")
            
            # This should trigger the warning
            PlotExplainer(api_keys={'groq': 'test-key'})
            
            # Verify warning was raised
            assert len(warning_list) == 1
            assert "Groq Python client not installed" in str(warning_list[0].message)
            assert warning_list[0].category == ImportWarning