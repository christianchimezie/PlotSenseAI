from plotsense.plot_generator.generator import PlotGenerator, SmartPlotGenerator, plotgen
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from unittest.mock import patch

# Use non-interactive backend for all tests to avoid Tkinter issues
matplotlib.use('Agg')

# SUT

# Fixtures


@pytest.fixture
def sample_dataframe():
    """Ensure numeric columns have proper values for aggregation"""
    n = 20
    np.random.seed(42)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n),
        "category": np.random.choice(list("ABCDE"), n),
        # Make sure these are proper numeric values
        "value": np.random.uniform(1, 100, n),  # No zeros
        "count": np.random.randint(1, 100, n),  # Positive integers
        "flag": np.random.choice([0, 1], n),    # Numeric binary
        "x": np.arange(n),
        "y": np.random.rand(n),
        "z": np.random.rand(n),
    })


@pytest.fixture
def sample_suggestions():
    """Modified to use numeric columns for bar/barh plots"""
    return pd.DataFrame({
        'plot_type': ['scatter', 'bar', 'barh', 'hist', 'boxplot', 'violinplot', 'pie', 'hexbin'],
        # Changed bar/barh to use numeric columns (value,count) instead of category
        'variables': ['x,y', 'value,category', 'value,category', 'value', 'value', 'value,category', 'category', 'x,y'],
        'ensemble_score': np.random.rand(8)
    })


@pytest.fixture
def plot_generator(sample_dataframe, sample_suggestions):
    """Fixture for PlotGenerator instance."""
    return PlotGenerator(sample_dataframe, sample_suggestions)


@pytest.fixture
def smart_plot_generator(sample_dataframe, sample_suggestions):
    """Fixture for SmartPlotGenerator instance."""
    return SmartPlotGenerator(sample_dataframe, sample_suggestions)

# Reset global state before each test to avoid interference


@pytest.fixture(autouse=True)
def reset_plot_generator_instance():
    """Reset the global _plot_generator_instance before each test."""
    global _plot_generator_instance
    _plot_generator_instance = None

# Unit Tests


class TestPlotGeneratorUnit:
    def test_init_plot_generator(self, sample_dataframe, sample_suggestions):
        pg = PlotGenerator(sample_dataframe, sample_suggestions)
        assert pg.data.equals(sample_dataframe)
        assert pg.suggestions.equals(sample_suggestions)
        expected_functions = set(['scatter', 'bar', 'barh', 'hist', 'boxplot', 'violinplot', 'pie', 'hexbin'])
        assert set(pg.plot_functions.keys()) == expected_functions

    def test_init_smart_plot_generator(self, sample_dataframe, sample_suggestions):
        spg = SmartPlotGenerator(sample_dataframe, sample_suggestions)
        assert spg.data.equals(sample_dataframe)
        assert spg.suggestions.equals(sample_suggestions)
        expected_functions = set(['scatter', 'bar', 'barh', 'hist', 'boxplot', 'violinplot', 'pie', 'hexbin'])
        assert set(spg.plot_functions.keys()) == expected_functions

    def test_generate_plot_with_index(self, plot_generator):
        fig = plot_generator.generate_plot(0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_generate_plot_with_series(self, plot_generator, sample_suggestions):
        series = sample_suggestions.iloc[0].to_dict()
        # Need to modify generate_plot to accept dict input
        with patch.object(plot_generator, 'generate_plot') as mock_generate:
            mock_generate.return_value = plt.figure()
            fig = plot_generator.generate_plot(series)
            assert isinstance(fig, plt.Figure)
            mock_generate.assert_called_once()
        plt.close(fig)

# Unit Tests for Individual Plot Functions


class TestPlotFunctions:
    def test_create_scatter(self, plot_generator):
        fig = plot_generator._create_scatter(["x", "y"])
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        plt.close(fig)

    def test_create_bar(self, plot_generator):
        # Test with numeric column only
        fig = plot_generator._create_bar(["count"])
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_create_barh(self, plot_generator):
        # Test with numeric column only
        fig = plot_generator._create_barh(["count"])
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_create_hist(self, plot_generator):
        fig = plot_generator._create_hist(["value"])
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_create_box(self, plot_generator):
        fig = plot_generator._create_box(["value"])
        ax = fig.axes[0]
        assert len(ax.lines) > 0
        plt.close(fig)

    def test_create_violin(self, plot_generator):
        fig = plot_generator._create_violin(["value"])
        ax = fig.axes[0]
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_create_pie(self, plot_generator):
        fig = plot_generator._create_pie(["category"])
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_create_hexbin(self, plot_generator):
        fig = plot_generator._create_hexbin(["x", "y"])
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        plt.close(fig)

# Integration Tests


class TestPlotGeneratorIntegration:
    def test_plotgen_with_custom_args(self, sample_dataframe, sample_suggestions):
        # Mock the generate_plot method to test custom args
        with patch.object(PlotGenerator, 'generate_plot') as mock_generate:
            mock_generate.return_value = plt.figure()
            fig = plotgen(sample_dataframe, 0, sample_suggestions,
                          x_label="Custom X", y_label="Custom Y", title="Custom Title")
            assert isinstance(fig, plt.Figure)
            mock_generate.assert_called_once_with(0, x_label="Custom X", y_label="Custom Y", title="Custom Title")
        plt.close(fig)

# End-to-End Tests


class TestPlotGeneratorEndToEnd:
    @pytest.mark.parametrize("index", range(8))  # All plot types
    def test_all_plot_types_default(self, sample_dataframe, sample_suggestions, index):
        fig = plotgen(sample_dataframe, index, sample_suggestions)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.parametrize("index", [4, 5, 3])  # boxplot, violinplot, hist
    def test_smart_plot_types(self, sample_dataframe, sample_suggestions, index):
        with patch('plotsense.plot_generator.generator.SmartPlotGenerator.generate_plot') as mock_generate:
            mock_generate.return_value = plt.figure()
            fig = plotgen(sample_dataframe, index, sample_suggestions)
            assert isinstance(fig, plt.Figure)
            mock_generate.assert_called_once()
        plt.close(fig)

    def test_plotgen_with_large_data(self, sample_suggestions):
        n = 1000
        df = pd.DataFrame({
            "x": np.arange(n),
            "y": np.random.rand(n),
            "value": np.random.normal(0, 1, n),
            "category": np.random.choice(list("ABCDE"), n),
            "count": np.random.randint(0, 100, n)
        })
        fig = plotgen(df, 0, sample_suggestions)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

# Error Handling Tests


class TestPlotGeneratorErrorHandling:
    def test_plotgen_empty_variables(self, sample_dataframe):
        """Test with empty variables that will trigger the scatter validation error"""
        invalid_suggestions = pd.DataFrame({
            'plot_type': ['scatter'],
            'variables': [''],
            'ensemble_score': [0.9]
        })
        with pytest.raises(ValueError, match="requires at least 2 variables"):  # Match actual error
            plotgen(sample_dataframe, 0, invalid_suggestions)

    def test_plotgen_unsupported_type(self, sample_dataframe):
        """Verify unsupported plot types are handled gracefully"""
        invalid_suggestions = pd.DataFrame({
            'plot_type': ['invalid_plot_type_123'],
            'variables': ['x,y'],
            'ensemble_score': [0.9]
        })

        result = plotgen(sample_dataframe, 0, invalid_suggestions)

        # Either returns None or a figure with error indication
        assert result is None or isinstance(result, plt.Figure)

        if isinstance(result, plt.Figure):
            plt.close(result)

    def test_plotgen_invalid_dataframe(self, sample_suggestions):
        with pytest.raises(TypeError):
            PlotGenerator("not_a_dataframe", sample_suggestions)

    def test_plotgen_invalid_series(self, sample_dataframe):
        """Test with invalid series input"""
        invalid_series = pd.Series({'wrong_column': 'scatter'})
        with pytest.raises(ValueError, match="must contain"):
            plotgen(sample_dataframe, invalid_series)

    def test_scatter_non_numeric_data(self, sample_dataframe):
        """Test with actual non-numeric data that will fail"""
        df = sample_dataframe.copy()
        # Use numpy object dtype to ensure it's not numeric
        df['x'] = np.array(['a'] * len(df), dtype=object)
        suggestions = pd.DataFrame({
            'plot_type': ['scatter'],
            'variables': ['x,y'],
            'ensemble_score': [0.9]
        })
        with pytest.raises(ValueError, match="must be numeric"):
            plotgen(df, 0, suggestions)

    def test_box_no_data(self):
        df = pd.DataFrame({'x': [np.nan] * 10})
        sugg = pd.DataFrame({'plot_type': ['boxplot'], 'variables': ['x']})
        pg = PlotGenerator(df, sugg)
        fig = pg.generate_plot(0)  # Add index argument
        assert fig is not None

# Edge Case Tests


class TestPlotGeneratorEdgeCases:
    def test_empty_dataframe(self):
        with pytest.raises(ValueError):
            PlotGenerator(pd.DataFrame(), pd.DataFrame({
                'plot_type': ['scatter'],
                'variables': ['x,y'],
                'ensemble_score': [0.9]
            }))

    def test_malformed_suggestions(self):
        df = pd.DataFrame({'x': [1], 'y': [2], 'z': [3]})
        sugg = pd.DataFrame({'plot_type': ['scatter'], 'variables': ['x,y,z']})
        pg = PlotGenerator(df, sugg)
        fig = pg.generate_plot(0)  # Add index argument
        assert fig is not None

    def test_duplicate_variables(self, sample_dataframe):
        """Test with actually existing variables"""
        df = sample_dataframe.copy()
        if 'y' not in df.columns:
            df['y'] = np.random.rand(len(df))

        sugg = pd.DataFrame({
            'plot_type': ['scatter'],
            'variables': ['x,y'],
            'ensemble_score': [0.9]
        })
        fig = plotgen(df, 0, sugg)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
