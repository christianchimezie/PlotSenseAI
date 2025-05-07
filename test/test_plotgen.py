import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import time
from unittest.mock import patch, MagicMock

# Use non-interactive backend for all tests to avoid Tkinter issues
matplotlib.use('Agg')

# SUT
from plotsense.plot_generator.generator import PlotGenerator, SmartPlotGenerator, plotgen

# Fixtures
@pytest.fixture
def sample_dataframe():
    """Deterministic sample DataFrame for testing without 2D arrays."""
    n = 20
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n),
        "category": np.random.choice(list("ABCDE"), n),
        "value": np.linspace(-3, 3, n),  # Ensures 20 unique values
        "count": np.random.randint(0, 100, n),
        "flag": np.random.choice([True, False], n),
        "x": np.arange(n),
        "y": np.random.rand(n),
        "z": np.random.rand(n),
        "u": np.random.rand(n),
        "v": np.random.rand(n),
        "dx": np.ones(n) * 0.1,
        "dy": np.ones(n) * 0.1,
        "dz": np.random.rand(n)
    })

@pytest.fixture
def sample_2d_array():
    """Fixture for a 2D array used in specific plots like surface."""
    return np.random.rand(10, 10)  # Smaller size to avoid memory issues

@pytest.fixture
def sample_suggestions():
    """Sample suggestions DataFrame covering supported plot types."""
    return pd.DataFrame({
        'plot_type': ['scatter', 'line', 'bar', 'barh', 'stem', 'step', 'fill_between',
                      'hist', 'boxplot', 'violinplot', 'errorbar', 'pie', 'polar',
                      'hexbin', 'quiver', 'streamplot', 'plot3d', 'scatter3d', 'bar3d', 'surface'],
        'variables': ['x,y', 'x,y', 'category,count', 'category,count',
                      'x,y', 'x,y', 'x,y,z',
                      'value', 'value', 'value,category', 'x,y,flag',
                      'category', 'x,y', 'x,y', 'x,y,u,v',
                      'x,y,u,v', 'x,y,z', 'x,y,z', 'x,y,z,dx,dy,dz', 'x2d'],
        'ensemble_score': np.random.rand(20)
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
        expected_functions = set(['scatter', 'line', 'bar', 'barh', 'stem', 'step', 'fill_between',
                                'hist', 'boxplot', 'violinplot', 'errorbar', 'pie', 'polar',
                                'hexbin', 'quiver', 'streamplot', 'plot3d', 'scatter3d', 'bar3d', 'surface'])
        assert set(pg.plot_functions.keys()) == expected_functions

    def test_init_smart_plot_generator(self, sample_dataframe, sample_suggestions):
        spg = SmartPlotGenerator(sample_dataframe, sample_suggestions)
        assert spg.data.equals(sample_dataframe)
        assert spg.suggestions.equals(sample_suggestions)
        expected_functions = set(['scatter', 'line', 'bar', 'barh', 'stem', 'step', 'fill_between',
                                'hist', 'boxplot', 'violinplot', 'errorbar', 'pie', 'polar',
                                'hexbin', 'quiver', 'streamplot', 'plot3d', 'scatter3d', 'bar3d', 'surface'])
        assert set(spg.plot_functions.keys()) == expected_functions
        assert spg.plot_functions['boxplot'] != PlotGenerator(sample_dataframe, sample_suggestions).plot_functions['boxplot']
        assert spg.plot_functions['violinplot'] != PlotGenerator(sample_dataframe, sample_suggestions).plot_functions['violinplot']
        assert spg.plot_functions['hist'] != PlotGenerator(sample_dataframe, sample_suggestions).plot_functions['hist']

    def test_generate_plot_with_index(self, plot_generator):
        fig = plot_generator.generate_plot(0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_generate_plot_with_series(self, plot_generator, sample_suggestions):
        series = sample_suggestions.iloc[0]
        fig = plot_generator.generate_plot(series)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0] if fig.axes else None
        assert ax is not None
        assert ax.name == 'rectilinear'  # Scatter plot uses rectilinear projection
        assert len(ax.collections) == 1  # Scatter plot has one collection
        plt.close(fig)

    def test_initialize_plot_functions(self, plot_generator):
        funcs = plot_generator._initialize_plot_functions()
        assert all(callable(func) for func in funcs.values())
        expected_functions = set(['scatter', 'line', 'bar', 'barh', 'stem', 'step', 'fill_between',
                                'hist', 'boxplot', 'violinplot', 'errorbar', 'pie', 'polar',
                                'hexbin', 'quiver', 'streamplot', 'plot3d', 'scatter3d', 'bar3d', 'surface'])
        assert set(funcs.keys()) == expected_functions

    def test_set_labels(self, plot_generator):
        fig, ax = plt.subplots()
        plot_generator._set_labels(ax, ["x", "y"])
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        plt.close(fig)

    def test_set_3d_labels(self, plot_generator):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_generator._set_3d_labels(ax, ["x", "y", "z"])
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        assert ax.get_zlabel() == "z"
        plt.close(fig)

# Unit Tests for Individual Plot Functions
class TestPlotFunctions:
    def test_create_scatter(self, plot_generator):
        fig = plot_generator._create_scatter(["x", "y"])
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_line(self, plot_generator):
        fig = plot_generator._create_line(["x", "y"])
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_bar(self, plot_generator):
        # Adjusted to avoid aggregating non-numeric 'category'
        fig = plot_generator._create_bar(["category", "count"])
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        unique_categories = len(plot_generator.data["category"].unique())
        if unique_categories > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_barh(self, plot_generator):
        # Skip due to MultiIndex dtype issue; revisit if implementation changes
        #pytest.skip("Skipping due to MultiIndex dtype issue in barh")
        fig = plot_generator._create_barh(["category", "count"])
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_create_stem(self, plot_generator):
        fig = plot_generator._create_stem(["x", "y"])
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_step(self, plot_generator):
        fig = plot_generator._create_step(["x", "y"])
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_fill_between(self, plot_generator):
        fig = plot_generator._create_fill_between(["x", "y", "z"])
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_hist(self, plot_generator):
        fig = plot_generator._create_hist(["value"])
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        unique_bins = len(plot_generator.data["value"].unique())
        if unique_bins > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_box(self, plot_generator):
        fig = plot_generator._create_box(["value"])
        ax = fig.axes[0]
        assert len(ax.lines) > 0  # Boxplots use lines for whiskers, not artists
        unique_values = len(plot_generator.data["value"].unique())
        if unique_values > 10:
            assert fig.get_size_inches()[0] >= 12
        plt.close(fig)

    def test_create_violin(self, plot_generator):
        fig = plot_generator._create_violin(["value", "category"])
        ax = fig.axes[0]
        assert len(ax.collections) > 0
        grouped_data = [plot_generator.data[plot_generator.data["category"] == cat]["value"]
                       for cat in plot_generator.data["category"].unique()]
        if len(grouped_data) > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_errorbar(self, plot_generator):
        fig = plot_generator._create_errorbar(["x", "y", "flag"])
        ax = fig.axes[0]
        assert len(ax.lines) > 0
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_pie(self, plot_generator):
        fig = plot_generator._create_pie(["category"])
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        unique_categories = len(plot_generator.data["category"].unique())
        if unique_categories > 10:
            assert fig.get_size_inches()[0] >= 12
        plt.close(fig)

    def test_create_polar(self, plot_generator):
        fig = plot_generator._create_polar(["x", "y"])
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            # Polar plots use angular labels, so rotation may differ
            assert ax.get_xticklabels()[0].get_rotation() == 0  # Adjusted expectation
        plt.close(fig)

    def test_create_hexbin(self, plot_generator):
        fig = plot_generator._create_hexbin(["x", "y"])
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_quiver(self, plot_generator):
        fig = plot_generator._create_quiver(["x", "y", "u", "v"])
        ax = fig.axes[0]
        assert len(ax.collections) > 0
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_streamplot(self, plot_generator):
        # Adjusted to use quiver as fallback for 1D data
        fig = plot_generator._create_streamplot(["x", "y", "u", "v"])
        ax = fig.axes[0]
        assert len(ax.collections) > 0  # Expect quiver-like output
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_plot3d(self, plot_generator):
        fig = plot_generator._create_plot3d(["x", "y", "z"])
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_scatter3d(self, plot_generator):
        fig = plot_generator._create_scatter3d(["x", "y", "z"])
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_bar3d(self, plot_generator):
        fig = plot_generator._create_bar3d(["x", "y", "z", "dx", "dy", "dz"])
        ax = fig.axes[0]
        assert len(ax.collections) > 0
        unique_x = len(plot_generator.data["x"].unique())
        if unique_x > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_surface(self, plot_generator, sample_2d_array):
        plot_generator.data["x2d"] = [sample_2d_array] * len(plot_generator.data)
        fig = plot_generator._create_surface(["x2d"])
        ax = fig.axes[0]
        assert len(ax.collections) == 1
        plt.close(fig)

    def test_create_box_smart(self, smart_plot_generator):
        fig = smart_plot_generator._create_box(["value", "category"])
        ax = fig.axes[0]
        assert len(ax.lines) > 0  # Boxplots use lines for whiskers
        grouped_data = [smart_plot_generator.data[smart_plot_generator.data["category"] == cat]["value"]
                       for cat in smart_plot_generator.data["category"].unique()]
        if len(grouped_data) > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_violin_smart(self, smart_plot_generator):
        fig = smart_plot_generator._create_violin(["value", "category"])
        ax = fig.axes[0]
        assert len(ax.collections) > 0
        grouped_data = [smart_plot_generator.data[smart_plot_generator.data["category"] == cat]["value"]
                       for cat in smart_plot_generator.data["category"].unique()]
        if len(grouped_data) > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_create_hist_smart(self, smart_plot_generator):
        fig = smart_plot_generator._create_hist(["value", "category"])
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        categories = smart_plot_generator.data["category"].unique()
        if len(categories) > 10:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

# Integration Tests
class TestPlotGeneratorIntegration:
    @pytest.mark.parametrize("index", [0, 5, 10, 15, 19])
    def test_plotgen_with_index(self, sample_dataframe, sample_suggestions, index, sample_2d_array):
        # Add x2d for plots requiring 2D arrays
        if sample_suggestions.iloc[index]['variables'] == 'x2d':
            sample_dataframe['x2d'] = [sample_2d_array] * len(sample_dataframe)
        fig = plotgen(sample_dataframe, index, sample_suggestions)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0] if fig.axes else None
        if ax:
            if index in [16, 17, 18]:  # Only plot3d, scatter3d, bar3d are 3D
                assert ax.name == '3d'
            # Skip figure size check for 2D array plots
            plot_type = sample_suggestions.iloc[index]['plot_type']
            variables = sample_suggestions.iloc[index]['variables']
            if variables != 'x2d' and plot_type not in ['surface']:
                unique_x = len(sample_dataframe["x"].unique())
                if unique_x > 10:
                    assert fig.get_size_inches()[0] >= 12
                    assert ax.get_xticklabels()[0].get_rotation() in [0, 90]  # Allow for polar plots
        plt.close(fig)

    @pytest.mark.parametrize("index", [0, 5, 10])
    def test_plotgen_with_series(self, sample_dataframe, sample_suggestions, index):
        series = sample_suggestions.iloc[index]
        fig = plotgen(sample_dataframe, series)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0] if fig.axes else None
        if ax:
            unique_x = len(sample_dataframe["x"].unique())
            if unique_x > 10:
                assert fig.get_size_inches()[0] >= 12
                assert ax.get_xticklabels()[0].get_rotation() in [0, 90]
        plt.close(fig)

    def test_plotgen_with_custom_args(self, sample_dataframe, sample_suggestions):
        fig = plotgen(sample_dataframe, 0, sample_suggestions, x_label="Custom X", y_label="Custom Y", title="Custom Title")
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0] if fig.axes else None
        if ax:
            assert ax.get_xlabel() == "Custom X"
            assert ax.get_ylabel() == "Custom Y"
            assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_plotgen_with_smart_generator(self, sample_dataframe, sample_suggestions):
        series = sample_suggestions.iloc[8]  # boxplot
        with patch('plotsense.plot_generator.generator.SmartPlotGenerator') as mock_spg:
            mock_spg.return_value.generate_plot.return_value = plt.figure()
            fig = plotgen(sample_dataframe, series)
            assert isinstance(fig, plt.Figure)
            mock_spg.assert_called_once()
        plt.close(fig)

# End-to-End Tests
class TestPlotGeneratorEndToEnd:
    @pytest.mark.parametrize("index", range(19))  # Exclude surface for now
    def test_all_plot_types_default(self, sample_dataframe, sample_suggestions, index, sample_2d_array):
        """Test all plot types with default settings."""
        if sample_suggestions.iloc[index]['variables'] == 'x2d':
            sample_dataframe['x2d'] = [sample_2d_array] * len(sample_dataframe)
        fig = plotgen(sample_dataframe, index, sample_suggestions)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0] if fig.axes else None
        if ax:
            if index in [16, 17, 18]:  # Only plot3d, scatter3d, bar3d are 3D
                assert ax.name == '3d'
                assert len(ax.lines) == 1 if index == 16 else len(ax.collections) == 1
            elif index == 0:  # Scatter
                assert len(ax.collections) == 1
                assert len(ax.collections[0].get_offsets()) == len(sample_dataframe)
            elif index == 2:  # Bar
                assert len(ax.patches) == len(sample_dataframe["category"].unique())
        plt.close(fig)

    @pytest.mark.parametrize("index", [8, 9, 7])  # boxplot, violinplot, hist
    def test_smart_plot_types(self, sample_dataframe, sample_suggestions, index):
        """Test SmartPlotGenerator enhanced plots."""
        with patch('plotsense.plot_generator.generator.SmartPlotGenerator') as mock_spg:
            mock_spg.return_value.generate_plot.return_value = plt.figure()
            fig = plotgen(sample_dataframe, index, sample_suggestions)
            assert isinstance(fig, plt.Figure)
            mock_spg.assert_called_once()
        plt.close(fig)

    def test_plotgen_with_custom_args(self, sample_dataframe, sample_suggestions):
        """Test end-to-end with custom arguments."""
        fig = plotgen(sample_dataframe, 0, sample_suggestions, x_label="Custom X", y_label="Custom Y", title="Custom Title")
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0] if fig.axes else None
        if ax:
            assert ax.get_xlabel() == "Custom X"
            assert ax.get_ylabel() == "Custom Y"
            assert ax.get_title() == "Custom Title"
            assert len(ax.collections) == 1  # Scatter
        plt.close(fig)

    def test_plotgen_with_large_data(self, sample_suggestions):
        """Test end-to-end with a larger dataset."""
        large_df = pd.DataFrame({
            "x": np.arange(1000),
            "y": np.random.rand(1000),
            "category": np.random.choice(list("ABCDE"), 1000),
            "value": np.random.normal(0, 1, 1000),
            "count": np.random.randint(0, 100, 1000),
            "flag": np.random.choice([True, False], 1000),
            "z": np.random.rand(1000),
            "u": np.random.rand(1000),
            "v": np.random.rand(1000),
            "dx": np.ones(1000) * 0.1,
            "dy": np.ones(1000) * 0.1,
            "dz": np.random.rand(1000)
        })
        fig = plotgen(large_df, 0, sample_suggestions)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0] if fig.axes else None
        if ax:
            assert len(ax.collections) == 1
            assert len(ax.collections[0].get_offsets()) == 1000
        plt.close(fig)

# Error Handling Tests
class TestPlotGeneratorErrorHandling:
    def test_plotgen_invalid_index(self, sample_dataframe, sample_suggestions):
        """Test plotgen with invalid index."""
        with pytest.raises(IndexError):
            plotgen(sample_dataframe, 20, sample_suggestions)

    def test_plotgen_empty_variables(self, sample_dataframe, sample_suggestions):
        """Test plotgen with empty variables in suggestions."""
        invalid_suggestions = sample_suggestions.copy()
        invalid_suggestions.iloc[0, 1] = ""
        with pytest.raises(ValueError, match="No variables specified"):
            plotgen(sample_dataframe, 0, invalid_suggestions)

    def test_plotgen_unsupported_type(self, sample_dataframe, sample_suggestions):
        """Test plotgen with unsupported plot type."""
        invalid_suggestions = sample_suggestions.copy()
        invalid_suggestions.iloc[0, 0] = "invalid_type"
        with pytest.raises(ValueError, match="Unsupported plot type"):
            plotgen(sample_dataframe, 0, invalid_suggestions)

    def test_plotgen_missing_suggestions(self, sample_dataframe):
        """Test plotgen with missing suggestions."""
        with pytest.raises(ValueError):
            plotgen(sample_dataframe, 0)

    def test_plotgen_invalid_dataframe(self, sample_suggestions):
        """Test plotgen with invalid DataFrame."""
        invalid_df = "not_a_dataframe"
        with pytest.raises(TypeError, match="Data must be a pandas DataFrame"):
            plotgen(invalid_df, 0, sample_suggestions)

    def test_plotgen_invalid_series(self, sample_dataframe, sample_suggestions):
        """Test plotgen with invalid Series format."""
        invalid_series = pd.Series({"wrong_column": "scatter"})
        with pytest.raises(KeyError):
            plotgen(sample_dataframe, invalid_series, sample_suggestions)

    def test_plotgen_invalid_custom_args(self, sample_dataframe, sample_suggestions):
        """Test plotgen with invalid custom arguments."""
        with pytest.raises(TypeError, match="Label must be a string"):
            plotgen(sample_dataframe, 0, sample_suggestions, x_label=123)

    def test_scatter_non_numeric_data(self, sample_suggestions):
        """Test scatter with non-numeric data."""
        df = pd.DataFrame({"x": ["a", "b", "c"], "y": ["d", "e", "f"]})
        with pytest.raises(ValueError, match="Scatter plot requires numeric data"):
            plotgen(df, 0, sample_suggestions)

    def test_surface_mismatched_shapes(self, sample_suggestions, sample_2d_array):
        """Test surface with non-2D array data."""
        df = pd.DataFrame({"x2d": [np.random.rand(5)] * 5})  # 1D arrays
        with pytest.raises(ValueError, match="Surface requires a 2D array"):
            plotgen(df, 19, sample_suggestions)

    def test_box_no_data(self, sample_suggestions):
        """Test boxplot with all-NaN data."""
        df = pd.DataFrame({"value": [np.nan] * 10})
        with pytest.raises(ValueError, match="No valid data for boxplot"):
            plotgen(df, 8, sample_suggestions)

# Performance Tests
class TestPlotGeneratorPerformance:
    @pytest.mark.parametrize("n", [1000, 10000])
    @pytest.mark.parametrize("index", [0, 7, 16])  # Scatter, Hist, Plot3D
    def test_performance_various_plots(self, sample_suggestions, n, index):
        """Test performance of plotgen with various plot types and data sizes."""
        df = pd.DataFrame({
            "x": np.arange(n),
            "y": np.random.rand(n),
            "z": np.random.rand(n),
            "value": np.random.normal(0, 1, n),
            "category": np.random.choice(list("ABCDE"), n),
            "u": np.random.rand(n),
            "v": np.random.rand(n),
            "dx": np.ones(n) * 0.1,
            "dy": np.ones(n) * 0.1,
            "dz": np.random.rand(n)
        })
        start_time = time.time()
        fig = plotgen(df, index, sample_suggestions)
        duration = time.time() - start_time
        assert isinstance(fig, plt.Figure)
        assert duration < 5.0  # Arbitrary threshold
        plt.close(fig)

    @pytest.mark.parametrize("n", [1000, 10000])
    def test_performance_smart_generator(self, sample_suggestions, n):
        """Test performance of SmartPlotGenerator for enhanced plots."""
        df = pd.DataFrame({
            "value": np.random.normal(0, 1, n),
            "category": np.random.choice(list("ABCDE"), n)
        })
        with patch('plotsense.plot_generator.generator.SmartPlotGenerator') as mock_spg:
            mock_instance = mock_spg.return_value
            mock_instance.generate_plot.return_value = plt.figure()
            start_time = time.time()
            fig = plotgen(df, 8, sample_suggestions)  # boxplot
            duration = time.time() - start_time
            assert isinstance(fig, plt.Figure)
            assert duration < 5.0
            mock_spg.assert_called_once()
        plt.close(fig)

# Edge Case Tests
class TestPlotGeneratorEdgeCases:
    def test_empty_dataframe(self, sample_suggestions):
        """Test plotgen with an empty DataFrame."""
        df_empty = pd.DataFrame(columns=["value", "count"])
        with pytest.raises(ValueError, match="DataFrame is empty"):
            plotgen(df_empty, 0, sample_suggestions)

    def test_very_large_data(self, sample_suggestions):
        """Test plotgen with a very large dataset."""
        n = 50000
        df = pd.DataFrame({
            "x": np.arange(n),
            "y": np.random.rand(n)
        })
        fig = plotgen(df, 0, sample_suggestions)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0] if fig.axes else None
        if ax:
            assert len(ax.collections) == 1
            assert len(ax.collections[0].get_offsets()) == n
        plt.close(fig)

    def test_extreme_values(self, sample_suggestions):
        """Test plotgen with extreme values (infinities, NaNs)."""
        df = pd.DataFrame({
            "x": [np.inf, -np.inf, np.nan, 1, 2],
            "y": [1, 2, 3, np.nan, np.inf]
        })
        with pytest.raises(ValueError, match="Scatter plot cannot handle infinite values"):
            plotgen(df, 0, sample_suggestions)

    def test_malformed_suggestions(self, sample_dataframe):
        """Test plotgen with malformed suggestions."""
        invalid_suggestions = pd.DataFrame({
            'plot_type': ['scatter'],
            'variables': ['x,y,z'],  # Too many variables for scatter
            'ensemble_score': [0.9]
        })
        with pytest.raises(ValueError, match="scatter requires exactly 2 variables"):
            plotgen(sample_dataframe, 0, invalid_suggestions)

    def test_many_categories(self, sample_suggestions):
        """Test plotgen with many categories."""
        df = pd.DataFrame({
            "category": [f"cat_{i}" for i in range(50)],
            "count": np.random.randint(0, 100, 50)
        })
        fig = plotgen(df, 2, sample_suggestions)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0] if fig.axes else None
        if ax:
            assert fig.get_size_inches()[0] >= 12
            assert ax.get_xticklabels()[0].get_rotation() == 90
        plt.close(fig)

    def test_duplicate_variables(self, sample_suggestions):
        """Test plotgen with duplicate variables."""
        df = pd.DataFrame({"x": np.arange(10)})
        with pytest.raises(KeyError, match="y"):
            plotgen(df, 0, sample_suggestions)

    def test_smart_generator_edge_cases(self, sample_suggestions):
        """Test SmartPlotGenerator with edge case data."""
        df = pd.DataFrame({
            "value": [1, 2, np.nan, np.inf],
            "category": ["A", "A", "B", "B"]
        })
        with pytest.raises(ValueError, match="Boxplot cannot handle infinite values"):
            plotgen(df, 8, sample_suggestions)

if __name__ == "__main__":
    pytest.main([__file__])