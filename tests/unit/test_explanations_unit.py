"""
Unit tests for PlotExplainer functionality.

The PlotExplainer has been refactored with a new provider-based architecture.
These tests verify the basic structure and imports work correctly.
"""

import pytest
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
load_dotenv()


class TestExplanationsImports:
    """Test that explanation modules can be imported."""

    def test_import_plot_explainer(self):
        """Test that PlotExplainer can be imported."""
        from plotsense.explanations.explanations import PlotExplainer
        assert PlotExplainer is not None

    def test_import_explainer_function(self):
        """Test that explainer function can be imported."""
        from plotsense.explanations.explanations import explainer
        assert explainer is not None

    def test_plot_creation(self):
        """Test that plots can be created for explanation testing."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        assert fig is not None
        assert ax is not None
        plt.close(fig)
