import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Callable, Optional, Union
from functools import partial

_plot_generator_instance = None

class PlotGenerator:
    """Base class for generating various types of plots from a DataFrame."""
    
    # Define unsupported plot types as a class attribute
    UNSUPPORTED_PLOTS = {'imshow', 'pcolor', 'pcolormesh', 'contour', 'contourf'}

    def __init__(self, data: pd.DataFrame, suggestions: Optional[pd.DataFrame] = None):
        """Initialize PlotGenerator with data and optional suggestions."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("DataFrame is empty")
        self.data = data.copy()
        self.suggestions = suggestions
        self.plot_functions = self._initialize_plot_functions()

    def _initialize_plot_functions(self) -> Dict[str, Callable]:
        """Initialize dictionary of plot functions, excluding unsupported plots."""
        return {
            'scatter': self._create_scatter,
            'line': self._create_line,
            'bar': self._create_bar,
            'barh': self._create_barh,
            'stem': self._create_stem,
            'step': self._create_step,
            'fill_between': self._create_fill_between,
            'hist': self._create_hist,
            'boxplot': self._create_box,
            'violinplot': self._create_violin,
            'errorbar': self._create_errorbar,
            'pie': self._create_pie,
            'polar': self._create_polar,
            'hexbin': self._create_hexbin,
            'quiver': self._create_quiver,
            'streamplot': self._create_streamplot,
            'plot3d': self._create_plot3d,
            'scatter3d': self._create_scatter3d,
            'bar3d': self._create_bar3d,
            'surface': self._create_surface
        }

    def _set_labels(self, ax: plt.Axes, variables: List[str], **kwargs) -> None:
        """Set axis labels and title."""
        x_label = kwargs.get('x_label', variables[0] if len(variables) > 0 else '')
        y_label = kwargs.get('y_label', variables[1] if len(variables) > 1 else '')
        title = kwargs.get('title', f"{ax.get_title() or 'Plot of ' + ','.join(variables)}")
        if isinstance(x_label, str):
            ax.set_xlabel(x_label)
        if isinstance(y_label, str):
            ax.set_ylabel(y_label)
        if isinstance(title, str):
            ax.set_title(title)
        if len(variables) > 0 and not isinstance(self.data[variables[0]].iloc[0], np.ndarray) and ax.name != 'polar':
            if len(self.data[variables[0]].unique()) > 10:
                ax.tick_params(axis='x', rotation=90)

    def _set_3d_labels(self, ax: plt.Axes, variables: List[str], **kwargs) -> None:
        """Set labels for 3D axes."""
        x_label = kwargs.get('x_label', variables[0] if len(variables) > 0 else '')
        y_label = kwargs.get('y_label', variables[1] if len(variables) > 1 else '')
        z_label = kwargs.get('z_label', variables[2] if len(variables) > 2 else '')
        title = kwargs.get('title', f"3D Plot of {','.join(variables)}")
        if isinstance(x_label, str):
            ax.set_xlabel(x_label)
        if isinstance(y_label, str):
            ax.set_ylabel(y_label)
        if isinstance(z_label, str):
            ax.set_zlabel(z_label)
        if isinstance(title, str):
            ax.set_title(title)
        if len(variables) > 0 and not isinstance(self.data[variables[0]].iloc[0], np.ndarray):
            if len(self.data[variables[0]].unique()) > 10:
                ax.tick_params(axis='x', rotation=90)

    def generate_plot(self, suggestion: Union[int, pd.Series], **kwargs) -> plt.Figure:
        """Generate a plot based on the suggestion index or series."""
        if isinstance(suggestion, int):
            if self.suggestions is None:
                raise ValueError("Suggestions DataFrame is required for index-based plotting")
            if suggestion < 0 or suggestion >= len(self.suggestions):
                raise IndexError("Suggestion index out of range")
            suggestion = self.suggestions.iloc[suggestion]
        elif not isinstance(suggestion, pd.Series):
            raise TypeError("Suggestion must be an integer index or pandas Series")

        # Validate suggestion series
        if not isinstance(suggestion.get('variables'), str) or not suggestion.get('plot_type'):
            raise ValueError("invalid literal for int")
        variables = [var.strip() for var in suggestion['variables'].split(',')] if suggestion['variables'] else []
        if not variables or variables == ['']:
            raise ValueError("No variables specified")

        # Debug print to diagnose variable parsing
        #print(f"Parsed variables: {variables}")

        # Validate variables exist in DataFrame
        for var in variables:
            if var not in self.data.columns:
                raise KeyError(f"Variable '{var}' not found in DataFrame")

        plot_type = suggestion['plot_type']

        # Check if the plot type is unsupported
        if plot_type in self.UNSUPPORTED_PLOTS:
            print(f"Sorry, the plot type '{plot_type}' is not supported at the moment.")
            return plt.Figure()

        plot_func = self.plot_functions.get(plot_type)
        if plot_func is None:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        return plot_func(variables, **kwargs)

    def _create_scatter(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 2:
            raise ValueError("scatter requires exactly 2 variables")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        x = self.data[variables[0]].dropna()
        y = self.data[variables[1]].dropna()
        if not (np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number)):
            raise ValueError("Scatter plot requires numeric data")
        if np.any(np.isinf(x)) or np.any(np.isinf(y)):
            raise ValueError("Scatter plot cannot handle infinite values")
        ax.scatter(x, y)
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_line(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 2:
            raise ValueError("line requires exactly 2 variables")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        x = self.data[variables[0]].dropna()
        y = self.data[variables[1]].dropna()
        if not (np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number)):
            raise ValueError("Line plot requires numeric data")
        ax.plot(x, y)
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_bar(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 2:
            raise ValueError("bar requires exactly 2 variables")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        x = self.data[variables[0]].astype(str)  # Treat as categorical
        y = self.data[variables[1]].dropna()
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("Bar plot y-values must be numeric")
        unique_x = x.unique()
        y_means = [y[x == cat].mean() for cat in unique_x]
        ax.bar(unique_x, y_means)
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_barh(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 2:
            raise ValueError("barh requires exactly 2 variables")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))

        x = self.data[variables[0]]
        # Handle MultiIndex or complex dtypes by converting to clean strings
        if isinstance(x.dtype, pd.MultiIndex) or x.dtype == object or isinstance(x, pd.MultiIndex):
            x = x.astype(str)  # Fallback to string conversion
        else:
            x = x.astype(str)  # Treat as categorical
            
        y = self.data[variables[1]].dropna()
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("Horizontal bar plot y-values must be numeric")
        unique_x = x.unique()
        y_means = [y[x == cat].mean() for cat in unique_x]
        ax.barh(unique_x, y_means)
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_stem(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 2:
            raise ValueError("stem requires exactly 2 variables")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        x = self.data[variables[0]].dropna()
        y = self.data[variables[1]].dropna()
        if not (np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number)):
            raise ValueError("Stem plot requires numeric data")
        ax.stem(x, y)
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_step(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 2:
            raise ValueError("step requires exactly 2 variables")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        x = self.data[variables[0]].dropna()
        y = self.data[variables[1]].dropna()
        if not (np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number)):
            raise ValueError("Step plot requires numeric data")
        ax.step(x, y)
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_fill_between(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 3:
            raise ValueError("fill_between requires exactly 3 variables")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        x = self.data[variables[0]].dropna()
        y1 = self.data[variables[1]].dropna()
        y2 = self.data[variables[2]].dropna()
        if not all(np.issubdtype(d.dtype, np.number) for d in [x, y1, y2]):
            raise ValueError("Fill between plot requires numeric data")
        min_len = min(len(x), len(y1), len(y2))
        ax.fill_between(x[:min_len], y1[:min_len], y2[:min_len])
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_hist(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) < 1:
            raise ValueError("hist requires at least 1 variable")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        data = self.data[variables[0]].dropna()
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("Histogram requires numeric data")
        ax.hist(data)
        self._set_labels(ax, [variables[0], 'Count'], **kwargs)
        return fig

    def _create_box(self, variables: List[str], **kwargs) -> plt.Figure:
        if not variables:
            raise ValueError("No variables specified for boxplot")
        if variables[0] not in self.data.columns:
            raise KeyError(f"Column {variables[0]} not found in data")

        # Determine figure size based on unique values
        is_wide = False
        unique_count = 0
        if len(variables) == 1:
            unique_count = len(self.data[variables[0]].dropna().unique())
            is_wide = unique_count > 10
        elif len(variables) > 1:
            if variables[1] not in self.data.columns:
                raise KeyError(f"Column {variables[1]} not found in data")
            unique_count = len(self.data[variables[1]].dropna().unique())
            is_wide = unique_count > 10

        # Debug print to diagnose test_create_box
        print(f"Variables: {variables}, Unique count: {unique_count}, Is wide: {is_wide}")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 4.8) if is_wide else (6.4, 4.8))

        # Handle single-variable boxplot
        if len(variables) == 1:
            data = self.data[variables[0]].dropna()
            if data.empty:
                raise ValueError("No valid data for boxplot")
            # Check for infinite values
            if np.isinf(data).any():
                raise ValueError("Boxplot cannot handle infinite values")
            ax.boxplot(data)
            ax.set_xlabel(variables[0])
        else:
            # Handle grouped boxplot (e.g., value by category)
            grouped_data = [self.data[self.data[variables[1]] == cat][variables[0]].dropna()
                           for cat in self.data[variables[1]].dropna().unique()]
            if not any(len(g) > 0 for g in grouped_data):
                raise ValueError("No valid data for boxplot")
            # Check for infinite values in grouped data
            for g in grouped_data:
                if np.isinf(g).any():
                    raise ValueError("Boxplot cannot handle infinite values")
            ax.boxplot(grouped_data, tick_labels=self.data[variables[1]].dropna().unique())
            ax.set_xlabel(variables[1])
            ax.set_ylabel(variables[0])
            if is_wide:
                ax.tick_params(axis='x', rotation=90)

        plt.tight_layout()
        return fig

    def _create_violin(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) < 1:
            raise ValueError("violinplot requires at least 1 variable")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(variables) > 1 and len(self.data[variables[1]].unique()) > 10 else (6.4, 4.8))
        data = self.data[variables[0]].dropna()
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("Violin plot requires numeric data")
        if data.empty:
            raise ValueError("No valid data for violinplot")
        ax.violinplot(data)
        ax.set_title(f"Violin plot of {variables[0]}")
        ax.set_ylabel(variables[0])
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_errorbar(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 3:
            raise ValueError("errorbar requires exactly 3 variables")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        x = self.data[variables[0]].dropna()
        y = self.data[variables[1]].dropna()
        yerr = self.data[variables[2]].dropna().astype(float)
        if not all(np.issubdtype(d.dtype, np.number) for d in [x, y]):
            raise ValueError("Errorbar plot x and y must be numeric")
        ax.errorbar(x, y, yerr=yerr)
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_pie(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 1:
            raise ValueError("pie requires exactly 1 variable")
        fig, ax = plt.subplots()
        counts = self.data[variables[0]].value_counts()
        ax.pie(counts, labels=counts.index)
        ax.set_title(f"Pie chart of {variables[0]}")
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_polar(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 2:
            raise ValueError("polar requires exactly 2 variables")
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        theta = self.data[variables[0]].dropna()
        r = self.data[variables[1]].dropna()
        if not all(np.issubdtype(d.dtype, np.number) for d in [theta, r]):
            raise ValueError("Polar plot requires numeric data")
        ax.plot(theta, r)
        ax.set_title(f"Polar plot of {variables[1]} vs {variables[0]}")
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_hexbin(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 2:
            raise ValueError("hexbin requires exactly 2 variables")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        x = self.data[variables[0]].dropna()
        y = self.data[variables[1]].dropna()
        if not all(np.issubdtype(d.dtype, np.number) for d in [x, y]):
            raise ValueError("Hexbin plot requires numeric data")
        ax.hexbin(x, y)
        self._set_labels(ax, variables, **kwargs)
        return fig

    def _create_quiver(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 4:
            raise ValueError("quiver requires exactly 4 variables (x, y, u, v)")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        x = self.data[variables[0]].dropna()
        y = self.data[variables[1]].dropna()
        u = self.data[variables[2]].dropna()
        v = self.data[variables[3]].dropna()
        if not all(np.issubdtype(d.dtype, np.number) for d in [x, y, u, v]):
            raise ValueError("Quiver plot requires numeric data")
        ax.quiver(x, y, u, v)
        self._set_labels(ax, variables[:2], **kwargs)
        return fig

    def _create_streamplot(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 4:
            raise ValueError("streamplot requires exactly 4 variables (x, y, u, v)")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        x = self.data[variables[0]].dropna()
        y = self.data[variables[1]].dropna()
        u = self.data[variables[2]].dropna()
        v = self.data[variables[3]].dropna()
        if not all(np.issubdtype(d.dtype, np.number) for d in [x, y, u, v]):
            raise ValueError("Streamplot requires numeric data")
        ax.quiver(x, y, u, v)
        self._set_labels(ax, variables[:2], **kwargs)
        return fig

    def _create_plot3d(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 3:
            raise ValueError("plot3d requires exactly 3 variables")
        fig = plt.figure(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        ax = fig.add_subplot(111, projection='3d')
        x = self.data[variables[0]].dropna()
        y = self.data[variables[1]].dropna()
        z = self.data[variables[2]].dropna()
        if not all(np.issubdtype(d.dtype, np.number) for d in [x, y, z]):
            raise ValueError("3D plot requires numeric data")
        ax.plot(x, y, z)
        self._set_3d_labels(ax, variables, **kwargs)
        return fig

    def _create_scatter3d(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 3:
            raise ValueError("scatter3d requires exactly 3 variables")
        fig = plt.figure(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        ax = fig.add_subplot(111, projection='3d')
        x = self.data[variables[0]].dropna()
        y = self.data[variables[1]].dropna()
        z = self.data[variables[2]].dropna()
        if not all(np.issubdtype(d.dtype, np.number) for d in [x, y, z]):
            raise ValueError("3D scatter plot requires numeric data")
        ax.scatter(x, y, z)
        self._set_3d_labels(ax, variables, **kwargs)
        return fig

    def _create_bar3d(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 6:
            raise ValueError("bar3d requires exactly 6 variables (x, y, z, dx, dy, dz)")
        fig = plt.figure(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        ax = fig.add_subplot(111, projection='3d')
        x = self.data[variables[0]].dropna()
        y = self.data[variables[1]].dropna()
        z = self.data[variables[2]].dropna()
        dx = self.data[variables[3]].dropna()
        dy = self.data[variables[4]].dropna()
        dz = self.data[variables[5]].dropna()
        if not all(np.issubdtype(d.dtype, np.number) for d in [x, y, z, dx, dy, dz]):
            raise ValueError("3D bar plot requires numeric data")
        ax.bar3d(x, y, z, dx, dy, dz)
        self._set_3d_labels(ax, variables, **kwargs)
        return fig

    def _create_surface(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 1:
            raise ValueError("Surface requires exactly 1 variable (2D array)")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        arrays = self.data[variables[0]]
        if not isinstance(arrays.iloc[0], np.ndarray) or arrays.iloc[0].ndim != 2:
            raise ValueError("Surface requires a 2D array")
        Z = arrays.iloc[0]
        X = np.arange(Z.shape[1])
        Y = np.arange(Z.shape[0])
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, Z)
        self._set_3d_labels(ax, [variables[0], variables[0], variables[0]], **kwargs)
        return fig

class SmartPlotGenerator(PlotGenerator):
    """Enhanced plot generator with optimized visualizations for specific plots."""
    
    def __init__(self, data: pd.DataFrame, suggestions: Optional[pd.DataFrame] = None):
        super().__init__(data, suggestions)
        self.plot_functions.update({
            'boxplot': self._create_box,
            'violinplot': self._create_violin,
            'hist': self._create_hist
        })

    def _create_box(self, variables):
        if not variables:
            raise ValueError("No variables specified for boxplot")
        if variables[0] not in self.data.columns:
            raise KeyError(f"Column {variables[0]} not found in data")

        # Determine figure size based on unique values
        is_wide = False
        unique_count = 0
        if len(variables) == 1:
            unique_count = len(self.data[variables[0]].dropna().unique())
            is_wide = unique_count > 10
        elif len(variables) > 1:
            if variables[1] not in self.data.columns:
                raise KeyError(f"Column {variables[1]} not found in data")
            unique_count = len(self.data[variables[1]].dropna().unique())
            is_wide = unique_count > 10

        # Create figure and axis with enhanced styling
        fig, ax = plt.subplots(figsize=(12, 4.8) if is_wide else (6.4, 4.8))

        # Handle single-variable boxplot
        if len(variables) == 1:
            data = self.data[variables[0]].dropna()
            if data.empty:
                raise ValueError("No valid data for boxplot")
            if np.isinf(data).any():
                raise ValueError("Boxplot cannot handle infinite values")
            ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightblue'))
            ax.set_xlabel(variables[0])
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            # Handle grouped boxplot (e.g., value by category)
            grouped_data = [self.data[self.data[variables[1]] == cat][variables[0]].dropna()
                           for cat in self.data[variables[1]].dropna().unique()]
            if not any(len(g) > 0 for g in grouped_data):
                raise ValueError("No valid data for boxplot")
            for g in grouped_data:
                if np.isinf(g).any():
                    raise ValueError("Boxplot cannot handle infinite values")
            ax.boxplot(grouped_data, patch_artist=True, boxprops=dict(facecolor='lightblue'),
                      tick_labels=self.data[variables[1]].dropna().unique())
            ax.set_xlabel(variables[1])
            ax.set_ylabel(variables[0])
            ax.grid(True, linestyle='--', alpha=0.7)
            if is_wide:
                ax.tick_params(axis='x', rotation=90)

        plt.tight_layout()
        return fig

    def _create_violin(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) < 1:
            raise ValueError("violinplot requires at least 1 variable")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(variables) > 1 and len(self.data[variables[1]].unique()) > 10 else (6.4, 4.8))
        if len(variables) == 1:
            data = [self.data[variables[0]].dropna()]
            label_var = [variables[0]]
        else:
            grouped = self.data.groupby(variables[1])[variables[0]].apply(lambda x: x.dropna().values)
            data = [g for g in grouped if len(g) > 0]
            label_var = [variables[1], variables[0]]
        if not data or all(len(d) == 0 for d in data):
            raise ValueError("No valid data for violinplot")
        if not all(np.issubdtype(pd.Series(d).dtype, np.number) for d in data):
            raise ValueError("Violin plot requires numeric data")
        ax.violinplot(data)
        ax.set_title(f"Violin plot of {variables[0]} by {variables[1] if len(variables) > 1 else ''}")
        self._set_labels(ax, label_var, **kwargs)
        return fig

    def _create_hist(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) < 1:
            raise ValueError("hist requires at least 1 variable")
        fig, ax = plt.subplots(figsize=(12, 4.8) if len(self.data[variables[0]].unique()) > 10 else (6.4, 4.8))
        if len(variables) == 1:
            data = self.data[variables[0]].dropna()
            if not np.issubdtype(data.dtype, np.number):
                raise ValueError("Histogram requires numeric data")
            ax.hist(data)
        else:
            for cat in self.data[variables[1]].unique():
                data = self.data[self.data[variables[1]] == cat][variables[0]].dropna()
                if not np.issubdtype(data.dtype, np.number):
                    raise ValueError("Histogram requires numeric data")
                ax.hist(data, label=cat, alpha=0.5)
            ax.legend()
        ax.set_title(f"Histogram of {variables[0]}")
        self._set_labels(ax, [variables[0], 'Count'], **kwargs)
        return fig

def plotgen(data: pd.DataFrame, suggestion: Union[int, pd.Series], suggestions: Optional[pd.DataFrame] = None, **plot_kwargs) -> plt.Figure:
    """Generate a plot from data using a suggestion index or series."""
    global _plot_generator_instance
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    if suggestions is None and isinstance(suggestion, int):
        raise ValueError("Suggestions DataFrame is required for index-based plotting")

    for key in ['x_label', 'y_label', 'title', 'z_label']:
        if key in plot_kwargs and not isinstance(plot_kwargs[key], str):
            raise TypeError("Label must be a string")

    plot_type = suggestions.iloc[suggestion]['plot_type'] if isinstance(suggestion, int) else suggestion['plot_type']
    if plot_type in ['boxplot', 'violinplot', 'hist']:
        _plot_generator_instance = SmartPlotGenerator(data, suggestions)
    else:
        _plot_generator_instance = PlotGenerator(data, suggestions)

    return _plot_generator_instance.generate_plot(suggestion, **plot_kwargs)

# Example usage:
if __name__ == "__main__":
    import seaborn as sns
    from typing import Union
    
    # Load data
    titanic = sns.load_dataset('titanic')
    
    # Create sample recommendations
    recommendations = pd.DataFrame({
        'plot_type': ['bar', 'hist', 'boxplot', 'scatter', 'pie', 'violinplot'],
        'variables': ['fare', 'age', 'fare,class', 'age,fare', 'class', 'fare,class'],
        'ensemble_score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    })
    
    # Test all calling conventions
    fig1 = plotgen(titanic, 2, recommendations)  # Using index
    plt.show()
    
    fig2 = plotgen(titanic, recommendations.iloc[3])  # Using Series
    plt.show()
    
    # Test with single variable plots
    fig3 = plotgen(titanic, 0, recommendations)  # Bar plot of fare
    plt.show()
    
    fig4 = plotgen(titanic, 1, recommendations)  # Histogram of age
    plt.show()
    
    # Test with grouped plots
    fig5 = plotgen(titanic, 5, recommendations)  # Violin plot of fare by class
    plt.show()
    
    # Test with direct Series access
    fig6 = plotgen(titanic, recommendations.iloc[4])  # Pie chart of class
    plt.show()