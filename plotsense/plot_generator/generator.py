import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Union


class PlotGenerator:
    """
    A class to generate various types of plots based on suggestions.
    It uses matplotlib for plotting and can handle both univariate and bivariate cases.
    """

    def __init__(self, data: pd.DataFrame, suggestions: Optional[pd.DataFrame] = None):
        """
        Initialize with data and plot suggestions.

        Args:
            data: DataFrame containing the actual data
            suggestions: DataFrame with plot suggestions
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("DataFrame is empty")
        if not isinstance(suggestions, pd.DataFrame):
            raise TypeError("Suggestions must be a pandas DataFrame")
        if suggestions.empty:
            raise ValueError("Suggestions DataFrame is empty")
        if 'plot_type' not in suggestions.columns or 'variables' not in suggestions.columns:
            raise ValueError("Suggestions DataFrame must contain 'plot_type' and 'variables' columns")

        self.data = data.copy()
        self.suggestions = suggestions
        self.plot_functions = self._initialize_plot_functions()

    def generate_plot(self, suggestion_index: int, **kwargs) -> plt.Figure:
        """
        Generate a plot based on the suggestion at given index.

        Args:
            suggestion_index: Index of the suggestion in dataframe
            **kwargs: Additional arguments for the plot

        Returns:
            matplotlib Figure object
        """
        # if suggestion_index < 0 or suggestion_index >= len(self.suggestions):
        #     raise IndexError("Suggestion index out of range")
        if not isinstance(suggestion_index, int):
            raise TypeError("Suggestion index must be an integer")
        if not isinstance(kwargs, dict):
            raise TypeError("Additional arguments must be provided as a dictionary")
        if self.suggestions.empty:
            raise ValueError("No suggestions available to generate a plot")
        if self.data.empty:
            raise ValueError("No data available to generate a plot")
        if not isinstance(self.suggestions, pd.DataFrame):
            raise TypeError("Suggestions must be a pandas DataFrame")
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        if self.suggestions.empty:
            raise ValueError("Suggestions DataFrame is empty")
        if self.data.empty:
            raise ValueError("DataFrame is empty")

        suggestion = self.suggestions.iloc[suggestion_index]
        plot_type = suggestion['plot_type'].lower()
        variables = [v.strip() for v in suggestion['variables'].split(',')]

        if plot_type not in self.plot_functions:
            print(f"This version of PlotSense does not support plot type: {plot_type}")
            return None

        plot_func = self.plot_functions[plot_type]
        return plot_func(variables, **kwargs)

    def _initialize_plot_functions(self) -> Dict[str, callable]:
        """Initialize all matplotlib plot functions with their requirements."""
        return {
            # Basic plots
            'scatter': self._create_scatter,
            'bar': self._create_bar,
            'barh': self._create_barh,

            # Statistical plots
            'hist': self._create_hist,
            'boxplot': self._create_box,
            'violinplot': self._create_violin,

            # Specialized plots
            'pie': self._create_pie,
            'hexbin': self._create_hexbin

        }

    # ========== Basic Plot Functions ==========

    def _create_scatter(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) < 2:
            raise ValueError("scatter requires at least 2 variables (x, y)")
        fig, ax = plt.subplots()
        ax.scatter(self.data[variables[0]], self.data[variables[1]], **kwargs)
        self._set_labels(ax, variables)
        ax.set_title(f"Scatter: {variables[0]} vs {variables[1]}")
        return fig

    def _create_bar(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)

        # Define font sizes
        tick_fontsize = kwargs.pop('tick_fontsize', 12)
        label_fontsize = kwargs.pop('label_fontsize', 14)
        title_fontsize = kwargs.pop('title_fontsize', 16)

        if len(variables) == 1:
            # Single variable - show value counts
            value_counts = self.data[variables[0]].value_counts().sort_values(ascending=False)
            ax.bar(value_counts.index.astype(str), value_counts.values, **kwargs)
            ax.set_xlabel(variables[0] if x_label is None else x_label, fontsize=label_fontsize)
            ax.set_ylabel('Count' if y_label is None else y_label, fontsize=label_fontsize)
            ax.set_title(f"Bar plot of {variables[0]}" if title is None else title, fontsize=title_fontsize)
            ax.tick_params(axis='x', labelsize=tick_fontsize)
            ax.tick_params(axis='y', labelsize=tick_fontsize)

            if len(value_counts) > 10:
                fig.set_size_inches(max(12, len(value_counts)), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        else:
            # First variable is numeric, second is categorical
            grouped = self.data.groupby(variables[1])[variables[0]].mean().sort_values(ascending=False)
            ax.bar(grouped.index.astype(str), grouped.values, **kwargs)
            ax.set_xlabel(variables[1] if x_label is None else x_label, fontsize=label_fontsize)
            ax.set_ylabel(f"{variables[0]}" if y_label is None else y_label, fontsize=label_fontsize)
            ax.set_title(f"{variables[0]} by {variables[1]}" if title is None else title, fontsize=title_fontsize)
            ax.tick_params(axis='x', labelsize=tick_fontsize)
            ax.tick_params(axis='y', labelsize=tick_fontsize)

            if len(grouped) > 10:
                fig.set_size_inches(max(12, len(grouped)), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        return fig

    def _create_barh(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)

        # Define font sizes
        tick_fontsize = kwargs.pop('tick_fontsize', 12)
        label_fontsize = kwargs.pop('label_fontsize', 14)
        title_fontsize = kwargs.pop('title_fontsize', 16)

        if len(variables) == 1:
            # Single variable - show value counts
            value_counts = self.data[variables[0]].value_counts()
            ax.barh(value_counts.index.astype(str), value_counts.values, **kwargs)
            ax.set_xlabel(variables[0] if x_label is None else x_label, fontsize=label_fontsize)
            ax.set_ylabel('Count' if y_label is None else y_label, fontsize=label_fontsize)
            ax.set_title(f"Bar plot of {variables[0]}" if title is None else title, fontsize=title_fontsize)
            ax.tick_params(axis='x', labelsize=tick_fontsize)
            ax.tick_params(axis='y', labelsize=tick_fontsize)

            if len(value_counts) > 10:
                fig.set_size_inches(max(12, len(value_counts)), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        else:
            # First variable is numeric, second is categorical
            grouped = self.data.groupby(variables[1])[variables[0]].mean()
            ax.barh(grouped.index.astype(str), grouped.values, **kwargs)
            ax.set_xlabel(variables[1] if x_label is None else x_label, fontsize=label_fontsize)
            ax.set_ylabel(f"{variables[0]}" if y_label is None else y_label, fontsize=label_fontsize)
            ax.set_title(f"{variables[0]} by {variables[1]}" if title is None else title, fontsize=title_fontsize)
            ax.tick_params(axis='x', labelsize=tick_fontsize)
            ax.tick_params(axis='y', labelsize=tick_fontsize)

            if len(grouped) > 10:
                fig.set_size_inches(max(12, len(grouped)), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        return fig

    # ========== Statistical Plot Functions ==========

    def _create_hist(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.hist(self.data[variables[0]], **kwargs)
        ax.set_xlabel(variables[0])
        ax.set_ylabel('Frequency')
        ax.set_title(f"Histogram of {variables[0]}")
        return fig

    def _create_box(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        ax.boxplot(self.data[variables[0]], **kwargs)
        ax.set_ylabel(variables[0])
        ax.set_title(f"Box plot of {variables[0]}")

        return fig

    def _create_violin(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        ax.violinplot(self.data[variables[0]], **kwargs)
        ax.set_ylabel(variables[0])
        ax.set_title(f"Violin plot of {variables[0]}")
        return fig

    # ========== Specialized Plot Functions ==========

    def _create_pie(self, variables: List[str], **kwargs) -> plt.Figure:
        value_counts = self.data[variables[0]].value_counts()
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', **kwargs)
        ax.set_title(f"Pie chart of {variables[0]}")
        return fig

    def _create_hexbin(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.hexbin(self.data[variables[0]], self.data[variables[1]], **kwargs)
        self._set_labels(ax, variables)
        ax.set_title(f"Hexbin: {variables[0]} vs {variables[1]}")
        return fig

    # ========== Helper Methods ==========

    def _set_labels(self, ax, variables: List[str]):
        """Set labels for x and y axes based on variables."""
        if len(variables) > 0:
            ax.set_xlabel(variables[0])
        if len(variables) > 1:
            ax.set_ylabel(variables[1])


class SmartPlotGenerator(PlotGenerator):
    def _create_box(self, variables: List[str], **kwargs) -> plt.Figure:
        """Enhanced boxplot that handles both univariate and bivariate cases with NaN handling."""
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        if len(variables) == 1:
            # Univariate case - single numerical variable
            data = self.data[variables[0]].dropna()  # Remove NaN values
            if len(data) == 0:
                raise ValueError(f"No valid data remaining after dropping NaN values for {variables[0]}")
            ax.boxplot(data, **kwargs)
            ax.set_ylabel(variables[0])
            ax.set_title(f"Box plot of {variables[0]}")
        elif len(variables) >= 2:
            # Bivariate case - numerical vs categorical
            numerical_var = variables[0]
            categorical_var = variables[1]

            # Clean data - remove rows where either variable is NaN
            clean_data = self.data[[numerical_var, categorical_var]].dropna()
            if len(clean_data) == 0:
                raise ValueError(f"No valid data remaining after cleaning {numerical_var} and {categorical_var}")

            # Group data by categorical variable
            grouped_data = [clean_data[clean_data[categorical_var] == cat][numerical_var]
                            for cat in clean_data[categorical_var].unique()]

            # Filter out empty groups
            grouped_data = [group for group in grouped_data if len(group) > 0]
            if not grouped_data:
                raise ValueError("No valid groups remaining after filtering")

            ax.boxplot(grouped_data, **kwargs)
            ax.set_xticklabels(clean_data[categorical_var].unique())
            ax.set_xlabel(categorical_var)
            ax.set_ylabel(numerical_var)
            ax.set_title(f"Box plot of {numerical_var} by {categorical_var}")
        else:
            raise ValueError("Box plot requires at least 1 variable")

        return fig

    def _create_violin(self, variables: List[str], **kwargs) -> plt.Figure:
        """Enhanced violin plot that handles both univariate and bivariate cases with NaN handling."""
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        if len(variables) == 1:
            # Univariate case - single numerical variable
            data = self.data[variables[0]].dropna()  # Remove NaN values
            if len(data) == 0:
                raise ValueError(f"No valid data remaining after dropping NaN values for {variables[0]}")
            ax.violinplot(data, **kwargs)
            ax.set_ylabel(variables[0])
            ax.set_title(f"Violin plot of {variables[0]}")
        elif len(variables) >= 2:
            # Bivariate case - numerical vs categorical
            numerical_var = variables[0]
            categorical_var = variables[1]

            # Clean data - remove rows where either variable is NaN
            clean_data = self.data[[numerical_var, categorical_var]].dropna()
            if len(clean_data) == 0:
                raise ValueError(f"No valid data remaining after cleaning {numerical_var} and {categorical_var}")

            # Group data by categorical variable
            grouped_data = [clean_data[clean_data[categorical_var] == cat][numerical_var]
                            for cat in clean_data[categorical_var].unique()]

            # Filter out empty groups
            grouped_data = [group for group in grouped_data if len(group) > 0]
            if not grouped_data:
                raise ValueError("No valid groups remaining after filtering")

            ax.violinplot(grouped_data, **kwargs)
            ax.set_xticks(np.arange(1, len(grouped_data) + 1))
            ax.set_xticklabels(clean_data[categorical_var].unique())
            ax.set_xlabel(categorical_var)
            ax.set_ylabel(numerical_var)
            ax.set_title(f"Violin plot of {numerical_var} by {categorical_var}")
        else:
            raise ValueError("Violin plot requires at least 1 variable")

        return fig

    def _create_hist(self, variables: List[str], **kwargs) -> plt.Figure:
        """Enhanced histogram that can handle grouping by a second variable."""
        fig, ax = plt.subplots(figsize=(12, 8))

        if len(variables) == 1:
            # Simple histogram
            data = self.data[variables[0]].dropna()
            if len(data) == 0:
                raise ValueError(f"No valid data remaining for {variables[0]}")

            ax.hist(data, **kwargs)
            ax.set_xlabel(variables[0])
            ax.set_ylabel('Frequency')
            ax.set_title(f"Histogram of {variables[0]}")
        elif len(variables) >= 2:
            # Grouped histogram
            numerical_var = variables[0]
            categorical_var = variables[1]

            # Clean data
            clean_data = self.data[[numerical_var, categorical_var]].dropna()
            if len(clean_data) == 0:
                raise ValueError(f"No valid data remaining after cleaning {numerical_var} and {categorical_var}")

            # Get unique categories
            categories = clean_data[categorical_var].unique()

            # Set default colors if not provided
            if 'color' not in kwargs and 'colors' not in kwargs:
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            else:
                colors = [kwargs.pop('color')] * len(categories) if 'color' in kwargs else kwargs.pop('colors')

            # Plot each group
            for i, cat in enumerate(categories):
                ax.hist(clean_data[clean_data[categorical_var] == cat][numerical_var],
                        alpha=0.5,
                        label=str(cat),
                        color=colors[i % len(colors)],
                        **kwargs)

            ax.set_xlabel(numerical_var)
            ax.set_ylabel('Frequency')
            ax.set_title(f"Histogram of {numerical_var} by {categorical_var}")
            ax.legend()
        else:
            raise ValueError("Histogram requires at least 1 variable")

        return fig

    def _create_scatter(self, variables: List[str],
                        size_scale: float = 100.0,
                        **kwargs) -> plt.Figure:
        """
        Create a scatter plot with optional color and size dimensions.

        Parameters:
        -----------
        variables : List[str]
            - 2 variables: x, y
            - 3 variables: x, y, color
            - 4 variables: x, y, color, size
        size_scale : float
            Scaling factor for bubble sizes (default: 100)

        Returns:
        --------
        matplotlib.figure.Figure
        """
        if len(variables) < 2:
            raise ValueError("Scatter requires at least 2 variables (x, y)")
        if len(variables) > 4:
            raise ValueError("Scatter supports maximum 4 variables (x, y, color, size)")

        # Check data types
        for var in variables[:2]:  # x and y must be numeric
            try:
                is_numeric = np.issubdtype(self.data[var].dtype, np.number)
            except TypeError:
                # Handle StringDtype and other pandas types that np.issubdtype can't parse
                is_numeric = pd.api.types.is_numeric_dtype(self.data[var])

            if not is_numeric:
                raise ValueError(f"Variable '{var}' must be numeric")

        fig, ax = plt.subplots()
        scatter_params = {
            'x': self.data[variables[0]],
            'y': self.data[variables[1]],
        }

        # Handle color (3rd variable)
        if len(variables) >= 3:
            color_data = self.data[variables[2]]
            if pd.api.types.is_numeric_dtype(color_data):
                # For numeric color data, use continuous colormap
                scatter_params['c'] = color_data
                kwargs.setdefault('cmap', 'viridis')
            else:
                # For categorical data, convert to numeric codes
                scatter_params['c'] = pd.factorize(color_data)[0]
                kwargs.setdefault('cmap', 'tab10')

        # Handle size (4th variable)
        if len(variables) == 4:
            size_data = self.data[variables[3]]
            if not pd.api.types.is_numeric_dtype(size_data):
                raise ValueError(f"Size variable '{variables[3]}' must be numeric")

            # Normalize and scale sizes
            sizes = np.abs(size_data)  # Ensure positive
            sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-8) * size_scale
            scatter_params['s'] = sizes

        # Apply any additional kwargs
        scatter_params.update(kwargs)

        scatter = ax.scatter(**scatter_params)

        # Set labels and title
        self._set_labels(ax, variables[:2])  # Assuming this sets x and y labels
        title = f"Scatter: {variables[0]} vs {variables[1]}"
        if len(variables) >= 3:
            title += f" (colored by {variables[2]})"
            # Add colorbar for continuous data
            if pd.api.types.is_numeric_dtype(self.data[variables[2]]):
                fig.colorbar(scatter, ax=ax, label=variables[2])
        if len(variables) == 4:
            title += f" (sized by {variables[3]})"
        ax.set_title(title)

        return fig


# Global instance of the plot generator
_plot_generator_instance = None


def plotgen(
    df: pd.DataFrame,
    suggestion: Union[int, pd.Series],
    suggestions_df: Optional[pd.DataFrame] = None,
    **plot_kwargs
) -> plt.Figure:
    """
    Generate a plot based on visualization suggestions.

    Args:
        df: Input DataFrame containing the data to plot
        suggestion: Either an integer index or a pandas Series containing the suggestion row
        suggestions_df: DataFrame containing visualization suggestions (required if suggestion is an index)
        **plot_kwargs: Additional arguments to pass to the plot function

    Returns:
        matplotlib.Figure: The generated figure

    Example:
        # Using index (requires suggestions_df)
        fig = plotgen(df, 7, suggestions_df=recommendations)

        # Using direct row access with additional plot arguments
        fig = plotgen(df, recommendations.iloc[7], bins=30, color='red')

        # Using specific variable names
        fig = plotgen(df, recommendations.iloc[7], x='age', y='fare')
    """
    global _plot_generator_instance

    # Handle case where suggestion is a row from recommendations
    if isinstance(suggestion, pd.Series):
        # Create a temporary single-row suggestions DataFrame
        temp_df = pd.DataFrame([suggestion])
        # Initialize the plot generator with this single suggestion
        _plot_generator_instance = SmartPlotGenerator(df, temp_df)

        # Get the variables from the suggestion
        variables = [v.strip() for v in suggestion['variables'].split(',')]

        # Handle x, y, z arguments if provided
        if 'x' in plot_kwargs:
            variables[0] = plot_kwargs.pop('x')
        if 'y' in plot_kwargs and len(variables) > 1:
            variables[1] = plot_kwargs.pop('y')
        if 'z' in plot_kwargs and len(variables) > 2:
            variables[2] = plot_kwargs.pop('z')

        # Create a new suggestion with updated variables
        updated_suggestion = suggestion.copy()
        updated_suggestion['variables'] = ','.join(variables)
        temp_df = pd.DataFrame([updated_suggestion])
        _plot_generator_instance.suggestions = temp_df

        # Generate the plot
        return _plot_generator_instance.generate_plot(0, **plot_kwargs)

    # Handle case where suggestion is an index
    elif isinstance(suggestion, int):
        if suggestions_df is None:
            raise ValueError("suggestions_df must be provided when using an index")

        # Initialize the plot generator if it doesn't exist
        if _plot_generator_instance is None:
            _plot_generator_instance = SmartPlotGenerator(df, suggestions_df)
        else:
            # Update the data if the generator exists but the data changed
            if not _plot_generator_instance.data.equals(df):
                _plot_generator_instance.data = df

        # Get the variables from the suggestion
        suggestion_row = suggestions_df.iloc[suggestion]
        variables = [v.strip() for v in suggestion_row['variables'].split(',')]

        # Handle x, y, z arguments if provided
        if 'x' in plot_kwargs:
            variables[0] = plot_kwargs.pop('x')
        if 'y' in plot_kwargs and len(variables) > 1:
            variables[1] = plot_kwargs.pop('y')
        if 'z' in plot_kwargs and len(variables) > 2:
            variables[2] = plot_kwargs.pop('z')

        # Create a new suggestion with updated variables
        updated_suggestion = suggestion_row.copy()
        updated_suggestion['variables'] = ','.join(variables)
        suggestions_df.iloc[suggestion] = updated_suggestion
        _plot_generator_instance.suggestions = suggestions_df

        # Generate the plot
        return _plot_generator_instance.generate_plot(suggestion, **plot_kwargs)
    # else:
    #     raise TypeError("suggestion must be either an integer index or a pandas Series")
