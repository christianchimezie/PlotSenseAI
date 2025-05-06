import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from mpl_toolkits.mplot3d import Axes3D


class PlotGenerator:
    def __init__(self, data: pd.DataFrame, suggestions: pd.DataFrame):
        """
        Initialize with data and plot suggestions.
        
        Args:
            data: DataFrame containing the actual data
            suggestions: DataFrame with plot suggestions
        """
        self.data = data
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
        suggestion = self.suggestions.iloc[suggestion_index]
        plot_type = suggestion['plot_type'].lower()
        variables = [v.strip() for v in suggestion['variables'].split(',')]
        
        if plot_type not in self.plot_functions:
            raise ValueError(f"Unsupported plot type: {plot_type}")
            
        plot_func = self.plot_functions[plot_type]
        return plot_func(variables, **kwargs)
    
    def _initialize_plot_functions(self) -> Dict[str, callable]:
        """Initialize all matplotlib plot functions with their requirements."""
        return {
            # Basic plots
            'scatter': self._create_scatter,
            'line': self._create_line,
            'bar': self._create_bar,
            'barh': self._create_barh,
            'stem': self._create_stem,
            'step': self._create_step,
            'fill_between': self._create_fill_between,
            
            # Statistical plots
            'hist': self._create_hist,
            'boxplot': self._create_box,
            'violinplot': self._create_violin,
            'errorbar': self._create_errorbar,
            
            # 2D plots
            'imshow': self._create_imshow,
            'pcolor': self._create_pcolor,
            'pcolormesh': self._create_pcolormesh,
            'contour': self._create_contour,
            'contourf': self._create_contourf,
            
            # Specialized plots
            'pie': self._create_pie,
            'polar': self._create_polar,
            'hexbin': self._create_hexbin,
            'quiver': self._create_quiver,
            'streamplot': self._create_streamplot,
            
            # 3D plots
            'plot3d': self._create_plot3d,
            'scatter3d': self._create_scatter3d,
            'bar3d': self._create_bar3d,
            'surface': self._create_surface,
        }
    
    # ========== Basic Plot Functions ==========
    def _create_scatter(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.scatter(self.data[variables[0]], self.data[variables[1]], **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel(variables[1] if y_label is None else y_label)
        ax.set_title(f"Scatter: {variables[0]} vs {variables[1]}" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_line(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.plot(self.data[variables[0]], self.data[variables[1]], **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel(variables[1] if y_label is None else y_label)
        ax.set_title(f"Line: {variables[0]} vs {variables[1]}" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_bar(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
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
            ax.bar(value_counts.index.astype(str), value_counts.values, **kwargs)
            ax.set_xlabel(variables[0] if x_label is None else x_label, fontsize=label_fontsize)
            ax.set_ylabel('Count' if y_label is None else y_label, fontsize=label_fontsize)
            ax.set_title(f"Bar plot of {variables[0]}" if title is None else title, fontsize=title_fontsize)
            ax.tick_params(axis='x', labelsize=tick_fontsize)
            ax.tick_params(axis='y', labelsize=tick_fontsize)
            ax.grid(True, linestyle='--', alpha=0.7)
            if len(value_counts) > 10:
                fig.set_size_inches(max(12, len(value_counts) * 0.5), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        else:
            # First variable is numeric, second is categorical
            grouped = self.data.groupby(variables[1])[variables[0]].mean()
            ax.bar(grouped.index.astype(str), grouped.values, **kwargs)
            ax.set_xlabel(variables[1] if x_label is None else x_label, fontsize=label_fontsize)
            ax.set_ylabel(f"Mean {variables[0]}" if y_label is None else y_label, fontsize=label_fontsize)
            ax.set_title(f"Mean {variables[0]} by {variables[1]}" if title is None else title, fontsize=title_fontsize)
            ax.tick_params(axis='x', labelsize=tick_fontsize)
            ax.tick_params(axis='y', labelsize=tick_fontsize)
            ax.grid(True, linestyle='--', alpha=0.7)
            if len(grouped) > 10:
                fig.set_size_inches(max(12, len(grouped) * 0.5), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
        
    def _create_barh(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        if len(variables) == 1:
            # Single variable - show value_counts
            value_counts = self.data[variables[0]].value_counts()
            ax.barh(value_counts.index.astype(str), value_counts.values, **kwargs)
            ax.set_ylabel(variables[0] if y_label is None else y_label)
            ax.set_xlabel('Count' if x_label is None else x_label)
            unique_y = value_counts.index
            if len(unique_y) > 10:
                fig.set_size_inches(12, max(8, len(unique_y) * 0.5))
                plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
        else:
            # First variable is numeric, second is categorical
            grouped = self.data.groupby(variables[1])[variables[0]].value_counts()
            # Convert MultiIndex to string by joining levels
            labels = [','.join(map(str, idx)) for idx in grouped.index]
            ax.barh(labels, grouped.values, **kwargs)
            ax.set_ylabel(variables[1] if y_label is None else y_label)
            ax.set_xlabel(f"Mean {variables[0]}" if x_label is None else x_label)
            if len(labels) > 10:
                fig.set_size_inches(12, max(8, len(labels) * 0.5))
                plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
            
        ax.set_title(f"Horizontal bar plot: {variables[0]}" + 
                    (f" by {variables[1]}" if len(variables) > 1 else "") if title is None else title)
        return fig
    
    def _create_stem(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.stem(self.data[variables[0]], self.data[variables[1]], **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel(variables[1] if y_label is None else y_label)
        ax.set_title(f"Stem: {variables[0]} vs {variables[1]}" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_step(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.step(self.data[variables[0]], self.data[variables[1]], **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel(variables[1] if y_label is None else y_label)
        ax.set_title(f"Step: {variables[0]} vs {variables[1]}" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_fill_between(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) < 3:
            raise ValueError("fill_between requires at least 3 variables (x, y1, y2)")
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.fill_between(self.data[variables[0]], 
                       self.data[variables[1]], 
                       self.data[variables[2]], **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel(variables[1] if y_label is None else y_label)
        ax.set_title(f"Fill Between: {variables[1]} and {variables[2]}" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    # ========== Statistical Plot Functions ==========
    def _create_hist(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.hist(self.data[variables[0]], **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel('Frequency' if y_label is None else y_label)
        ax.set_title(f"Histogram of {variables[0]}" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_box(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.boxplot(self.data[variables[0]], **kwargs)
        ax.set_xlabel(x_label if x_label is not None else '')
        ax.set_ylabel(variables[0] if y_label is None else y_label)
        ax.set_title(f"Box plot of {variables[0]}" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_violin(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.violinplot(self.data[variables[0]], **kwargs)
        ax.set_xlabel(x_label if x_label is not None else '')
        ax.set_ylabel(variables[0] if y_label is None else y_label)
        ax.set_title(f"Violin plot of {variables[0]}" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_errorbar(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) < 3:
            raise ValueError("errorbar requires at least 3 variables (x, y, yerr)")
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.errorbar(self.data[variables[0]], 
                    self.data[variables[1]], 
                    yerr=self.data[variables[2]], **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel(variables[1] if y_label is None else y_label)
        ax.set_title(f"Errorbar: {variables[0]} vs {variables[1]}" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    # ========== 2D Plot Functions ==========
    def _create_imshow(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 1:
            raise ValueError("imshow requires exactly 1 variable (2D array)")
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.imshow(self.data[variables[0]], **kwargs)
        ax.set_xlabel(x_label if x_label is not None else '')
        ax.set_ylabel(y_label if y_label is not None else '')
        ax.set_title(f"Image show of {variables[0]}" if title is None else title)
        return fig
    
    def _create_pcolor(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 1:
            raise ValueError("pcolor requires exactly 1 variable (2D array)")
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.pcolor(self.data[variables[0]], **kwargs)
        ax.set_xlabel(x_label if x_label is not None else '')
        ax.set_ylabel(y_label if y_label is not None else '')
        ax.set_title(f"Pcolor of {variables[0]}" if title is None else title)
        return fig
    
    def _create_pcolormesh(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 1:
            raise ValueError("pcolormesh requires exactly 1 variable (2D array)")
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.pcolormesh(self.data[variables[0]], **kwargs)
        ax.set_xlabel(x_label if x_label is not None else '')
        ax.set_ylabel(y_label if y_label is not None else '')
        ax.set_title(f"Pcolormesh of {variables[0]}" if title is None else title)
        return fig
    
    def _create_contour(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 1:
            raise ValueError("contour requires exactly 1 variable (2D array)")
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.contour(self.data[variables[0]], **kwargs)
        ax.set_xlabel(x_label if x_label is not None else '')
        ax.set_ylabel(y_label if y_label is not None else '')
        ax.set_title(f"Contour plot of {variables[0]}" if title is None else title)
        return fig
    
    def _create_contourf(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) != 1:
            raise ValueError("contourf requires exactly 1 variable (2D array)")
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.contourf(self.data[variables[0]], **kwargs)
        ax.set_xlabel(x_label if x_label is not None else '')
        ax.set_ylabel(y_label if y_label is not None else '')
        ax.set_title(f"Filled contour plot of {variables[0]}" if title is None else title)
        return fig
    
    # ========== Specialized Plot Functions ==========
    def _create_pie(self, variables: List[str], **kwargs) -> plt.Figure:
        value_counts = self.data[variables[0]].value_counts()
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        title = kwargs.pop('title', None)
        
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', **kwargs)
        ax.set_title(f"Pie chart of {variables[0]}" if title is None else title)
        unique_labels = value_counts.index
        if len(unique_labels) > 10:
            fig.set_size_inches(max(12, len(unique_labels) * 0.5), 8)
        return fig
    
    def _create_polar(self, variables: List[str], **kwargs) -> plt.Figure:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        # Extract label-related kwargs if provided
        title = kwargs.pop('title', None)
        
        ax.plot(self.data[variables[0]], self.data[variables[1]], **kwargs)
        ax.set_title(f"Polar plot: {variables[0]} vs {variables[1]}" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_hexbin(self, variables: List[str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.hexbin(self.data[variables[0]], self.data[variables[1]], **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel(variables[1] if y_label is None else y_label)
        ax.set_title(f"Hexbin: {variables[0]} vs {variables[1]}" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_quiver(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) < 4:
            raise ValueError("quiver requires at least 4 variables (x, y, u, v)")
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        ax.quiver(self.data[variables[0]], 
                  self.data[variables[1]], 
                  self.data[variables[2]], 
                  self.data[variables[3]], **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel(variables[1] if y_label is None else y_label)
        ax.set_title(f"Quiver plot" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_streamplot(self, variables: List[str], **kwargs) -> plt.Figure:
        if len(variables) < 4:
            raise ValueError("streamplot requires at least 4 variables (x, y, u, v)")
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        x = np.linspace(0, 1, len(self.data[variables[0]]))
        y = np.linspace(0, 1, len(self.data[variables[1]]))
        X, Y = np.meshgrid(x, y)
        ax.streamplot(X, Y, 
                     self.data[variables[2]].values.reshape(len(x), len(y)),
                     self.data[variables[3]].values.reshape(len(x), len(y)),
                     **kwargs)
        ax.set_xlabel(x_label if x_label is not None else '')
        ax.set_ylabel(y_label if y_label is not None else '')
        ax.set_title(f"Stream plot" if title is None else title)
        unique_x = x
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    # ========== 3D Plot Functions ==========
    def _create_plot3d(self, variables: List[str], **kwargs) -> plt.Figure:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        z_label = kwargs.pop('z_label', None)
        title = kwargs.pop('title', None)
        
        ax.plot(self.data[variables[0]], 
                self.data[variables[1]], 
                self.data[variables[2]], **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel(variables[1] if y_label is None else y_label)
        ax.set_zlabel(variables[2] if z_label is None else z_label)
        ax.set_title(f"3D Line Plot" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_scatter3d(self, variables: List[str], **kwargs) -> plt.Figure:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        z_label = kwargs.pop('z_label', None)
        title = kwargs.pop('title', None)
        
        ax.scatter(self.data[variables[0]], 
                   self.data[variables[1]], 
                   self.data[variables[2]], **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel(variables[1] if y_label is None else y_label)
        ax.set_zlabel(variables[2] if z_label is None else z_label)
        ax.set_title(f"3D Scatter Plot" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_bar3d(self, variables: List[str], **kwargs) -> plt.Figure:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        z_label = kwargs.pop('z_label', None)
        title = kwargs.pop('title', None)
        
        # For bar3d we need x, y, z positions and sizes
        if len(variables) < 6:
            raise ValueError("bar3d requires 6 variables (x, y, z, dx, dy, dz)")
            
        x = self.data[variables[0]]
        y = self.data[variables[1]]
        z = self.data[variables[2]]
        dx = self.data[variables[3]]
        dy = self.data[variables[4]]
        dz = self.data[variables[5]]
        
        ax.bar3d(x, y, z, dx, dy, dz, **kwargs)
        ax.set_xlabel(variables[0] if x_label is None else x_label)
        ax.set_ylabel(variables[1] if y_label is None else y_label)
        ax.set_zlabel(variables[2] if z_label is None else z_label)
        ax.set_title(f"3D Bar Plot" if title is None else title)
        unique_x = self.data[variables[0]].dropna().unique()
        if len(unique_x) > 10:
            fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig
    
    def _create_surface(self, variables: List[str], **kwargs) -> plt.Figure:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        z_label = kwargs.pop('z_label', None)
        title = kwargs.pop('title', None)
        
        # For surface plot we need a grid of values
        if len(variables) != 1:
            raise ValueError("surface requires exactly 1 variable (2D array)")
            
        Z = self.data[variables[0]].values
        X = np.arange(Z.shape[1])
        Y = np.arange(Z.shape[0])
        X, Y = np.meshgrid(X, Y)
        
        ax.plot_surface(X, Y, Z, **kwargs)
        ax.set_xlabel(x_label if x_label is not None else '')
        ax.set_ylabel(y_label if y_label is not None else '')
        ax.set_zlabel(z_label if z_label is not None else '')
        ax.set_title(f"3D Surface Plot" if title is None else title)
        return fig
    
    # ========== Helper Methods ==========
    def _set_labels(self, ax, variables: List[str], **kwargs):
        """Set labels for x and y axes based on variables."""
        x_label = kwargs.get('x_label', None)
        y_label = kwargs.get('y_label', None)
        if len(variables) > 0:
            ax.set_xlabel(variables[0] if x_label is None else x_label)
        if len(variables) > 1:
            ax.set_ylabel(variables[1] if y_label is None else y_label)
    
    def _set_3d_labels(self, ax, variables: List[str], **kwargs):
        """Set labels for 3D plots."""
        x_label = kwargs.get('x_label', None)
        y_label = kwargs.get('y_label', None)
        z_label = kwargs.get('z_label', None)
        if len(variables) > 0:
            ax.set_xlabel(variables[0] if x_label is None else x_label)
        if len(variables) > 1:
            ax.set_ylabel(variables[1] if y_label is None else y_label)
        if len(variables) > 2:
            ax.set_zlabel(variables[2] if z_label is None else z_label)


class SmartPlotGenerator(PlotGenerator):
    def _create_box(self, variables: List[str], **kwargs) -> plt.Figure:
        """Enhanced boxplot that handles both univariate and bivariate cases with NaN handling."""
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        if len(variables) == 1:
            # Univariate case - single numerical variable
            data = self.data[variables[0]].dropna()  # Remove NaN values
            if len(data) == 0:
                raise ValueError(f"No valid data remaining after dropping NaN values for {variables[0]}")
            ax.boxplot(data, **kwargs)
            ax.set_xlabel(x_label if x_label is not None else '')
            ax.set_ylabel(variables[0] if y_label is None else y_label)
            ax.set_title(f"Box plot of {variables[0]}" if title is None else title)
            unique_x = data.unique()
            if len(unique_x) > 10:
                fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
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
            ax.set_xlabel(categorical_var if x_label is None else x_label)
            ax.set_ylabel(numerical_var if y_label is None else y_label)
            ax.set_title(f"Box plot of {numerical_var} by {categorical_var}" if title is None else title)
            if len(clean_data[categorical_var].unique()) > 10:
                fig.set_size_inches(max(12, len(clean_data[categorical_var].unique()) * 0.5), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig

    def _create_violin(self, variables: List[str], **kwargs) -> plt.Figure:
        """Enhanced violin plot that handles both univariate and bivariate cases with NaN handling."""
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        if len(variables) == 1:
            # Univariate case - single numerical variable
            data = self.data[variables[0]].dropna()  # Remove NaN values
            if len(data) == 0:
                raise ValueError(f"No valid data remaining after dropping NaN values for {variables[0]}")
            ax.violinplot(data, **kwargs)
            ax.set_xlabel(x_label if x_label is not None else '')
            ax.set_ylabel(variables[0] if y_label is None else y_label)
            ax.set_title(f"Violin plot of {variables[0]}" if title is None else title)
            unique_x = data.unique()
            if len(unique_x) > 10:
                fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
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
            ax.set_xticks(np.arange(1, len(grouped_data)+1))
            ax.set_xticklabels(clean_data[categorical_var].unique())
            ax.set_xlabel(categorical_var if x_label is None else x_label)
            ax.set_ylabel(numerical_var if y_label is None else y_label)
            ax.set_title(f"Violin plot of {numerical_var} by {categorical_var}" if title is None else title)
            if len(clean_data[categorical_var].unique()) > 10:
                fig.set_size_inches(max(12, len(clean_data[categorical_var].unique()) * 0.5), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        return fig

    def _create_hist(self, variables: List[str], **kwargs) -> plt.Figure:
        """Enhanced histogram that can handle grouping by a second variable."""
        fig, ax = plt.subplots()
        # Extract label-related kwargs if provided
        x_label = kwargs.pop('x_label', None)
        y_label = kwargs.pop('y_label', None)
        title = kwargs.pop('title', None)
        
        if len(variables) == 1:
            # Simple histogram
            data = self.data[variables[0]].dropna()
            if len(data) == 0:
                raise ValueError(f"No valid data remaining for {variables[0]}")
                
            ax.hist(data, **kwargs)
            ax.set_xlabel(variables[0] if x_label is None else x_label)
            ax.set_ylabel('Frequency' if y_label is None else y_label)
            ax.set_title(f"Histogram of {variables[0]}" if title is None else title)
            unique_x = data.unique()
            if len(unique_x) > 10:
                fig.set_size_inches(max(12, len(unique_x) * 0.5), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
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
                
            ax.set_xlabel(numerical_var if x_label is None else x_label)
            ax.set_ylabel('Frequency' if y_label is None else y_label)
            ax.set_title(f"Histogram of {numerical_var} by {categorical_var}" if title is None else title)
            ax.legend()
            if len(categories) > 10:
                fig.set_size_inches(max(12, len(categories) * 0.5), 8)
                plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
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
        plot_type = suggestion['plot_type'].lower()
        
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
        plot_type = suggestion_row['plot_type'].lower()
        
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
        
    else:
        raise TypeError("suggestion must be either an integer index or a pandas Series")


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