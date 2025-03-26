import matplotlib.pyplot as plt
import pandas as pd


class PlotMatplot:
    """
    Initialize the plot generator with a dataframe
    """
    
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input should be a pandas DataFrame")
        self.df = df

    def generate_plot(self, plot_type='bar', x_col=None, y_col=None, **kwargs):
        """
        Generate a plot based on the specified plot type and configuration.
        
        Parameters:
        - plot_type (str): Type of plot ('bar', 'line', 'scatter').
        - x_col (str): Column name for x-axis.
        - y_col (str): Column name for y-axis.
        - kwargs: Additional parameters for matplotlib plotting functions.
        """
        
        if self.df is None or x_col is None or y_col is None:
            raise ValueError("Invalid input. Ensure DataFrame, x_col, and y_col are specified.")

        try:
            plt.figure(figsize=kwargs.get('figsize', (8, 5)))

            if plot_type == 'bar':
                plt.bar(self.df[x_col], self.df[y_col], color=kwargs.get('color', 'skyblue'),
                        edgecolor=kwargs.get('edgecolor', 'black'), linewidth=kwargs.get('linewidth', 1.5))
            elif plot_type == 'line':
                plt.plot(self.df[x_col], self.df[y_col], color=kwargs.get('color', 'blue'),
                         marker=kwargs.get('marker', 'o'), linestyle=kwargs.get('linestyle', '-'))
            elif plot_type == 'scatter':
                plt.scatter(self.df[x_col], self.df[y_col], color=kwargs.get('color', 'red'))
            else:
                raise ValueError("Unsupported plot type. Use 'bar', 'line', or 'scatter'.")

            plt.xlabel(kwargs.get('xlabel', x_col))
            plt.ylabel(kwargs.get('ylabel', y_col))
            plt.title(kwargs.get('title', f"{plot_type.capitalize()} Plot"))
            plt.grid(kwargs.get('grid', True))
            plt.show()
        
        except Exception as e:
            print(f"Error generating plot: {e}")
            return None

