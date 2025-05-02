import pandas as pd
import inspect
import matplotlib.pyplot as plt


class PlotMatplot:

    
    def plot_check(self,df: pd.DataFrame, plot: str, *args, **kwargs):
        """
        this is the function that does the actual plotting
        
        """
        #fig = plt.figure(figsize=kwargs.get('figsize', (8, 5)))
        # Get the actual plot function
        plot_func = getattr(plt, plot)

        processed_args = tuple(
        df[col] if isinstance(col, str) and col in df.columns else col for col in args
        )

        # Do the actual plot
        fig = plot_func(*processed_args, **kwargs)

        # Title and label params
        title = kwargs.pop('title', None)
        xlabel = kwargs.pop('xlabel', None)
        ylabel = kwargs.pop('ylabel', None)

        # Default title to "x vs y" using args[0] and args[1]
        #default_title = f"plot of {args[0]}" if len(args) <= 1 else f"Plot of {args[0]} vs {args[1]}" if len(args) >= 2 else None

        
        if args[0] and args[1]:
            default_title = f"Plot of {args[0]} vs {args[1]}"
        elif args[0]:
            default_title = f"plot of {args[0]}"
        else:
            default_title = "Plot"
            
        plt.title(title if title is not None else default_title)

        # Default xlabel to args[0], ylabel to args[1]
        default_xlabel = args[0] if len(args) >= 1 else None
        default_ylabel = args[1] if len(args) >= 2 else None
        plt.xlabel(xlabel if xlabel is not None else default_xlabel)
        plt.ylabel(ylabel if ylabel is not None else default_ylabel)
        

        return fig

def generate_plot(df, suggestion_series, **kwargs):
    """
    Generate a plot based on the sugestion result of the llm
    It uses the suggested axis as the default x and y except overridden by the users

    parameters:
    df: pandas dataframe containing the data
    suggestion series: indexed series of a plot with column names 'plot_type' and 'variables'
    **kwargs: optional overrides for x and y, additional matplotlib arguments for custom plot

    returns plt.figure
    """

    PLOT_TYPE_MAP = {
    "scatter plot": "scatter",
    "line plot": "plot",
    "bar chart": "bar",
    "bar plot": "bar",
    "histogram": "hist",
    "box plot": "boxplot",
    "pie chart": "pie",
    "area chart": "fill_between",
    # Add more mappings as needed
    }

    raw_plot_type = suggestion_series['plot_type'].strip().lower()
    suggested_plot = PLOT_TYPE_MAP.get(raw_plot_type)

    variables = [col.strip() for col in suggestion_series['variables'].split(',')]
    x_suggested = variables[0] if variables[0] else None
    y_suggested = variables[1] if len(variables) > 1 and variables[1] else None

    # User override takes priority
    x = kwargs.get('x', x_suggested)
    y = kwargs.get('y', y_suggested)

    # Build positional args ONLY if not already passed via kwargs
    args = []
    if x:
        args.append(x)
    if y:
        args.append(y)

    # Remove x and y from kwargs to avoid duplication
    kwargs.pop('x', None)
    kwargs.pop('y', None)

    plot_instance = PlotMatplot()

    plot_instance.plot_check(df, suggested_plot, *args, **kwargs)