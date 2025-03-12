import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

@TimeoutError
class SeabornPlotGenerator:

    def __init__(self):
        pass

    def set_seaborn_style(style="darkgrid", context="notebook", palette="deep"):
        """Sets the default Seaborn style and context."""
        sns.set_style(style)
        sns.set_context(context)
        sns.set_palette(palette)

    # Line Plot
    def generate_seaborn_lineplot(data, x, y, hue=None, title="Line Plot"):
        """Generates a line plot using Seaborn."""
        #set_seaborn_style()
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x=x, y=y, hue=hue)
        plt.title(title)
        plt.show()

    # Scatter Plot
    def generate_seaborn_scatterplot(data, x, y, hue=None, title="Scatter Plot"):
        """Generates a scatter plot using Seaborn."""
        #set_seaborn_style()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=x, y=y, hue=hue)
        plt.title(title)
        plt.show()

    # Histogram
    def generate_seaborn_histogram(data, column, bins=30, title="Histogram"):
        """Generates a histogram using Seaborn."""
        #set_seaborn_style()
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], bins=bins, kde=True)
        plt.title(title)
        plt.show()

    # Bar Plot
    def generate_seaborn_barplot(data, x, y, hue=None, title="Bar Plot"):
        """Generates a bar plot using Seaborn."""
        #set_seaborn_style()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=data, x=x, y=y, hue=hue)
        plt.title(title)
        plt.show()

        