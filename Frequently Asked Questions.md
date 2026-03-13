## Frequently Asked Questions
1. Why is my API key not working?

Ensure your API key is correctly set in your environment variables.

Example:

export GROQ_API_KEY="your_api_key_here"

You can also pass it directly in the code using the api_keys parameter.

2. Can I use my own dataset?

Yes. PlotSense works with any pandas DataFrame.

Example:

import pandas as pd
df = pd.read_csv("your_dataset.csv")
recommendation = recommender(df)


3. Which plots are recommended for categorical variables?

Common plots include:

Bar charts

Pie charts

Boxplots (categorical vs numerical)

PlotSense automatically suggests appropriate plots based on the dataset structure.

4. Do I need a Groq API key to use PlotSense?

Yes. PlotSense uses Groq-hosted LLMs to generate visualization recommendations and explanations.

You can get a free API key at:
https://console.groq.com/keys


5. Can I control which plot gets generated?

Yes. Each recommendation has an index. You can generate a specific plot using that index.

Example:

fig = plotgen(df, 0, recommendation)

6. How do I contribute to PlotSense?

You can contribute by:

Fixing bugs

Adding new plot types

Improving documentation

Suggesting new features

Feel free to open a pull request or issue on GitHub.

7. Why does the plot type sometimes slightly differ from my expectation?

PlotSense uses LLM-based recommendations, which analyze the dataset and propose plots based on statistical relationships and patterns. Results may vary slightly between runs.

8. Can I add my own LLM model?

Currently PlotSense is configured for Groq-hosted models, but the architecture allows adding additional providers by extending the model handlers.

9. What Python version is supported?

PlotSense works best with Python 3.9 or later. Earlier versions may cause dependency issues.

10. What does “ensemble score” mean in the results?

The ensemble score reflects how strongly the models agree on a visualization suggestion.
Higher scores indicate stronger agreement between models.

11. Can I use PlotSense in Jupyter Notebook?

Yes. PlotSense works well inside Jupyter Notebook, Google Colab, and Python scripts.

Example:

recommendation = recommender(df)
fig = plotgen(df, 0, recommendation)

12. What happens if the API request fails?

If the API request fails, PlotSense will raise an error such as:

PlotSenseAPIError

PlotSenseDataError

PlotSenseConfigError

Ensure your API key and internet connection are working

13. Does PlotSense modify my original dataset?

No. PlotSense works on a copy of the DataFrame, so the original dataset remains unchanged.

14. Can PlotSense work with time-series data?

Yes. If the dataset contains datetime columns, PlotSense may recommend visualizations such as:

Line plots

Scatter plots

Distribution plots over time.

15. Can I use PlotSense in Jupyter Notebook?

Yes. PlotSense works well inside Jupyter Notebook, Google Colab, and Python scripts.

Example:

recommendation = recommender(df)
fig = plotgen(df, 0, recommendation)