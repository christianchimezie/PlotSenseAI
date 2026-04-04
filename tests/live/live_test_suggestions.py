import os
from dotenv import load_dotenv
from plotsense.visual_suggestion.suggestions import recommender
import pandas as pd

load_dotenv()

# Example: create a simple DataFrame
df = pd.DataFrame({
    "Year": [2020, 2021, 2022, 2023],
    "Sales": [150, 200, 250, 300],
    "Profit": [40, 50, 65, 80]
})

# Load API keys from environment (only include if present)
api_keys = {}
if os.getenv("GROQ_API_KEY"):
    api_keys["groq"] = os.getenv("GROQ_API_KEY")
if os.getenv("OPENAI_API_KEY"):
    api_keys["openai"] = os.getenv("OPENAI_API_KEY")
if os.getenv("AZURE_API_KEY"):
    api_keys["azure"] = os.getenv("AZURE_API_KEY")

# Run the recommender
recommendations = recommender(
    df,
    n=3,  # number of visualizations to recommend
    api_keys=api_keys,
    # selected_models=[("openai", "gpt-5")],
)

# Display the recommendations
print("📊 Recommended visualizations:")
print(recommendations)
