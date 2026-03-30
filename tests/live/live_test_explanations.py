import os
from dotenv import load_dotenv
from plotsense.explanations.explanations import explainer
from matplotlib import pyplot as plt

load_dotenv()

# Example: generate a simple plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

# Load API keys from environment
api_keys = {
    "groq": os.getenv("GROQ_API_KEY"),
    # "openai": os.getenv("OPENAI_API_KEY")
}

# Run explainer
result = explainer(
    fig,
    prompt="Explain this simple line plot",
    api_keys=api_keys,
    # selected_models=[("openai", "gpt-4.1")],
)
print(result)
