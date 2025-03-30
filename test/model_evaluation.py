import os
import time
import pandas as pd
import random
from dotenv import load_dotenv
from plotsense.explanations import generate_explanation  # Ensure this function is implemented

# Load environment variables
load_dotenv()

# Path to synthetic dataset
data_path = "test/synthetic_sales_data.csv"
df = pd.read_csv(data_path)

# List of 20 models to evaluate
models = [f"model_{i}" for i in range(1, 21)]

# Function to evaluate a model
def evaluate_model(model_name, sample_data):
    start_time = time.time()
    explanation = generate_explanation(model_name, sample_data)  # Model generates an explanation
    end_time = time.time()
    
    # Mock accuracy scoring (replace with real evaluation metric)
    accuracy = random.uniform(0.7, 1.0)  # Simulated accuracy score
    latency = end_time - start_time
    
    return {"model": model_name, "accuracy": accuracy, "latency": latency}

# Select a sample from the dataset for testing
sample_data = df.sample(n=10, random_state=42)  # Adjust sample size if needed

# Evaluate all models
results = [evaluate_model(model, sample_data) for model in models]

# Convert results to DataFrame and sort by accuracy (descending) and latency (ascending)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["accuracy", "latency"], ascending=[False, True])

# Select the top 5 models
best_models = results_df.head(5)

# Save results
results_df.to_csv("test/model_evaluation_results.csv", index=False)
best_models.to_csv("test/best_models.csv", index=False)

print("Model evaluation completed. Results saved in 'test/model_evaluation_results.csv' and 'test/best_models.csv'")
