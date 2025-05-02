import os
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from dotenv import load_dotenv
import pandas as pd
import warnings
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import textwrap
import builtins
from pprint import pprint
from groq import Groq


load_dotenv()

class VisualizationRecommender:
    DEFAULT_MODELS = {
        'groq': [
            ('llama3-70b-8192', 0.5),  # (model_name, weight)
            ('mistral-saba-24b', 0.5),
            ('llama3-8b-8192', 0.5),
            ('llama-3.3-70b-versatile', 0.5)
        ],
        # Add other providers here
    }

    def __init__(self, api_keys: Optional[Dict[str, str]] = None, timeout: int = 30, interactive: bool = True, debug: bool = False):
        """
        Initialize VisualizationRecommender with API keys and configuration.
        
        Args:
            api_keys: Optional dictionary of API keys. If not provided,
                     keys will be loaded from environment variables.
            timeout: Timeout in seconds for API requests
            interactive: Whether to prompt for missing API keys
            debug: Enable debug output
        """   
        self.interactive = interactive
        self.debug = debug
        api_keys = api_keys or {}
        self.api_keys = {
            'groq': os.getenv('GROQ_API_KEY') 
            # Add other services here
        }

        self.timeout = timeout
        self.clients = {}
        self.available_models = []
        self.df = None
        self.model_weights = {}
        self.n_to_request = 5 

        self.api_keys.update(api_keys)

        self._validate_keys()
        self._initialize_clients()
        self._detect_available_models()
        self._initialize_model_weights()
       

        if self.debug:
            print("\n[DEBUG] Initialization Complete")
            print(f"Available models: {self.available_models}")
            print(f"Model weights: {self.model_weights}")
            if hasattr(self, 'clients'):
                print(f"Clients initialized: {bool(self.clients)}")
                
    def _validate_keys(self):
        """Validate that required API keys are present"""
        for service in ['groq']:
            if not self.api_keys.get(service):
                if self.interactive:
                    try:
                        self.api_keys[service] = builtins.input(f"Enter {service.upper()} API key: ").strip()
                        if not self.api_keys[service]:
                            raise ValueError(f"{service.upper()} API key is required")
                    except (EOFError, OSError):
                            # Handle cases where input is not available
                        raise ValueError(f"{service.upper()} API key is required")
                else:
                    raise ValueError(f"{service.upper()} API key is required. Set it in the environment or pass it as an argument.")

    def _initialize_clients(self):
        """Initialize API clients"""
        self.clients = {}
        if self.api_keys.get('groq'):
            try:
                self.clients['groq'] = Groq(api_key=self.api_keys['groq'])
            except ImportError:
                warnings.warn("Groq Python client not installed. pip install groq")

    def _detect_available_models(self):
        self.available_models = []
        for provider, client in self.clients.items():
            if client and provider in self.DEFAULT_MODELS:
                # For now we'll assume all DEFAULT_MODELS are available
                # In a real implementation, you might want to check which models are actually available
                self.available_models.extend([m[0] for m in self.DEFAULT_MODELS[provider]])
        
        if self.debug:
            print(f"[DEBUG] Detected available models: {self.available_models}")

    def _initialize_model_weights(self):
        total_weight = 0
        self.model_weights = {}

        # Only include weights for available models
        for provider in self.DEFAULT_MODELS:
            for model, weight in self.DEFAULT_MODELS[provider]:
                if model in self.available_models:
                    self.model_weights[model] = weight
                    total_weight += weight

        # Normalize weights to sum to 1
        if total_weight > 0:
            for model in self.model_weights:
                self.model_weights[model] /= total_weight

        if self.debug:
            print(f"[DEBUG] Model weights: {self.model_weights}")

    def set_dataframe(self, df: pd.DataFrame):
        """Set the DataFrame to analyze and provide debug info"""
        self.df = df
        if self.debug:
            print("\n[DEBUG] DataFrame Info:")
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            print("\nSample data:")
            print(df.head(2))

    def recommend_visualizations(self, n: int = 5, custom_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Generate visualization recommendations using weighted ensemble approach.
        
        Args:
            n: Number of recommendations to return (default: 3)
            custom_weights: Optional dictionary to override default model weights
            
        Returns:
            pd.DataFrame: Recommended visualizations with ensemble scores
            
        Raises:
            ValueError: If no DataFrame is set or no models are available
        """
        """Generate visualization recommendations using weighted ensemble approach."""
        self.n_to_request = max(n, 5)
        
        if self.df is None:
            raise ValueError("No DataFrame set. Call set_dataframe() first.")

        if not self.available_models:
            raise ValueError("No available models detected")
        
        if self.debug:
            print("\n[DEBUG] Starting recommendation process")
            print(f"Using models: {self.available_models}")
        
        # Use custom weights if provided, otherwise use defaults
        weights = custom_weights if custom_weights else self.model_weights

        # Get recommendations from all models in parallel
        all_recommendations = self._get_all_recommendations()

        if self.debug:
            print("\n[DEBUG] Raw recommendations from models:")
            pprint(all_recommendations)

        # Apply weighted ensemble scoring
        ensemble_results = self._apply_ensemble_scoring(all_recommendations, weights)

        # If we don't have enough results, try to supplement
        if len(ensemble_results) < n:
            if self.debug:
                print(f"\n[DEBUG] Only got {len(ensemble_results)} recommendations, trying to supplement")
            return self._supplement_recommendations(ensemble_results, n)
        
        if self.debug:
            print("\n[DEBUG] Ensemble results before filtering:")
            print(ensemble_results)
        
        return ensemble_results.head(n)
            

    def _supplement_recommendations(self, existing: pd.DataFrame, target: int) -> pd.DataFrame:
        """Generate additional recommendations if we didn't get enough initially."""
        if len(existing) >= target:
            return existing.head(target)
        
        needed = target - len(existing)
        df_description = self._describe_dataframe()
        
        # Try to get more recommendations from the best-performing model
        best_model = existing.iloc[0]['source_models'][0] if not existing.empty else self.available_models[0]
        
        prompt = textwrap.dedent(f"""
            You already recommended these visualizations:
            {existing[['plot_type', 'variables']].to_string()}
            
            Please recommend {needed} ADDITIONAL different visualizations for:
            {df_description}
            
            Use the same format but ensure they're distinct from the above.
        """)
        
        try:
            response = self._query_llm(prompt, best_model)
            new_recs = self._parse_recommendations(response, f"{best_model}-supplement")
            
            # Combine with existing
            combined = pd.concat([existing, pd.DataFrame(new_recs)], ignore_index=True)
            combined = combined.drop_duplicates(subset=['plot_type', 'variables'])
            
            if self.debug:
                print(f"\n[DEBUG] Supplemented with {len(new_recs)} new recommendations")
            
            return combined.head(target)
        except Exception as e:
            if self.debug:
                print(f"\n[WARNING] Couldn't supplement recommendations: {str(e)}")
            return existing.head(target)  # Return what we have

    def _get_all_recommendations(self) -> Dict[str, List[Dict]]:
        df_description = self._describe_dataframe()
        prompt = self._create_prompt(df_description)
        
        if self.debug:
            print("\n[DEBUG] Prompt being sent to models:")
            print(prompt)

        model_handlers = {
            'llama': self._query_llm,
            'mistral': self._query_llm,  # Same handler as llama
            # Add other model handlers here
        }

        all_recommendations = {}

        with ThreadPoolExecutor() as executor:
            futures = {}
            for model in self.available_models:
                model_type = model.split('-')[0].lower()
                if model_type.startswith(("llama", "mistral")):
                    model_type = "llama" if "llama" in model_type else "mistral"
                query_func = model_handlers[model_type]
                futures[executor.submit(self._get_model_recommendations, model, prompt, query_func)] = model

            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                try:
                    result = future.result()
                    all_recommendations[model] = result
                    if self.debug:
                        print(f"\n[DEBUG] Got {len(result)} recommendations from {model}")
                except Exception as e:
                    warnings.warn(f"Failed to get recommendations from {model}: {str(e)}")
                    if self.debug:
                        print(f"\n[ERROR] Failed to process {model}: {str(e)}")

        return all_recommendations

    def _get_model_recommendations(self, model: str, prompt: str, query_func: Callable[[str, str], str]) -> List[Dict]:
        try:
            response = query_func(prompt, model)
            
            if self.debug:
                print(f"\n[DEBUG] Raw response from {model}:")
                print(response)
            
            return self._parse_recommendations(response, model)
        except Exception as e:
            warnings.warn(f"Error processing model {model}: {str(e)}")
            if self.debug:
                print(f"\n[ERROR] Failed to parse response from {model}: {str(e)}")
            return []

    def _apply_ensemble_scoring(self, all_recommendations: Dict[str, List[Dict]], weights: Dict[str, float]) -> pd.DataFrame:
        output_columns = ['plot_type', 'variables', 'ensemble_score', 'model_agreement', 'source_models']
        
        if self.debug:
            print("\n[DEBUG] Applying ensemble scoring with weights:")
            pprint(weights)
        
        recommendation_weights = defaultdict(float)
        recommendation_details = {}

        for model, recs in all_recommendations.items():
            model_weight = weights.get(model, 0)
            if model_weight <= 0:
                continue

            for rec in recs:
                # Create a consistent key for the recommendation
                variables = rec['variables']
                if isinstance(variables, str):
                    variables = [v.strip() for v in variables.split(',')]
                
                # Filter variables to only those in the DataFrame
                valid_vars = [var for var in variables if var in self.df.columns]
                if not valid_vars:
                    if self.debug:
                        print(f"\n[DEBUG] Skipping recommendation from {model} with invalid variables: {variables}")
                    continue
                
                var_key = ', '.join(sorted(valid_vars))
                rec_key = (rec['plot_type'].lower(), var_key)
                
                model_score = rec.get('score', 1.0)
                total_weight = model_weight * model_score
                recommendation_weights[rec_key] += total_weight

                if rec_key not in recommendation_details:
                    recommendation_details[rec_key] = {
                        'plot_type': rec['plot_type'],
                        'variables': var_key,
                        'source_models': [model],
                        'raw_weight': total_weight
                    }
                else:
                    recommendation_details[rec_key]['source_models'].append(model)
                    recommendation_details[rec_key]['raw_weight'] += total_weight

        if not recommendation_details:
            if self.debug:
                print("\n[DEBUG] No valid recommendations after filtering")
            return pd.DataFrame(columns=output_columns)

        results = pd.DataFrame(list(recommendation_details.values()))

        if self.debug:
            print("\n[DEBUG] Recommendations before scoring:")
            print(results)

        if not results.empty:
            total_possible = sum(weights.values())
            results['ensemble_score'] = results['raw_weight'] / total_possible
            results['ensemble_score'] = results['ensemble_score'].round(2)
            results['model_agreement'] = results['source_models'].apply(len)
            results = results.sort_values(['ensemble_score', 'model_agreement'], ascending=[False, False]).reset_index(drop=True)
            return results[output_columns]

        return pd.DataFrame(columns=output_columns)

    def _describe_dataframe(self) -> str:
        num_cols = len(self.df.columns)
        sample_size = min(3, len(self.df))

        desc = [
            f"DataFrame Shape: {self.df.shape}",
            f"Columns ({num_cols}): {', '.join(self.df.columns)}",
            "\nColumn Details:",
        ]

        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            sample_values = self.df[col].dropna().head(sample_size).tolist()

            desc.append(
                f"- {col}: {dtype} ({unique_count} unique values), sample: {sample_values}"
            )

        return "\n".join(desc)

    def _create_prompt(self, df_description: str) -> str:
        return textwrap.dedent(f"""
            You are a data visualization expert analyzing this dataset:

            {df_description}

            Recommend {self.n_to_request} insightful visualizations using matplotlib's plotting functions.
            For each suggestion, follow this format:

            Plot Type: <matplotlib function name - exact, like bar, scatter, hist, boxplot, pie>
            Variables: <comma-separated list of variables>
            Rationale: <1-2 sentences explaining why this visualization is useful, based on column data types and insight potential>
            ---

            Focus your visualizations on:
            1. Revealing meaningful patterns or relationships
            2. Choosing appropriate plot types for the column data types:
                - Numerical → hist, scatter, boxplot, line, hexbin
                - Categorical → bar, pie
                - Date/Time → line, area, bar
                - Multivariate combinations → scatter, pairplot, heatmap
            3. Avoiding misleading representations
            4. Providing practical, actionable insights

            Important guidelines:
            1. Use ONLY these matplotlib plot types (use exact names) such as  bar, scatter, hist, boxplot, pie, line, heatmap, violinplot, area, hexbin, pairplot, jointplot and more.
            2. Variables must exist in the dataset
            3. Prioritize more informative plot types
            4. Include both univariate and multivariate plots
            5. Never use general terms like "chart" or "graph" - always use the exact matplotlib function name
            6. For seaborn-style plots that wrap matplotlib, use the underlying matplotlib function
            7. Avoid recommending plots that mismatch with variable types

            Example correct responses:
            Plot Type: scatter
            Variables: temperature, humidity
            Rationale: Shows relationship between temperature and humidity
            ---
            Plot Type: bar
            Variables: location, sales
            Rationale: Compares sales across different locations
            ---
            Plot Type: hist
            Variables: temperature
            Rationale: Shows distribution of temperature values
        """)

    def _query_llm(self, prompt: str, model: str) -> str:
        if not self.clients.get('groq'):
            raise ValueError("Groq client not initialized")
        
        try:
            response = self.clients['groq'].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1000,
                timeout=self.timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq API query failed for {model}: {str(e)}")


    def _parse_recommendations(self, response: str, model: str) -> List[Dict]:
        """Parse the LLM response into structured recommendations"""
        recommendations = []

        # Split response into recommendation blocks
        blocks = [b.strip() for b in response.split('---') if b.strip()]
        
        if self.debug:
            print(f"\n[DEBUG] Parsing {len(blocks)} blocks from {model}")
        
        for block in blocks:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if not lines:
                continue
                
            try:
                rec = {'source_model': model}
                for line in lines:
                    if line.lower().startswith('plot type:'):
                        rec['plot_type'] = line.split(':', 1)[1].strip().lower()
                    elif line.lower().startswith('variables:'):
                        raw_vars = line.split(':', 1)[1].strip()
                        # Get all variables first

                        # Filter variables to only those that exist in DataFrame
                        variables = [v.strip() for v in raw_vars.split(',') if v.strip() in self.df.columns]
                        rec['variables'] = ', '.join([var for var in variables if var in self.df.columns])
                
                if 'plot_type' in rec and 'variables' in rec and rec['variables']:
                    recommendations.append(rec)
            except Exception as e:
                warnings.warn(f"Failed to parse recommendation from {model}: {str(e)}")
                continue
        
        return recommendations
    


# Package-level convenience function
_recommender_instance = None

def recommender(
    df: pd.DataFrame,
    n: int = 5,
    api_keys: dict = {},
    custom_weights: Optional[Dict[str, float]] = None,
    debug: bool = False
) -> pd.DataFrame:
    """
    Generate visualization recommendations using weighted ensemble of LLMs.
    
    Args:
        df: Input DataFrame to analyze
        n: Number of recommendations to return (default: 3)
        api_keys: Dictionary of API keys
        custom_weights: Optional dictionary to override default model weights
        debug: Enable debug output
        
    Returns:
        pd.DataFrame: Recommended visualizations with ensemble scores
    """
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = VisualizationRecommender(api_keys=api_keys, debug=debug)
    
    _recommender_instance.set_dataframe(df)
    return _recommender_instance.recommend_visualizations(
        n=n,
        custom_weights=custom_weights
    )
