import os
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import warnings
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import textwrap
import getpass
from pprint import pprint
from groq import Groq
from plotsense.exceptions import PlotSenseAPIError, PlotSenseDataError, PlotSenseConfigError


load_dotenv()


class VisualizationRecommender:
    DEFAULT_MODELS = {
        'groq': [
            ('llama-3.3-70b-versatile', 0.5),  # (model_name, weight)
            ('llama-3.1-8b-instant', 0.5)

        ],

    }

    def __init__(self,
                 api_keys: Optional[Dict[str,
                                         str]] = None,
                 timeout: int = 30,
                 interactive: bool = True,
                 debug: bool = False):
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
        service_links = {
            'groq': 'https://console.groq.com/keys'
        }

        for service in ['groq']:
            if not self.api_keys.get(service):
                if self.interactive:
                    try:
                        link = service_links.get(
                            service, f"the {service.upper()} website")
                        message = (
                            f"Enter {service.upper()} API key (get it at {link}): ")
                        # Use getpass to hide input (avoid key appearing in logs/terminal)
                        self.api_keys[service] = getpass.getpass(
                            message).strip()
                        if not self.api_keys[service]:
                            raise PlotSenseDataError(
                                f"{service.upper()} API key is required")
                    except (EOFError, OSError):
                        # Handle cases where input is not available
                        raise PlotSenseDataError(
                            f"{service.upper()} API key is required "
                            f"(get it at {service_links.get(service)})"
                        )
                else:
                    raise PlotSenseDataError(
                        f"{service.upper()} API key is required. "
                        f"Set it in the environment or pass it as an argument. "
                        f"You can get it at {service_links.get(service)}"
                    )

    def _initialize_clients(self):
        """Initialize API clients"""
        self.clients = {}
        if self.api_keys.get('groq'):
            try:
                self.clients['groq'] = Groq(api_key=self.api_keys['groq'])
            except ImportError:
                raise PlotSenseConfigError("Python client not installed. Install with: pip install groq")

    def _detect_available_models(self):
        self.available_models = []
        for provider, client in self.clients.items():
            if client and provider in self.DEFAULT_MODELS:
                self.available_models.extend(
                    [m[0] for m in self.DEFAULT_MODELS[provider]])

        if self.debug:
            print(
                f"[DEBUG] Detected available models: {self.available_models}")

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

    def recommend_visualizations(self,
                                 n: int = 5,
                                 custom_weights: Optional[Dict[str,
                                                               float]] = None) -> pd.DataFrame:
        """
        Generate visualization recommendations using weighted ensemble approach.

        Args:
            n: Number of recommendations to return (default: 3)
            custom_weights: Optional dictionary to override default model weights

        Returns:
            pd.DataFrame: Recommended visualizations with ensemble scores

        Raises:
            ValueError:
            If no DataFrame is set or no models are available
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
        ensemble_results = self._apply_ensemble_scoring(
            all_recommendations, weights)

        # Validate and correct variable order
        if not ensemble_results.empty:
            ensemble_results = self._validate_variable_order(ensemble_results)

        # If we don't have enough results, try to supplement
        if len(ensemble_results) < n:
            if self.debug:
                print(
                    f"\n[DEBUG] Only got {len(ensemble_results)} recommendations, trying to supplement")
            return self._supplement_recommendations(ensemble_results, n)

        if self.debug:
            print("\n[DEBUG] Ensemble results before filtering:")
            print(ensemble_results)

        return ensemble_results.head(n)

    def _supplement_recommendations(
            self,
            existing: pd.DataFrame,
            target: int) -> pd.DataFrame:
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
            new_recs = self._parse_recommendations(
                response, f"{best_model}-supplement")

            # Combine with existing
            combined = pd.concat(
                [existing, pd.DataFrame(new_recs)], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=['plot_type', 'variables'])

            if self.debug:
                print(
                    f"\n[DEBUG] Supplemented with {len(new_recs)} new recommendations")

            return combined.head(target)
        except Exception as e:
            if self.debug:
                print(
                    f"\n[WARNING] Couldn't supplement recommendations: {str(e)}")
            return existing.head(target)  # Return what we have

    def _get_all_recommendations(self) -> Dict[str, List[Dict]]:
        df_description = self._describe_dataframe()
        prompt = self._create_prompt(df_description)

        if self.debug:
            print("\n[DEBUG] Prompt being sent to models:")
            print(prompt)

        model_handlers = {
            'llama': self._query_llm

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
                futures[executor.submit(
                    self._get_model_recommendations, model, prompt, query_func)] = model

            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                try:
                    result = future.result()
                    all_recommendations[model] = result
                    if self.debug:
                        print(
                            f"\n[DEBUG] Got {len(result)} recommendations from {model}")
                except Exception as e:
                    # Surface API/client failures as configuration/API errors
                    raise PlotSenseAPIError(
                        f"Failed to get recommendations from {model}: {str(e)}"
                    ) from e

        return all_recommendations

    def _get_model_recommendations(self,
                                   model: str,
                                   prompt: str,
                                   query_func: Callable[[str,
                                                         str],
                                                        str]) -> List[Dict]:
        try:
            response = query_func(prompt, model)

            if self.debug:
                print(f"\n[DEBUG] Raw response from {model}:")
                print(response)

            return self._parse_recommendations(response, model)
        except Exception as e:
            warnings.warn(f"Error processing models {model}: {str(e)}")
            if self.debug:
                print(
                    f"\n[ERROR] Failed to parse response from {model}: {str(e)}")
            return []

    def _process_recommendation_from_model(
        self, rec: Dict, valid_vars: List[str],
        model: str, model_weight: float,
        recommendation_weights: Dict,
        recommendation_details: Dict
    ):
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

    def _collect_recommendations_from_models(self, all_recommendations: Dict[str, List[Dict]],
                                             weights: Dict[str, float]) -> Dict:
        """Collect recommendations from all models."""
        recommendation_weights = defaultdict(float)
        recommendation_details = {}

        for model, recs in all_recommendations.items():
            model_weight = weights.get(model, 0)
            if model_weight <= 0:
                continue

            for rec in recs:
                variables = rec['variables']
                if isinstance(variables, str):
                    variables = [v.strip() for v in variables.split(',')]

                valid_vars = [var for var in variables if var in self.df.columns]
                if not valid_vars:
                    if self.debug:
                        print(f"\n[DEBUG] Skipping recommendation from {model} with invalid variables: {variables}")
                    continue

                self._process_recommendation_from_model(
                    rec, valid_vars, model, model_weight,
                    recommendation_weights, recommendation_details
                )

        return recommendation_details

    def _apply_ensemble_scoring(self,
                                all_recommendations: Dict[str,
                                                          List[Dict]],
                                weights: Dict[str,
                                              float]) -> pd.DataFrame:
        output_columns = ['plot_type', 'variables',
                          'ensemble_score', 'model_agreement', 'source_models']

        if self.debug:
            print("\n[DEBUG] Applying ensemble scoring with weights:")
            pprint(weights)

        recommendation_details = self._collect_recommendations_from_models(
            all_recommendations, weights
        )

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
            results = results.sort_values(['ensemble_score', 'model_agreement'], ascending=[
                                          False, False]).reset_index(drop=True)
            return results[output_columns]

        return pd.DataFrame(columns=output_columns)

    def _describe_dataframe(self) -> str:
        num_cols = len(self.df.columns)
        sample_size = min(3, len(self.df))
        desc: List[str] = []

        # --- Basic Metadata ---
        desc.append(f"DataFrame Shape: {self.df.shape}")
        desc.append(f"Columns ({num_cols}): {', '.join(self.df.columns)}")
        desc.append("\nColumn Details:")

        # --- Column-Level Analysis ---
        for col in self.df.columns:
            # Determine semantic type (more granular than dtype)
            if pd.api.types.is_datetime64_dtype(self.df[col]):
                col_type = "datetime"
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                col_type = "numerical"
            elif self.df[col].nunique() / len(self.df[col]) < 0.05:  # Low cardinality
                col_type = "categorical"
            else:
                col_type = "text/other"

            # Basic info
            unique_count = self.df[col].nunique()
            sample_values = self.df[col].dropna().head(sample_size).tolist()
            desc.append(
                f"- {col}: {col_type} ({unique_count} unique values), sample: {sample_values}")

            # Add stats for numerical/datetime
            if col_type == "numerical":
                desc.append(
                    f"  Stats: min={self.df[col].min()}, max={self.df[col].max()}, "
                    f"mean={self.df[col].mean():.2f}, missing={self.df[col].isna().sum()}")
            elif col_type == "datetime":
                desc.append(
                    f"  Range: {self.df[col].min()} to {self.df[col].max()}, "
                    f"missing={self.df[col].isna().sum()}"
                )

        # --- Relationship Analysis ---
        numerical_cols = self.df.select_dtypes(
            include=np.number).columns.tolist()
        if len(numerical_cols) > 1:
            desc.append("\nNumerical Variable Correlations (Pearson):")
            corr = self.df[numerical_cols].corr().round(2)
            desc.append(str(corr))

        # Categorical-numerical potential groupings
        categorical_cols = [
            col for col in self.df.columns
            if self.df[col].nunique() / len(self.df[col]) < 0.05
        ]
        if categorical_cols and numerical_cols:
            desc.append("\nPotential Groupings (categorical vs numerical):")
            desc.append(f"  - Could group by: {categorical_cols}")
            desc.append(f"  - To analyze: {numerical_cols}")

        return "\n".join(desc)

    def _create_prompt(self, df_description: str) -> str:
        return textwrap.dedent(f"""
            You are a data visualization expert analyzing this dataset:

            {df_description}

            Recommend {self.n_to_request} insightful visualizations using matplotlib's plotting functions.
            For each suggestion, follow this exact format:

            Plot Type: <matplotlib function name - exact, like bar, scatter, hist, boxplot, pie, contour, quiver, etc.>
            Variables: <comma-separated list of variables WITH NUMERICAL VARIABLES FIRST>
            Rationale: <1-2 sentences explaining why this visualization is useful>
            ---

            CRITICAL VARIABLE ORDERING RULES:
            1. If a suggestion includes both numerical and categorical variables, NUMERICAL VARIABLES MUST COME FIRST.
            - Correct: "income, gender"
            - Incorrect: "gender, income"
            2. For plots requiring two numerical variables (e.g., scatter), order by analysis priority (dependent variable first).
            3. For single-variable plots, use natural order (e.g., "age" for a histogram).

            GENERAL RULES FOR ALL PLOT TYPES:
            1. Ensure the plot type is a valid matplotlib function
            2. The plot type must be appropriate for the variables' data types
            3. The number of variables must match what the plot type requires
            4. Variables must exist in the dataset
            5. Never combine incompatible variables
            6. Always specify complete variable sets
            7. Ensure plot type names are in lowercase and match matplotlib's naming conventions eg hist for histogram, bar for barplot
            8. Ensure the common plot types requirements are met including the data types

            COMMON PLOT TYPE REQUIREMENTS (non-exhaustive):
            1. bar: 1 categorical (x) + 1 numerical (y)  → Variables: [numerical], [categorical]
            2. scatter: Exactly 2 numerical → Variables: [independent], [dependent]
            3. hist: Exactly 1 numerical → Variables: [numerical]
            4. boxplot: 1 numerical OR 1 numerical + 1 categorical → Variables: [numerical], [categorical] (if grouped)
            5. pie: Exactly 1 categorical → Variables: [categorical]
            6. line: 1 numerical (y) OR 1 numerical (y) + 1 datetime (x) → Variables: [y], [x] (if applicable)
            7. heatmap: 2 categorical + 1 numerical OR correlation matrix → Variables: [numerical], [categorical], [categorical]
            8. violinplot: Same as boxplot
            9. hexbin: Exactly 2 numerical variables
            10. pairplot: 2+ numerical variables
            11. jointplot: Exactly 2 numerical variables
            12. contour: 2 numerical variables for grid + 1 for values
            13. quiver: 2 numerical variables for grid + 2 for vectors
            14. imshow: 2D array of numerical values
            15. errorbar: 1 numerical (x) + 1 numerical (y) + error values
            16. stackplot: 1 numerical (x) + multiple numerical (y)
            17. stem: 1 numerical (x) + 1 numerical (y)
            18. fill_between: 1 numerical (x) + 2 numerical (y)
            19. pcolormesh: 2D grid of numerical values
            20. polar: Angular and radial coordinates

            If suggesting a plot not listed above, ensure:
            - The function exists in matplotlib
            - Variable types and counts are explicitly compatible
            - The rationale clearly explains the insight provided

            Additional Requirements:
            1. For specialized plots (like quiver, contour), ensure all required components are specified
            2. Consider the statistical properties and relationships of the variables
            3. Suggest plots that would reveal meaningful insights about the data
            4. Include both common and advanced plots when appropriate

            Example CORRECT suggestions (NUMERICAL FIRST):
            Plot Type: boxplot
            Variables: income, gender
            Rationale: Compares income distribution across genders
            ---
            Plot Type: scatter
            Variables: age, income
            Rationale: Shows relationship between age and income
            ---
            Plot Type: bar
            Variables: revenue, product_category
            Rationale: Compares revenue across product categories

            Example INCORRECT suggestions (REJECT THESE):
            Plot Type: boxplot
            Variables: gender, income  # WRONG - categorical listed first
            ---
            Plot Type: scatter
            Variables: price, weight  # WRONG - no clear priority order
            Rationale: Should specify independent/dependent variable order
        """)

    def _query_llm(self, prompt: str, model: str) -> str:
        if not self.clients.get('groq'):
            raise PlotSenseDataError("Groq client not initialized")

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
            raise PlotSenseAPIError(f"Groq API query failed for {model}: {str(e)}")

    def _validate_variable_order(
            self, recommendations: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and correct the order of variables in recommendations,
        ensuring numerical variables come first.

        Args:
            recommendations: DataFrame of visualization recommendations

        Returns:
            DataFrame with corrected variable order
        """
        def _reorder_variables(row):
            # Split variables
            variables = [var.strip() for var in row['variables'].split(',')]

            # Identify numerical and non-numerical variables
            numerical_vars = [
                var for var in variables
                if pd.api.types.is_numeric_dtype(self.df[var])
            ]

            date_vars = [
                var for var in variables
                if pd.api.types.is_datetime64_any_dtype(self.df[var])
            ]

            non_numerical_vars = [
                var for var in variables
                if var not in numerical_vars and var not in date_vars
            ]

            # Combine with numerical variables first
            corrected_vars = date_vars + numerical_vars + non_numerical_vars

            # Update the row with corrected variable order
            row['variables'] = ', '.join(corrected_vars)
            return row

        # Apply reordering
        corrected_recommendations = recommendations.apply(
            _reorder_variables, axis=1)

        if self.debug:
            print("\n[DEBUG] Variable Order Validation:")
            for orig, corrected in zip(
                    recommendations['variables'], corrected_recommendations['variables']):
                if orig != corrected:
                    print(f"  Corrected: {orig} → {corrected}")

        return corrected_recommendations

    def _parse_recommendations(self, response: str, model: str) -> List[Dict]:
        """Parse the LLM response into structured recommendations"""
        recommendations = []

        # Split response into recommendation blocks
        blocks = [b.strip() for b in response.split('---') if b.strip()]

        if self.debug:
            print(f"\n[DEBUG] Parsing {len(blocks)} blocks from {model}")

        for block in blocks:
            lines = [line.strip()
                     for line in block.split('\n') if line.strip()]
            if not lines:
                continue

            try:
                rec = {'source_model': model}
                for line in lines:
                    if line.lower().startswith('plot type:'):
                        rec['plot_type'] = line.split(
                            ':', 1)[1].strip().lower()
                    elif line.lower().startswith('variables:'):
                        raw_vars = line.split(':', 1)[1].strip()
                        # Filter variables to only those that exist in
                        # DataFrame
                        variables = [v.strip() for v in raw_vars.split(
                            ',') if v.strip() in self.df.columns]
                        rec['variables'] = ', '.join(
                            [var for var in variables if var in self.df.columns])
                        # rec['variables'] = self._reorder_variables(',
                        # '.join(variables))  # Keep original order for now

                if 'plot_type' in rec and 'variables' in rec and rec['variables']:
                    recommendations.append(rec)
            except Exception as e:
                warnings.warn(
                    f"Failed to parse recommendation from {model}: {str(e)}")
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
        _recommender_instance = VisualizationRecommender(
            api_keys=api_keys, debug=debug)

    _recommender_instance.set_dataframe(df)
    return _recommender_instance.recommend_visualizations(
        n=n,
        custom_weights=custom_weights
    )
