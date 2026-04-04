import os
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Tuple, Union, Optional, Dict
from dotenv import load_dotenv

from plotsense.core.ai_interface import AIModelInterface
from plotsense.core.enums.strategy import StrategyName
from plotsense.core.providers.provider_manager import ProviderManager
from plotsense.core.utils import encode_image, save_plot_to_image
from plotsense.exceptions import PlotSenseDataError

load_dotenv()


class PlotExplainer:
    """
    A class to generate and refine explanations for plots using LLMs.
    """

    def __init__(
        self,
        api_keys: Optional[Dict[str, str]],
        strategy: StrategyName = StrategyName.ROUND_ROBIN,
        selected_models: Optional[List[Tuple[str, str]]] = None,
        max_iterations: int = 3,
        interactive: bool = True,
        timeout: int = 30,
    ):
        self.timeout = timeout  # timeout for API calls
        self.max_iterations = max_iterations  # max iterations for refinement
        self.interactive = interactive
        self.strategy_name = strategy  # strategy for provider selection

        # selected_models is the source of truth for provider selection
        selected_providers = {p for p, _ in (selected_models or [])}

        self.manager = ProviderManager(
            api_keys=api_keys or {},
            interactive=interactive,
            selected_providers=list(selected_providers) if selected_providers else None
        )
        self.ai_interface = AIModelInterface(self.manager, timeout=self.timeout)

        all_models = self.manager.list_all_models()
        self.available_models = [
            (provider, model)
            for provider, models in all_models.items()
            for model in models
        ]

        if selected_models:
            # Expand vendor-level selections to include all variants
            # E.g., ("openai", "gpt-4o-mini") should match both:
            # - ("openai_chat", "gpt-4o-mini")
            # - ("openai_response", "gpt-4o-mini")
            expanded_selected = set()
            unsupported_models = {}  # Track which models aren't in registry

            for vendor, model in selected_models:
                # Check if vendor has variants (contains underscore in available_models)
                vendor_variants = {
                    prov for prov, _ in self.available_models
                    if prov.startswith(vendor + "_") or prov == vendor
                }

                if not vendor_variants:
                    # Vendor not available at all
                    raise ValueError(
                        f"Provider '{vendor}' is not initialized or has no available variants. "
                        f"Check that you have a valid API key for '{vendor}'."
                    )

                # Check if model is supported by this vendor
                supported_models = {
                    m for prov, m in self.available_models
                    if prov.startswith(vendor + "_") or prov == vendor
                }

                if model not in supported_models:
                    unsupported_models[vendor] = (model, sorted(supported_models))

                for variant in vendor_variants:
                    expanded_selected.add((variant, model))

            # If any models are unsupported, raise clear error
            if unsupported_models:
                error_lines = ["Unsupported model(s) requested:"]
                for vendor, (requested, supported) in unsupported_models.items():
                    error_lines.append(
                        f"\n  Provider '{vendor}': model '{requested}' not found."
                        f"\n  Supported models: {', '.join(supported[:3])}"
                        + (f", ... ({len(supported)} total)" if len(supported) > 3 else "")
                    )
                raise ValueError("\n".join(error_lines))

            self.available_models = [
                pair for pair in self.available_models if pair in expanded_selected
            ]

        if not self.available_models:
            raise ValueError(
                "No available models detected — check API keys or selection input."
            )

        self.strategy = self.ai_interface._init_strategy(
            self.strategy_name, self.available_models
        )

    def refine_plot_explanation(
        self,
        plot_object: Union[Figure, Axes],
        prompt: str = "Explain this data visualization",
        temp_image_path: str = "temp_plot.jpg",
        custom_parameters: Optional[Dict] = None
    ) -> str:
        """Generate and iteratively refine an explanation of a matplotlib/seaborn plot"""
        if not self.available_models:
            raise PlotSenseDataError("No available models detected")

        # Save plot to temporary image file
        image_path = save_plot_to_image(plot_object, temp_image_path)

        try:
            # Iterative refinement process
            current_explanation = None

            for iteration in range(self.max_iterations):
                provider, current_model = self.strategy.select_model(
                    iteration, current_explanation
                )

                if current_explanation is None:
                    current_explanation = self._generate_initial_explanation(
                        provider, current_model, image_path, prompt, custom_parameters
                    )
                else:
                    critique = self._generate_critique(
                        provider,
                        current_model,
                        image_path,
                        current_explanation,
                        prompt,
                        custom_parameters,
                    )

                    current_explanation = self._generate_refinement(
                        provider,
                        current_model,
                        image_path,
                        current_explanation,
                        critique,
                        prompt,
                        custom_parameters,
                    )

            if current_explanation is None:
                raise RuntimeError(
                    "Failed to generate an explanation — no models available or initial step failed."
                )

            return current_explanation

        finally:
            # Clean up temporary image file
            if os.path.exists(image_path):
                os.remove(image_path)

    def _generate_initial_explanation(
        self,
        provider: str,
        model: str,
        image_path: str,
        original_prompt: str,
        custom_parameters: Optional[Dict] = None
    ) -> str:
        """Generate initial plot explanation with structured format"""
        base_prompt = f"""
        Explanation Generation Requirements:
        - Provide a comprehensive analysis of the data visualization
        - Use a structured format with these sections:
        1. Overview
        2. Key Features
        3. Insights and Patterns
        4. Conclusion
        - Be specific and data-driven
        - Highlight key statistical and visual elements

        Specific Prompt: {original_prompt}

        Formatting Instructions:
        - Use markdown-style headers
        - Include bullet points for clarity
        - Provide quantitative insights
        - Explain the significance of visual elements
        """

        return self._query_model(
            provider=provider,
            model=model,
            prompt=base_prompt,
            image_path=image_path,
            custom_parameters=custom_parameters,
        )

    def _generate_critique(
        self,
        provider: str,
        model: str,
        image_path: str,
        current_explanation: str,
        original_prompt: str,
        custom_parameters: Optional[Dict] = None
    ) -> str:
        """Generate critique of current explanation"""
        critique_prompt = f"""
        Explanation Critique Guidelines:

        Current Explanation:
        {current_explanation}

        Original Prompt:
        {original_prompt}

        Evaluation Criteria:
        1. Assess the completeness of each section
        - Overview: Clarity and conciseness of plot description
        - Key Features: Depth of visual and statistical analysis
        - Insights and Patterns: Identification of meaningful trends
        - Conclusion: Relevance and forward-looking perspective

        2. Identify areas for improvement:
        - Are there missing key observations?
        - Is the language precise and data-driven?
        - Are statistical insights thoroughly explained?
        - Do the insights connect logically?

        3. Suggest specific enhancements:
        - Add more quantitative details
        - Clarify any ambiguous statements
        - Provide deeper context
        - Ensure comprehensive coverage of plot elements

        Provide a constructive critique that will help refine the explanation.
        """

        return self._query_model(
            provider=provider,
            model=model,
            prompt=critique_prompt,
            image_path=image_path,
            custom_parameters=custom_parameters
        )

    def _generate_refinement(
        self,
        provider: str,
        model: str,
        image_path: str,
        current_explanation: str,
        critique: str,
        original_prompt: str,
        custom_parameters: Optional[Dict] = None
    ) -> str:
        """Generate refined explanation based on critique"""
        refinement_prompt = f"""
        Explanation Refinement Instructions:

        Original Explanation:
        {current_explanation}

        Critique Received:
        {critique}

        original Prompt:
        {original_prompt}

        Refinement Guidelines:
        1. Address all points in the critique
        2. Maintain the original structured format
        3. Enhance depth and precision of analysis
        4. Add more quantitative insights
        5. Improve clarity and readability

        Specific Refinement Objectives:
        - Elaborate on key statistical observations
        - Provide more context for insights
        - Ensure each section is comprehensive
        - Use precise, data-driven language
        - Connect insights logically

        Produce a refined explanation that elevates the original analysis.
        - Be concise but thorough in your critique.
        - Use markdown-style headers for clarity
        - Include bullet points for clarity
        - Provide quantitative insights
        - Ensure the explanation is comprehensive and insightful
        """

        return self._query_model(
            provider=provider,
            model=model,
            prompt=refinement_prompt,
            image_path=image_path,
            custom_parameters=custom_parameters
        )

    def _query_model(
        self, provider: str, model: str, prompt: str, image_path: str,
        custom_parameters: Optional[Dict] = None
    ) -> str:
        base64_image = encode_image(image_path)
        return self.ai_interface.query_model(
            provider=provider,
            model=model,
            prompt=prompt,
            base64_image=base64_image,
            custom_parameters=custom_parameters
        )


# Package-level convenience function
_explainer_instance = None


def explainer(
    plot_object: Union[Figure, Axes],
    prompt: str = "Explain this data visualization",
    *,  # force keyword args after this

    custom_parameters: Optional[Dict] = None,
    strategy: StrategyName = StrategyName.ROUND_ROBIN,
    selected_models: Optional[List[Tuple[str, str]]] = None,

    api_keys: Optional[Dict[str, str]] = None,
    max_iterations: int = 3,
    interactive: bool = True,
    timeout: int = 30,
    temp_image_path: str = "temp_plot.jpg",
) -> str:
    """
    Convenience function to generate and refine plot explanations
    Uses a singleton PlotExplainer instance for efficiency.

    Args:
        - plot_object: Matplotlib Figure or Axes object to explain
        - prompt: Initial prompt for explanation generation
        - custom_parameters: Optional dict of custom parameters for the model
        - strategy: StrategyName enum for model selection strategy
        - selected_models: Optional list of (provider, model) tuples to restrict models
        - api_keys: Optional dict of API keys for providers
        - max_iterations: Max refinement iterations
        - interactive: Whether to prompt user for input when needed
        - timeout: Timeout in seconds for API calls
        - temp_image_path: Path to save temporary plot image

    Returns:
        A comprehensive, refined explanation generated by the chosen AI models.
    """

    global _explainer_instance

    if _explainer_instance is None:
        _explainer_instance = PlotExplainer(
            api_keys=api_keys,
            strategy=strategy,
            selected_models=selected_models,
            max_iterations=max_iterations,
            interactive=interactive,
            timeout=timeout,
        )

    return _explainer_instance.refine_plot_explanation(
        plot_object=plot_object,
        prompt=prompt,
        custom_parameters=custom_parameters,
        temp_image_path=temp_image_path
    )
