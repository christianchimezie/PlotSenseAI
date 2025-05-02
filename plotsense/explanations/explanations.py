import base64
import os
from io import BytesIO
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, List
from dotenv import load_dotenv
from groq import Groq
import warnings
import builtins

load_dotenv()
class PlotExplainer:
    DEFAULT_MODELS = {
        'groq': ['meta-llama/llama-4-scout-17b-16e-instruct', 'llama-3.2-90b-vision-preview'],
        # Add other providers here
    }
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None,interactive: bool = True, timeout: int = 30):
        """
        Initialize PlotExplainer with API keys and configuration.
        
        Args:
            api_keys: Optional dictionary of API keys. If not provided,
                     keys will be loaded from environment variables.
            timeout: Timeout in seconds for API requests
        """
        # Default to empty dict if None
        api_keys = api_keys or {}

         # Set up default keys from environment variables
        self.api_keys = {
            'groq': os.getenv('GROQ_API_KEY')
            # Add other services here
        }
        self.api_keys.update(api_keys)
    
        self.interactive = interactive
        self.timeout = timeout
        self.clients = {}
        self.available_models = []
        
        self.api_keys.update(api_keys)
        
        self._validate_keys()
        self._initialize_clients()
        self._detect_available_models()


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
                import groq
                self.clients['groq'] = Groq(api_key=self.api_keys['groq'])
            except ImportError:
                warnings.warn("Groq Python client not installed. pip install groq", ImportWarning)

    def _detect_available_models(self):
        self.available_models = []
        for provider, client in self.clients.items():
            if client and provider in self.DEFAULT_MODELS:
                # For now we'll assume all DEFAULT_MODELS are available
                # In a real implementation, you might want to check which models are actually available
                self.available_models.extend(self.DEFAULT_MODELS[provider])
    
    def refine_plot_explanation(
        self, 
        plot_object: Union[plt.Figure, plt.Axes],
        prompt: str = "Explain this data visualization",
        iterations: int = 2,
        model_rotation: Optional[List[str]] = None
    ) -> str:
        """
        Generate and iteratively refine an explanation of a matplotlib/seaborn plot
        
        Args:
            plot_object: Matplotlib Figure or Axes object
            prompt: Initial explanation prompt
            iterations: Number of refinement cycles (1-5)
            model_rotation: Optional list of models to use (default: auto-select)
            
        Returns:
            str: Refined explanation text
            
        Raises:
            ValueError: If no models are available or invalid parameters provided
        """
        if not self.available_models:
            raise ValueError("No available models detected")
        
        if iterations < 1 or iterations > 5:
            raise ValueError("Iterations must be between 1 and 5")
        
        img_bytes = self._plot_to_bytes(plot_object)
        models = model_rotation or self._select_model_rotation()
        
        explanation = self._generate_explanation(img_bytes, prompt, models[0])
        
        for i in range(iterations - 1):
            critic = models[i % len(models)]
            refiner = models[(i + 1) % len(models)]
            
            critique = self._generate_critique(
                img_bytes, explanation, prompt, critic
            )
            
            explanation = self._generate_refinement(
                img_bytes, explanation, critique, prompt, refiner
            )
        
        return explanation
    
    def _plot_to_bytes(self, plot_object: Union[plt.Figure, plt.Axes]) -> bytes:
        """Convert matplotlib plot to bytes"""
        if isinstance(plot_object, plt.Axes):
            fig = plot_object.figure
        else:
            fig = plot_object

        # Standardize image generation
        fig.set_size_inches(8, 6)   
        buf = BytesIO()
        fig.savefig(buf, format='png',dpi=100, bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue()
    
    def _select_model_rotation(self):
        """Select models to use based on availability"""
        priority_order = [
            'meta-llama/llama-4-scout-17b-16e-instruct', 'llama-3.2-90b-vision-preview'
        ]
        return [m for m in priority_order if m in self.available_models]
    
    def _query_model(self, img_bytes: bytes, prompt: str, model: str) -> str:
        """Generic method to query different models"""
        if model in ['meta-llama/llama-4-scout-17b-16e-instruct', 'llama-3.2-90b-vision-preview']:
            response = self._query_llama3(img_bytes, prompt)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        return response
        
    def _generate_explanation(self, img_bytes: bytes, prompt: str, model: str) -> str:
        """Generate initial explanation"""
        return self._query_model(img_bytes, prompt, model)
    
    def _generate_critique(self, img_bytes: bytes, explanation: str, prompt: str, model: str) -> str:
        """Generate critique of current explanation"""
        critique_prompt = f"""
        Critique this plot explanation based on the required structure:
        
        Original Prompt: {prompt}
        Current Explanation: {explanation}

        Evaluate whether the explanation contains all required sections:
        1. Overview
        2. Key Features
        3. Insights and Patterns
        4. Conclusion
            
        For each section, provide specific feedback on:
        - Completeness of information
        - Accuracy of observations
        - Clarity of presentation
        - Depth of analysis
        
        Also note any:
        - Missing important patterns
        - Technical inaccuracies
        - Unclear statements

        Be concise but thorough in your critique.
        
        Provide your critique in a bullet-point format.
        """
        return self._query_model(img_bytes, critique_prompt, model)
    
    def _generate_refinement(self, img_bytes: bytes, explanation: str, critique: str, prompt: str, model: str) -> str:
        """Generate refined explanation"""
        refinement_prompt = f"""
        Improve this plot explanation based on the critique while maintaining the required structure:
        
        Original Prompt: {prompt}
        Current Explanation: {explanation}
        Critique: {critique}
        
        Create an improved version that maintains these clear sections:
        - Overview
        - Key Features
        - Insights and Patterns
        - Conclusion
        
        Specifically:
        1. Address all valid critique points
        2. Ensure each section is well-developed
        3. Maintain accurate information
        4. Improve clarity and insightfulness
        5. Keep technical correctness
        
        Return the improved explanation with the same section headers.
        """
        return self._query_model(img_bytes, refinement_prompt, model)
    
    def _query_llama3(self, img_bytes: bytes, prompt: str) -> str:
        """Query Groq's Llama3 model with plot image"""
        # Initialize client with API key if not already done
        if 'groq' not in self.clients:
            raise ValueError("Groq client not initialized")
        
        client = self.clients['groq']  # Use the client from initialization
        
        # Convert image to base64
        base64_image = base64.b64encode(img_bytes).decode('utf-8')

        # Structured prompt template
        structured_prompt = f"""
        {prompt}
        
        Please structure your response with these clear sections:
        
        **Overview**:
        Provide a high-level description of what the visualization shows
        
        **Key Features**:
        - Describe the main visual elements
        - Note any important data points or ranges
        - Highlight how variables are represented
        
        **Insights and Patterns**:
        - Identify trends, clusters, or outliers
        - Note any interesting relationships between variables
        - Point out any surprising or notable observations
        
        **Conclusion**:
        - Summarize the main takeaways
        - Suggest any implications or next steps for analysis
        
        Keep the response clear, concise, and focused on the data.
        """
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": structured_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq API: {str(e)}")
            raise
    
    # Add similar methods for other models (Claude, Gemini, etc.)

# Package-level convenience function
_explainer_instance = None

def explainer(
    plot_object: Union[plt.Figure, plt.Axes],
    prompt: str = "Explain this data visualization",
    iterations: int = 2,
    api_keys: dict = {}
) -> str:
    """
    Generate an AI-refined explanation of a matplotlib/seaborn plot
    
    Args:
        plot_object: Matplotlib Figure or Axes object
        prompt: Explanation prompt (default generic)
        iterations: Refinement cycles (default 2)
        api_keys: Dictionary of API keys
        
    Returns:
        str: Refined explanation text
    """
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = PlotExplainer(api_keys)
    return _explainer_instance.refine_plot_explanation(
        plot_object=plot_object,
        prompt=prompt,
        iterations=iterations
    )

