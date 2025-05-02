import base64
import os
import json
import hashlib
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, List
from dotenv import load_dotenv
from groq import Groq
import warnings

load_dotenv()

class PlotExplainer:
    DEFAULT_MODELS = {
        'groq': ['meta-llama/llama-4-scout-17b-16e-instruct', 'llama-3.2-90b-vision-preview'],
        # Add other providers here
    }
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None, timeout: int = 30, cache_dir: str = ".plot_explainer_cache"):
        """
        Initialize PlotExplainer with API keys and configuration.
        
        Args:
            api_keys: Optional dictionary of API keys. If not provided,
                     keys will be loaded from environment variables.
            timeout: Timeout in seconds for API requests
            cache_dir: Directory to store cached responses
        """
        api_keys = api_keys or {}
        self.api_keys = {
            'groq': os.getenv('GROQ_API_KEY')
            # Add other services here
        }
        self.api_keys.update(api_keys)
        
        self.timeout = timeout
        self.clients = {}
        self.available_models = []
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self._validate_keys()
        self._initialize_clients()
        self._detect_available_models()

    def _validate_keys(self):
        """Validate that required API keys are present"""
        for service in ['groq']:  # Add other required services here
            if not self.api_keys.get(service):
                self.api_keys[service] = input(f"Enter {service.upper()} API key: ").strip()
                if not self.api_keys[service]:
                    raise ValueError(f"{service.upper()} API key is required")

    def _initialize_clients(self):
        """Initialize API clients"""
        if self.api_keys.get('groq'):
            try:
                self.clients['groq'] = Groq(api_key=self.api_keys['groq'])
            except ImportError:
                warnings.warn("Groq Python client not installed. pip install groq")
        
    def _detect_available_models(self):
        """Detect which models are available based on configured clients"""
        self.available_models = []
        for provider, client in self.clients.items():
            if client and provider in self.DEFAULT_MODELS:
                self.available_models.extend(self.DEFAULT_MODELS[provider])
    
    def _get_cache_key(self, img_bytes: bytes, prompt: str, model: str, context: str = "") -> str:
        """Generate unique hash for caching"""
        combined = f"{model}{context}{prompt}{hashlib.md5(img_bytes).hexdigest()}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def refine_plot_explanation(
        self, 
        plot_object: Union[plt.Figure, plt.Axes],
        prompt: str = "Explain this data visualization",
        iterations: int = 2,
        model_rotation: Optional[List[str]] = None,
        temperature: float = 0.3,
        deterministic: bool = False
    ) -> str:
        """
        Generate and iteratively refine an explanation of a matplotlib/seaborn plot
        
        Args:
            plot_object: Matplotlib Figure or Axes object
            prompt: Initial explanation prompt
            iterations: Number of refinement cycles (1-5)
            model_rotation: Optional list of models to use
            temperature: Controls randomness (0 for deterministic)
            deterministic: If True, forces temperature=0 and uses caching
            
        Returns:
            str: Refined explanation text
        """
        if not self.available_models:
            raise ValueError("No available models detected")
        
        if iterations < 1 or iterations > 5:
            raise ValueError("Iterations must be between 1 and 5")
        
        if deterministic:
            temperature = 0
            
        img_bytes = self._plot_to_bytes(plot_object)
        models = model_rotation or self._select_model_rotation()
        
        explanation = self._generate_explanation(img_bytes, prompt, models[0], temperature)
        
        for i in range(iterations - 1):
            critic = models[i % len(models)]
            refiner = models[(i + 1) % len(models)]
            
            critique = self._generate_critique(
                img_bytes, explanation, prompt, critic, temperature
            )
            
            explanation = self._generate_refinement(
                img_bytes, explanation, critique, prompt, refiner, temperature
            )
        
        return explanation
    
    def _plot_to_bytes(self, plot_object: Union[plt.Figure, plt.Axes]) -> bytes:
        """Convert matplotlib plot to consistent bytes"""
        if isinstance(plot_object, plt.Axes):
            fig = plot_object.figure
        else:
            fig = plot_object
            
        # Standardize image generation
        fig.set_size_inches(8, 6)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue()
    
    def _select_model_rotation(self):
        """Select models to use based on availability"""
        priority_order = [
            'meta-llama/llama-4-scout-17b-16e-instruct', 
            'llama-3.2-90b-vision-preview'
        ]
        return [m for m in priority_order if m in self.available_models]
    
    def _query_model(
        self, 
        img_bytes: bytes, 
        prompt: str, 
        model: str, 
        temperature: float = 0.3,
        context: str = ""
    ) -> str:
        """Generic method to query different models with caching"""
        cache_key = self._get_cache_key(img_bytes, prompt, model, context)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            return json.loads(cache_file.read_text())["response"]
        
        if model in ['meta-llama/llama-4-scout-17b-16e-instruct', 'llama-3.2-90b-vision-preview']:
            response = self._query_llama3(img_bytes, prompt, temperature)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        cache_file.write_text(json.dumps({
            "response": response,
            "model": model,
            "temperature": temperature,
            "prompt": prompt
        }))
        return response
        
    def _generate_explanation(
        self, 
        img_bytes: bytes, 
        prompt: str, 
        model: str,
        temperature: float
    ) -> str:
        """Generate initial explanation"""
        return self._query_model(img_bytes, prompt, model, temperature)
    
    def _generate_critique(
        self, 
        img_bytes: bytes, 
        explanation: str, 
        prompt: str, 
        model: str,
        temperature: float
    ) -> str:
        """Generate critique of current explanation"""
        critique_prompt = f"""
        Critique this plot explanation:
        
        Original Prompt: {prompt}
        Current Explanation: {explanation}
        
        Provide specific feedback on:
        1. Data interpretation accuracy
        2. Clarity of key insights
        3. Missing important patterns
        4. Technical correctness
        
        Be concise but thorough in your critique.
        """
        return self._query_model(img_bytes, critique_prompt, model, temperature, "critique")
    
    def _generate_refinement(
        self, 
        img_bytes: bytes, 
        explanation: str, 
        critique: str, 
        prompt: str, 
        model: str,
        temperature: float
    ) -> str:
        """Generate refined explanation"""
        refinement_prompt = f"""
        Improve this plot explanation based on the critique:
        
        Original Prompt: {prompt}
        Current Explanation: {explanation}
        Critique: {critique}
        
        Create an improved version that:
        1. Addresses all valid critique points
        2. Maintains accurate information
        3. Improves clarity and insightfulness
        4. Preserves technical correctness
        """
        return self._query_model(img_bytes, refinement_prompt, model, temperature, "refinement")
    
    def _query_llama3(
        self, 
        img_bytes: bytes, 
        prompt: str, 
        temperature: float = 0.3,
        seed: int = 42
    ) -> str:
        """Query Groq's Llama3 model with plot image"""
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        try:
            response = self.clients['groq'].chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
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
                temperature=temperature,
                seed=seed if temperature == 0 else None,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq API: {str(e)}")
            raise

# Package-level convenience function
_explainer_instance = None

def refine_plot_explanation(
    plot_object: Union[plt.Figure, plt.Axes],
    prompt: str = "Explain this data visualization",
    iterations: int = 2,
    api_keys: dict = {},
    deterministic: bool = False
) -> str:
    """
    Generate an AI-refined explanation of a matplotlib/seaborn plot
    
    Args:
        plot_object: Matplotlib Figure or Axes object
        prompt: Explanation prompt (default generic)
        iterations: Refinement cycles (default 2)
        api_keys: Dictionary of API keys
        deterministic: If True, ensures consistent outputs
        
    Returns:
        str: Refined explanation text
    """
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = PlotExplainer(api_keys)
    return _explainer_instance.refine_plot_explanation(
        plot_object=plot_object,
        prompt=prompt,
        iterations=iterations,
        deterministic=deterministic
    )