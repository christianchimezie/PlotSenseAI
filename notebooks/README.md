# PlotKit Notebooks

This directory contains Jupyter notebooks for learning, testing, and exploring PlotKit's features.

## Available Notebooks

### 1. **getting-started.ipynb**
Introductory guide to PlotKit. Start here if you're new to the package.

### 2. **01-testing-explanations.ipynb** 
Test and demonstrate the **Explanations API** (`explainer()` function).

**What you'll learn:**
- Generate explanations for plots using LLMs
- Select specific providers and models
- Handle API keys (environment, direct, interactive)
- Use error messages to debug issues
- Reference all available models

**Perfect for:**
- Testing OpenAI, Groq, Anthropic, Gemini providers
- Understanding multi-provider support
- Learning model selection syntax
- Testing error handling

### 3. **02-testing-suggestions.ipynb**
Test and demonstrate the **Suggestions API** (`recommender()` function).

**What you'll learn:**
- Get visualization recommendations for your data
- Use different LLM providers for recommendations
- Work with various data types (timeseries, categories, etc.)
- Configure recommendation count and providers
- Test with sample datasets

**Perfect for:**
- Testing recommendation generation
- Understanding what visualizations suit your data
- Comparing recommendations across providers

### 4. **03-testing-multi-provider.ipynb**
Advanced notebook exploring **multi-provider infrastructure**.

**What you'll learn:**
- How the dynamic registry system works
- Available models across all providers
- Cost and performance metrics
- Provider variant system (e.g., groq_default vs groq_openai)
- Error handling and validation
- Cross-provider querying

**Perfect for:**
- Understanding the architecture
- Cost optimization (comparing provider prices)
- Performance analysis
- Advanced debugging

## How to Use

### Quick Start

1. **Start Jupyter:**
   ```bash
   cd /home/dyung/Projects/PlotKit
   jupyter notebook notebooks/
   ```

2. **Open a notebook:**
   - Click the notebook file in the Jupyter interface
   - Or navigate to `http://localhost:8888` and select the notebook

3. **Run cells:**
   - Click a cell and press `Shift+Enter` to run it
   - Or use the Run button in the toolbar

### Setting Up API Keys

Before running any notebook, set up your API keys:

**Option 1: Environment Variables (Recommended)**
```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk-..."
export ANTHROPIC_API_KEY="..."
jupyter notebook notebooks/
```

**Option 2: .env File**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk-...
ANTHROPIC_API_KEY=...
```

**Option 3: Interactive Mode**
Leave `api_keys` empty and set `interactive=True`:
```python
explainer(fig, interactive=True)  # Will prompt for keys
```

## Notebook Organization

### Explanations Notebook (01)
```
1. Setup imports
2. Create test plot
3. Configure API keys
4. Test basic explanation
5. Test with specific provider (OpenAI)
6. Test with Groq
7. Test error handling
8. Reference available models
```

### Suggestions Notebook (02)
```
1. Setup imports
2. Create sample data
3. Configure API keys
4. Test basic recommendations
5. Test with OpenAI
6. Test with Groq
7. Test with different data types
8. Reference available models
```

### Multi-Provider Notebook (03)
```
1. Setup imports
2. Explore registry system
3. List all available models
4. Check costs and performance
5. Understand provider variants
6. Create test plot and data
7. Test error handling (invalid model)
8. Test error handling (invalid provider)
9. Query different providers
```

## Common Tasks

### Test a Specific Provider
```python
# In any notebook, use selected_models:
result = explainer(
    fig,
    api_keys=api_keys,
    selected_models=[("groq", "llama-3.1-8b-instant")]
)
```

### Check What Models Are Available
```python
from plotsense.core.registry_loader import get_registry_loader
registry = get_registry_loader()

# Get models for a specific provider
models = registry.get_provider_models("openai", "chat")
print(models)  # ['gpt-4o', 'gpt-4o-mini', ...]
```

### Get Cost Information
```python
registry = get_registry_loader()
costs = registry.get_model_costs()
# Compare prices across providers
```

### Test Error Handling
```python
# This will show a clear error with supported models
try:
    explainer(
        fig,
        selected_models=[("openai", "invalid-model")]
    )
except ValueError as e:
    print(e)  # Shows supported models
```

## Tips

### Jupyter Shortcuts
- `Shift+Enter`: Run cell
- `Ctrl+M`: Toggle markdown/code mode
- `A`: Insert cell above
- `B`: Insert cell below
- `DD`: Delete cell
- `Z`: Undo cell deletion

### Best Practices
1. **Don't edit .ipynb files directly** - always use Jupyter UI
2. **Save frequently** - Jupyter auto-saves, but manual save is good too
3. **Clear outputs** - Use "Cell > All Output > Clear" to reduce file size
4. **Add comments** - Document what you're testing and why
5. **Use markdown cells** - Structure your notebooks with headers and explanations

### Debugging Tips
- **Check API keys**: Print them (masked) with `api_keys.keys()`
- **Read error messages**: They now tell you exactly what's wrong
- **Use print statements**: Add `print()` to debug values
- **Restart kernel**: If something weird happens, restart the kernel
- **Check logs**: Look at terminal output for additional context

## Environment Variables Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `GROQ_API_KEY` | Groq API key | `gsk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `GOOGLE_API_KEY` | Google Gemini key | `AIza...` |
| `AZURE_API_KEY` | Azure OpenAI key | `...` |
| `PLOTSENSE_REGISTRY_URL` | Custom registry URL | (optional) |

## File Size Notes

Notebooks can get large with outputs and images. To keep them manageable:
- Clear cell outputs before committing: `Cell > All Output > Clear`
- Use `.gitignore` to exclude large notebook outputs
- Or store outputs separately in a `results/` directory

## Contributing

If you create useful test notebooks:
1. Follow the naming convention: `NN-description.ipynb`
2. Add a docstring at the top explaining what it tests
3. Include markdown sections with clear structure
4. Test all code before committing
5. Clear outputs before committing

## Support

For issues or questions:
1. Check the error message - it should guide you
2. Refer to the appropriate notebook for examples
3. Check `docs/` for detailed documentation
4. Review GitHub issues for similar problems
