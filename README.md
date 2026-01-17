# 🌟 PlotSense: AI-Powered Data Visualization Assistant

[![PyPI version](https://img.shields.io/pypi/v/plotsense.svg)](https://pypi.org/project/plotsense/)
[![Downloads](https://static.pepy.tech/badge/plotsense)](https://pepy.tech/project/plotsense)
[![Python Tests](https://github.com/PlotSenseAI/PlotSenseAI/actions/workflows/python-tests.yml/badge.svg)](https://github.com/PlotSenseAI/PlotSenseAI/actions/workflows/python-tests.yml)
[![Frontend Tests](https://github.com/PlotSenseAI/PlotSenseAI/actions/workflows/frontend-tests.yml/badge.svg)](https://github.com/PlotSenseAI/PlotSenseAI/actions/workflows/frontend-tests.yml)
[![codecov](https://codecov.io/gh/PlotSenseAI/PlotSenseAI/branch/main/graph/badge.svg)](https://codecov.io/gh/PlotSenseAI/PlotSenseAI)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-black.svg)](https://flake8.pycqa.org/)

## 📌 Overview

**PlotSense** is an AI-powered assistant that helps data professionals and analysts make smarter, faster, and more explainable data visualizations. Whether you're exploring a new dataset or building dashboards, PlotSense simplifies the process with:

- ✅ Smart Visualization Suggestions - Recommends the best plots based on your data structure and relationships.
- 📊 Visualization Plot - Generates suggested plot with ease.
- 🧠 Natural Language Explanations – Automatically explains charts in plain English.
- 🔗 Seamless Integration – Works out of the box with pandas, matplotlib, and seaborn.

Let AI supercharge your EDA (Exploratory Data Analysis).

## 📚 Documentation
- 📖 **[Technical Roadmap](https://plotsenseai.gitbook.io/plotsense-technical-roadmap/)** - Future features and development plans
- 🏗️ **[Architecture & Methodology](https://plotsenseai.gitbook.io/plotsense-technical-roadmap/plotsense-technical-documentation-v2)** - System design and technical implementation

## 💬 Community

Join our community to get help, share ideas, and connect with other PlotSense users:

- 💭 **[Discord Server](https://discord.gg/CacGryW4HR)** - Chat with the community and get real-time support

## ⚡ Quickstart

### 🔧 Install the package

Using pip:
```bash
pip install plotsense
```

Using uv (recommended for faster installation):
```bash
uv pip install plotsense
```

### 🧠 Import PlotSense:

```bash
import plotsense as ps
from plotsense import recommender, plotgen, explainer
```
### 🔐 Authenticate with Groq API:
Get your free API key from Groq Cloud https://console.groq.com/home

```bash
import os
# Set GROQ_API_KEY environment variable
os.environ['GROQ_API_KEY'] = 'your-api-key-here'

#or

# Set API key (one-time setup)
ps.set_api_key("your-api-key-here")
```

## 🚀 Core Features
### 🎯 1. AI-Recommended Visualizations
Let PlotSense analyze your data and suggest optimal charts.

```bash
import pandas as pd
# Load your dataset (e.g., pandas DataFrame)
df = pd.read_csv("data.csv")

# Get AI-recommended visualizations
suggestions = recommender(df) # default number of suggestions is 5
print(suggestions)
```
### 📊 Sample Output:

![alt text](image.png)

🎛️ Want more suggestions?

``` bash
suggestions = recommender(df, n=10)  
```

### 📈 2. One-Click Plot Generation
Generate recommended charts instantly using .iloc

```bash
plot1 = plotgen(df, suggestions.iloc[0]) # This will plot a bar chart with variables 'survived', 'pclass'
plot2 = plotgen(df, suggestions.iloc[1]) # This will plot a bar chart with variables 'survived', 'sex'
plot3 = plotgen(df, suggestions.iloc[2]) # This will plot a histogram with variable 'age'
```

or Generate recommended charts instantly using three argurments

```bash
plot1 = plotgen(df, 0, suggestions) # This will plot a bar chart with variables 'survived', 'pclass'
plot2 = plotgen(df, 1, suggestions) # This will plot a bar chart with variables 'survived', 'sex'
plot3 = plotgen(df, 2, suggestions) # This will plot a histogram with variable 'age'
```
🎛️ Want more control?

``` bash
plot1 = plotgen(df, suggestions.iloc[0], x='pclass', y='survived')
```
Supported Plots
- scatter
- bar
- barh
- histogram
- boxplot
- violinplot
- pie
- hexbin

### 🧾 3. AI-Powered Plot Explanation
Turn your visualizations into stories with natural language insights:

``` bash
explanation = explainer(plot1)

print(explanation)
```

### ⚙️ Advanced Options
- Custom Prompts: You can provide your own prompt to guide the explanation

``` bash
explanation = explainer(
    fig,
    prompt="Explain the key trends in this sales data visualization"
)
```
- Multiple Refinement Iterations: Increase the number of refinement cycles for more polished explanations:

```bash  
explanation = explainer(fig, max_iterations=3)  # Default is 2
```

## 🔄 Combined Workflow: Suggest → Plot → Explain
``` bash
suggestions = recommender(df)
plot = plotgen(df, suggestions.iloc[0])
insight = explainer(plot)
```

## 🧪 Testing

PlotSenseAI has comprehensive test coverage for both Python and frontend code.

### Quick Test Commands

**Python Tests:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=plotsense --cov-report=html

# Run fast tests only (skip slow tests)
pytest -m "not slow"
```

**Frontend Tests:**
```bash
cd web

# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run in watch mode
npm test -- --watch
```

### Test Infrastructure

- **Backend**: pytest with 950+ lines of tests covering:
  - Unit tests for individual functions
  - Integration tests for component interaction
  - End-to-end workflow tests
  - Mock external API calls (Groq)

- **Frontend**: Vitest + React Testing Library
  - Component tests for UI elements
  - User interaction testing
  - Accessibility testing

- **CI/CD**: GitHub Actions runs tests automatically on every push and PR
- **Pre-commit Hooks**: Automated linting and quick tests before commits

For detailed testing documentation, see [TESTING.md](TESTING.md).

### Setting Up Development Environment

Using uv (recommended):
```bash
# Install uv if not already installed
# Visit https://docs.astral.sh/uv/getting-started/installation/

# Sync all dependencies including dev extras
uv sync --all-extras

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or macOS:
source .venv/bin/activate

# Install frontend dependencies
cd web && npm install

# Install pre-commit hooks (optional but recommended)
uv run pre-commit install
```

Using pip (traditional):
```bash
# Install Python dependencies
pip install -e .
pip install pytest pytest-cov pytest-mock

# Install frontend dependencies
cd web && npm install

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Start for Contributors

1. **Fork and clone** the repository
2. **Set up development environment:**

   Using uv (recommended):
   ```bash
   # Install uv: https://docs.astral.sh/uv/getting-started/installation/
   uv sync --all-extras
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv run pre-commit install
   ```

   Using pip:
   ```bash
   pip install -e .
   pip install pytest pytest-cov pytest-mock pre-commit
   pre-commit install
   ```
3. **Create a feature branch:** `git checkout -b feature/your-feature`
4. **Make changes and add tests**
5. **Run tests:** `uv run pytest` (or `pytest`) and `cd web && npm test`
6. **Submit a Pull Request**

### Branching Strategy
- `main` → Production-ready version
- `dev` → Active development
- `feature/<feature-name>` → New features
- `fix/<bug-name>` → Bug fixes

### 💡 How to Help
- 🐞 **Bug Reports** → [Open an issue](https://github.com/PlotSenseAI/PlotSenseAI/issues/new?template=bug_report.md)
- 💡 **Feature Requests** → [Request a feature](https://github.com/PlotSenseAI/PlotSenseAI/issues/new?template=feature_request.md)
- ❓ **Questions** → [Ask a question](https://github.com/PlotSenseAI/PlotSenseAI/issues/new?template=question.md)
- 🚀 **Submit PRs** → See [CONTRIBUTING.md](CONTRIBUTING.md)

### 📅 Roadmap

Upcoming features:
- More model integrations
- Automated insight highlighting
- Jupyter widget support
- Features/target analysis
- More supported plots
- PlotSense web interface
- PlotSense customised notebook template

### 📥 Install or Update
Using pip:
``` bash
pip install --upgrade plotsense  # Get the latest features!
```

Using uv:
``` bash
uv pip install --upgrade plotsense  # Get the latest features faster!
```

## 👥 Maintainers

PlotSense is maintained by a dedicated team of developers and data scientists:

| Name | GitHub | Email |
|------|--------|-------|
| **Christian Chimezie** | [@christianchimezie](https://github.com/christianchimezie) | chimeziechristiancc@gmail.com |
| **Toluwaleke Ogidan** | [@T-leke](https://github.com/T-leke) | gbemilekeogidan@gmail.com |
| **Onyekachukwu Ojumah** | [@ojumah20](https://github.com/ojumah20) | Onyekaojumah22@gmail.com |
| **Grace Farayola** | [@Itsmeright](https://github.com/Itsmeright) | gracefarayola@gmail.com |
| **Amaka Iduwe** | [@Nwaamaka-Iduwe](https://github.com/Nwaamaka-Iduwe) | nwaamaka_iduwe@yahoo.com |
| **Nelson Ogbeide** | [@Nelsonchris1](https://github.com/Nelsonchris1) | Ogbeide331@gmail.com |
| **Abayomi Olagunju** | [@jerryola1](https://github.com/jerryola1) | https://abayomiolagunju.net/|
| **Olamilekan Ajao** | [@olamilekanajao](https://github.com/olamilekanajao) | olamilekan011@gmail.com |

We welcome contributions from the community! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🛡 License
Apache License 2.0

## 🔐 API & Privacy Notes
- Your API key is securely held in memory for your current Python session.
- All requests are processed via Groq's API servers—no data is stored locally by PlotSense.
- Requires an internet connection for model-backed features.

Let your data speak—with clarity, power, and PlotSense.
📊✨

## Your Feedback
[Feedback Form](https://forms.gle/QEjipzHiMagpAQU99)
