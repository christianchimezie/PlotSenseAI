# Testing Guide for PlotSenseAI

This document provides comprehensive information about the testing infrastructure for PlotSenseAI, including how to run tests, write new tests, and understand the testing architecture.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Python Testing](#python-testing)
- [Frontend Testing](#frontend-testing)
- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [CI/CD Integration](#cicd-integration)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Coverage Reports](#coverage-reports)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

PlotSenseAI uses a comprehensive testing infrastructure that includes:

- **Backend**: pytest with coverage tracking for Python code
- **Frontend**: Vitest with React Testing Library for TypeScript/React code
- **CI/CD**: GitHub Actions for automated testing on every push/PR
- **Pre-commit hooks**: Automated checks before commits
- **Code quality**: Linting with flake8 (Python) and ESLint (TypeScript)

## Quick Start

### Install Dependencies

**Python:**
```bash
pip install -e .
pip install pytest pytest-cov pytest-mock
```

**Frontend:**
```bash
cd web
npm install
```

### Run All Tests

**Python:**
```bash
pytest
```

**Frontend:**
```bash
cd web
npm test
```

## Python Testing

### Framework and Tools

- **Test Runner**: pytest 7.0+
- **Coverage**: pytest-cov
- **Mocking**: pytest-mock, unittest.mock
- **Configuration**: `pytest.ini`, `pyproject.toml`, `.coveragerc`

### Test Structure

The test suite is organized into three categories:

```
tests/
├── conftest.py                           # Shared fixtures and configuration
├── unit/                                 # Fast, isolated unit tests (mocked APIs)
│   ├── conftest.py                       # Unit test specific fixtures
│   ├── test_exceptions.py                # Exception handling tests
│   ├── test_plot_generator.py            # Plot generation unit tests
│   ├── test_explanations_unit.py         # PlotExplainer unit tests (mocked)
│   └── test_suggestions_unit.py          # VisualizationRecommender unit tests (mocked)
├── integration/                          # Integration tests (component interaction)
│   ├── conftest.py                       # Integration test fixtures
│   └── (currently placeholder)
└── live/                                 # Live tests (real API calls)
    ├── conftest.py                       # Live test fixtures
    ├── test_live_suggestions.py          # Real Groq/OpenAI API tests
    └── test_live_explanations.py         # Real Groq/OpenAI API tests
```

**Test Coverage**:
- **Unit Tests**: 56 tests covering core functionality with mocked APIs
- **Live Tests**: 5 tests requiring real API keys (Groq, OpenAI)

### Running Python Tests

```bash
# Run all tests (unit and integration only, live tests skipped)
pytest

# Run only non-live tests (same as above)
pytest -m "not live"

# Run only live tests (requires GROQ_API_KEY and OPENAI_API_KEY)
pytest -m live

# Run only unit tests
pytest tests/unit

# Run with coverage
pytest --cov=plotsense --cov-report=html

# Run specific test file
pytest tests/unit/test_plot_generator.py

# Run specific test class
pytest tests/unit/test_plot_generator.py::TestPlotFunctions

# Run specific test
pytest tests/unit/test_plot_generator.py::TestPlotFunctions::test_create_scatter

# Run tests by marker
pytest -m unit                  # Fast unit tests only
pytest -m integration          # Integration tests only
pytest -m "not live"           # All tests except live (default)
pytest -m live                 # Only live tests

# Verbose output with details
pytest -v

# Stop on first failure
pytest -x

# Show local variables in tracebacks
pytest -l

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

### Test Markers

Tests are organized with markers for selective execution:

- `@pytest.mark.unit`: Fast, isolated unit tests (mocked APIs)
- `@pytest.mark.integration`: Tests that combine multiple components
- `@pytest.mark.e2e`: End-to-end workflow tests
- `@pytest.mark.slow`: Tests that take significant time
- `@pytest.mark.api`: Tests that mock external API calls
- `@pytest.mark.requires_api_key`: Tests requiring valid credentials
- `@pytest.mark.plotting`: Tests that generate visualizations
- `@pytest.mark.live`: Live tests requiring real API keys and actual API calls (opt-in)

### Shared Fixtures

The `tests/conftest.py` file provides shared fixtures:

- `test_api_key`: Test API key from environment
- `sample_dataframe`: Standard 100-row DataFrame for testing
- `small_dataframe`: 20-row DataFrame for quick tests
- `large_dataframe`: 1000-row DataFrame for performance tests
- `sample_suggestions`: Mock visualization suggestions
- `simple_plot`, `sample_plot`, `scatter_plot`, `bar_plot`: Matplotlib plot fixtures
- `mock_groq_client`: Mocked Groq API client
- `mock_groq_completion`: Mocked Groq completion response
- `temp_image_path`: Temporary image file for testing
- `reset_matplotlib`: Auto-use fixture that closes all plots after each test
- `reset_random_seed`: Auto-use fixture that resets NumPy seed for reproducibility

### Example Test

```python
import pytest
from plotsense import plotgen

@pytest.mark.unit
def test_scatter_plot(sample_dataframe, sample_suggestions):
    """Test scatter plot generation."""
    fig = plotgen(sample_dataframe, 0, sample_suggestions)
    assert fig is not None
    assert len(fig.axes[0].collections) == 1
```

## Frontend Testing

### Framework and Tools

- **Test Runner**: Vitest 2.1+
- **Testing Library**: React Testing Library 16.1+
- **User Interactions**: @testing-library/user-event
- **Coverage**: @vitest/coverage-v8
- **Configuration**: `vitest.config.ts`

### Test Structure

```
web/src/
├── test/
│   ├── setup.ts              # Global test setup
│   └── test-utils.tsx        # Custom render utilities
└── components/
    └── ui/
        ├── button.tsx
        ├── button.test.tsx   # Button component tests
        ├── card.tsx
        └── card.test.tsx     # Card component tests
```

### Running Frontend Tests

```bash
cd web

# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- button.test.tsx

# Run tests matching pattern
npm test -- --grep "Button"
```

### Example Test

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Button from './button'

describe('Button', () => {
  it('should call onClick when clicked', async () => {
    const handleClick = vi.fn()
    const user = userEvent.setup()

    render(<Button onClick={handleClick}>Click me</Button>)
    await user.click(screen.getByRole('button'))

    expect(handleClick).toHaveBeenCalledTimes(1)
  })
})
```

## Test Organization

### Test Types

1. **Unit Tests** (`tests/unit/`)
   - Test individual functions/components in isolation
   - Fast execution (< 100ms per test)
   - All external APIs are mocked
   - No real API calls
   - Example: Testing plot generation, exception handling

2. **Integration Tests** (`tests/integration/`)
   - Test interaction between components
   - May use mocked external services
   - Example: Testing recommendation → plot generation flow

3. **Live Tests** (`tests/live/`)
   - Test with real API calls (Groq, OpenAI)
   - Require valid API credentials (GROQ_API_KEY, OPENAI_API_KEY)
   - Skipped by default with `pytest`
   - Run explicitly with `pytest -m live`
   - Example: Real end-to-end workflow with actual API calls

## Running Tests

### Local Development

```bash
# Python: Run fast unit tests during development
pytest tests/unit

# Python: Run all non-live tests
pytest -m "not live"

# Python: Watch mode (requires pytest-watch)
ptw tests/unit

# Frontend: Run in watch mode
cd web && npm test -- --watch
```

### Before Commit

```bash
# Run pre-commit hooks manually
pre-commit run --all-files

# Or just run tests
pytest -v
cd web && npm test
```

### CI/CD

Tests run automatically on:
- Every push to `main`, `dev`, or `feature/*` branches
- Every pull request to `main` or `dev`

See `.github/workflows/` for CI configuration.

## Writing Tests

### Python Test Template

```python
import pytest
from plotsense.module import function_to_test

class TestFeatureName:
    """Test suite for feature X."""

    @pytest.mark.unit
    def test_basic_functionality(self, sample_dataframe):
        """Test basic use case (runs by default)."""
        result = function_to_test(sample_dataframe)
        assert result is not None

    @pytest.mark.integration
    def test_integration_scenario(self, mock_groq_client):
        """Test integration with mocked external service."""
        # Test implementation
        pass

    @pytest.mark.live
    def test_real_api_call(self, sample_dataframe):
        """Test with real API call (opt-in via pytest -m live)."""
        # This requires real API keys
        pass

    def test_error_handling(self):
        """Test that errors are handled properly."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

**Key Points:**
- By default, unit and integration tests run: `pytest`
- Live tests are skipped by default
- To run live tests: `pytest -m live` (requires API keys)
- To exclude live tests explicitly: `pytest -m "not live"`

### Frontend Test Template

```tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Component from './Component'

describe('Component', () => {
  it('should render correctly', () => {
    render(<Component />)
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('should handle user interaction', async () => {
    const user = userEvent.setup()
    render(<Component />)

    await user.click(screen.getByRole('button'))
    expect(screen.getByText('Updated')).toBeInTheDocument()
  })
})
```

## CI/CD Integration

### GitHub Actions Workflows

**Python Tests** (`.github/workflows/python-tests.yml`):
- Runs on: Ubuntu, Windows, macOS
- Python versions: 3.9, 3.10, 3.11, 3.12
- Includes: Linting, testing, coverage reporting

**Frontend Tests** (`.github/workflows/frontend-tests.yml`):
- Runs on: Ubuntu
- Node version: 20
- Includes: Linting, type checking, testing, build verification

### Required Secrets

Add these to GitHub repository secrets:
- `GROQ_API_KEY`: For testing API integrations (optional - tests use mocks)
- `CODECOV_TOKEN`: For coverage reporting (optional)

## Pre-commit Hooks

### Installation

```bash
pip install pre-commit
pre-commit install
```

### What Gets Checked

- **File checks**: Trailing whitespace, end-of-file, large files
- **Python**: flake8 linting, security checks with bandit
- **Frontend**: ESLint, TypeScript type checking
- **Tests**: Fast unit tests run before commit
- **Commit messages**: Conventional commit format

### Skip Hooks (Use Sparingly)

```bash
git commit --no-verify
```

## Coverage Reports

### Generate Coverage Reports

**Python:**
```bash
pytest --cov=plotsense --cov-report=html --cov-report=term
open htmlcov/index.html  # View HTML report
```

**Frontend:**
```bash
cd web
npm run test:coverage
open coverage/index.html  # View HTML report
```

### Coverage Goals

- **Minimum**: 70% line and branch coverage
- **Target**: 80%+ coverage for critical modules
- **Exclusions**: Test files, configuration, `__repr__` methods

## Best Practices

### General

1. **Write tests first** (TDD) when fixing bugs
2. **Keep tests focused**: One concept per test
3. **Use descriptive names**: `test_scatter_plot_with_missing_data`
4. **Avoid test interdependence**: Each test should run independently
5. **Mock external services**: Don't make real API calls in tests

### Python

1. **Use fixtures**: Leverage `conftest.py` fixtures
2. **Mark tests appropriately**: Use `@pytest.mark.*` decorators
3. **Test edge cases**: Empty data, NaN values, invalid inputs
4. **Clean up resources**: Close figures with `plt.close(fig)`
5. **Use parametrize**: Test multiple inputs efficiently

```python
@pytest.mark.parametrize("plot_type", ["scatter", "bar", "hist"])
def test_plot_types(plot_type, sample_dataframe):
    # Test implementation
    pass
```

### Frontend

1. **Query by role/label**: Prefer `getByRole`, `getByLabelText`
2. **Test user behavior**: Use `userEvent` for interactions
3. **Avoid implementation details**: Don't test internal state
4. **Test accessibility**: Ensure proper ARIA attributes
5. **Use data-testid sparingly**: Only when semantic queries fail

## Troubleshooting

### Common Issues

**Issue**: Tests fail with "No module named 'plotsense'"
```bash
# Solution: Install package in editable mode
pip install -e .
```

**Issue**: Frontend tests fail with "Cannot find module '@/...'"
```bash
# Solution: Check path alias in vite.config.ts and tsconfig.json
```

**Issue**: Matplotlib tests fail on CI
```bash
# Solution: Ensure non-interactive backend in test
matplotlib.use('Agg')
```

**Issue**: Tests are too slow
```bash
# Solution: Run fast tests only
pytest -m "not slow"
```

**Issue**: Coverage report not generated
```bash
# Solution: Install coverage tools
pip install pytest-cov
```

### Debug Mode

**Python:**
```bash
# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s
```

**Frontend:**
```bash
# Run with UI for debugging
npm run test:ui
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [Vitest documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/react)
- [Testing Best Practices](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)

## Contributing

When adding new features:

1. Write tests for new functionality
2. Place unit tests in `tests/unit/` with `@pytest.mark.unit` or no marker
3. Place integration tests in `tests/integration/` with `@pytest.mark.integration`
4. Place live API tests in `tests/live/` with `@pytest.mark.live`
5. Ensure all tests pass:
   - `pytest` - runs unit and integration tests
   - `pytest -m live` - runs live tests (requires API keys)
6. Check coverage: Aim for 80%+ on new code
   - `pytest --cov=plotsense --cov-report=html`
7. Run pre-commit hooks: `pre-commit run --all-files`
8. Update this document if adding new test patterns

---

For questions or issues with testing, please open an issue on GitHub.
