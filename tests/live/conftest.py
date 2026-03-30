"""
Live test specific fixtures and configuration.

Live tests require real API keys and make actual API calls.
These tests are marked with @pytest.mark.live and can be skipped with:
    pytest -m "not live"
"""

import pytest
import os
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def groq_api_key():
    """Get Groq API key from environment."""
    key = os.getenv('GROQ_API_KEY')
    if not key:
        pytest.skip("GROQ_API_KEY not set in environment")
    return key


@pytest.fixture(scope="session")
def openai_api_key():
    """Get OpenAI API key from environment."""
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        pytest.skip("OPENAI_API_KEY not set in environment")
    return key
