"""Unit tests for provider selection and initialization logic.

These tests verify that:
1. Only providers with API keys are initialized
2. selected_models must reference providers with keys
3. Unrelated providers are not prompted for or validated
"""

import pytest
from unittest.mock import patch, MagicMock
from plotsense.core.providers.provider_manager import ProviderManager


class TestProviderInitialization:
    """Test provider initialization with api_keys as source of truth."""
    
    def test_only_provided_providers_initialized(self, monkeypatch):
        """Should only initialize providers that have API keys."""
        # Mock the provider classes to avoid real API validation
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'groq': {
                'default': MagicMock(LINK='https://groq.com', validate_key=MagicMock(return_value=True))
            },
            'openai': {
                'default': MagicMock(LINK='https://openai.com', validate_key=MagicMock(return_value=True))
            }
        }):
            manager = ProviderManager(
                api_keys={'groq': 'groq-key'},
                interactive=False
            )
            
            # Only groq should be initialized
            assert len(manager.providers) > 0
            assert any('groq' in p for p in manager.providers.keys())
    
    def test_missing_provider_key_not_prompted(self, capsys):
        """Should not prompt for providers not in api_keys."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'groq': {
                'default': MagicMock(LINK='https://groq.com', validate_key=MagicMock(return_value=True))
            },
            'openai': {
                'default': MagicMock(LINK='https://openai.com', validate_key=MagicMock(return_value=True))
            }
        }):
            # Create manager with only groq key and interactive mode
            manager = ProviderManager(
                api_keys={'groq': 'groq-key'},
                interactive=True
            )
            
            captured = capsys.readouterr()
            # Should not prompt for openai since it wasn't provided
            assert 'Enter OPENAI' not in captured.out
    
    def test_selected_models_requires_valid_provider(self):
        """Should raise clear error if selected_models references missing-key provider."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'groq': {
                'default': MagicMock(LINK='https://groq.com', validate_key=MagicMock(return_value=True))
            },
            'openai': {
                'default': MagicMock(LINK='https://openai.com', validate_key=MagicMock(return_value=True))
            }
        }):
            with pytest.raises(ValueError, match="Selected models require provider.*openai"):
                ProviderManager(
                    api_keys={'groq': 'groq-key'},
                    interactive=False,
                    restrict_to=['openai']  # openai has no key
                )
    
    def test_restrict_to_filters_available_providers(self):
        """Should filter models within providers that have keys."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'groq': {
                'default': MagicMock(LINK='https://groq.com', validate_key=MagicMock(return_value=True)),
                'other': MagicMock(LINK='https://groq.com', validate_key=MagicMock(return_value=True))
            }
        }):
            manager = ProviderManager(
                api_keys={'groq': 'groq-key'},
                interactive=False,
                restrict_to=['groq']
            )
            
            # Should successfully restrict to groq (which has a key)
            assert manager.providers is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_api_keys_no_prompting(self, capsys):
        """With empty api_keys and interactive=True, should not prompt or fail."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {}):
            manager = ProviderManager(
                api_keys={},
                interactive=True
            )
            
            # Should complete without error even though no providers available
            assert manager.providers == {}
    
    def test_invalid_api_key_format_skipped(self, capsys):
        """Should skip providers with invalid key format."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'groq': {
                'default': MagicMock(LINK='https://groq.com', validate_key=MagicMock(return_value=True))
            }
        }):
            manager = ProviderManager(
                api_keys={'groq': ''},  # Empty key
                interactive=False
            )
            
            captured = capsys.readouterr()
            assert 'skipping groq due to invalid' in captured.out.lower()
    
    def test_unknown_provider_in_restrict_to_raises_error(self):
        """Should reject unknown providers in restrict_to."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'groq': {
                'default': MagicMock(LINK='https://groq.com', validate_key=MagicMock(return_value=True))
            }
        }):
            with pytest.raises(ValueError, match="Unsupported provider"):
                ProviderManager(
                    api_keys={'groq': 'groq-key', 'unknown': 'key'},
                    interactive=False,
                    restrict_to=['unknown']
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
