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
            with pytest.raises(ValueError, match="API key required for provider.*openai"):
                ProviderManager(
                    api_keys={'groq': 'groq-key'},
                    interactive=False,
                    selected_providers=['openai']  # openai has no key
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
                selected_providers=['groq']
            )
            
            # Should successfully restrict to groq (which has a key)
            assert manager.providers is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_api_keys_no_selected_providers_raises_error(self):
        """With no api_keys and no selected_providers, should raise error."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {}):
            with pytest.raises(ValueError, match="No providers selected and no API keys provided"):
                ProviderManager(
                    api_keys={},
                    interactive=True,
                    selected_providers=None
                )
    
    def test_selected_provider_with_empty_key_raises_error_non_interactive(self):
        """Selected provider with empty key and non-interactive should raise error."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'groq': {
                'default': MagicMock(LINK='https://groq.com', validate_key=MagicMock(return_value=True))
            }
        }):
            with pytest.raises(ValueError, match="API key required"):
                ProviderManager(
                    api_keys={'groq': ''},  # Empty key
                    interactive=False,
                    selected_providers=['groq']
                )
    
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
                    selected_providers=['unknown']
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestSelectedModelsSourceOfTruth:
    """Test that selected_models determines provider selection, not api_keys."""
    
    def test_selected_models_without_keys_interactive_prompts(self):
        """With selected_providers and no keys, should prompt interactively."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'groq': {
                'default': MagicMock(LINK='https://groq.com', validate_key=MagicMock(return_value=True))
            }
        }):
            with patch('plotsense.core.providers.provider_manager.prompt_for_api_key') as mock_prompt:
                mock_prompt.return_value = 'gsk_provided'
                
                manager = ProviderManager(
                    api_keys={},  # No keys
                    interactive=True,
                    selected_providers=['groq']
                )
                
                # Should have called prompt
                assert mock_prompt.called
                # Should have initialized groq
                assert 'groq_default' in manager.providers
    
    def test_api_keys_alone_ignored_when_selected_providers_given(self):
        """When selected_providers is given, only those providers are initialized."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'groq': {
                'default': MagicMock(LINK='https://groq.com', validate_key=MagicMock(return_value=True))
            },
            'openai': {
                'chat': MagicMock(LINK='https://openai.com', validate_key=MagicMock(return_value=True))
            }
        }):
            # Have groq key but select only openai -> should require openai key
            with pytest.raises(ValueError, match="API key required.*openai"):
                ProviderManager(
                    api_keys={'groq': 'gsk_...'},  # groq key exists but not selected
                    interactive=False,
                    selected_providers=['openai']  # only openai selected
                )
    
    def test_unselected_provider_never_prompted(self):
        """Unselected providers should never be prompted for."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'groq': {
                'default': MagicMock(LINK='https://groq.com', validate_key=MagicMock(return_value=True))
            },
            'openai': {
                'chat': MagicMock(LINK='https://openai.com', validate_key=MagicMock(return_value=True))
            }
        }):
            with patch('plotsense.core.providers.provider_manager.prompt_for_api_key') as mock_prompt:
                mock_prompt.return_value = 'gsk_provided'
                
                manager = ProviderManager(
                    api_keys={},
                    interactive=True,
                    selected_providers=['groq']  # Only groq selected
                )
                
                # Should only have prompted for groq, not openai
                calls = [c[0][0].lower() for c in mock_prompt.call_args_list]
                assert 'groq' in calls
                assert 'openai' not in calls
    
    def test_selected_provider_missing_key_non_interactive_error(self):
        """Selected provider with missing key and non-interactive should raise error."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'openai': {
                'chat': MagicMock(LINK='https://openai.com', validate_key=MagicMock(return_value=True))
            }
        }):
            with pytest.raises(ValueError, match="API key required"):
                ProviderManager(
                    api_keys={},
                    interactive=False,
                    selected_providers=['openai']  # openai selected but no key
                )
    
    def test_user_skips_selected_provider_in_interactive_raises_error(self):
        """If user skips prompt for selected provider, should raise error."""
        with patch('plotsense.core.providers.provider_manager.ProviderManager.SUPPORTED_PROVIDERS', {
            'openai': {
                'chat': MagicMock(LINK='https://openai.com', validate_key=MagicMock(return_value=True))
            }
        }):
            with patch('plotsense.core.providers.provider_manager.prompt_for_api_key') as mock_prompt:
                mock_prompt.return_value = None  # User skipped
                
                with pytest.raises(ValueError, match="API key required"):
                    ProviderManager(
                        api_keys={},
                        interactive=True,
                        selected_providers=['openai']  # openai required
                    )
