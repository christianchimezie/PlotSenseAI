"""Tests for hidden API key input using getpass."""

import pytest
from unittest.mock import patch, MagicMock
from plotsense.core.utils import prompt_for_api_key


class TestHiddenAPIKeyInput:
    """Tests that API key input is hidden from terminal/logs."""
    
    def test_uses_getpass_not_input(self):
        """Verify that getpass.getpass is used instead of input()."""
        with patch('getpass.getpass', return_value='hidden_key') as mock_getpass:
            key = prompt_for_api_key('groq', 'https://example.com', interactive=True)
            
            # Verify getpass was called
            mock_getpass.assert_called_once()
            assert key == 'hidden_key'
    
    def test_getpass_masks_input(self):
        """Verify getpass hides input from terminal."""
        # getpass.getpass() does not echo input to terminal
        # This is verified by the fact that getpass module is imported
        # and used instead of builtins.input
        with patch('getpass.getpass', return_value='secret_api_key') as mock_getpass:
            key = prompt_for_api_key('openai', 'https://openai.com', interactive=True)
            
            # The getpass function is called with appropriate prompt
            call_args = mock_getpass.call_args[0][0]
            assert 'OPENAI' in call_args.upper()
            assert key == 'secret_api_key'
    
    def test_hidden_input_with_skip(self):
        """Hidden input should work with skip_if_missing."""
        with patch('getpass.getpass', return_value='') as mock_getpass:
            key = prompt_for_api_key(
                'anthropic', 
                'https://console.anthropic.com/keys',
                interactive=True,
                skip_if_missing=True
            )
            
            mock_getpass.assert_called_once()
            assert key is None
    
    def test_hidden_input_required_empty_raises(self):
        """Empty hidden input should raise when key is required."""
        with patch('getpass.getpass', return_value='') as mock_getpass:
            with pytest.raises(ValueError, match="API key is required"):
                prompt_for_api_key(
                    'anthropic',
                    'https://console.anthropic.com/keys',
                    interactive=True,
                    skip_if_missing=False
                )
            
            mock_getpass.assert_called_once()
    
    def test_hidden_input_strips_whitespace(self):
        """Hidden input should have whitespace stripped."""
        with patch('getpass.getpass', return_value='  api_key_with_spaces  ') as mock_getpass:
            key = prompt_for_api_key('groq', 'https://console.groq.com/keys', interactive=True)
            
            # Whitespace should be stripped
            assert key == 'api_key_with_spaces'
            assert not key.startswith(' ')
            assert not key.endswith(' ')
    
    def test_hidden_input_non_interactive_no_getpass(self):
        """Non-interactive mode should not call getpass."""
        with patch('getpass.getpass') as mock_getpass:
            with pytest.raises(ValueError):
                prompt_for_api_key('groq', 'https://console.groq.com/keys', interactive=False)
            
            # getpass should not be called in non-interactive mode
            mock_getpass.assert_not_called()
    
    def test_hidden_input_eof_error(self):
        """Handle EOF when getpass is used."""
        with patch('getpass.getpass', side_effect=EOFError):
            with pytest.raises(ValueError, match="API key is required"):
                prompt_for_api_key(
                    'gemini',
                    'https://aistudio.google.com/app/apikey',
                    interactive=True
                )
    
    def test_hidden_input_os_error(self):
        """Handle OSError when getpass is used."""
        with patch('getpass.getpass', side_effect=OSError):
            with pytest.raises(ValueError, match="API key is required"):
                prompt_for_api_key(
                    'azure',
                    'https://portal.azure.com',
                    interactive=True
                )
    
    def test_hidden_input_multiple_calls(self):
        """Multiple hidden inputs should each use getpass."""
        with patch('getpass.getpass', side_effect=['key1', 'key2', 'key3']):
            key1 = prompt_for_api_key('groq', 'https://example.com', interactive=True)
            key2 = prompt_for_api_key('openai', 'https://example.com', interactive=True)
            key3 = prompt_for_api_key('anthropic', 'https://example.com', interactive=True)
            
            assert key1 == 'key1'
            assert key2 == 'key2'
            assert key3 == 'key3'
    
    def test_hidden_input_with_special_characters(self):
        """Hidden input should handle special characters in API keys."""
        special_key = 'sk-key_with-special.chars+/=abc123'
        
        with patch('getpass.getpass', return_value=special_key):
            key = prompt_for_api_key('openai', 'https://platform.openai.com/api-keys', interactive=True)
            
            assert key == special_key
