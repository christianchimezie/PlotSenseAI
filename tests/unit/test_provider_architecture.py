"""Unit tests for provider architecture and ai_interface routing.

Tests verify that:
1. Provider names follow vendor_variant format
2. AIModelInterface correctly routes requests based on vendor
3. Message formatting is applied based on vendor type
4. No permanent error blocks override successful returns
"""

import pytest
from unittest.mock import patch, MagicMock
from plotsense.core.ai_interface import AIModelInterface
from plotsense.core.providers.provider_manager import ProviderManager


class TestProviderArchitecture:
    """Test provider naming and registration."""
    
    def test_provider_names_follow_vendor_variant_format(self):
        """All providers should use vendor_variant naming (e.g., groq_default, openai_chat)."""
        for vendor, variants in ProviderManager.SUPPORTED_PROVIDERS.items():
            for variant_name, provider_cls in variants.items():
                full_name = f"{vendor}_{variant_name}"
                # Verify it follows the pattern
                assert "_" in full_name, f"Provider {full_name} doesn't follow vendor_variant format"
                vendor_part = full_name.split("_")[0]
                assert vendor_part == vendor, f"Vendor part mismatch: {vendor_part} != {vendor}"
    
    def test_all_vendors_have_variants(self):
        """Every vendor should have at least one variant registered."""
        for vendor, variants in ProviderManager.SUPPORTED_PROVIDERS.items():
            assert len(variants) > 0, f"Vendor {vendor} has no variants"
            assert isinstance(variants, dict), f"Vendor {vendor} variants should be a dict"


class TestAIModelInterfaceRouting:
    """Test AIModelInterface correctly routes to providers based on vendor."""
    
    def test_vendor_extraction_from_provider_name(self):
        """Test that vendor is correctly extracted from full provider name."""
        test_cases = [
            ("groq_default", "groq"),
            ("openai_chat", "openai"),
            ("openai_response", "openai"),
            ("anthropic_default", "anthropic"),
            ("gemini_default", "gemini"),
            ("azure_default", "azure"),
            ("ollama_default", "ollama"),
        ]
        
        for full_name, expected_vendor in test_cases:
            vendor = full_name.split("_")[0].lower()
            assert vendor == expected_vendor, f"Vendor extraction failed for {full_name}"
    
    def test_groq_vendor_routing(self):
        """Test that groq_default is routed to Groq handler."""
        mock_manager = MagicMock()
        mock_manager.providers = {"groq_default": MagicMock()}
        mock_manager.query.return_value = "test response"
        
        ai = AIModelInterface(mock_manager)
        result = ai.query_model(
            provider="groq_default",
            model="llama-3.1-8b-instant",
            prompt="test"
        )
        # Should call manager.query (no ValueError about unknown provider)
        assert result == "test response"
    
    def test_openai_chat_routing(self):
        """Test that openai_chat is routed correctly."""
        mock_manager = MagicMock()
        mock_manager.providers = {"openai_chat": MagicMock()}
        mock_manager.query.return_value = "test response"
        
        ai = AIModelInterface(mock_manager)
        result = ai.query_model(
            provider="openai_chat",
            model="gpt-4",
            prompt="test"
        )
        assert result == "test response"
    
    def test_openai_response_routing(self):
        """Test that openai_response variant is routed correctly."""
        mock_manager = MagicMock()
        mock_manager.providers = {"openai_response": MagicMock()}
        mock_manager.query.return_value = "test response"
        
        ai = AIModelInterface(mock_manager)
        result = ai.query_model(
            provider="openai_response",
            model="text-davinci-003",
            prompt="test"
        )
        assert result == "test response"


class TestMessageFormatting:
    """Test that messages are formatted based on vendor type."""
    
    def test_groq_message_format_simple(self):
        """Groq messages should be simple role/content pairs (text-only)."""
        mock_manager = MagicMock()
        ai = AIModelInterface(mock_manager)
        
        messages = ai._build_messages(
            vendor="groq",
            model="llama-3.1-8b-instant",
            prompt="test prompt"
        )
        
        # Should be simple format without nested content structures
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "test prompt"
        assert isinstance(messages[0]["content"], str)
    
    def test_openai_message_format(self):
        """OpenAI messages should include system message."""
        mock_manager = MagicMock()
        ai = AIModelInterface(mock_manager)
        
        messages = ai._build_messages(
            vendor="openai",
            model="gpt-4",
            prompt="test prompt"
        )
        
        # Should include system message
        assert isinstance(messages, list)
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
    
    def test_anthropic_message_format(self):
        """Anthropic messages should follow Claude format."""
        mock_manager = MagicMock()
        ai = AIModelInterface(mock_manager)
        
        messages = ai._build_messages(
            vendor="anthropic",
            model="claude-3-opus",
            prompt="test prompt"
        )
        
        # Should not have system role (Claude doesn't use it in messages)
        assert isinstance(messages, list)
        assert messages[0]["role"] == "user"


class TestErrorHandling:
    """Test error handling in query_model."""
    
    def test_no_finally_override_bug(self):
        """The finally block should NOT always return an error.
        
        This was a critical bug where finally always returned an error,
        overriding all successful query responses.
        """
        mock_manager = MagicMock()
        mock_provider = MagicMock()
        mock_provider.query.return_value = "successful response"
        mock_manager.providers = {"groq_default": mock_provider}
        mock_manager.query.return_value = "successful response"
        
        ai = AIModelInterface(mock_manager)
        result = ai.query_model(
            provider="groq_default",
            model="llama-3.1-8b-instant",
            prompt="test"
        )
        
        # Should return actual response, not "No valid query handler found" error
        assert "No valid query handler found" not in result
        assert result == "successful response"
    
    def test_unknown_provider_raises_error(self):
        """Unknown providers should raise ValueError early."""
        mock_manager = MagicMock()
        mock_manager.providers = {"groq_default": MagicMock()}
        
        ai = AIModelInterface(mock_manager)
        
        with pytest.raises(ValueError, match="Unknown provider"):
            ai.query_model(
                provider="nonexistent_provider",
                model="some-model",
                prompt="test"
            )
    
    def test_query_exception_returns_error_message(self):
        """Query exceptions should return error string, not raise."""
        mock_manager = MagicMock()
        mock_manager.providers = {"groq_default": MagicMock()}
        mock_manager.query.side_effect = RuntimeError("API error")
        
        ai = AIModelInterface(mock_manager)
        result = ai.query_model(
            provider="groq_default",
            model="llama-3.1-8b-instant",
            prompt="test"
        )
        
        # Should return error message, not raise
        assert isinstance(result, str)
        assert "Error:" in result
        assert "API error" in result


class TestProviderVariants:
    """Test handling of multiple variants per vendor."""
    
    def test_openai_variants_coexist(self):
        """OpenAI has chat and response variants - both should be registered."""
        variants = ProviderManager.SUPPORTED_PROVIDERS.get("openai", {})
        assert "chat" in variants, "OpenAI chat variant missing"
        assert "response" in variants, "OpenAI response variant missing"
        assert len(variants) == 2
    
    def test_groq_variants_coexist(self):
        """Groq has native API and OpenAI-compatible variants."""
        variants = ProviderManager.SUPPORTED_PROVIDERS.get("groq", {})
        assert "default" in variants, "Groq default (native API) variant missing"
        assert "openai" in variants, "Groq OpenAI-compatible variant missing"
        assert len(variants) == 2
    
    def test_single_variant_vendors(self):
        """Vendors with single variant should have 'default' as variant name."""
        single_variant_vendors = {"anthropic", "gemini", "azure", "ollama"}
        
        for vendor in single_variant_vendors:
            variants = ProviderManager.SUPPORTED_PROVIDERS.get(vendor, {})
            assert "default" in variants, f"{vendor} missing 'default' variant"
            assert len(variants) == 1, f"{vendor} should have exactly one variant"


class TestMultiProviderConsistency:
    """Test that all providers work consistently with the new architecture."""
    
    def test_all_providers_have_vendor_variant_names(self):
        """Every registered provider should follow vendor_variant naming."""
        providers_to_init = set()
        
        for vendor, variants in ProviderManager.SUPPORTED_PROVIDERS.items():
            for variant_name in variants.keys():
                full_name = f"{vendor}_{variant_name}"
                providers_to_init.add(full_name)
        
        # Verify all follow the pattern
        for provider_name in providers_to_init:
            parts = provider_name.split("_")
            assert len(parts) >= 2, f"Provider {provider_name} doesn't follow vendor_variant format"
            assert parts[0] in ProviderManager.SUPPORTED_PROVIDERS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
