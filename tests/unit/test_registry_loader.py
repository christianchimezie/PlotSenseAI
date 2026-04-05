"""Tests for the registry metadata loader system."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from plotsense.core.registry_loader import RegistryLoader, get_registry_loader


class TestRegistryLoaderBundled:
    """Tests for bundled registry loading (always available)."""

    def test_bundled_registry_loads(self):
        """Bundled registry should load successfully without network."""
        loader = RegistryLoader()
        registry = loader.registry

        assert registry is not None
        assert "providers" in registry
        assert "modelMetadata" in registry
        assert loader.loaded_from == "bundled"

    def test_bundled_registry_has_required_structure(self):
        """Bundled registry must have valid structure."""
        loader = RegistryLoader()
        registry = loader.registry

        # Check providers
        providers = registry.get("providers", {})
        assert isinstance(providers, dict)
        assert len(providers) > 0

        # Check each provider has variants
        for vendor, vendor_data in providers.items():
            assert "variants" in vendor_data
            variants = vendor_data["variants"]
            assert isinstance(variants, dict)
            assert len(variants) > 0

            # Check each variant has models
            for variant_name, variant_data in variants.items():
                assert "models" in variant_data
                assert isinstance(variant_data["models"], list)
                assert len(variant_data["models"]) > 0

    def test_bundled_registry_has_metadata(self):
        """Bundled registry must have cost and performance metadata."""
        loader = RegistryLoader()
        registry = loader.registry

        metadata = registry.get("modelMetadata", {})
        assert "costs" in metadata
        assert "performance" in metadata

        costs = metadata["costs"]
        performance = metadata["performance"]

        assert isinstance(costs, dict)
        assert isinstance(performance, dict)
        assert len(costs) > 0
        assert len(performance) > 0


class TestRegistryLoaderValidation:
    """Tests for registry validation logic."""

    def test_validate_registry_rejects_invalid_structure(self):
        """Should reject registries with missing required fields."""
        loader = RegistryLoader()

        # Missing providers
        assert not loader._validate_registry({"modelMetadata": {}})

        # Missing modelMetadata
        assert not loader._validate_registry({"providers": {}})

        # Missing variants in provider
        assert not loader._validate_registry({
            "providers": {"groq": {}},
            "modelMetadata": {"costs": {}, "performance": {}}
        })

        # Missing models in variant
        assert not loader._validate_registry({
            "providers": {"groq": {"variants": {"default": {}}}},
            "modelMetadata": {"costs": {}, "performance": {}}
        })

    def test_validate_registry_accepts_valid_structure(self):
        """Should accept valid registry structures."""
        loader = RegistryLoader()

        valid_registry = {
            "providers": {
                "groq": {
                    "variants": {
                        "default": {"models": ["model1", "model2"]}
                    }
                }
            },
            "modelMetadata": {
                "costs": {"model1": 0.1},
                "performance": {"model1": 10.0}
            }
        }

        assert loader._validate_registry(valid_registry)


class TestRegistryLoaderCache:
    """Tests for cache loading/saving."""

    def test_cache_is_not_used_when_missing(self):
        """When cache doesn't exist, should load from bundled."""
        with patch("plotsense.core.registry_loader.CACHE_FILE") as mock_cache:
            mock_cache.exists.return_value = False

            loader = RegistryLoader()
            registry = loader.registry

            assert registry is not None
            assert loader.loaded_from == "bundled"

    @patch("plotsense.core.registry_loader.CACHE_EXPIRY_HOURS", 24)
    def test_cache_is_used_when_fresh(self):
        """Fresh cache should be used instead of remote/bundled."""
        cache_data = {
            "providers": {
                "test": {
                    "variants": {
                        "default": {"models": ["test-model"]}
                    }
                }
            },
            "modelMetadata": {
                "costs": {"test-model": 0.1},
                "performance": {"test-model": 8.0}
            }
        }

        with patch("plotsense.core.registry_loader.CACHE_FILE") as mock_cache:
            mock_cache.exists.return_value = True
            mock_cache.stat.return_value.st_mtime = 9999999999  # Recent time

            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(cache_data)

                # Mock time.time() to return a recent time
                with patch("plotsense.core.registry_loader.time.time", return_value=10000000000):
                    loader = RegistryLoader()
                    registry = loader.registry

                    assert registry is not None
                    assert "test" in registry.get("providers", {})

    def test_save_to_cache_creates_directory(self):
        """Saving cache should create directory if needed."""
        loader = RegistryLoader()
        test_data = {
            "providers": {},
            "modelMetadata": {"costs": {}, "performance": {}}
        }

        with patch("plotsense.core.registry_loader.CACHE_DIR") as mock_dir:
            loader._save_to_cache(test_data)
            mock_dir.mkdir.assert_called_once()


class TestRegistryLoaderRemote:
    """Tests for remote registry fetching."""

    def test_remote_fetch_falls_back_on_failure(self):
        """Failed remote fetch should fall back to bundled."""
        with patch("plotsense.core.registry_loader.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Network error")

            loader = RegistryLoader()
            registry = loader.registry

            assert registry is not None
            # Should have loaded from bundled since remote failed
            assert loader.loaded_from == "bundled"

    def test_remote_fetch_validates_response(self):
        """Remote response should be validated before use."""
        invalid_data = {"invalid": "structure"}

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(invalid_data).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_response

            loader = RegistryLoader()
            _ = loader.registry

            # Should fall back to bundled since remote was invalid
            assert loader.loaded_from == "bundled"


class TestRegistryLoaderAccessors:
    """Tests for registry data accessor methods."""

    def test_get_provider_models(self):
        """Should return correct model list for provider/variant."""
        loader = RegistryLoader()

        # Groq default should have models
        models = loader.get_provider_models("groq", "default")
        assert isinstance(models, list)
        assert len(models) > 0
        assert "llama-3.1-8b-instant" in models

    def test_get_provider_models_nonexistent(self):
        """Should return empty list for nonexistent provider."""
        loader = RegistryLoader()

        models = loader.get_provider_models("nonexistent", "variant")
        assert models == []

    def test_get_all_provider_models(self):
        """Should return all models by vendor_variant name."""
        loader = RegistryLoader()

        all_models = loader.get_all_provider_models()
        assert isinstance(all_models, dict)
        assert len(all_models) > 0

        # Should have vendor_variant keys
        assert any("_" in key for key in all_models.keys())

        # Each should map to a list
        for key, models in all_models.items():
            assert isinstance(models, list)

    def test_get_model_costs(self):
        """Should return model cost map."""
        loader = RegistryLoader()

        costs = loader.get_model_costs()
        assert isinstance(costs, dict)
        assert len(costs) > 0

        # Values should be floats
        for model, cost in costs.items():
            assert isinstance(cost, (int, float))

    def test_get_model_performance(self):
        """Should return model performance map."""
        loader = RegistryLoader()

        performance = loader.get_model_performance()
        assert isinstance(performance, dict)
        assert len(performance) > 0

        # Values should be floats
        for model, score in performance.items():
            assert isinstance(score, (int, float))

    def test_get_provider_display_name(self):
        """Should return display name for provider variant."""
        loader = RegistryLoader()

        name = loader.get_provider_display_name("groq", "default")
        assert isinstance(name, str)
        assert len(name) > 0
        assert "Groq" in name or "groq" in name.lower()

    def test_get_provider_display_name_nonexistent(self):
        """Should return vendor_variant for nonexistent provider."""
        loader = RegistryLoader()

        name = loader.get_provider_display_name("nonexistent", "variant")
        assert name == "nonexistent_variant"


class TestRegistryLoaderSingleton:
    """Tests for global singleton instance."""

    def test_get_registry_loader_singleton(self):
        """Should return same instance on multiple calls."""
        loader1 = get_registry_loader()
        loader2 = get_registry_loader()

        assert loader1 is loader2

    def test_singleton_registry_cached(self):
        """Singleton should cache loaded registry."""
        loader = get_registry_loader()
        registry1 = loader.registry
        registry2 = loader.registry

        # Should be same object
        assert registry1 is registry2


class TestRegistryLoaderIntegration:
    """Integration tests for registry loading."""

    def test_provider_models_are_accessible(self):
        """All providers in registry should have accessible models."""
        loader = RegistryLoader()
        registry = loader.registry

        for vendor, vendor_data in registry.get("providers", {}).items():
            for variant_name in vendor_data.get("variants", {}).keys():
                models = loader.get_provider_models(vendor, variant_name)
                assert len(models) > 0, f"No models for {vendor}_{variant_name}"

    def test_cost_performance_coverage(self):
        """Models in registry should ideally have cost/performance data."""
        loader = RegistryLoader()
        all_models = loader.get_all_provider_models()
        costs = loader.get_model_costs()
        _ = loader.get_model_performance()

        # Collect all unique models from all providers
        all_unique_models = set()
        for models in all_models.values():
            all_unique_models.update(models)

        # Check coverage (not all models need data, but most should)
        coverage = len(all_unique_models & set(costs.keys())) / len(all_unique_models)
        assert coverage > 0.5, "Less than 50% of models have cost data"
