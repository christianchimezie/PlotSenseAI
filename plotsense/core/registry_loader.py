"""
Registry loader for managing provider and model metadata.

Provides a unified interface for loading provider/model metadata from:
1. Local cache (if fresh)
2. Remote GitHub registry (if available)
3. Bundled fallback (always available)

This allows updates to provider/model metadata without requiring a package release.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import urllib.request
import urllib.error


REMOTE_REGISTRY_URL = os.getenv(
    "PLOTSENSE_REGISTRY_URL",
    "https://raw.githubusercontent.com/dyung/PlotKit/main/plotsense/data/providers.json"
)
CACHE_DIR = Path.home() / ".cache" / "plotsense"
CACHE_FILE = CACHE_DIR / "providers.json"
CACHE_EXPIRY_HOURS = 24


class RegistryLoader:
    """
    Loads and manages provider/model metadata from multiple sources.
    
    Resolution order:
    1. Local cache (if fresh and valid)
    2. Remote GitHub registry (if fetch succeeds)
    3. Bundled package fallback (always available)
    """
    
    def __init__(self):
        self._registry: Optional[Dict[str, Any]] = None
        self._loaded_from: Optional[str] = None
    
    @property
    def registry(self) -> Dict[str, Any]:
        """Load and return the provider/model registry."""
        if self._registry is None:
            self._registry = self._load_registry()
        return self._registry
    
    @property
    def loaded_from(self) -> Optional[str]:
        """
        Indicates where the registry was loaded from.
        
        Returns:
            One of: "cache", "remote", "bundled", or None if not loaded yet
        """
        return self._loaded_from
    
    def _load_registry(self) -> Dict[str, Any]:
        """
        Load registry with fallback order:
        1. Try local cache (if fresh)
        2. Try remote fetch (if available)
        3. Fall back to bundled registry
        """
        # Try cache first
        cached = self._load_from_cache()
        if cached:
            self._loaded_from = "cache"
            return cached
        
        # Try remote
        remote = self._fetch_remote_registry()
        if remote:
            self._loaded_from = "remote"
            self._save_to_cache(remote)
            return remote
        
        # Fall back to bundled
        bundled = self._load_bundled_registry()
        self._loaded_from = "bundled"
        return bundled
    
    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """
        Load registry from local cache if it exists and is fresh.
        
        Returns:
            Registry dict if valid and fresh, None otherwise
        """
        if not CACHE_FILE.exists():
            return None
        
        try:
            # Check if cache is fresh
            file_mtime = CACHE_FILE.stat().st_mtime
            file_age_hours = (time.time() - file_mtime) / 3600
            
            if file_age_hours > CACHE_EXPIRY_HOURS:
                return None
            
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
            
            # Validate structure
            if self._validate_registry(data):
                return data
        except (IOError, json.JSONDecodeError, OSError):
            pass
        
        return None
    
    def _fetch_remote_registry(self) -> Optional[Dict[str, Any]]:
        """
        Fetch registry from remote GitHub URL.
        
        Returns:
            Registry dict if fetch succeeds and is valid, None otherwise
        """
        try:
            with urllib.request.urlopen(REMOTE_REGISTRY_URL, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
            
            # Validate before returning
            if self._validate_registry(data):
                return data
        except Exception:
            # Silently handle any fetch/validation errors
            pass
        
        return None
    
    def _load_bundled_registry(self) -> Dict[str, Any]:
        """
        Load registry bundled with the package.
        
        Returns:
            Registry dict from bundled JSON file
            
        Raises:
            FileNotFoundError: If bundled file cannot be found
            json.JSONDecodeError: If bundled file is invalid JSON
        """
        bundled_path = Path(__file__).parent.parent / "data" / "providers.json"
        
        with open(bundled_path, "r") as f:
            data = json.load(f)
        
        if not self._validate_registry(data):
            raise ValueError("Bundled registry is invalid")
        
        return data
    
    def _validate_registry(self, data: Any) -> bool:
        """
        Validate registry structure.
        
        Checks:
        - Top-level keys exist (providers, modelMetadata)
        - Providers structure is valid
        - Model metadata exists
        
        Args:
            data: Registry data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, dict):
            return False
        
        # Check top-level structure
        if "providers" not in data or "modelMetadata" not in data:
            return False
        
        providers = data.get("providers")
        metadata = data.get("modelMetadata")
        
        if not isinstance(providers, dict) or not isinstance(metadata, dict):
            return False
        
        # Check providers structure
        for vendor, vendor_data in providers.items():
            if not isinstance(vendor_data, dict) or "variants" not in vendor_data:
                return False
            
            variants = vendor_data.get("variants")
            if not isinstance(variants, dict):
                return False
            
            for variant_name, variant_data in variants.items():
                if not isinstance(variant_data, dict):
                    return False
                if "models" not in variant_data or not isinstance(variant_data["models"], list):
                    return False
        
        # Check metadata structure
        if "costs" not in metadata or "performance" not in metadata:
            return False
        
        costs = metadata.get("costs")
        performance = metadata.get("performance")
        
        if not isinstance(costs, dict) or not isinstance(performance, dict):
            return False
        
        return True
    
    def _save_to_cache(self, data: Dict[str, Any]) -> None:
        """
        Save registry to local cache.
        
        Args:
            data: Registry data to cache
        """
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except (IOError, OSError):
            # Silently fail - cache is optional
            pass
    
    def get_provider_models(self, vendor: str, variant: str = "default") -> list:
        """
        Get list of models for a provider variant.
        
        Args:
            vendor: Provider vendor name (e.g., "groq", "openai")
            variant: Provider variant (default: "default")
            
        Returns:
            List of model names, or empty list if not found
        """
        try:
            return self.registry["providers"][vendor]["variants"][variant]["models"]
        except KeyError:
            return []
    
    def get_all_provider_models(self) -> Dict[str, list]:
        """
        Get all available models by provider_variant name.
        
        Returns:
            Dict mapping "vendor_variant" to list of models
        """
        result = {}
        providers = self.registry.get("providers", {})
        
        for vendor, vendor_data in providers.items():
            variants = vendor_data.get("variants", {})
            for variant_name, variant_data in variants.items():
                key = f"{vendor}_{variant_name}"
                result[key] = variant_data.get("models", [])
        
        return result
    
    def get_model_costs(self) -> Dict[str, float]:
        """
        Get cost multipliers for all models.
        
        Returns:
            Dict mapping model name to cost multiplier
        """
        return self.registry.get("modelMetadata", {}).get("costs", {})
    
    def get_model_performance(self) -> Dict[str, float]:
        """
        Get performance scores for all models.
        
        Returns:
            Dict mapping model name to performance score
        """
        return self.registry.get("modelMetadata", {}).get("performance", {})
    
    def get_provider_display_name(self, vendor: str, variant: str = "default") -> str:
        """
        Get human-readable display name for a provider variant.
        
        Args:
            vendor: Provider vendor name
            variant: Provider variant (default: "default")
            
        Returns:
            Display name, or vendor_variant if not found
        """
        try:
            return self.registry["providers"][vendor]["variants"][variant]["displayName"]
        except KeyError:
            return f"{vendor}_{variant}"


# Global singleton instance
_loader: Optional[RegistryLoader] = None


def get_registry_loader() -> RegistryLoader:
    """
    Get or create the global registry loader instance.
    
    Returns:
        RegistryLoader instance
    """
    global _loader
    if _loader is None:
        _loader = RegistryLoader()
    return _loader
