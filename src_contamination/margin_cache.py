# File: src_contamination/margin_cache.py
"""
Boundary margin cache management.

Provides fast lookup of pre-measured contamination margins.
Auto-measures if not cached.
"""

import json
import sys
from pathlib import Path
from typing import Optional

from .of_hash import config_hash, config_name
from .contamination import measure_boundary_contamination


# Default cache location
CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "boundary_margins.json"


def cache_key(config: dict, magnitude: float) -> str:
    """
    Generate cache key for a configuration.
    
    Format: {hash}_mag{magnitude}
    """
    h = config_hash(config)
    return f"{h}_mag{magnitude}"


def load_cache() -> dict:
    """
    Load cache from disk.
    
    Returns:
        Cache dict (empty if file doesn't exist)
    """
    if not CACHE_FILE.exists():
        return {}
    
    with open(CACHE_FILE, 'r') as f:
        return json.load(f)


def save_cache(cache: dict) -> None:
    """
    Save cache to disk.
    
    Args:
        cache: Cache dict to save
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def save_margin(
    config: dict,
    magnitude: float,
    results: dict
) -> None:
    """
    Save margin measurement to cache.
    
    Args:
        config: Algorithm config dict
        magnitude: Perturbation magnitude used
        results: Measurement results from measure_boundary_contamination
    """
    cache = load_cache()
    
    key = cache_key(config, magnitude)
    
    # Add metadata
    entry = {
        'margin': results['margin'],
        'sizes': results['sizes'],
        'shift': results['shift'],
        'threshold': results['threshold'],
        'config_name': config_name(config),
        'config_hash': config_hash(config),
        'algorithm': config.get('algorithm', 'unknown'),
        'config': config
    }
    
    cache[key] = entry
    save_cache(cache)


def get_margin(
    config: dict,
    magnitude: float,
    auto_measure: bool = True
) -> int:
    """
    Get boundary margin for a configuration.
    
    If not cached and auto_measure=True, measures and caches result.
    
    Args:
        config: Algorithm config dict (must include 'algorithm' key)
        magnitude: Perturbation magnitude
        auto_measure: If True, measure if not cached
        
    Returns:
        Margin in pixels
    """
    cache = load_cache()
    key = cache_key(config, magnitude)
    
    if key in cache:
        return cache[key]['margin']
    
    if not auto_measure:
        print(f"❌ No cached margin for {key} and auto_measure=False")
        sys.exit(1)
    
    # Measure and cache
    print(f"📏 Measuring boundary margin for {config_name(config)}...")
    results = measure_boundary_contamination(config, shift=magnitude)
    
    save_margin(config, magnitude, results)
    print(f"   ✅ Margin: {results['margin']} px")
    
    return results['margin']


def get_cached_margin(
    config: dict,
    magnitude: float
) -> Optional[int]:
    """
    Get cached margin without auto-measuring.
    
    Args:
        config: Algorithm config dict
        magnitude: Perturbation magnitude
        
    Returns:
        Margin in pixels, or None if not cached
    """
    cache = load_cache()
    key = cache_key(config, magnitude)
    
    if key in cache:
        return cache[key]['margin']
    return None


def list_cached_configs(algorithm: Optional[str] = None) -> list[dict]:
    """
    List all cached configurations.
    
    Args:
        algorithm: Filter by algorithm (None for all)
        
    Returns:
        List of cache entries
    """
    cache = load_cache()
    
    entries = []
    for key, entry in cache.items():
        if algorithm is None or entry.get('algorithm') == algorithm:
            entries.append({
                'key': key,
                **entry
            })
    
    return entries


def clear_cache() -> None:
    """Remove all cached margins."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print("🗑️  Cache cleared")


if __name__ == "__main__":
    print("📦 Margin cache demo")
    print("=" * 40)
    
    # Example config (user's key names)
    config = {
        'algorithm': 'farneback',
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.1,
        'flags': 0
    }
    
    # Get margin (will measure if not cached)
    margin = get_margin(config, magnitude=1.0)
    print(f"\nMargin for {config_name(config)}: {margin} px")
    
    # List cached
    print("\nCached configurations:")
    for entry in list_cached_configs():
        print(f"  {entry['config_name']}: {entry['margin']} px")
