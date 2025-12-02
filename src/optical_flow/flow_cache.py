#!/usr/bin/env python3
# File: src/optical_flow/flow_cache.py
"""
In-memory FIFO cache for optical flow results.

Caches computed flows and metrics to avoid redundant computation when
exploring different ensemble weights or adding incremental configs.

Key features:
- FIFO eviction when max size reached
- O(1) lookup and insertion
- Session-scoped (no disk persistence)
- Simple and predictable
"""

import hashlib
import json
import numpy as np
from collections import deque
from typing import Optional, Dict, Any


class FlowCache:
    """
    In-memory FIFO cache for optical flow computation results.
    
    Cache key is based on algorithm config only (not images), since
    each pipeline run processes a single image pair.
    
    Attributes:
        max_size: Maximum number of configs to cache
        _cache: Dict mapping cache key → results
        _order: Deque tracking insertion order for FIFO
        stats: Dict of cache statistics (hits, misses, evictions)
    """
    
    def __init__(self, max_size: int = 1000, verbose: bool = False):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of configs to cache (default 1000)
            verbose: Print detailed cache operations (default False)
        """
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        
        self.max_size = max_size
        self.verbose = verbose
        
        self._cache: Dict[str, Any] = {}
        self._order = deque()  # Keys in insertion order
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'inserts': 0,
        }
    
    def make_key(self, config: dict, frame1: np.ndarray = None, frame2: np.ndarray = None) -> str:
        """
        Generate cache key from config and optionally image pair.
        
        Key includes:
        - Image hash (if frames provided) - distinguishes different test cases
        - Algorithm type
        - All algorithm parameters  
        - Perturbation configuration
        
        Args:
            config: Algorithm configuration dict
            frame1: First frame (optional, for multi-image sessions)
            frame2: Second frame (optional, for multi-image sessions)
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # If images provided, hash them
        if frame1 is not None and frame2 is not None:
            # Convert to bytes for hashing
            if frame1.dtype == np.float32:
                frame1_bytes = (frame1 * 255).astype(np.uint8).tobytes()
                frame2_bytes = (frame2 * 255).astype(np.uint8).tobytes()
            else:
                frame1_bytes = frame1.tobytes()
                frame2_bytes = frame2.tobytes()
            
            # Fast hash of image data
            h = hashlib.sha256()
            h.update(frame1_bytes)
            h.update(frame2_bytes)
            image_hash = h.hexdigest()[:16]
        else:
            image_hash = "noimage"
        
        # Hash config
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
        # Combined key
        key = f"{image_hash}_{config_hash}"
        
        if self.verbose:
            print(f"   Cache key: {key} for {config.get('algorithm', 'unknown')}")
        
        return key
    
    def has(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key is cached
        """
        return key in self._cache
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached results.
        
        Args:
            key: Cache key
            
        Returns:
            Cached results or None if not found
        """
        if key in self._cache:
            self.stats['hits'] += 1
            if self.verbose:
                print(f"   Cache HIT: {key}")
            return self._cache[key]
        else:
            self.stats['misses'] += 1
            if self.verbose:
                print(f"   Cache MISS: {key}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store results in cache with FIFO eviction.
        
        Args:
            key: Cache key
            value: Results to cache
        """
        # If key already exists, update without changing order
        if key in self._cache:
            self._cache[key] = value
            if self.verbose:
                print(f"   Cache UPDATE: {key}")
            return
        
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = self._order.popleft()
            del self._cache[oldest_key]
            self.stats['evictions'] += 1
            
            if self.verbose:
                print(f"   ⚠️  Cache EVICTED: {oldest_key} (size={self.max_size})")
        
        # Add new entry
        self._cache[key] = value
        self._order.append(key)
        self.stats['inserts'] += 1
        
        if self.verbose:
            print(f"   Cache INSERT: {key} (size={len(self._cache)}/{self.max_size})")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._order.clear()
        
        # Reset stats except evictions (historical)
        self.stats['hits'] = 0
        self.stats['misses'] = 0
        self.stats['inserts'] = 0
        
        if self.verbose:
            print("   Cache CLEARED")
    
    def size(self) -> int:
        """Return current number of cached entries."""
        return len(self._cache)
    
    def memory_estimate_mb(self) -> float:
        """
        Estimate memory usage in MB.
        
        Rough estimate: ~15 MB per config
        (flows + metrics at native resolution)
        
        Returns:
            Estimated memory usage in MB
        """
        return len(self._cache) * 15.0
    
    def report(self) -> None:
        """Print cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = 100 * self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        print(f"\n{'='*80}")
        print("CACHE STATISTICS")
        print(f"{'='*80}")
        print(f"Size: {len(self._cache)} / {self.max_size} configs")
        print(f"Memory (est): {self.memory_estimate_mb():.1f} MB")
        print(f"\nOperations:")
        print(f"  Hits:      {self.stats['hits']:6d} ({hit_rate:.1f}%)")
        print(f"  Misses:    {self.stats['misses']:6d}")
        print(f"  Inserts:   {self.stats['inserts']:6d}")
        print(f"  Evictions: {self.stats['evictions']:6d}")
        print(f"{'='*80}\n")
    
    def __len__(self) -> int:
        """Return current cache size."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Support 'key in cache' syntax."""
        return key in self._cache


if __name__ == "__main__":
    # Simple test
    print("Testing FlowCache...")
    
    cache = FlowCache(max_size=3, verbose=True)
    
    # Test insertions
    print("\n1. Insert configs 1-3:")
    for i in range(1, 4):
        config = {'algorithm': 'test', 'param': i}
        key = cache.make_key(config)
        cache.set(key, {'result': i})
    
    print(f"\nCache size: {len(cache)}")
    
    # Test eviction
    print("\n2. Insert config 4 (should evict config 1):")
    config4 = {'algorithm': 'test', 'param': 4}
    key4 = cache.make_key(config4)
    cache.set(key4, {'result': 4})
    
    # Test hits and misses
    print("\n3. Test lookups:")
    config1 = {'algorithm': 'test', 'param': 1}
    key1 = cache.make_key(config1)
    result1 = cache.get(key1)
    print(f"   Config 1: {'HIT' if result1 else 'MISS'} (expected MISS)")
    
    config2 = {'algorithm': 'test', 'param': 2}
    key2 = cache.make_key(config2)
    result2 = cache.get(key2)
    print(f"   Config 2: {'HIT' if result2 else 'MISS'} (expected HIT)")
    
    # Report
    cache.report()
    
    print("✅ FlowCache test complete")
