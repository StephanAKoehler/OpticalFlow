# File: src_contamination/contamination.py
"""
Boundary contamination measurement logic.

Measures how far optical flow boundary artifacts penetrate into valid regions
by using test patterns where border tiles have zero ground truth flow.
"""

import numpy as np
import sys
from pathlib import Path

from .test_pattern import create_test_frames

# Import from user's optical flow module
from src.optical_flow.algorithms import compute_optical_flow


def measure_contamination(
    config: dict,
    size: int,
    shift: float = 1.0,
    threshold: float = 0.01
) -> tuple[int, np.ndarray]:
    """
    Measure boundary contamination depth for a single tile size.
    
    The test creates a 3x3 tiled image where only the center tile has motion.
    Border tiles are static. Any flow detected at the boundary between center
    and border is algorithmic contamination.
    
    We measure from the CENTER TILE boundary inward to find where flow becomes clean.
    
    Args:
        config: Algorithm config dict (must include 'algorithm' key)
        size: Center tile size in pixels
        shift: Applied motion in x direction
        threshold: Error threshold for "clean" (pixels)
        
    Returns:
        (depth, error_map) where:
            depth: Contamination depth in pixels from center tile edge
            error_map: Error field for center tile only (size x size)
    """
    if 'algorithm' not in config:
        print(f"❌ ERROR: config must have 'algorithm' key")
        sys.exit(1)
    
    # Create test frames
    frame1, frame2 = create_test_frames(size, shift)
    
    # Compute flow
    u, v = compute_optical_flow(frame1, frame2, config)
    
    # Extract center tile
    u_center = u[size:2*size, size:2*size]
    v_center = v[size:2*size, size:2*size]
    
    # Error from expected (shift, 0)
    error = np.sqrt((u_center - shift)**2 + v_center**2)
    
    # Find depth where interior becomes clean
    # Start from edge, move inward until max error < threshold
    for depth in range(1, size // 2):
        interior = error[depth:-depth, depth:-depth]
        if interior.size > 0 and np.max(interior) < threshold:
            return depth, error
    
    # Contamination extends to center - return max possible
    return size // 2, error


def measure_boundary_contamination(
    config: dict,
    shift: float = 1.0,
    threshold: float = 0.01,
    sizes: list[int] = [100, 200, 400]
) -> dict:
    """
    Measure boundary contamination across multiple tile sizes.
    
    Returns the maximum contamination depth across all sizes (conservative).
    
    Args:
        config: Algorithm config dict (must include 'algorithm' key)
        shift: Applied motion in x direction
        threshold: Error threshold for "clean"
        sizes: List of center tile sizes to test
        
    Returns:
        Dict with per-size depths and overall margin
    """
    if 'algorithm' not in config:
        print(f"❌ ERROR: config must have 'algorithm' key")
        sys.exit(1)
    
    results = {
        'algorithm': config['algorithm'],
        'shift': shift,
        'threshold': threshold,
        'sizes': {}
    }
    
    max_depth = 0
    for size in sizes:
        depth, _ = measure_contamination(config, size, shift, threshold)
        results['sizes'][str(size)] = depth
        max_depth = max(max_depth, depth)
    
    results['margin'] = max_depth
    
    return results


if __name__ == "__main__":
    print("🔬 Contamination measurement demo")
    print("=" * 40)
    
    # Default Farneback config (using user's key names)
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
    
    print(f"Config: {config}")
    print()
    
    # Measure at multiple sizes
    results = measure_boundary_contamination(config, shift=1.0)
    
    print("Results:")
    for size, depth in results['sizes'].items():
        print(f"  Size {size:>3}: {depth:>2} px contamination")
    print(f"\n  ➜ Margin: {results['margin']} px (max across sizes)")
