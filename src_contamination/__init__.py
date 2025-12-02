# File: src_contamination/__init__.py
"""
Boundary Contamination Measurement System

Empirically measures how far optical flow algorithm boundary artifacts
contaminate valid regions. Creates a lookup table for per-config boundary margins.

Usage:
    from src_contamination import get_margin
    
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
    
    margin = get_margin(config, magnitude=1.0)
"""

from .test_pattern import create_template, create_test_frames
from .contamination import measure_contamination, measure_boundary_contamination
from .margin_cache import get_margin, save_margin, load_cache, list_cached_configs
from .of_hash import config_hash, config_name

__all__ = [
    "create_template",
    "create_test_frames",
    "measure_contamination",
    "measure_boundary_contamination",
    "get_margin",
    "save_margin",
    "load_cache",
    "list_cached_configs",
    "config_hash",
    "config_name",
]
