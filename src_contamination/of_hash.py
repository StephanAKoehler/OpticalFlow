# File: src_contamination/of_hash.py
"""
Optical flow configuration hashing and naming utilities.
"""

import hashlib
import json


def config_hash(config: dict) -> str:
    """
    Compute deterministic hash for OF config.
    
    Args:
        config: Algorithm configuration dict
        
    Returns:
        12-character hash string
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def config_name(config: dict) -> str:
    """
    Generate human-readable name for OF config.
    
    Example: "farneback_win15_pyr3_iter5_poly5"
    
    Args:
        config: Algorithm configuration dict
        
    Returns:
        Human-readable config name
    """
    parts = []
    
    # Algorithm name first
    if 'algorithm' in config:
        parts.append(config['algorithm'])
    
    # Farneback parameters
    if 'winsize' in config:
        parts.append(f"win{config['winsize']}")
    if 'levels' in config:
        parts.append(f"pyr{config['levels']}")
    if 'iterations' in config:
        parts.append(f"iter{config['iterations']}")
    if 'poly_n' in config:
        parts.append(f"poly{config['poly_n']}")
    if 'poly_sigma' in config:
        parts.append(f"sig{config['poly_sigma']:.1f}")
    
    # DIS parameters
    if 'preset' in config:
        preset = config['preset']
        # Handle both integer and string presets
        if isinstance(preset, int):
            preset_names = {0: 'ultra', 1: 'medium', 2: 'fast'}
            parts.append(preset_names.get(preset, f"preset{preset}"))
        else:
            parts.append(f"{preset.lower()}")
    if 'finest_scale' in config:
        parts.append(f"fine{config['finest_scale']}")
    if 'patch_size' in config:
        parts.append(f"patch{config['patch_size']}")
    if 'patch_stride' in config:
        parts.append(f"stride{config['patch_stride']}")
    
    if not parts:
        # Fallback to hash if no recognized params
        return config_hash(config)
    
    return "_".join(parts)


if __name__ == "__main__":
    # Example usage
    test_config = {
        "algorithm": "farneback",
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 5,
        "poly_n": 5,
        "poly_sigma": 1.1
    }
    
    print(f"Config name: {config_name(test_config)}")
    print(f"Config hash: {config_hash(test_config)}")
    
    # Test DIS with integer preset
    dis_config = {
        "algorithm": "dis",
        "preset": 1,
        "finest_scale": 0,
        "patch_size": 8,
        "patch_stride": 4
    }
    print(f"DIS config name: {config_name(dis_config)}")
