# File: src/optical_flow/config_naming.py
"""
Configuration naming utilities for optical flow algorithms.

Generates human-readable names for algorithm configurations.
"""


def generate_config_name(config: dict) -> str:
    """
    Generate human-readable config name.

    Handles both Farneback and DIS algorithm parameters.

    Args:
        config: Algorithm configuration dict with 'algorithm' key

    Returns:
        Human-readable string identifying the configuration

    Examples:
        Farneback: "FB_win15_lev3_iter5_poly5_s1.1"
        DIS: "DIS_FAST_scale0_iter12_patch8"
    """
    algorithm = config.get('algorithm', 'unknown')
    parts = []

    if algorithm == 'farneback':
        parts.append("FB")

        # Core parameters in order of importance
        if 'winsize' in config:
            parts.append(f"win{config['winsize']}")
        if 'levels' in config:
            parts.append(f"lev{config['levels']}")
        if 'iterations' in config:
            parts.append(f"iter{config['iterations']}")
        if 'poly_n' in config:
            parts.append(f"poly{config['poly_n']}")
        if 'poly_sigma' in config:
            sigma_val = config['poly_sigma']
            # Format sigma nicely (remove unnecessary decimals)
            if sigma_val == int(sigma_val):
                parts.append(f"s{int(sigma_val)}")
            else:
                parts.append(f"s{sigma_val:.1f}")
        if 'pyr_scale' in config:
            parts.append(f"pyr{config['pyr_scale']:.2f}")

    elif algorithm == 'dis':
        parts.append("DIS")

        # Core parameters in order of importance
        if 'preset' in config:
            preset = config['preset']
            # Abbreviate preset for brevity
            preset_abbrev = {
                'ULTRAFAST': 'UF',
                'FAST': 'F',
                'MEDIUM': 'M',
                'ultrafast': 'UF',
                'fast': 'F',
                'medium': 'M'
            }
            parts.append(preset_abbrev.get(preset, preset))

        if 'finest_scale' in config:
            parts.append(f"sc{config['finest_scale']}")
        if 'iterations' in config:
            parts.append(f"it{config['iterations']}")
        if 'patch_size' in config:
            parts.append(f"p{config['patch_size']}")
        if 'patch_stride' in config:
            parts.append(f"st{config['patch_stride']}")

    elif algorithm == 'dualtvl1':
        parts.append("TVL1")

        # Core parameters
        if 'scales_number' in config:
            parts.append(f"sc{config['scales_number']}")
        if 'outer_iterations' in config:
            parts.append(f"out{config['outer_iterations']}")
        if 'inner_iterations' in config:
            parts.append(f"in{config['inner_iterations']}")
        if 'lambda_' in config:
            parts.append(f"L{config['lambda_']:.2f}")

    elif algorithm == 'deepflow':
        parts.append("DEEP")
        # DeepFlow typically uses default parameters
        # No tunable parameters in OpenCV implementation

    elif algorithm == 'brox':
        parts.append("BROX")

        # Core parameters in order of importance
        if 'outer_iterations' in config:
            parts.append(f"out{config['outer_iterations']}")
        if 'inner_iterations' in config:
            parts.append(f"in{config['inner_iterations']}")
        if 'alpha' in config:
            alpha_val = config['alpha']
            parts.append(f"a{alpha_val:.3f}")
        if 'gamma' in config:
            gamma_val = config['gamma']
            if gamma_val == int(gamma_val):
                parts.append(f"g{int(gamma_val)}")
            else:
                parts.append(f"g{gamma_val:.1f}")
        if 'scale_factor' in config:
            parts.append(f"s{config['scale_factor']:.2f}")

    elif algorithm == 'lucas_kanade':
        parts.append("LK")
        # Add Lucas-Kanade specific parameters when implemented

    else:
        parts.append(algorithm.upper())
        # Fallback: add any numeric parameters
        for key, value in sorted(config.items()):
            if key != 'algorithm' and isinstance(value, (int, float)):
                parts.append(f"{key[:3]}{value}")

    return "_".join(parts) if parts else "default"


if __name__ == "__main__":
    # Test config naming
    print("🧪 Testing config naming...")

    # Farneback configs
    fb_configs = [
        {
            'algorithm': 'farneback',
            'winsize': 15,
            'levels': 3,
            'iterations': 5,
            'poly_n': 5,
            'poly_sigma': 1.1,
            'pyr_scale': 0.5
        },
        {
            'algorithm': 'farneback',
            'winsize': 21,
            'levels': 4,
            'iterations': 3,
            'poly_n': 7,
            'poly_sigma': 1.5
        }
    ]

    print("\nFarneback configs:")
    for cfg in fb_configs:
        name = generate_config_name(cfg)
        print(f"  {name}")

    # DIS configs
    dis_configs = [
        {
            'algorithm': 'dis',
            'preset': 'FAST',
            'finest_scale': 0,
            'iterations': 12,
            'patch_size': 8,
            'patch_stride': 4
        },
        {
            'algorithm': 'dis',
            'preset': 'MEDIUM',
            'finest_scale': 1,
            'iterations': 16,
            'patch_size': 12,
            'patch_stride': 6
        },
        {
            'algorithm': 'dis',
            'preset': 'ULTRAFAST',
            'finest_scale': 2
        }
    ]

    print("\nDIS configs:")
    for cfg in dis_configs:
        name = generate_config_name(cfg)
        print(f"  {name}")

    # Brox configs
    brox_configs = [
        {
            'algorithm': 'brox',
            'alpha': 0.197,
            'gamma': 50.0,
            'outer_iterations': 150,
            'inner_iterations': 5,
            'scale_factor': 0.8
        },
        {
            'algorithm': 'brox',
            'alpha': 0.3,
            'gamma': 100.0,
            'outer_iterations': 100,
            'inner_iterations': 10
        }
    ]

    print("\nBrox configs:")
    for cfg in brox_configs:
        name = generate_config_name(cfg)
        print(f"  {name}")

    # Edge cases
    print("\nEdge cases:")
    print(f"  Empty: {generate_config_name({})}")
    print(f"  Unknown algo: {generate_config_name({'algorithm': 'mystery', 'param1': 42})}")

    print("\n✨ All naming tests passed!")