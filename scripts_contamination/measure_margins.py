# File: scripts_contamination/measure_margins.py
"""
CLI entry point for boundary contamination measurement.

Usage:
    python scripts_contamination/measure_margins.py config.toml
    python scripts_contamination/measure_margins.py config.toml --plot
    python scripts_contamination/measure_margins.py --list
"""

import sys
import argparse
import tomllib
from pathlib import Path
from itertools import product

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src_contamination import (
    measure_boundary_contamination,
    get_margin,
    save_margin,
    load_cache,
    list_cached_configs,
    config_hash,
    config_name,
)
from src_contamination.visualization import (
    plot_single_config,
    plot_config_comparison,
    plot_error_profile,
)


def parse_config(config_path: Path) -> dict:
    """
    Parse TOML config file.
    
    Expected structure:
    [algorithm]
    name = "farneback"
    
    [parameters]
    pyr_scale = 0.5
    levels = [3, 4]
    winsize = [5, 9, 15, 21]
    iterations = [3, 5]
    poly_n = [5, 7]
    poly_sigma = [1.1]
    flags = 0
    
    [perturbations]
    magnitude = 1.0
    
    [measurement]
    threshold = 0.01
    sizes = [100, 200, 400]
    """
    with open(config_path, 'rb') as f:
        return tomllib.load(f)


def expand_parameters(algorithm: str, params: dict) -> list[dict]:
    """
    Expand parameter lists into all combinations.
    
    Args:
        algorithm: Algorithm name to include in each config
        params: Dict with parameter names -> list of values
        
    Returns:
        List of config dicts, one per combination
    """
    # Ensure all values are lists
    param_lists = {}
    for key, value in params.items():
        if isinstance(value, list):
            param_lists[key] = value
        else:
            param_lists[key] = [value]
    
    # Generate all combinations
    keys = list(param_lists.keys())
    values = [param_lists[k] for k in keys]
    
    configs = []
    for combo in product(*values):
        config = {'algorithm': algorithm}
        config.update(dict(zip(keys, combo)))
        configs.append(config)
    
    return configs


def measure_all(
    configs: list[dict],
    magnitude: float,
    threshold: float,
    sizes: list[int],
    output_dir: Path,
    do_plot: bool = False
) -> list[dict]:
    """
    Measure boundary contamination for all configs.
    
    Args:
        configs: List of parameter configs (each includes 'algorithm')
        magnitude: Perturbation magnitude
        threshold: Error threshold
        sizes: Tile sizes to test
        output_dir: Results directory
        do_plot: Whether to generate plots
        
    Returns:
        List of result dicts
    """
    results = []
    n_configs = len(configs)
    
    print(f"\n🔬 Measuring {n_configs} configurations...")
    print("=" * 50)
    
    for i, config in enumerate(configs):
        name = config_name(config)
        h = config_hash(config)
        
        print(f"\n[{i+1}/{n_configs}] {name}")
        
        # Measure
        result = measure_boundary_contamination(
            config,
            shift=magnitude,
            threshold=threshold,
            sizes=sizes
        )
        
        # Print summary
        for size, depth in result['sizes'].items():
            print(f"    {size}px: {depth}px")
        print(f"    ➜ Margin: {result['margin']}px")
        
        # Save to cache
        save_margin(config, magnitude, result)
        
        # Store result
        result['config'] = config
        result['config_name'] = name
        result['config_hash'] = h
        results.append(result)
        
        # Generate plots if requested
        if do_plot:
            config_dir = output_dir / f"{config['algorithm']}_{h}"
            for size in sizes:
                plot_single_config(
                    config, size, magnitude,
                    output_dir=config_dir,
                    threshold=threshold
                )
            plot_error_profile(
                config, max(sizes), magnitude,
                output_path=config_dir / "error_profile.png",
                threshold=threshold
            )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure optical flow boundary contamination"
    )
    parser.add_argument(
        'config',
        nargs='?',
        type=Path,
        help='TOML configuration file'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List cached configurations'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Generate comparison plot of all configs'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('results_contamination'),
        help='Output directory (default: results_contamination)'
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("📦 Cached boundary margins:")
        print("=" * 50)
        entries = list_cached_configs()
        if not entries:
            print("  (no cached entries)")
        else:
            for entry in sorted(entries, key=lambda x: x['margin']):
                print(f"  {entry['config_name']:40s}  margin={entry['margin']:2d}px")
        return
    
    # Require config file for other modes
    if args.config is None:
        parser.error("Config file required (or use --list)")
    
    if not args.config.exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Parse config
    print(f"📄 Loading config: {args.config}")
    toml_config = parse_config(args.config)
    
    algorithm = toml_config['algorithm']['name']
    params = toml_config.get('parameters', {})
    magnitude = toml_config.get('perturbations', {}).get('magnitude', 1.0)
    threshold = toml_config.get('measurement', {}).get('threshold', 0.01)
    sizes = toml_config.get('measurement', {}).get('sizes', [100, 200, 400])
    
    # Expand parameters (includes algorithm in each config)
    configs = expand_parameters(algorithm, params)
    
    print(f"Algorithm:    {algorithm}")
    print(f"Configs:      {len(configs)}")
    print(f"Magnitude:    {magnitude}")
    print(f"Threshold:    {threshold}")
    print(f"Sizes:        {sizes}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Measure all
    results = measure_all(
        configs, magnitude, threshold, sizes,
        args.output, do_plot=args.plot
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary (sorted by margin):")
    print("=" * 50)
    
    for r in sorted(results, key=lambda x: x['margin']):
        print(f"  {r['config_name']:40s}  margin={r['margin']:2d}px")
    
    # Comparison plot
    if args.compare and len(configs) > 1:
        print("\n📈 Generating comparison plot...")
        plot_config_comparison(
            configs, magnitude,
            output_path=args.output / f"{algorithm}_comparison.png",
            sizes=sizes
        )
        print(f"✅ Saved: {args.output}/{algorithm}_comparison.png")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
