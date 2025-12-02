#!/usr/bin/env python3
# File: scripts/evaluate_weights.py
"""
Evaluate fixed weights from config.toml on a dataset.

Tests transfer learning: weights trained elsewhere applied to this dataset.
Produces line plot comparing oracle vs 4 ensemble methods across frame pairs.

Usage:
    python scripts/evaluate_weights.py config.toml
"""

import sys
import json
import pickle
import hashlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tomli

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_loader import load_movie_sequence
from src.evaluation.ground_truth import compute_epe


# =============================================================================
# Auto-detection
# =============================================================================

def auto_detect_experiment(data_dir: Path) -> tuple[str, str]:
    """
    Auto-detect movie_hash and of_hash from data directory.
    
    Returns (movie_hash, of_hash) or exits with error.
    """
    if not data_dir.exists():
        print(f"❌ ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Find movie hashes (directories in data/)
    movie_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if len(movie_dirs) == 0:
        print(f"❌ ERROR: No experiments found in {data_dir}")
        sys.exit(1)
    elif len(movie_dirs) > 1:
        print(f"❌ ERROR: Multiple movie hashes found in {data_dir}:")
        for d in sorted(movie_dirs):
            print(f"   - {d.name}")
        print(f"\nSpecify which one by removing others or using --movie-hash")
        sys.exit(1)
    
    movie_hash = movie_dirs[0].name
    analysis_dir = movie_dirs[0] / 'analysis'
    
    if not analysis_dir.exists():
        print(f"❌ ERROR: No analysis directory in {movie_dirs[0]}")
        sys.exit(1)
    
    # Find of hashes (directories in analysis/)
    of_dirs = [d for d in analysis_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if len(of_dirs) == 0:
        print(f"❌ ERROR: No OF analysis found in {analysis_dir}")
        sys.exit(1)
    elif len(of_dirs) > 1:
        print(f"❌ ERROR: Multiple OF hashes found in {analysis_dir}:")
        for d in sorted(of_dirs):
            print(f"   - {d.name}")
        print(f"\nSpecify which one by removing others or using --of-hash")
        sys.exit(1)
    
    of_hash = of_dirs[0].name
    
    return movie_hash, of_hash


# =============================================================================
# Weight Hash
# =============================================================================

def compute_weight_hash(selection_configs: dict) -> str:
    """
    Compute 8-char hash from all selection configs.
    
    Args:
        selection_configs: Dict of {method_name: {param: value, ...}}
    
    Returns:
        8-character hex hash
    """
    # Sort for deterministic hashing
    sorted_config = {}
    for method in sorted(selection_configs.keys()):
        sorted_config[method] = dict(sorted(selection_configs[method].items()))
    
    config_str = json.dumps(sorted_config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


# =============================================================================
# Ensemble Selection
# =============================================================================

def apply_selection(results_full: list, 
                    method_config: dict,
                    valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply weighted selection to get ensemble flow.
    
    Args:
        results_full: List of config result dicts
        method_config: Selection config with normalize, aggregation, power, weights
        valid_mask: Boolean mask for valid pixels
    
    Returns:
        (u_ensemble, v_ensemble) flow fields
    """
    n_configs = len(results_full)
    H, W = valid_mask.shape
    
    # Extract method parameters
    normalize = method_config.get('normalize', 'none')
    aggregation = method_config.get('aggregation', 'sum')
    power = method_config.get('power', 2)
    
    # Metric names and weights
    metric_names = ['perturbation_rms', 'consistency', 'photometric']
    weights = {m: method_config.get(m, 0.0) for m in metric_names}
    
    # Map metric names to result keys (nested under 'metrics')
    metric_key_map = {
        'perturbation_rms': 'displacements_sensitivity_A2B',
        'consistency': 'consistency_A',
        'photometric': 'photometric_A'
    }
    
    # Build metric stacks
    metric_stacks = {}
    for metric_name in metric_names:
        result_key = metric_key_map[metric_name]
        stack = np.stack([r['metrics'][result_key] for r in results_full], axis=0)
        metric_stacks[metric_name] = stack
    
    # Normalize if MAD
    if normalize == 'mad':
        for metric_name, stack in metric_stacks.items():
            # Compute MAD per pixel across configs
            median = np.median(stack, axis=0, keepdims=True)
            mad = np.median(np.abs(stack - median), axis=0, keepdims=True)
            mad = np.maximum(mad, 1e-10)  # Avoid division by zero
            metric_stacks[metric_name] = (stack - median) / mad
    
    # Compute penalty per config per pixel
    penalty_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for metric_name, weight in weights.items():
        if weight > 0:
            stack = metric_stacks[metric_name]
            if aggregation == 'sum':
                penalty_stack += weight * np.power(np.abs(stack), power)
            else:  # max
                penalty_stack = np.maximum(penalty_stack, weight * np.power(np.abs(stack), power))
    
    # Select best config per pixel (lowest penalty)
    selection = np.argmin(penalty_stack, axis=0)
    
    # Build ensemble flow
    u_stack = np.stack([r['flows']['u_AB'] for r in results_full], axis=0)
    v_stack = np.stack([r['flows']['v_AB'] for r in results_full], axis=0)
    
    u_ensemble = np.zeros((H, W), dtype=np.float32)
    v_ensemble = np.zeros((H, W), dtype=np.float32)
    
    for i in range(n_configs):
        mask = selection == i
        u_ensemble[mask] = u_stack[i][mask]
        v_ensemble[mask] = v_stack[i][mask]
    
    return u_ensemble, v_ensemble


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_method(results_full: list,
                    method_config: dict,
                    u_truth: np.ndarray,
                    v_truth: np.ndarray,
                    valid_mask: np.ndarray,
                    epe_power: float) -> float:
    """
    Evaluate a single method on a single pair.
    
    Returns mean EPE^power over valid pixels.
    """
    u_ensemble, v_ensemble = apply_selection(results_full, method_config, valid_mask)
    
    epe_map = compute_epe(u_ensemble, v_ensemble, u_truth, v_truth, valid_mask, power=epe_power)
    
    return float(np.nanmean(epe_map))


# =============================================================================
# Figure Generation
# =============================================================================

def generate_figure(pair_results: dict,
                    selection_configs: dict,
                    weight_hash: str,
                    epe_power: float,
                    output_path: Path):
    """
    Generate line plot comparing oracle vs methods across pairs.
    
    Args:
        pair_results: {method_name: [epe_pair0, epe_pair1, ...], 'oracle': [...]}
        selection_configs: Config dict for annotation
        weight_hash: For filename
        epe_power: EPE power used
        output_path: Where to save
    """
    n_pairs = len(pair_results['oracle'])
    x = np.arange(n_pairs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors for methods
    colors = {
        'oracle': 'black',
        'mad_sum': 'tab:blue',
        'mad_max': 'tab:cyan', 
        'raw_sum': 'tab:orange',
        'raw_max': 'tab:red'
    }
    
    # Plot oracle as dashed
    ax.plot(x, pair_results['oracle'], 'k--', linewidth=2, marker='o', 
            markersize=8, label='Oracle', zorder=10)
    
    # Plot methods as solid
    for method in ['mad_sum', 'mad_max', 'raw_sum', 'raw_max']:
        if method in pair_results:
            ax.plot(x, pair_results[method], linewidth=2, marker='s',
                    markersize=6, label=method, color=colors[method])
    
    ax.set_xlabel('Pair Index', fontsize=12)
    ax.set_ylabel(f'EPE^{epe_power}', fontsize=12)
    ax.set_title('EPE per Frame Pair', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Build weight table
    methods = ['mad_sum', 'mad_max', 'raw_sum', 'raw_max']
    header = "Weights:    pert  cons  phot  trac"
    rows = [header]
    for method in methods:
        if method in selection_configs:
            cfg = selection_configs[method]
            p = cfg.get('perturbation_rms', 0)
            c = cfg.get('consistency', 0)
            ph = cfg.get('photometric', 0)
            t = cfg.get('traction', 0)
            rows.append(f"{method:11s} {p:4.2f}  {c:4.2f}  {ph:4.2f}  {t:4.2f}")
    
    weight_text = "\n".join(rows)
    
    ax.text(0.98, 0.98, weight_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def run_evaluation(config: dict, data_dir: Path = Path('data'), use_optimized: bool = False):
    """
    Run weight evaluation on detected experiment.
    
    Args:
        config: Full config dict with [selection.*] sections
        data_dir: Base data directory
        use_optimized: Load weights from optimization results
    
    Returns:
        Dict with evaluation results
    """
    print("=" * 60)
    print("📊 WEIGHT EVALUATION")
    print("=" * 60)
    
    # Auto-detect experiment
    movie_hash, of_hash = auto_detect_experiment(data_dir)
    
    # Paths
    movie_dir = data_dir / movie_hash
    analysis_dir = movie_dir / 'analysis' / of_hash
    sweep_dir = analysis_dir / 'sweep'
    figures_dir = analysis_dir / 'figures'
    optimization_dir = analysis_dir / 'optimization'
    figures_dir.mkdir(exist_ok=True)
    
    print(f"Experiment: {analysis_dir}")
    
    # Required methods
    required_methods = ['mad_sum', 'mad_max', 'raw_sum', 'raw_max']
    
    # Default weights
    default_weights = {
        'perturbation_rms': 1.0,
        'consistency': 1.0,
        'photometric': 1.0,
        'power': 2,
        'traction': 0.0
    }
    
    # Load weights based on priority
    selection_configs = {}
    weight_source = None
    
    if use_optimized:
        # Load from optimization results
        if not optimization_dir.exists():
            print(f"❌ ERROR: Optimization directory not found: {optimization_dir}")
            sys.exit(1)
        
        for method in required_methods:
            weights_path = optimization_dir / method / 'best_weights.json'
            if weights_path.exists():
                with open(weights_path, 'r') as f:
                    data = json.load(f)
                
                # Use best_selection_config which has all the params
                selection_configs[method] = data['best_selection_config']
            else:
                # Method wasn't optimized, use defaults
                norm, agg = method.split('_')
                selection_configs[method] = {
                    **default_weights,
                    'normalize': 'mad' if norm == 'mad' else 'none',
                    'aggregation': agg
                }
        
        weight_source = "optimized"
        print("   (using optimized weights)")
    
    else:
        # Check for config [selection.*] sections
        has_config_weights = any(
            f'selection.{m}' in config or 
            ('selection' in config and m in config.get('selection', {}))
            for m in required_methods
        )
        
        if has_config_weights:
            for method in required_methods:
                section_name = f'selection.{method}'
                if section_name in config:
                    selection_configs[method] = config[section_name]
                elif 'selection' in config and method in config['selection']:
                    selection_configs[method] = config['selection'][method]
                else:
                    # Method not in config, use defaults
                    norm, agg = method.split('_')
                    selection_configs[method] = {
                        **default_weights,
                        'normalize': 'mad' if norm == 'mad' else 'none',
                        'aggregation': agg
                    }
            weight_source = "config"
            print("   (using config weights)")
        else:
            # Use defaults
            for method in required_methods:
                norm, agg = method.split('_')
                selection_configs[method] = {
                    **default_weights,
                    'normalize': 'mad' if norm == 'mad' else 'none',
                    'aggregation': agg
                }
            weight_source = "default"
            print("   (using default weights)")
    
    # Compute weight hash
    weight_hash = compute_weight_hash(selection_configs)
    
    # Get EPE power from config
    eval_config = config.get('evaluation', {})
    epe_power = eval_config.get('epe_power', 2.0)
    
    # Load movie sequence for ground truth (silently)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        movie = load_movie_sequence(movie_dir)
    n_pairs = len(movie.pairs)
    
    print(f"\nEvaluating {n_pairs} pairs...")
    
    # Initialize results
    pair_results = {
        'oracle': [],
        'mad_sum': [],
        'mad_max': [],
        'raw_sum': [],
        'raw_max': []
    }
    
    # Process each pair
    for pair_idx in range(n_pairs):
        pair_dir = sweep_dir / f'pair_{pair_idx:03d}'
        
        # Load sweep results
        results_path = pair_dir / 'results_full.pkl'
        if not results_path.exists():
            print(f"❌ ERROR: Missing {results_path}")
            sys.exit(1)
        
        with open(results_path, 'rb') as f:
            results_full = pickle.load(f)
        
        # Load oracle
        oracle_path = pair_dir / 'oracle.npz'
        if not oracle_path.exists():
            print(f"❌ ERROR: Missing {oracle_path}")
            sys.exit(1)
        
        oracle_data = np.load(oracle_path)
        oracle_epe = float(oracle_data['oracle_epe_forward'])
        pair_results['oracle'].append(oracle_epe)
        
        # Get ground truth
        pair = movie.pairs[pair_idx]
        u_truth = pair.u_truth
        v_truth = pair.v_truth
        valid_mask = pair.valid_mask
        
        # Evaluate each method
        for method in required_methods:
            epe = evaluate_method(
                results_full, 
                selection_configs[method],
                u_truth, v_truth, valid_mask,
                epe_power
            )
            pair_results[method].append(epe)
    
    # Summary table
    print(f"\n{'Method':<12} {'Mean EPE':<12} {'Std':<12}")
    print("-" * 36)
    
    for method in ['oracle'] + required_methods:
        values = pair_results[method]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{method:<12} {mean_val:<12.6f} {std_val:<12.6f}")
    
    # Generate figure
    output_path = figures_dir / f'epe_per_pair_{weight_hash}.png'
    generate_figure(pair_results, selection_configs, weight_hash, epe_power, output_path)
    
    print(f"\n📊 {output_path}")
    print("=" * 60)
    
    return pair_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate fixed weights on dataset'
    )
    parser.add_argument('config', type=Path, help='TOML config file')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                       help='Base data directory (default: data/)')
    parser.add_argument('--use-optimized', action='store_true',
                       help='Load weights from optimization results instead of config')
    
    args = parser.parse_args()
    
    # Load config
    if not args.config.exists():
        print(f"❌ ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'rb') as f:
        config = tomli.load(f)
    
    # Run evaluation
    run_evaluation(config, args.data_dir, args.use_optimized)


if __name__ == "__main__":
    main()
