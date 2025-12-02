#!/usr/bin/env python3
# File: scripts/optimize_weights.py
"""
Sequence-Level Weight Optimization

Optimizes metric weights across ALL frame pairs in a movie sequence.
Finds a single weight vector that minimizes mean EPE across the sequence.

Supports:
- Optimizable weights (varied by Optuna)
- Fixed weights (forced to exact value)
- Sum-to-1 constraint on optimizable weights
- Initial trial seeding from config

Usage:
    python scripts/optimize_weights.py config.toml --movie-hash abc123 --of-hash def456
    python scripts/optimize_weights.py config.toml --movie-hash abc123 --of-hash def456 --trials 200
"""

import sys
import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import tomli
import tomli_w

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("❌ ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_loader import load_movie_sequence
from src.ensemble.selection import (
    ALL_METRICS,
    GAIN_METRICS,
    build_metric_stacks,
    normalize_stacks,
    compute_penalty,
    select_ensemble,
    gather_flow,
    compute_epe_stats,
    validate_weight_config,
    normalize_weights_to_sum_one,
)


# =============================================================================
# Configuration
# =============================================================================

SELECTION_METHODS = {
    'raw_sum': {'normalize': 'raw', 'aggregation': 'sum'},
    'raw_max': {'normalize': 'raw', 'aggregation': 'max'},
    'mad_sum': {'normalize': 'mad', 'aggregation': 'sum'},
    'mad_max': {'normalize': 'mad', 'aggregation': 'max'},
}


def compute_optimization_hash(config: dict) -> str:
    """Compute hash from optimization config."""
    # Extract relevant sections
    opt_sections = {}
    for key in ['optimization', 'evaluation']:
        if key in config:
            opt_sections[key] = config[key]
    config_str = tomli_w.dumps(opt_sections)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


# =============================================================================
# Data Loading
# =============================================================================

def load_pair_data(pair_dir: Path, pair: 'FramePair', epe_power: float) -> dict:
    """
    Load all data for one frame pair.
    
    Args:
        pair_dir: Directory containing results_full.pkl
        pair: FramePair object with ground truth
        epe_power: EPE power
    
    Returns dict with:
        results_full, u_truth, v_truth, valid_mask, H, W
    """
    results_path = pair_dir / 'results_full.pkl'
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    with open(results_path, 'rb') as f:
        results_full = pickle.load(f)
    
    return {
        'results_full': results_full,
        'u_truth': pair.u_truth,
        'v_truth': pair.v_truth,
        'valid_mask': pair.valid_mask,
        'H': pair.metadata['H'],
        'W': pair.metadata['W'],
    }


def build_flow_stacks(results_full: list) -> tuple[np.ndarray, np.ndarray]:
    """Build u and v flow stacks from results."""
    n_configs = len(results_full)
    first_flow = results_full[0]['flows']['u_AB']
    H, W = first_flow.shape
    
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, r in enumerate(results_full):
        u_stack[i] = r['flows']['u_AB']
        v_stack[i] = r['flows']['v_AB']
    
    return u_stack, v_stack


# =============================================================================
# Precomputation for Fast Objective Evaluation  
# =============================================================================

def precompute_pair(
    pair_data: dict,
    enabled_metrics: list[str],
    normalize_method: str
) -> dict:
    """
    Precompute normalized metric stacks and flow stacks for one pair.
    
    Args:
        pair_data: Dict with results_full, valid_mask, etc.
        enabled_metrics: List of metrics to include
        normalize_method: 'raw' or 'mad'
        
    Returns:
        Dict with precomputed stacks ready for fast objective evaluation
    """
    results_full = pair_data['results_full']
    
    # Build metric stacks (handles gain metric inversion)
    stacks = build_metric_stacks(results_full, enabled_metrics)
    
    # Normalize
    stacks = normalize_stacks(stacks, normalize_method)
    
    # Build flow stacks
    u_stack, v_stack = build_flow_stacks(results_full)
    
    return {
        'metric_stacks': stacks,
        'u_stack': u_stack,
        'v_stack': v_stack,
        'u_truth': pair_data['u_truth'],
        'v_truth': pair_data['v_truth'],
        'valid_mask': pair_data['valid_mask'],
        'H': pair_data['H'],
        'W': pair_data['W'],
    }


def compute_epe_from_precomputed(
    precomputed: dict,
    weights: dict,
    aggregation: str,
    epe_power: float
) -> dict:
    """
    Compute EPE stats from precomputed data.
    
    Args:
        precomputed: Output from precompute_pair
        weights: {metric_name: float}
        aggregation: 'sum' or 'max'
        epe_power: Power for EPE
        
    Returns:
        {'mean': float, 'std': float, 'median': float}
    """
    # Compute penalty
    penalty = compute_penalty(precomputed['metric_stacks'], weights, aggregation)
    
    # Select best config per pixel
    selection = select_ensemble(penalty)
    
    # Gather flow
    u_ens, v_ens = gather_flow(
        precomputed['u_stack'],
        precomputed['v_stack'],
        selection
    )
    
    # Compute EPE stats
    return compute_epe_stats(
        u_ens, v_ens,
        precomputed['u_truth'], precomputed['v_truth'],
        precomputed['valid_mask'],
        epe_power
    )


# =============================================================================
# Optuna Optimization
# =============================================================================

def create_objective(
    all_precomputed: list[dict],
    optimize_metrics: list[str],
    fixed_weights: dict[str, float],
    aggregation: str,
    epe_power: float
):
    """
    Create Optuna objective function.
    
    Args:
        all_precomputed: Precomputed data for all pairs
        optimize_metrics: Metrics to optimize (sample weights for these)
        fixed_weights: Fixed weight values (not optimized)
        aggregation: 'sum' or 'max'
        epe_power: EPE power
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial: optuna.Trial) -> float:
        # Sample weights for optimizable metrics in [0, 1]
        raw_weights = {}
        for metric in optimize_metrics:
            raw_weights[metric] = trial.suggest_float(metric, 0.0, 1.0)
        
        # Normalize to sum = 1
        total = sum(raw_weights.values())
        if total < 1e-10:
            return float('inf')
        
        opt_weights = {k: v / total for k, v in raw_weights.items()}
        
        # Merge with fixed weights
        all_weights = {**fixed_weights, **opt_weights}
        
        # Compute mean EPE across all pairs
        pair_epes = []
        for precomputed in all_precomputed:
            stats = compute_epe_from_precomputed(
                precomputed, all_weights, aggregation, epe_power
            )
            pair_epes.append(stats['mean'])
        
        return float(np.mean(pair_epes))
    
    return objective


def run_optimization(
    method_name: str,
    all_precomputed: list[dict],
    optimize_metrics: list[str],
    fixed_weights: dict[str, float],
    initial_weights: dict[str, float],
    aggregation: str,
    n_trials: int,
    epe_power: float,
    output_dir: Path = None,
    show_progress: bool = True
) -> dict:
    """
    Run Optuna optimization for one method.
    
    Uses SQLite storage for persistence - optimization can resume across runs.
    
    Returns:
        Dict with best_weights, best_epe, n_trials_total, etc.
    """
    # Set up persistent storage if output_dir provided
    storage = None
    study_name = f"optimize_{method_name}"
    
    if output_dir is not None:
        db_path = output_dir / "optuna_studies.db"
        storage = f"sqlite:///{db_path}"
    
    # Load or create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True  # Resume from previous runs
    )
    
    # Track how many trials existed before this run
    n_existing = len(study.trials)
    
    # Enqueue initial trial only if study is fresh (no trials yet)
    if n_existing == 0 and initial_weights:
        init_normalized = normalize_weights_to_sum_one(initial_weights)
        study.enqueue_trial(init_normalized)
    
    # Create and run objective
    objective = create_objective(
        all_precomputed,
        optimize_metrics,
        fixed_weights,
        aggregation,
        epe_power
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress,
        gc_after_trial=True
    )
    
    n_total = len(study.trials)
    
    # Extract best weights
    best_raw = study.best_params.copy()
    
    # Normalize to sum=1
    total = sum(best_raw.values())
    best_opt_weights = {k: v / total for k, v in best_raw.items()}
    
    # Merge with fixed weights for final result
    best_weights = {**fixed_weights, **best_opt_weights}
    
    # Compute final EPE stats
    pair_stats = []
    for precomputed in all_precomputed:
        stats = compute_epe_from_precomputed(
            precomputed, best_weights, aggregation, epe_power
        )
        pair_stats.append(stats)
    
    mean_epe = float(np.mean([s['mean'] for s in pair_stats]))
    mean_std = float(np.mean([s['std'] for s in pair_stats]))
    
    return {
        'method_name': method_name,
        'best_weights': best_weights,
        'best_epe': mean_epe,
        'best_std': mean_std,
        'n_trials_this_run': n_trials,
        'n_trials_total': n_total,
        'n_trials_previous': n_existing,
        'best_trial': study.best_trial.number,
    }


# =============================================================================
# Optimization Hashing and Config Saving
# =============================================================================

def compute_optimization_hash(
    optimize_weights: dict,
    fixed_weights: dict,
    methods: list[str],
    epe_power: float
) -> str:
    """
    Compute hash of optimization configuration.
    
    Hash is based on:
    - Which metrics to optimize and their initial weights
    - Fixed weight values
    - Optimization methods (raw_sum, etc.)
    - EPE power
    
    Returns:
        12-character hash string
    """
    # Build canonical representation (sorted for determinism)
    hash_dict = {
        'optimize_weights': dict(sorted(optimize_weights.items())),
        'fixed_weights': dict(sorted(fixed_weights.items())),
        'methods': sorted(methods),
        'epe_power': epe_power,
    }
    
    # Convert to canonical JSON string
    hash_str = json.dumps(hash_dict, sort_keys=True, separators=(',', ':'))
    
    # Compute hash
    return hashlib.sha256(hash_str.encode()).hexdigest()[:12]


def save_optimization_config(output_dir: Path, config: dict, opt_hash: str) -> None:
    """
    Save optimization config to output directory for reproducibility.
    
    Saves only the optimization-relevant sections of the config.
    """
    config_path = output_dir / 'config.toml'
    
    # Only save if doesn't exist (don't overwrite on resume)
    if config_path.exists():
        return
    
    # Extract relevant sections
    opt_config = {
        'optimization': config.get('optimization', {}),
        'evaluation': {
            'epe_power': config.get('evaluation', {}).get('epe_power', 2.0)
        },
        '_metadata': {
            'opt_hash': opt_hash,
        }
    }
    
    # Save as TOML
    with open(config_path, 'wb') as f:
        tomli_w.dump(opt_config, f)


# =============================================================================
# Main Entry Points
# =============================================================================

def reset_optimization(output_dir: Path) -> None:
    """Delete optimization studies database to start fresh."""
    db_path = output_dir / "optuna_studies.db"
    if db_path.exists():
        db_path.unlink()
        print(f"🗑️  Deleted optimization database: {db_path}")
    else:
        print(f"   No database found at: {db_path}")


def run_sequence_optimization(
    config: dict,
    movie_hash: str,
    of_hash: str,
    data_dir: str = 'data',
    n_trials: Optional[int] = None,
    reset: bool = False,
    show_progress: bool = True
) -> dict:
    """
    Run optimization for a movie sequence.
    
    Uses SQLite storage for persistence - optimization can resume across runs.
    Output directory includes hash of optimization config for reproducibility.
    
    Args:
        config: Full config dict
        movie_hash: Movie hash
        of_hash: Optical flow hash
        data_dir: Data directory
        n_trials: Override number of trials
        reset: If True, delete previous optimization results and start fresh
        show_progress: Show progress bars
        
    Returns:
        Optimization results dict (method -> {best_weights, best_epe, ...})
    """
    data_path = Path(data_dir)
    analysis_dir = data_path / movie_hash / 'analysis' / of_hash
    
    # Extract config
    opt_config = config.get('optimization', {})
    eval_config = config.get('evaluation', {})
    
    methods = opt_config.get('methods', ['raw_sum', 'raw_max', 'mad_sum', 'mad_max'])
    n_trials_config = opt_config.get('n_trials', 100)
    n_trials = n_trials or n_trials_config
    epe_power = eval_config.get('epe_power', 2.0)
    
    # Get weight configuration
    optimize_weights = opt_config.get('weights', {})
    fixed_weights = opt_config.get('fixed_weights', {})
    
    # Validate all metrics are accounted for
    validate_weight_config(optimize_weights, fixed_weights)
    
    # Compute optimization hash based on config that affects results
    opt_hash = compute_optimization_hash(
        optimize_weights=optimize_weights,
        fixed_weights=fixed_weights,
        methods=methods,
        epe_power=epe_power
    )
    
    # Create output directory with hash
    output_dir = analysis_dir / 'optimization' / opt_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config copy for reproducibility (only optimization-relevant parts)
    save_optimization_config(output_dir, config, opt_hash)
    
    # Handle reset
    if reset:
        reset_optimization(output_dir)
    
    # Determine which metrics to optimize (non-zero initial weight)
    optimize_metrics = [k for k, v in optimize_weights.items() if v > 0]
    initial_weights = {k: v for k, v in optimize_weights.items() if v > 0}
    
    # Add zero-weight metrics to fixed
    for k, v in optimize_weights.items():
        if v == 0:
            fixed_weights[k] = 0.0
    
    # Get enabled metrics (all with non-zero weight after merging)
    all_weights = {**fixed_weights, **initial_weights}
    enabled_metrics = [k for k, v in all_weights.items() if v != 0]
    
    if not enabled_metrics:
        raise ValueError("No metrics enabled (all weights are 0)")
    
    print("=" * 80)
    print("🔍 SEQUENCE WEIGHT OPTIMIZATION")
    print("=" * 80)
    print(f"Movie hash: {movie_hash}")
    print(f"OF hash: {of_hash}")
    print(f"Opt hash: {opt_hash}")
    print(f"Output: {output_dir}")
    print()
    
    # Load movie sequence
    print("📂 Loading movie and sweep results...")
    movie = load_movie_sequence(data_path / movie_hash)
    print(f"   Pairs: {len(movie.pairs)}")
    
    # Check for ground truth via first pair
    has_ground_truth = movie.pairs[0].has_gt if movie.pairs else False
    print(f"   Ground truth: {'Available' if has_ground_truth else 'Not available'}")
    
    if not has_ground_truth:
        print("❌ ERROR: Ground truth required for optimization")
        sys.exit(1)
    
    # Load pair data
    sweep_dir = analysis_dir / 'sweep'
    if not sweep_dir.exists():
        print(f"❌ ERROR: Sweep directory not found: {sweep_dir}")
        sys.exit(1)
    
    pair_dirs = sorted(sweep_dir.glob('pair_*'))
    if not pair_dirs:
        print("❌ ERROR: No pair directories found. Run sweep first.")
        sys.exit(1)
    
    # Check number of configs
    test_results = pickle.load(open(pair_dirs[0] / 'results_full.pkl', 'rb'))
    n_configs = len(test_results)
    print(f"   Configs per pair: {n_configs}")
    print()
    
    print("📋 Optimization configuration:")
    print(f"   Methods: {methods}")
    print(f"   Trials: {n_trials}")
    print(f"   EPE power: {epe_power}")
    print(f"   Optimize metrics: {optimize_metrics}")
    print(f"   Fixed weights: {fixed_weights}")
    print()
    
    # Load all pair data
    all_pair_data = []
    for pair_idx, pair_dir in enumerate(pair_dirs):
        pair = movie.pairs[pair_idx]
        pair_data = load_pair_data(pair_dir, pair, epe_power)
        all_pair_data.append(pair_data)
    
    # Run optimization for each method
    print("=" * 80)
    print(f"🔄 RUNNING {len(methods)} OPTIMIZATIONS")
    print("=" * 80)
    print()
    
    results = {}
    
    for i, method_name in enumerate(methods):
        method_config = SELECTION_METHODS[method_name]
        normalize_method = method_config['normalize']
        aggregation = method_config['aggregation']
        
        print(f"[{i+1}/{len(methods)}] {method_name}")
        print("   Precomputing metric stacks...")
        
        # Precompute for this normalization method
        all_precomputed = []
        for pair_data in all_pair_data:
            precomputed = precompute_pair(pair_data, enabled_metrics, normalize_method)
            all_precomputed.append(precomputed)
        
        print(f"   Running {n_trials} trials...")
        
        result = run_optimization(
            method_name=method_name,
            all_precomputed=all_precomputed,
            optimize_metrics=optimize_metrics,
            fixed_weights=fixed_weights,
            initial_weights=initial_weights,
            aggregation=aggregation,
            n_trials=n_trials,
            epe_power=epe_power,
            output_dir=output_dir,
            show_progress=False  # Disable Optuna progress bar
        )
        
        results[method_name] = result
        
        # Format weights for display
        w = result['best_weights']
        weight_strs = []
        for m in ALL_METRICS:
            if m in w:
                abbrev = {'traction': 'trac', 'perturbation_rms': 'pert', 
                         'consistency': 'cons', 'photometric': 'phot',
                         'photometric_rgb': 'pRGB', 'photometric_rgb_log': 'pLog',
                         'speed_sym': 'sSpd'}[m]
                weight_strs.append(f"{abbrev}={w[m]:.2f}")
        
        # Show trial counts
        n_prev = result['n_trials_previous']
        n_total = result['n_trials_total']
        trial_info = f"(total: {n_total})" if n_prev > 0 else ""
        if n_prev > 0:
            print(f"   📊 Resumed from {n_prev} previous trials {trial_info}")
        
        print(f"   ✅ EPE^{epe_power}: {result['best_epe']:.6f} ± {result['best_std']:.6f}")
        print(f"   Weights: {', '.join(weight_strs)}")
        print()
    
    # Build summary
    summary = {
        'movie_hash': movie_hash,
        'of_hash': of_hash,
        'opt_hash': opt_hash,
        'n_trials_this_run': n_trials,
        'epe_power': epe_power,
        'optimize_metrics': optimize_metrics,
        'fixed_weights': fixed_weights,
        'results': results,
    }
    
    # Save summary
    summary_path = output_dir / 'optimization_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final results
    print("=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print()
    
    # Sort by EPE
    sorted_methods = sorted(results.keys(), key=lambda m: results[m]['best_epe'])
    
    # Check if any have previous trials
    any_resumed = any(results[m]['n_trials_previous'] > 0 for m in sorted_methods)
    
    if any_resumed:
        print(f"{'Method':<15} {'EPE^' + str(epe_power):<20} {'Trials':<12} {'Weights':<50}")
        print("-" * 100)
    else:
        print(f"{'Method':<15} {'EPE^' + str(epe_power):<20} {'Weights':<60}")
        print("-" * 95)
    
    for method in sorted_methods:
        r = results[method]
        epe_str = f"{r['best_epe']:.6f} ± {r['best_std']:.4f}"
        
        w = r['best_weights']
        weight_strs = []
        for m in ALL_METRICS:
            if m in w:
                abbrev = {'traction': 'trac', 'perturbation_rms': 'pert',
                         'consistency': 'cons', 'photometric': 'phot',
                         'photometric_rgb': 'pRGB', 'photometric_rgb_log': 'pLog',
                         'speed_sym': 'sSpd'}[m]
                weight_strs.append(f"{abbrev}={w[m]:.2f}")
        
        if any_resumed:
            trial_str = f"{r['n_trials_total']}"
            print(f"{method:<15} {epe_str:<20} {trial_str:<12} {', '.join(weight_strs)}")
        else:
            print(f"{method:<15} {epe_str:<20} {', '.join(weight_strs)}")
    
    print()
    best_method = sorted_methods[0]
    print(f"🏆 Best method: {best_method} (EPE^{epe_power} = {results[best_method]['best_epe']:.6f})")
    
    # Add opt_hash to each result for external access
    for method in results:
        results[method]['opt_hash'] = opt_hash
    
    # Return results dict (summary is saved to JSON file)
    # Each result now includes opt_hash for callers that need it
    return results


# Alias for backward compatibility with run_experiment.py
optimize_sequence = run_sequence_optimization


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize ensemble weights')
    parser.add_argument('config', type=str, help='Path to config TOML')
    parser.add_argument('--movie-hash', type=str, required=True, help='Movie hash')
    parser.add_argument('--of-hash', type=str, required=True, help='Optical flow hash')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--trials', type=int, default=None, help='Number of trials')
    parser.add_argument('--reset', action='store_true', help='Reset optimization (delete previous trials)')
    
    args = parser.parse_args()
    
    # Handle reset
    if args.reset:
        data_path = Path(args.data_dir)
        output_dir = data_path / args.movie_hash / 'analysis' / args.of_hash / 'optimization'
        reset_optimization(output_dir)
        print()
    
    # Load config
    with open(args.config, 'rb') as f:
        config = tomli.load(f)
    
    run_sequence_optimization(
        config=config,
        movie_hash=args.movie_hash,
        of_hash=args.of_hash,
        data_dir=args.data_dir,
        n_trials=args.trials,
    )


if __name__ == '__main__':
    main()
