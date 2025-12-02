#!/usr/bin/env python3
# File: scripts/optimize_weights.py
"""
Sequence-Level Weight Optimization

Optimizes metric weights across ALL frame pairs in a movie sequence.
Finds a single weight vector that minimizes mean EPE across the sequence.

Runs 4 Optuna studies:
- mad_sum: MAD normalize, sum aggregation
- mad_max: MAD normalize, max aggregation  
- raw_sum: raw values, sum aggregation
- raw_max: raw values, max aggregation

Usage:
    python scripts/optimize_weights.py config.toml --movie-hash abc123 --of-hash def456
    python scripts/optimize_weights.py config.toml --movie-hash abc123 --of-hash def456 --trials 200
"""

import sys
import csv
import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tomli
import tomli_w

try:
    import optuna
except ImportError:
    print("❌ ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_loader import load_movie_sequence
from src.evaluation.ground_truth import compute_epe


# =============================================================================
# Configuration
# =============================================================================

OPTIMIZATION_SECTIONS = ['selection', 'optimization', 'evaluation']

# Loss metrics (lower = better)
LOSS_METRIC_NAMES = ['traction', 'perturbation_rms', 'consistency', 'photometric', 'photometric_rgb', 'photometric_rgb_log']

# Gain metrics (higher = better, inverted at selection time: 1 - value/max)
GAIN_METRIC_NAMES = ['speed_sym']

# All metric names
ALL_METRIC_NAMES = LOSS_METRIC_NAMES + GAIN_METRIC_NAMES

# Legacy alias for backward compatibility
METRIC_NAMES = ['perturbation_rms', 'consistency', 'photometric']

# Metric key mapping (metric name -> (A key, B key) in results)
METRIC_KEY_MAP = {
    'traction': ('traction_A', 'traction_B'),
    'perturbation_rms': ('displacements_sensitivity_A2B', 'displacements_sensitivity_B2A'),
    'consistency': ('consistency_A', 'consistency_B'),
    'photometric': ('photometric_A', 'photometric_B'),
    'photometric_rgb': ('photometric_rgb_A', 'photometric_rgb_B'),
    'photometric_rgb_log': ('photometric_rgb_log_A', 'photometric_rgb_log_B'),
    'speed_sym': ('speed_sym_A', 'speed_sym_B'),
}

# Selection method definitions
SELECTION_METHODS = {
    'mad_sum': {'normalize': 'mad', 'aggregation': 'sum', 'power': 2},
    'mad_max': {'normalize': 'mad', 'aggregation': 'max', 'power': 2},
    'raw_sum': {'normalize': 'none', 'aggregation': 'sum', 'power': 2},
    'raw_max': {'normalize': 'none', 'aggregation': 'max', 'power': 2},
}


def extract_optimization_config(config: dict) -> dict:
    """Extract optimization-relevant sections from full config."""
    opt_config = {}
    for section in OPTIMIZATION_SECTIONS:
        if section in config:
            opt_config[section] = config[section]
    return opt_config


def compute_optimization_hash(opt_config: dict) -> str:
    """Compute hash from optimization config."""
    config_str = tomli_w.dumps(opt_config)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def get_epe_power(config: dict) -> float:
    """Extract epe_power from config."""
    eval_config = config.get('evaluation', {})
    epe_power = eval_config.get('epe_power')
    if epe_power is None:
        print("❌ ERROR: [evaluation] section must specify epe_power")
        sys.exit(1)
    return float(epe_power)


# =============================================================================
# Data Loading
# =============================================================================

def load_all_pair_data(sweep_dir: Path, movie, epe_power: float) -> list[dict]:
    """
    Load sweep results and ground truth for all pairs.
    
    Args:
        sweep_dir: Directory containing pair_XXX subdirectories
        movie: MovieSequence object
        epe_power: EPE power for computations
        
    Returns:
        List of dicts, one per pair, containing:
            - results_full: List of config results
            - u_truth, v_truth: Ground truth flow
            - valid_mask: Valid pixel mask
            - H, W: Image dimensions
    """
    n_pairs = len(movie.pairs)
    all_data = []
    
    for pair_idx in range(n_pairs):
        pair = movie.pairs[pair_idx]
        pair_dir = sweep_dir / f'pair_{pair_idx:03d}'
        
        # Load results
        results_path = pair_dir / 'results_full.pkl'
        if not results_path.exists():
            print(f"❌ ERROR: Sweep results not found: {results_path}")
            print("   Run optical_flow_track.py first")
            sys.exit(1)
        
        with open(results_path, 'rb') as f:
            results_full = pickle.load(f)
        
        all_data.append({
            'results_full': results_full,
            'u_truth': pair.u_truth,
            'v_truth': pair.v_truth,
            'valid_mask': pair.valid_mask,
            'H': pair.metadata['H'],
            'W': pair.metadata['W'],
        })
    
    return all_data


# =============================================================================
# Metric Extraction and Normalization
# =============================================================================

def extract_metrics_from_result(result: dict) -> dict:
    """
    Extract and average A/B metrics from a single config result.
    
    Args:
        result: Config result dict (structured or flat format)
        
    Returns:
        Dict mapping metric_name -> (H, W) array
    """
    metrics = {}
    
    # Handle structured vs flat format
    if 'metrics' in result:
        raw_metrics = result['metrics']
    else:
        raw_metrics = result
    
    for metric_name, (key_A, key_B) in METRIC_KEY_MAP.items():
        val_A = raw_metrics.get(key_A)
        val_B = raw_metrics.get(key_B)
        
        if val_A is not None and val_B is not None:
            metrics[metric_name] = (val_A + val_B) / 2
        elif val_A is not None:
            metrics[metric_name] = val_A
        elif val_B is not None:
            metrics[metric_name] = val_B
    
    return metrics


def extract_flows_from_result(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract forward flow from result dict."""
    if 'flows' in result:
        return result['flows']['u_AB'], result['flows']['v_AB']
    else:
        return result['u_AB'], result['v_AB']


def mad_normalize_stack(metric_stack: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    MAD-normalize a metric across all configs.
    
    Args:
        metric_stack: (n_configs, H, W) array
        valid_mask: (H, W) boolean mask
        
    Returns:
        Normalized metric stack, clipped to >= 0
    """
    n_configs = metric_stack.shape[0]
    
    # Gather all valid values
    all_valid = []
    for i in range(n_configs):
        all_valid.append(metric_stack[i][valid_mask])
    all_valid = np.concatenate(all_valid)
    
    # Handle empty or constant data
    if len(all_valid) == 0:
        return np.zeros_like(metric_stack)
    
    median = np.median(all_valid)
    mad = np.median(np.abs(all_valid - median))
    
    if mad < 1e-10:
        return np.zeros_like(metric_stack)
    
    normalized = (metric_stack - median) / mad
    return np.maximum(0, normalized)


# =============================================================================
# Precomputation for Fast Objective Evaluation
# =============================================================================

def precompute_pair_data(pair_data: dict, method_config: dict) -> dict:
    """
    Precompute normalized metric stacks and flow stacks for one pair.
    
    This is done once before optimization to make objective evaluation fast.
    
    Args:
        pair_data: Dict with results_full, valid_mask, etc.
        method_config: Dict with 'normalize', 'aggregation', 'power'
        
    Returns:
        Dict with precomputed stacks
    """
    results_full = pair_data['results_full']
    valid_mask = pair_data['valid_mask']
    H, W = pair_data['H'], pair_data['W']
    n_configs = len(results_full)
    
    normalize = method_config['normalize']
    
    # Build metric stacks for all metrics
    metric_stacks = {}
    
    # Process loss metrics (lower = better)
    for metric_name in LOSS_METRIC_NAMES:
        stack = np.full((n_configs, H, W), np.inf, dtype=np.float32)
        
        for i, result in enumerate(results_full):
            metrics = extract_metrics_from_result(result)
            if metric_name in metrics:
                stack[i] = metrics[metric_name]
        
        # Skip if metric not available (e.g., RGB metrics for grayscale input)
        if np.all(np.isinf(stack)):
            continue
        
        # Normalize if requested
        if normalize == 'mad':
            stack = mad_normalize_stack(stack, valid_mask)
        
        metric_stacks[metric_name] = stack
    
    # Process gain metrics (higher = better, invert to: 1 - value/max)
    for metric_name in GAIN_METRIC_NAMES:
        stack = np.full((n_configs, H, W), 0.0, dtype=np.float32)  # Default 0 for gain
        
        for i, result in enumerate(results_full):
            metrics = extract_metrics_from_result(result)
            if metric_name in metrics:
                stack[i] = metrics[metric_name]
        
        # Skip if metric not available
        if np.all(stack == 0):
            continue
        
        # Invert: 1 - value/max (per-pixel max across configs)
        max_val = np.max(stack, axis=0, keepdims=True)
        eps = 1e-6
        stack = 1.0 - stack / (max_val + eps)
        
        # Normalize if requested (after inversion)
        if normalize == 'mad':
            stack = mad_normalize_stack(stack, valid_mask)
        
        metric_stacks[metric_name] = stack
    
    # Build flow stacks
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, result in enumerate(results_full):
        u, v = extract_flows_from_result(result)
        u_stack[i] = u
        v_stack[i] = v
    
    return {
        'metric_stacks': metric_stacks,
        'u_stack': u_stack,
        'v_stack': v_stack,
        'u_truth': pair_data['u_truth'],
        'v_truth': pair_data['v_truth'],
        'valid_mask': valid_mask,
        'H': H,
        'W': W,
    }


def compute_ensemble_epe(
    precomputed: dict,
    weights: dict,
    aggregation: str,
    power: float,
    epe_power: float
) -> tuple[float, float, float]:
    """
    Compute ensemble EPE for one pair with given weights.
    
    Args:
        precomputed: Precomputed metric/flow stacks
        weights: Dict of metric weights
        aggregation: 'sum' or 'max'
        power: Penalty power
        epe_power: EPE power for loss
        
    Returns:
        Tuple of (mean_epe, std_epe, median_epe) over valid pixels
    """
    H, W = precomputed['H'], precomputed['W']
    valid_mask = precomputed['valid_mask']
    metric_stacks = precomputed['metric_stacks']
    n_configs = precomputed['u_stack'].shape[0]
    
    # Compute weighted penalty
    if aggregation == 'sum':
        penalty_stack = np.zeros((n_configs, H, W), dtype=np.float32)
        for metric_name, weight in weights.items():
            if weight != 0 and metric_name in metric_stacks:
                penalty_stack += weight * metric_stacks[metric_name]
    else:  # max
        penalty_stack = np.full((n_configs, H, W), -np.inf, dtype=np.float32)
        for metric_name, weight in weights.items():
            if weight != 0 and metric_name in metric_stacks:
                weighted = weight * metric_stacks[metric_name]
                penalty_stack = np.maximum(penalty_stack, weighted)
        # Handle case where all weights are 0
        penalty_stack = np.where(np.isinf(penalty_stack), 0, penalty_stack)
    
    # Apply power
    penalty_stack = np.power(penalty_stack, power)
    
    # Per-pixel selection
    best_config = np.argmin(penalty_stack, axis=0)
    
    # Build ensemble flow using advanced indexing
    row_idx = np.arange(H)[:, None]
    col_idx = np.arange(W)[None, :]
    u_ensemble = precomputed['u_stack'][best_config, row_idx, col_idx]
    v_ensemble = precomputed['v_stack'][best_config, row_idx, col_idx]
    
    # Compute EPE
    u_truth = precomputed['u_truth']
    v_truth = precomputed['v_truth']
    epe = np.sqrt((u_ensemble - u_truth)**2 + (v_ensemble - v_truth)**2)
    
    # Compute stats over valid pixels
    epe_powered = epe[valid_mask] ** epe_power
    mean_epe = float(np.mean(epe_powered))
    std_epe = float(np.std(epe_powered))
    median_epe = float(np.median(epe_powered))
    
    return mean_epe, std_epe, median_epe


# =============================================================================
# Optuna Optimization
# =============================================================================

def create_sequence_objective(
    all_precomputed: list[dict],
    method_config: dict,
    epe_power: float
):
    """
    Create Optuna objective for sequence-level optimization.
    
    Args:
        all_precomputed: List of precomputed data, one per pair
        method_config: Dict with 'normalize', 'aggregation', 'power'
        epe_power: EPE power for loss
        
    Returns:
        Objective function for Optuna
    """
    aggregation = method_config['aggregation']
    power = method_config['power']
    
    def objective(trial: optuna.Trial) -> float:
        # Sample weights for all available metrics
        weights = {}
        
        # Get available metrics from first precomputed (they're all the same)
        available_metrics = set(all_precomputed[0]['metric_stacks'].keys())
        
        # Sample weights for all available metrics in [0, 1]
        # (non-negative to avoid degenerate solutions with _sum aggregation)
        for metric_name in ALL_METRIC_NAMES:
            if metric_name in available_metrics:
                weights[metric_name] = trial.suggest_float(metric_name, 0.0, 1.0)
        
        # Check for degenerate case (all near zero)
        total = sum(abs(v) for v in weights.values())
        if total < 1e-10:
            return float('inf')

        # Compute EPE for each pair (using raw weights)
        pair_epes = []
        for precomputed in all_precomputed:
            mean_epe, _, _ = compute_ensemble_epe(
                precomputed, weights, aggregation, power, epe_power
            )
            pair_epes.append(mean_epe)
        
        # Return mean across pairs (sequence objective)
        return float(np.mean(pair_epes))
    
    return objective


def run_optimization_for_method(
    method_name: str,
    method_config: dict,
    all_precomputed: list[dict],
    output_dir: Path,
    n_trials: int,
    epe_power: float,
    show_progress: bool = True
) -> dict:
    """
    Run Optuna optimization for one selection method.
    
    Args:
        method_name: e.g., 'mad_sum'
        method_config: Dict with normalize/aggregation/power
        all_precomputed: Precomputed data for all pairs
        output_dir: Output directory for this method
        n_trials: Number of trials
        epe_power: EPE power
        show_progress: Show progress bar
        
    Returns:
        Dict with optimization results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create objective
    objective = create_sequence_objective(all_precomputed, method_config, epe_power)
    
    # Create study
    storage_path = output_dir / 'optuna_study.db'
    storage = f'sqlite:///{storage_path}'
    
    sampler = optuna.samplers.TPESampler(seed=42)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        study_name=method_name,
        storage=storage,
        load_if_exists=True,
        direction='minimize',
        sampler=sampler
    )
    
    import os
    n_jobs = max(1, (os.cpu_count() or 1) - 1)

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
        gc_after_trial=True
    )
    
    # Extract best weights and normalize by max(abs) for interpretability
    # Largest weight becomes ±1.0, others are relative to it
    raw_best = study.best_params.copy()
    max_abs = max(abs(v) for v in raw_best.values())
    best_weights = {k: v / max_abs for k, v in raw_best.items()}
    
    # Compute per-pair EPEs with best weights
    aggregation = method_config['aggregation']
    power = method_config['power']
    
    pair_epes = []
    pair_stds = []
    pair_medians = []
    n_configs_used_list = []
    
    for precomputed in all_precomputed:
        mean_epe, std_epe, median_epe = compute_ensemble_epe(
            precomputed, best_weights, aggregation, power, epe_power
        )
        pair_epes.append(mean_epe)
        pair_stds.append(std_epe)
        pair_medians.append(median_epe)
        
        # Count configs used (for stats)
        H, W = precomputed['H'], precomputed['W']
        valid_mask = precomputed['valid_mask']
        n_configs = precomputed['u_stack'].shape[0]
        
        # Recompute selection for config count
        if aggregation == 'sum':
            penalty_stack = np.zeros((n_configs, H, W), dtype=np.float32)
            for metric_name, weight in best_weights.items():
                if weight != 0 and metric_name in precomputed['metric_stacks']:
                    penalty_stack += weight * precomputed['metric_stacks'][metric_name]
        else:  # max
            penalty_stack = np.full((n_configs, H, W), -np.inf, dtype=np.float32)
            for metric_name, weight in best_weights.items():
                if weight != 0 and metric_name in precomputed['metric_stacks']:
                    weighted = weight * precomputed['metric_stacks'][metric_name]
                    penalty_stack = np.maximum(penalty_stack, weighted)
            penalty_stack = np.where(np.isinf(penalty_stack), 0, penalty_stack)
        
        best_config = np.argmin(np.power(penalty_stack, power), axis=0)
        n_configs_used = len(np.unique(best_config[valid_mask]))
        n_configs_used_list.append(n_configs_used)
    
    # Build selection config (includes all available metrics)
    best_selection_config = {
        'normalize': method_config['normalize'],
        'aggregation': method_config['aggregation'],
        'power': method_config['power'],
    }
    # Add all metric weights
    for metric_name in ALL_METRIC_NAMES:
        best_selection_config[metric_name] = best_weights.get(metric_name, 0.0)
    
    # Results - use mean of pixel-level stds (not std across pairs)
    results = {
        'method_name': method_name,
        'best_weights': best_weights,
        'best_selection_config': best_selection_config,
        'best_epe': float(np.mean(pair_epes)),
        'best_epe_std': float(np.mean(pair_stds)),  # Mean of pixel-level stds
        'best_epe_median': float(np.mean(pair_medians)),  # Mean of pixel-level medians
        'per_pair_epes': pair_epes,
        'per_pair_stds': pair_stds,
        'per_pair_medians': pair_medians,
        'n_configs_used': float(np.mean(n_configs_used_list)),
        'n_trials': len(study.trials),
        'epe_power': epe_power,
    }
    
    # Save results
    with open(output_dir / 'best_weights.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save trials CSV
    trials_data = []
    for trial in study.trials:
        trial_data = {'number': trial.number, 'value': trial.value, **trial.params}
        trials_data.append(trial_data)
    
    if trials_data:
        with open(output_dir / 'trials.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trials_data[0].keys())
            writer.writeheader()
            writer.writerows(trials_data)
    
    return results


# =============================================================================
# Figure Generation
# =============================================================================

def generate_optimization_figures(
    optimization_results: dict,
    output_dir: Path,
    epe_power: float
):
    """Generate comparison figures across methods."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    methods = list(optimization_results.keys())
    n_methods = len(methods)
    
    if n_methods == 0:
        return
    
    # Method comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epes = [optimization_results[m]['best_epe'] for m in methods]
    stds = [optimization_results[m]['best_epe_std'] for m in methods]
    
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple'][:n_methods]
    bars = ax.bar(methods, epes, yerr=stds, capsize=5, color=colors, alpha=0.7)
    
    ax.set_ylabel(f'Mean EPE^{epe_power}')
    ax.set_title('Sequence Optimization Results')
    
    # Add value labels
    for bar, epe in zip(bars, epes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{epe:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'method_comparison.png', dpi=150)
    plt.close()
    
    # Per-pair EPE comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    
    n_pairs = len(optimization_results[methods[0]]['per_pair_epes'])
    x = np.arange(n_pairs)
    width = 0.8 / n_methods
    
    for i, method in enumerate(methods):
        pair_epes = optimization_results[method]['per_pair_epes']
        offset = (i - n_methods/2 + 0.5) * width
        ax.bar(x + offset, pair_epes, width, label=method, alpha=0.7)
    
    ax.set_xlabel('Pair Index')
    ax.set_ylabel(f'EPE^{epe_power}')
    ax.set_title('Per-Pair EPE by Method')
    ax.legend()
    ax.set_xticks(x)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'per_pair_comparison.png', dpi=150)
    plt.close()
    
    print(f"   📊 Saved optimization figures")


# =============================================================================
# Main Optimization Function
# =============================================================================

def optimize_sequence(
    config: dict,
    movie_hash: str,
    of_hash: str,
    data_dir: Path = Path('data'),
    n_trials: int = 100,
    methods: Optional[list[str]] = None
) -> dict:
    """
    Run sequence-level weight optimization for all selection methods.
    
    Args:
        config: Full TOML config dict
        movie_hash: Hash of movie
        of_hash: Hash of OF configuration
        data_dir: Base data directory
        n_trials: Number of Optuna trials per method
        methods: List of methods to optimize (default: all 4)
        
    Returns:
        Dict mapping method names to optimization results
    """
    # Get epe_power
    epe_power = get_epe_power(config)
    
    # Setup paths
    movie_dir = data_dir / movie_hash
    analysis_dir = movie_dir / 'analysis' / of_hash
    sweep_dir = analysis_dir / 'sweep'
    opt_dir = analysis_dir / 'optimization'
    
    print("=" * 80)
    print("🔍 SEQUENCE WEIGHT OPTIMIZATION")
    print("=" * 80)
    print(f"Movie hash: {movie_hash}")
    print(f"OF hash: {of_hash}")
    print(f"Output: {opt_dir}")
    print()
    
    # Validate paths
    if not sweep_dir.exists():
        print(f"❌ ERROR: Sweep directory not found: {sweep_dir}")
        print("   Run optical_flow_track.py first")
        sys.exit(1)
    
    # Create output directory
    opt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save optimization config
    opt_config = extract_optimization_config(config)
    with open(opt_dir / 'optimization.toml', 'wb') as f:
        tomli_w.dump(opt_config, f)
    
    # Determine boundary margin from movie
    gen_path = movie_dir / 'generation.toml'
    if gen_path.exists():
        with open(gen_path, 'rb') as f:
            gen_config = tomli.load(f)
        # Get boundary margin from eval config
        boundary_margin = config.get('evaluation', {}).get('boundary_margin', 15)
    else:
        boundary_margin = 15
    
    # Load movie
    print("📂 Loading movie and sweep results...")
    movie = load_movie_sequence(movie_dir, boundary_margin=boundary_margin)
    
    if not movie.metadata['has_gt']:
        print("❌ ERROR: Optimization requires ground truth")
        sys.exit(1)
    
    n_pairs = len(movie.pairs)
    print(f"   Pairs: {n_pairs}")
    print(f"   Ground truth: Available")
    print()
    
    # Load all pair data
    all_pair_data = load_all_pair_data(sweep_dir, movie, epe_power)
    n_configs = len(all_pair_data[0]['results_full'])
    print(f"   Configs per pair: {n_configs}")
    print()
    
    # Determine methods to optimize
    if methods is None:
        methods = list(SELECTION_METHODS.keys())
    
    # Override from config if specified
    opt_config_methods = config.get('optimization', {}).get('methods')
    if opt_config_methods is not None:
        methods = [m for m in opt_config_methods if m in SELECTION_METHODS]

    # Override n_trials from config only if not explicitly passed
    if n_trials is None:
        config_trials = config.get('optimization', {}).get('n_trials')
        if config_trials is not None:
            n_trials = config_trials
    
    print(f"📋 Optimization configuration:")
    print(f"   Methods: {methods}")
    print(f"   Trials: {n_trials}")
    print(f"   EPE power: {epe_power}")
    print()
    
    # Run optimizations
    print("=" * 80)
    print(f"🔄 RUNNING {len(methods)} OPTIMIZATIONS")
    print("=" * 80)
    print()
    
    optimization_results = {}
    
    for method_idx, method_name in enumerate(methods):
        print(f"[{method_idx + 1}/{len(methods)}] {method_name}")
        
        method_config = SELECTION_METHODS[method_name]
        method_dir = opt_dir / method_name
        
        # Precompute data for this method
        print(f"   Precomputing metric stacks...")
        all_precomputed = [
            precompute_pair_data(pd, method_config) for pd in all_pair_data
        ]
        
        # Run optimization
        print(f"   Running {n_trials} trials...")
        result = run_optimization_for_method(
            method_name=method_name,
            method_config=method_config,
            all_precomputed=all_precomputed,
            output_dir=method_dir,
            n_trials=n_trials,
            epe_power=epe_power,
            show_progress=(method_idx == 0)  # Only first shows progress
        )
        
        optimization_results[method_name] = result
        
        # Print result
        print(f"   ✅ EPE^{epe_power}: {result['best_epe']:.6f} ± {result['best_epe_std']:.6f}")
        # Show all available weights
        weights_str = ", ".join([f"{m}={result['best_weights'].get(m, 0):.2f}" 
                                 for m in ALL_METRIC_NAMES 
                                 if m in result['best_weights']])
        print(f"   Weights: {weights_str}")
        print()
    
    # Generate figures
    print("📊 Generating figures...")
    generate_optimization_figures(optimization_results, opt_dir, epe_power)
    print()
    
    # Print summary
    print("=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print()
    
    # Sort by EPE
    sorted_methods = sorted(optimization_results.items(), 
                           key=lambda x: x[1]['best_epe'])
    
    # Build header with abbreviated metric names
    metric_abbrevs = {
        'traction': 'trac',
        'perturbation_rms': 'pert',
        'consistency': 'cons',
        'photometric': 'phot',
        'photometric_rgb': 'pRGB',
        'photometric_rgb_log': 'pLog',
        'speed_sym': 'sSpd',
    }
    
    print(f"{'Method':<15} {'EPE^' + str(int(epe_power)):<20} {'Weights':<60}")
    print("-" * 95)
    
    for method_name, result in sorted_methods:
        epe_str = f"{result['best_epe']:.6f} ± {result['best_epe_std']:.4f}"
        weights_str = ", ".join([f"{metric_abbrevs.get(m, m[:4])}={result['best_weights'].get(m, 0):.2f}" 
                                for m in ALL_METRIC_NAMES
                                if m in result['best_weights']])
        print(f"{method_name:<15} {epe_str:<20} {weights_str}")
    
    print()
    
    # Best method
    best_method = sorted_methods[0][0]
    best_epe = sorted_methods[0][1]['best_epe']
    print(f"🏆 Best method: {best_method} (EPE^{epe_power} = {best_epe:.6f})")
    
    # Save summary
    summary = {
        'movie_hash': movie_hash,
        'of_hash': of_hash,
        'n_pairs': n_pairs,
        'n_configs': n_configs,
        'n_trials': n_trials,
        'epe_power': epe_power,
        'best_method': best_method,
        'results': {m: {
            'best_epe': r['best_epe'],
            'best_epe_std': r['best_epe_std'],
            'best_weights': r['best_weights'],
        } for m, r in optimization_results.items()}
    }
    
    with open(opt_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return optimization_results


# =============================================================================
# CLI Entry Point  
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Sequence-level weight optimization for optical flow ensemble'
    )
    parser.add_argument('config', type=Path, help='TOML configuration file')
    parser.add_argument('--movie-hash', type=str, required=True,
                       help='Hash of movie')
    parser.add_argument('--of-hash', type=str, required=True,
                       help='Hash of OF configuration')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                       help='Base data directory (default: data/)')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of Optuna trials (default: 100)')
    parser.add_argument('--methods', type=str, nargs='+',
                       choices=['mad_sum', 'mad_max', 'raw_sum', 'raw_max'],
                       help='Methods to optimize (default: all)')
    
    args = parser.parse_args()
    
    # Validate config exists
    if not args.config.exists():
        print(f"❌ ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    # Load config
    with open(args.config, 'rb') as f:
        config = tomli.load(f)
    
    # Run optimization
    optimize_sequence(
        config=config,
        movie_hash=args.movie_hash,
        of_hash=args.of_hash,
        data_dir=args.data_dir,
        n_trials=args.trials,
        methods=args.methods
    )


if __name__ == "__main__":
    main()
