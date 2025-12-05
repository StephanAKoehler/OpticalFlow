#!/usr/bin/env python3
# File: scripts/optimize_weights.py
"""
Sequence-Level Weight Optimization

Optimizes metric weights across ALL frame pairs in a movie sequence.
Finds a single weight vector that minimizes mean EPE across the sequence.

Supports two loss formulations:
- "additive": Legacy weighted sum (w_photometric * photometric + ...)
- "multiplicative": New hybrid with photometric gates and depth-scaled stability

Usage:
    # Additive (legacy)
    python scripts/optimize_weights.py config.toml --movie-hash abc123 --of-hash def456
    
    # Multiplicative (new)
    python scripts/optimize_weights.py config.toml --movie-hash abc123 --of-hash def456 --loss multiplicative
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
    compute_multiplicative_loss,
)
from src.ensemble.loss import get_loss_params, LOSS_FUNCTIONS


# =============================================================================
# Configuration
# =============================================================================

# Additive loss methods (legacy)
ADDITIVE_METHODS = {
    'raw_sum': {'normalize': 'raw', 'aggregation': 'sum'},
    'raw_max': {'normalize': 'raw', 'aggregation': 'max'},
    'mad_sum': {'normalize': 'mad', 'aggregation': 'sum'},
    'mad_max': {'normalize': 'mad', 'aggregation': 'max'},
}

# Multiplicative loss is a single method (no variants)
MULTIPLICATIVE_METHOD = 'multiplicative'

# All c_* parameters for multiplicative loss
MULTIPLICATIVE_PARAMS = [
    'c_gray', 'c_r', 'c_g', 'c_b', 'c_log',
    'c_traction', 'c_consistency', 'c_perturbation'
]


def compute_optimization_hash(config: dict) -> str:
    """Compute hash from optimization config."""
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
        results_full, u_truth, v_truth, valid_mask, H, W, frame_constants
    """
    results_path = pair_dir / 'results_full.pkl'
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    with open(results_path, 'rb') as f:
        results_full = pickle.load(f)
    
    # Load frame_constants if available
    frame_constants_path = pair_dir / 'frame_constants.json'
    if frame_constants_path.exists():
        with open(frame_constants_path, 'r') as f:
            frame_constants = json.load(f)
    else:
        frame_constants = None
    
    return {
        'results_full': results_full,
        'u_truth': pair.u_truth,
        'v_truth': pair.v_truth,
        'valid_mask': pair.valid_mask,
        'H': pair.metadata['H'],
        'W': pair.metadata['W'],
        'frame_constants': frame_constants,
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

def precompute_pair_additive(
    pair_data: dict,
    enabled_metrics: list[str],
    normalize_method: str
) -> dict:
    """
    Precompute normalized metric stacks for ADDITIVE loss.
    
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


def precompute_pair_multiplicative(pair_data: dict) -> dict:
    """
    Precompute data for MULTIPLICATIVE loss.
    
    Args:
        pair_data: Dict with results_full, frame_constants, etc.
        
    Returns:
        Dict with data ready for multiplicative loss computation
    """
    results_full = pair_data['results_full']
    
    # Build flow stacks
    u_stack, v_stack = build_flow_stacks(results_full)
    
    return {
        'results_full': results_full,
        'frame_constants': pair_data['frame_constants'],
        'u_stack': u_stack,
        'v_stack': v_stack,
        'u_truth': pair_data['u_truth'],
        'v_truth': pair_data['v_truth'],
        'valid_mask': pair_data['valid_mask'],
        'H': pair_data['H'],
        'W': pair_data['W'],
    }


def compute_epe_from_precomputed_additive(
    precomputed: dict,
    weights: dict,
    aggregation: str,
    epe_power: float
) -> dict:
    """
    Compute EPE stats from precomputed data (ADDITIVE loss).
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


def compute_epe_from_precomputed_multiplicative(
    precomputed: dict,
    params: dict,
    epe_power: float
) -> dict:
    """
    Compute EPE stats from precomputed data (MULTIPLICATIVE loss).
    """
    # Compute multiplicative loss
    loss_stack = compute_multiplicative_loss(
        precomputed['results_full'],
        params,
        precomputed['frame_constants']
    )
    
    # Select best config per pixel
    selection = select_ensemble(loss_stack)
    
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
# Optuna Optimization - ADDITIVE
# =============================================================================

def create_objective_additive(
    all_precomputed: list[dict],
    optimize_metrics: list[str],
    fixed_weights: dict[str, float],
    aggregation: str,
    epe_power: float
):
    """
    Create Optuna objective function for ADDITIVE loss.
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
            stats = compute_epe_from_precomputed_additive(
                precomputed, all_weights, aggregation, epe_power
            )
            pair_epes.append(stats['mean'])
        
        return float(np.mean(pair_epes))
    
    return objective


# =============================================================================
# Optuna Optimization - MULTIPLICATIVE
# =============================================================================

def create_objective_multiplicative(
    all_precomputed: list[dict],
    optimize_params: list[str],
    fixed_params: dict[str, float],
    epe_power: float
):
    """
    Create Optuna objective function for MULTIPLICATIVE loss.
    
    All c_* parameters are in [0, 1].
    """
    def objective(trial: optuna.Trial) -> float:
        # Sample c_* parameters in [0, 1]
        params = {}
        for param in optimize_params:
            params[param] = trial.suggest_float(param, 0.0, 1.0)
        
        # Merge with fixed params
        all_params = {**fixed_params, **params}
        
        # Compute mean EPE across all pairs
        pair_epes = []
        for precomputed in all_precomputed:
            stats = compute_epe_from_precomputed_multiplicative(
                precomputed, all_params, epe_power
            )
            pair_epes.append(stats['mean'])
        
        return float(np.mean(pair_epes))
    
    return objective


# =============================================================================
# Run Optimization
# =============================================================================

def run_optimization_additive(
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
    Run Optuna optimization for ADDITIVE loss.
    
    Uses SQLite storage for persistence.
    """
    storage = None
    study_name = f"optimize_{method_name}"
    
    if output_dir is not None:
        db_path = output_dir / "optuna_studies.db"
        storage = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True
    )
    
    n_existing = len(study.trials)
    
    # Enqueue initial trial only if study is fresh
    if n_existing == 0 and initial_weights:
        init_normalized = normalize_weights_to_sum_one(initial_weights)
        study.enqueue_trial(init_normalized)
    
    objective = create_objective_additive(
        all_precomputed,
        optimize_metrics,
        fixed_weights,
        aggregation,
        epe_power
    )
    
    n_new_trials = max(0, n_trials - n_existing)
    
    if n_new_trials > 0:
        if show_progress:
            study.optimize(objective, n_trials=n_new_trials, show_progress_bar=True)
        else:
            with tqdm(total=n_new_trials, desc=f"   {method_name}", ncols=80) as pbar:
                def callback(study, trial):
                    pbar.update(1)
                study.optimize(objective, n_trials=n_new_trials, callbacks=[callback])
    
    # Get best result
    best_trial = study.best_trial
    raw_weights = {k: v for k, v in best_trial.params.items()}
    total = sum(raw_weights.values())
    best_weights = {k: v / total for k, v in raw_weights.items()}
    
    # Merge with fixed weights
    all_weights = {**fixed_weights, **best_weights}
    
    return {
        'best_weights': all_weights,
        'best_epe': best_trial.value,
        'n_trials_total': len(study.trials),
        'n_trials_new': n_new_trials,
        'study_name': study_name,
    }


def run_optimization_multiplicative(
    all_precomputed: list[dict],
    optimize_params: list[str],
    fixed_params: dict[str, float],
    initial_params: dict[str, float],
    n_trials: int,
    epe_power: float,
    output_dir: Path = None,
    show_progress: bool = True
) -> dict:
    """
    Run Optuna optimization for MULTIPLICATIVE loss.
    
    Uses SQLite storage for persistence.
    """
    method_name = 'multiplicative'
    storage = None
    study_name = f"optimize_{method_name}"
    
    if output_dir is not None:
        db_path = output_dir / "optuna_studies.db"
        storage = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True
    )
    
    n_existing = len(study.trials)
    
    # Enqueue initial trial only if study is fresh
    if n_existing == 0 and initial_params:
        study.enqueue_trial(initial_params)
    
    objective = create_objective_multiplicative(
        all_precomputed,
        optimize_params,
        fixed_params,
        epe_power
    )
    
    n_new_trials = max(0, n_trials - n_existing)
    
    if n_new_trials > 0:
        if show_progress:
            study.optimize(objective, n_trials=n_new_trials, show_progress_bar=True)
        else:
            with tqdm(total=n_new_trials, desc=f"   {method_name}", ncols=80) as pbar:
                def callback(study, trial):
                    pbar.update(1)
                study.optimize(objective, n_trials=n_new_trials, callbacks=[callback])
    
    # Get best result
    best_trial = study.best_trial
    best_params = {k: v for k, v in best_trial.params.items()}
    
    # Merge with fixed params
    all_params = {**fixed_params, **best_params}
    
    return {
        'best_params': all_params,
        'best_epe': best_trial.value,
        'n_trials_total': len(study.trials),
        'n_trials_new': n_new_trials,
        'study_name': study_name,
    }


# =============================================================================
# Output Management
# =============================================================================

def reset_optimization(output_dir: Path):
    """Delete previous optimization results."""
    if output_dir.exists():
        db_path = output_dir / "optuna_studies.db"
        if db_path.exists():
            db_path.unlink()
            print(f"   🗑️  Deleted {db_path}")
        
        summary_path = output_dir / "optimization_summary.json"
        if summary_path.exists():
            summary_path.unlink()
            print(f"   🗑️  Deleted {summary_path}")


def save_optimization_config(output_dir: Path, config: dict, opt_hash: str):
    """Save optimization config for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / "config.toml"
    with open(config_path, 'wb') as f:
        tomli_w.dump(config, f)


# =============================================================================
# Main Entry Points
# =============================================================================

def run_sequence_optimization(
    config: dict,
    movie_hash: str,
    of_hash: str,
    data_dir: str = "data",
    n_trials: int = None,
    reset: bool = False
) -> dict:
    """
    Run sequence-level weight optimization.
    
    Supports both additive and multiplicative loss functions based on config.
    """
    # Parse config
    opt_config = config.get('optimization', {})
    eval_config = config.get('evaluation', {})
    
    # Determine loss function
    loss_function = opt_config.get('loss_function', 'additive')
    
    epe_power = eval_config.get('epe_power', 2.0)
    
    if n_trials is None:
        n_trials = opt_config.get('n_trials', 100)
    
    # Setup paths
    data_path = Path(data_dir)
    analysis_dir = data_path / movie_hash / 'analysis' / of_hash
    
    opt_hash = compute_optimization_hash(config)
    output_dir = analysis_dir / 'optimization' / opt_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    save_optimization_config(output_dir, config, opt_hash)
    
    # Handle reset
    if reset:
        reset_optimization(output_dir)
    
    print("=" * 80)
    print("🔍 SEQUENCE WEIGHT OPTIMIZATION")
    print("=" * 80)
    print(f"Movie hash: {movie_hash}")
    print(f"OF hash: {of_hash}")
    print(f"Opt hash: {opt_hash}")
    print(f"Loss function: {loss_function.upper()}")
    print(f"Output: {output_dir}")
    print()
    
    # Load movie sequence
    print("📂 Loading movie and sweep results...")
    movie = load_movie_sequence(data_path / movie_hash)
    print(f"   Pairs: {len(movie.pairs)}")
    
    # Check for ground truth
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
    
    # Load all pair data
    all_pair_data = []
    for pair_idx, pair_dir in enumerate(pair_dirs):
        pair = movie.pairs[pair_idx]
        pair_data = load_pair_data(pair_dir, pair, epe_power)
        all_pair_data.append(pair_data)
    
    # Branch based on loss function
    if loss_function == 'multiplicative':
        return _run_multiplicative_optimization(
            opt_config, all_pair_data, n_trials, epe_power, output_dir, opt_hash
        )
    else:
        return _run_additive_optimization(
            opt_config, all_pair_data, n_trials, epe_power, output_dir, opt_hash
        )


def _run_additive_optimization(
    opt_config: dict,
    all_pair_data: list,
    n_trials: int,
    epe_power: float,
    output_dir: Path,
    opt_hash: str
) -> dict:
    """Run ADDITIVE loss optimization."""
    
    # Get methods to run
    methods = opt_config.get('methods', ['raw_sum', 'mad_sum'])
    
    # Get weights
    optimize_weights = opt_config.get('weights', {})
    fixed_weights = opt_config.get('fixed_weights', {})
    
    # Validate
    try:
        validate_weight_config(optimize_weights, fixed_weights)
    except AssertionError as e:
        print(f"⚠️  Weight validation warning: {e}")
    
    # Determine which metrics to optimize
    optimize_metrics = [k for k, v in optimize_weights.items() if v > 0]
    initial_weights = {k: v for k, v in optimize_weights.items() if v > 0}
    
    # Add zero-weight metrics to fixed
    for k, v in optimize_weights.items():
        if v == 0:
            fixed_weights[k] = 0.0
    
    # Get enabled metrics
    all_weights = {**fixed_weights, **initial_weights}
    enabled_metrics = [k for k, v in all_weights.items() if v != 0]
    
    if not enabled_metrics:
        print("❌ ERROR: No metrics enabled (all weights are 0)")
        sys.exit(1)
    
    print("📋 Additive optimization configuration:")
    print(f"   Methods: {methods}")
    print(f"   Trials: {n_trials}")
    print(f"   EPE power: {epe_power}")
    print(f"   Optimize metrics: {optimize_metrics}")
    print(f"   Fixed weights: {fixed_weights}")
    print()
    
    # Run optimization for each method
    print("=" * 80)
    print(f"🔄 RUNNING {len(methods)} OPTIMIZATIONS")
    print("=" * 80)
    print()
    
    results = {}
    
    for i, method_name in enumerate(methods):
        method_config = ADDITIVE_METHODS[method_name]
        normalize_method = method_config['normalize']
        aggregation = method_config['aggregation']
        
        print(f"[{i+1}/{len(methods)}] {method_name}")
        print("   Precomputing metric stacks...")
        
        all_precomputed = []
        for pair_data in all_pair_data:
            precomputed = precompute_pair_additive(pair_data, enabled_metrics, normalize_method)
            all_precomputed.append(precomputed)
        
        print(f"   Running {n_trials} trials...")
        
        result = run_optimization_additive(
            method_name=method_name,
            all_precomputed=all_precomputed,
            optimize_metrics=optimize_metrics,
            fixed_weights=fixed_weights,
            initial_weights=initial_weights,
            aggregation=aggregation,
            n_trials=n_trials,
            epe_power=epe_power,
            output_dir=output_dir,
            show_progress=False
        )
        
        results[method_name] = result
        print(f"   ✅ Best EPE^{epe_power}: {result['best_epe']:.6f}")
        print()
    
    # Save summary
    _save_summary(results, output_dir, opt_hash, 'additive', epe_power)
    
    return results


def _run_multiplicative_optimization(
    opt_config: dict,
    all_pair_data: list,
    n_trials: int,
    epe_power: float,
    output_dir: Path,
    opt_hash: str
) -> dict:
    """Run MULTIPLICATIVE loss optimization."""
    
    # Check frame_constants availability
    for i, pair_data in enumerate(all_pair_data):
        if pair_data['frame_constants'] is None:
            print(f"❌ ERROR: frame_constants.json not found for pair {i}")
            print("   Re-run sweep with updated sweep.py to generate frame_constants")
            sys.exit(1)
    
    # Get c_* parameters to optimize
    optimize_params_config = opt_config.get('multiplicative_params', {})
    fixed_params = opt_config.get('multiplicative_fixed', {})
    
    # Default: optimize all c_* params starting at 0.5
    if not optimize_params_config:
        optimize_params_config = {p: 0.5 for p in MULTIPLICATIVE_PARAMS}
    
    # Determine which to optimize (non-zero initial)
    optimize_params = [k for k, v in optimize_params_config.items() if v > 0]
    initial_params = {k: v for k, v in optimize_params_config.items() if v > 0}
    
    # Add zero params to fixed
    for k, v in optimize_params_config.items():
        if v == 0:
            fixed_params[k] = 0.0
    
    print("📋 Multiplicative optimization configuration:")
    print(f"   Trials: {n_trials}")
    print(f"   EPE power: {epe_power}")
    print(f"   Optimize params: {optimize_params}")
    print(f"   Fixed params: {fixed_params}")
    print()
    
    # Precompute for all pairs
    print("   Precomputing data...")
    all_precomputed = []
    for pair_data in all_pair_data:
        precomputed = precompute_pair_multiplicative(pair_data)
        all_precomputed.append(precomputed)
    
    print("=" * 80)
    print("🔄 RUNNING MULTIPLICATIVE OPTIMIZATION")
    print("=" * 80)
    print()
    
    print(f"   Running {n_trials} trials...")
    
    result = run_optimization_multiplicative(
        all_precomputed=all_precomputed,
        optimize_params=optimize_params,
        fixed_params=fixed_params,
        initial_params=initial_params,
        n_trials=n_trials,
        epe_power=epe_power,
        output_dir=output_dir,
        show_progress=False
    )
    
    print(f"   ✅ Best EPE^{epe_power}: {result['best_epe']:.6f}")
    print()
    
    # Format params for display
    print("📊 Best parameters:")
    for p in MULTIPLICATIVE_PARAMS:
        val = result['best_params'].get(p, 0)
        print(f"   {p}: {val:.4f}")
    print()
    
    results = {'multiplicative': result}
    
    # Save summary
    _save_summary(results, output_dir, opt_hash, 'multiplicative', epe_power)
    
    return results


def _save_summary(results: dict, output_dir: Path, opt_hash: str, 
                  loss_function: str, epe_power: float):
    """Save optimization summary to JSON."""
    summary = {
        'opt_hash': opt_hash,
        'loss_function': loss_function,
        'epe_power': epe_power,
        'results': {}
    }
    
    for method, result in results.items():
        summary['results'][method] = {
            'best_epe': result['best_epe'],
            'n_trials': result['n_trials_total'],
        }
        if 'best_weights' in result:
            summary['results'][method]['best_weights'] = result['best_weights']
        if 'best_params' in result:
            summary['results'][method]['best_params'] = result['best_params']
    
    summary_path = output_dir / 'optimization_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Saved summary to {summary_path}")


# Alias for backward compatibility
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
    parser.add_argument('--loss', type=str, choices=['additive', 'multiplicative'],
                        default=None, help='Override loss function from config')
    parser.add_argument('--reset', action='store_true', help='Reset optimization')
    
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
    
    # Override loss function if specified
    if args.loss:
        if 'optimization' not in config:
            config['optimization'] = {}
        config['optimization']['loss_function'] = args.loss
    
    run_sequence_optimization(
        config=config,
        movie_hash=args.movie_hash,
        of_hash=args.of_hash,
        data_dir=args.data_dir,
        n_trials=args.trials,
        reset=args.reset
    )


if __name__ == '__main__':
    main()
