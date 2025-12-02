# File: src/optimization/weight_optimizer.py
"""
Weight optimization using Optuna.

Learns optimal metric weights by minimizing EPE on ground truth data.
"""

import numpy as np
import sys
import json
from pathlib import Path
from typing import Optional, Callable

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("❌ ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)


# Metric name mapping from selection config to results_full keys
METRIC_KEY_MAP = {
    'traction': ('traction_A', 'traction_B'),
    'perturbation_rms': ('displacements_sensitivity_A2B', 'displacements_sensitivity_B2A'),
    'consistency': ('consistency_A', 'consistency_B'),
    'photometric': ('photometric_A', 'photometric_B'),
}

# Metrics to optimize (exclude traction by default since it's proven unreliable)
DEFAULT_METRIC_NAMES = ['perturbation_rms', 'consistency', 'photometric']


def extract_metrics_and_flows(results_full: list[dict]) -> tuple[list[dict], list[tuple]]:
    """
    Extract metrics and flows from results_full format.
    
    Args:
        results_full: List of config result dicts from sweep
        
    Returns:
        (all_metrics, all_flows) where:
            all_metrics: List of {metric_name: (H, W) array}
            all_flows: List of (u, v) tuples
    """
    all_metrics = []
    all_flows = []
    
    for result in results_full:
        # Extract flows
        u = result.get('u_AB') 
        v = result.get('v_AB')
        if u is None or v is None:
            # Try alternate keys
            flows = result.get('flows', {})
            u = flows.get('u_AB', flows.get('u'))
            v = flows.get('v_AB', flows.get('v'))
        
        if u is None or v is None:
            print(f"❌ ERROR: Could not find flow fields in result")
            print(f"   Keys: {list(result.keys())}")
            sys.exit(1)
        
        all_flows.append((u, v))
        
        # Extract metrics - average A and B variants
        metrics = {}
        for metric_name, (key_A, key_B) in METRIC_KEY_MAP.items():
            val_A = result.get(key_A)
            val_B = result.get(key_B)
            
            # Try nested 'metrics' dict
            if val_A is None:
                metrics_dict = result.get('metrics', {})
                val_A = metrics_dict.get(key_A)
                val_B = metrics_dict.get(key_B)
            
            if val_A is not None and val_B is not None:
                metrics[metric_name] = (val_A + val_B) / 2
            elif val_A is not None:
                metrics[metric_name] = val_A
            elif val_B is not None:
                metrics[metric_name] = val_B
            # else: metric not available, skip
        
        all_metrics.append(metrics)
    
    return all_metrics, all_flows


def normalize_metric_mad(
    metric_stack: np.ndarray,
    valid_mask: np.ndarray
) -> np.ndarray:
    """
    MAD-normalize a metric across all configs.
    
    Args:
        metric_stack: (n_configs, H, W) array
        valid_mask: (H, W) boolean mask
        
    Returns:
        Normalized metric stack, clipped to >= 0
    """
    # Flatten all valid values across configs
    n_configs = metric_stack.shape[0]
    all_valid = []
    for i in range(n_configs):
        all_valid.append(metric_stack[i][valid_mask])
    all_valid = np.concatenate(all_valid)
    
    median = np.median(all_valid)
    mad = np.median(np.abs(all_valid - median))
    
    if mad == 0:
        return np.zeros_like(metric_stack)
    
    normalized = (metric_stack - median) / mad
    return np.maximum(0, normalized)


def create_objective(
    all_metrics: list[dict],
    all_flows: list[tuple],
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    normalize: str,
    aggregation: str,
    power: float,
    valid_mask: np.ndarray,
    metric_names: list[str],
    optimized_metric_names: list[str],
    epe_power: float
) -> Callable:
    """
    Create Optuna objective function for weight optimization.
    
    Args:
        metric_names: All metrics to include in penalty (4 metrics)
        optimized_metric_names: Metrics to sample weights for (3 metrics, traction excluded)
    """
    n_configs = len(all_flows)
    H, W = u_truth.shape
    
    # Precompute normalized metrics
    # Stack each metric across configs
    normalized_stacks = {}
    
    for metric_name in metric_names:
        # Check if metric exists in at least one config
        has_metric = any(metric_name in m for m in all_metrics)
        if not has_metric:
            continue
        
        # Build stack
        stack = np.zeros((n_configs, H, W))
        for i, metrics in enumerate(all_metrics):
            if metric_name in metrics:
                stack[i] = metrics[metric_name]
            else:
                stack[i] = np.inf  # Missing metric gets worst score
        
        # Normalize if requested
        if normalize == "mad":
            stack = normalize_metric_mad(stack, valid_mask)
        
        normalized_stacks[metric_name] = stack
    
    # Precompute flow stack
    u_stack = np.stack([f[0] for f in all_flows], axis=0)
    v_stack = np.stack([f[1] for f in all_flows], axis=0)
    
    def objective(trial: optuna.Trial) -> float:
        # Sample weights for the 3 optimizable metrics only
        raw_weights = {}
        for metric_name in optimized_metric_names:
            if metric_name in normalized_stacks:
                raw_weights[metric_name] = trial.suggest_float(metric_name, 0.0, 1.0)
        
        # Add traction with fixed raw weight of 1.0 (will be normalized)
        # This means traction gets the "remainder" after normalization
        if 'traction' in normalized_stacks:
            raw_weights['traction'] = 1.0
        
        # Normalize all weights to sum to 1
        total = sum(raw_weights.values())
        if total == 0:
            return float('inf')
        weights = {k: v / total for k, v in raw_weights.items()}
        
        # Compute weighted penalty for each config
        penalty_stack = np.zeros((n_configs, H, W))
        
        for metric_name, weight in weights.items():
            if metric_name in normalized_stacks:
                weighted = weight * normalized_stacks[metric_name]
                if aggregation == "sum":
                    penalty_stack += weighted
                else:  # max - need different approach
                    penalty_stack = np.maximum(penalty_stack, weighted)
        
        # Per-pixel best config selection
        best_config_idx = np.argmin(penalty_stack, axis=0)
        
        # Build ensemble flow using advanced indexing
        row_idx = np.arange(H)[:, None]
        col_idx = np.arange(W)[None, :]
        u_ensemble = u_stack[best_config_idx, row_idx, col_idx]
        v_ensemble = v_stack[best_config_idx, row_idx, col_idx]
        
        # Compute EPE
        epe = np.sqrt((u_ensemble - u_truth)**2 + (v_ensemble - v_truth)**2)
        valid_epe = epe[valid_mask]
        
        # Return mean(EPE ** epe_power)
        return np.mean(valid_epe ** epe_power)
    
    return objective


def optimize_weights(
    results_full: list[dict],
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    valid_mask: np.ndarray,
    selection_template: dict,
    output_dir: Path,
    n_trials: int = 100,
    epe_power: float = 2.0,
    show_progress: bool = True,
    method_name: str = "optimization"
) -> dict:
    """
    Optimize metric weights using Optuna.
    
    Args:
        results_full: List of config result dicts from sweep
        u_truth, v_truth: Ground truth flow
        valid_mask: Boolean mask of valid pixels
        selection_template: Dict with 'normalize', 'aggregation', 'power'
        output_dir: Directory for output (study.db, best_weights.json)
        n_trials: Number of optimization trials
        epe_power: Power for EPE in loss function
        show_progress: Whether to show progress bar
        method_name: Name for logging
        
    Returns:
        Dict with best_weights, best_epe, best_selection_config, n_configs_used
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract template params
    normalize = selection_template['normalize']
    aggregation = selection_template['aggregation']
    power = selection_template['power']
    
    # Extract metrics and flows
    all_metrics, all_flows = extract_metrics_and_flows(results_full)
    n_configs = len(all_flows)
    
    # Determine which metrics are available
    available_metrics = set()
    for metrics in all_metrics:
        available_metrics.update(metrics.keys())
    
    # Use default metrics that are available for optimization (3 DOF)
    # Traction is always included but NOT optimized - it gets the remainder weight
    optimized_metric_names = [m for m in DEFAULT_METRIC_NAMES if m in available_metrics]
    
    if not optimized_metric_names:
        print(f"❌ ERROR: No metrics available for optimization")
        print(f"   Available: {available_metrics}")
        sys.exit(1)
    
    # All metrics for stacking (include traction)
    all_metric_names = optimized_metric_names.copy()
    if 'traction' in available_metrics and 'traction' not in all_metric_names:
        all_metric_names.append('traction')
    
    # Create objective
    objective = create_objective(
        all_metrics=all_metrics,
        all_flows=all_flows,
        u_truth=u_truth,
        v_truth=v_truth,
        normalize=normalize,
        aggregation=aggregation,
        power=power,
        valid_mask=valid_mask,
        metric_names=all_metric_names,  # All 4 for stacking
        optimized_metric_names=optimized_metric_names,  # Only 3 for sampling
        epe_power=epe_power
    )
    
    # Create study with SQLite storage
    storage_path = output_dir / "optuna_study.db"
    storage = f"sqlite:///{storage_path}"
    
    sampler = TPESampler(seed=42)
    
    # Suppress verbose Optuna logging - only show progress bar
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        study_name=method_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress,
        gc_after_trial=True  # Clean up memory
    )
    
    # Extract best weights (renormalized with traction)
    raw_best = study.best_params.copy()
    # Add traction with fixed raw weight 1.0 (same as in objective)
    if 'traction' in available_metrics:
        raw_best['traction'] = 1.0
    total = sum(raw_best.values())
    best_weights = {k: v / total for k, v in raw_best.items()}
    
    # Build best selection config (now includes all 4 weights)
    best_selection_config = {
        'normalize': normalize,
        'aggregation': aggregation,
        'power': power,
        'traction': best_weights.get('traction', 0.0),
        'perturbation_rms': best_weights.get('perturbation_rms', 0.0),
        'consistency': best_weights.get('consistency', 0.0),
        'photometric': best_weights.get('photometric', 0.0)
    }
    
    # Compute EPE std for best result
    # Re-run selection with best weights to get ensemble
    H, W = u_truth.shape
    
    # Rebuild penalty stack with best weights (all 4 metrics)
    normalized_stacks = {}
    for metric_name in all_metric_names:
        stack = np.zeros((n_configs, H, W))
        for i, metrics in enumerate(all_metrics):
            if metric_name in metrics:
                stack[i] = metrics[metric_name]
            else:
                stack[i] = np.inf
        if normalize == "mad":
            stack = normalize_metric_mad(stack, valid_mask)
        normalized_stacks[metric_name] = stack
    
    penalty_stack = np.zeros((n_configs, H, W))
    for metric_name, weight in best_weights.items():
        if metric_name in normalized_stacks:
            weighted = weight * normalized_stacks[metric_name]
            if aggregation == "sum":
                penalty_stack += weighted
            else:
                penalty_stack = np.maximum(penalty_stack, weighted)
    
    best_config_idx = np.argmin(penalty_stack, axis=0)
    n_configs_used = len(np.unique(best_config_idx[valid_mask]))
    
    # Build ensemble flow
    u_stack = np.stack([f[0] for f in all_flows], axis=0)
    v_stack = np.stack([f[1] for f in all_flows], axis=0)
    row_idx = np.arange(H)[:, None]
    col_idx = np.arange(W)[None, :]
    u_ensemble = u_stack[best_config_idx, row_idx, col_idx]
    v_ensemble = v_stack[best_config_idx, row_idx, col_idx]
    
    # Compute EPE stats
    epe = np.sqrt((u_ensemble - u_truth)**2 + (v_ensemble - v_truth)**2)
    epe_powered = epe ** epe_power
    best_epe = float(np.mean(epe_powered[valid_mask]))
    best_epe_std = float(np.std(epe_powered[valid_mask]))
    
    # Save results
    results = {
        'best_weights': best_weights,
        'best_epe': best_epe,
        'best_epe_std': best_epe_std,
        'best_selection_config': best_selection_config,
        'n_configs_used': n_configs_used,
        'n_trials': len(study.trials),
        'normalize': normalize,
        'aggregation': aggregation,
        'power': power,
        'epe_power': epe_power,
    }
    
    # Save best weights JSON
    with open(output_dir / 'best_weights.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save trials CSV
    trials_data = []
    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'value': trial.value,
            **trial.params
        }
        trials_data.append(trial_data)
    
    import csv
    with open(output_dir / 'trials.csv', 'w', newline='') as f:
        if trials_data:
            writer = csv.DictWriter(f, fieldnames=trials_data[0].keys())
            writer.writeheader()
            writer.writerows(trials_data)
    
    return results


if __name__ == "__main__":
    print("🧪 Testing weight optimizer")
    print("=" * 50)
    
    # Create synthetic test data matching results_full format
    np.random.seed(42)
    H, W = 50, 50
    n_configs = 5
    
    # Ground truth
    u_truth = np.ones((H, W)) * 2.0
    v_truth = np.ones((H, W)) * 1.0
    valid_mask = np.ones((H, W), dtype=bool)
    
    # Simulate results_full
    results_full = []
    for i in range(n_configs):
        noise = 0.3 + i * 0.2
        u = u_truth + np.random.randn(H, W) * noise
        v = v_truth + np.random.randn(H, W) * noise
        epe = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        
        result = {
            'u_AB': u,
            'v_AB': v,
            'displacements_sensitivity_A2B': epe * (0.8 + np.random.rand(H, W) * 0.4),
            'displacements_sensitivity_B2A': epe * (0.9 + np.random.rand(H, W) * 0.3),
            'consistency_A': epe * (0.6 + np.random.rand(H, W) * 0.4),
            'consistency_B': epe * (0.5 + np.random.rand(H, W) * 0.5),
            'photometric_A': epe * (1.0 + np.random.rand(H, W) * 0.3),
            'photometric_B': epe * (1.1 + np.random.rand(H, W) * 0.2),
            'traction_A': np.random.rand(H, W) * 2,
            'traction_B': np.random.rand(H, W) * 2,
        }
        results_full.append(result)
    
    # Test optimization
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        results = optimize_weights(
            results_full=results_full,
            u_truth=u_truth,
            v_truth=v_truth,
            valid_mask=valid_mask,
            selection_template={'normalize': 'mad', 'aggregation': 'sum', 'power': 2},
            output_dir=Path(tmpdir),
            n_trials=20,
            epe_power=2.0,
            show_progress=False,
            method_name="test"
        )
    
    print(f"\n✅ Optimization test passed!")
    print(f"   Best EPE^2: {results['best_epe']:.4f}")
    print(f"   Weights: {results['best_weights']}")
    print(f"   Configs used: {results['n_configs_used']}")
