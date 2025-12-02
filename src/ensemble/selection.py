# File: src/ensemble/selection.py
"""
Unified ensemble selection module.

Single source of truth for:
- Building metric stacks from results
- Normalizing metrics (raw, MAD)
- Computing penalties (sum, max aggregation)
- Selecting configs (pixel-wise, config-wise)
- Computing EPE statistics

Used by both optimize_weights.py and ranking_comparison.py.
"""

import numpy as np
from typing import Optional

# =============================================================================
# CONSTANTS
# =============================================================================

# Mapping from weight name to key in results['metrics']
METRIC_KEY_MAP = {
    'traction': 'traction_A',
    'perturbation_rms': 'displacements_sensitivity_A2B',
    'consistency': 'consistency_A',
    'photometric': 'photometric_A',
    'photometric_rgb': 'photometric_rgb_A',
    'photometric_rgb_log': 'photometric_rgb_log_A',
    'speed_sym': 'speed_sym_A',
}

# All known metric names
ALL_METRICS = list(METRIC_KEY_MAP.keys())

# Gain metrics: higher = better (need inversion to "lower = better")
GAIN_METRICS = {'speed_sym'}


# =============================================================================
# METRIC STACK BUILDING
# =============================================================================

def build_metric_stacks(
    results_full: list,
    enabled_metrics: list[str]
) -> dict[str, np.ndarray]:
    """
    Load metric stacks from results.
    
    Args:
        results_full: List of result dicts, each with 'metrics' key
        enabled_metrics: Which metrics to include (e.g., ['photometric', 'photometric_rgb'])
    
    Returns:
        {metric_name: ndarray (n_configs, H, W)}
        
    Notes:
        - Only loads metrics in enabled_metrics
        - Automatically inverts gain metrics (speed_sym → 1 - val/max)
        - Raises error if requested metric not available in results
    """
    if not results_full:
        raise ValueError("results_full is empty")
    
    n_configs = len(results_full)
    
    # Get shape from first result
    first_metrics = results_full[0]['metrics']
    sample_key = METRIC_KEY_MAP[enabled_metrics[0]]
    if sample_key not in first_metrics:
        # Try to find any available metric for shape
        for m in enabled_metrics:
            if METRIC_KEY_MAP[m] in first_metrics:
                sample_key = METRIC_KEY_MAP[m]
                break
        else:
            raise ValueError(f"No enabled metrics found in results")
    
    H, W = first_metrics[sample_key].shape
    
    stacks = {}
    
    for metric_name in enabled_metrics:
        metric_key = METRIC_KEY_MAP[metric_name]
        
        # Check availability
        if metric_key not in first_metrics:
            raise ValueError(f"Metric '{metric_name}' (key: {metric_key}) not found in results")
        
        # Build stack
        stack = np.zeros((n_configs, H, W), dtype=np.float32)
        for i, r in enumerate(results_full):
            stack[i] = r['metrics'][metric_key]
        
        # Invert gain metrics: higher value → lower penalty
        if metric_name in GAIN_METRICS:
            max_val = np.max(stack, axis=0, keepdims=True)
            eps = 1e-6
            stack = 1.0 - stack / (max_val + eps)
        
        stacks[metric_name] = stack
    
    return stacks


def compute_config_means(stacks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Compute per-config mean for each metric.
    
    Args:
        stacks: {metric_name: (n_configs, H, W)}
        
    Returns:
        {metric_name: (n_configs,)} mean values
    """
    return {
        name: np.nanmean(stack, axis=(1, 2))
        for name, stack in stacks.items()
    }


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_stacks(
    stacks: dict[str, np.ndarray],
    method: str,
    config_means: Optional[dict[str, np.ndarray]] = None
) -> dict[str, np.ndarray]:
    """
    Apply normalization to metric stacks.
    
    Args:
        stacks: {metric_name: (n_configs, H, W)}
        method: 'raw' (no change) or 'mad' (divide by MAD)
        config_means: Optional precomputed means for MAD calculation.
                      If None and method='mad', will compute internally.
        
    Returns:
        New dict with normalized stacks (does not modify input)
    """
    if method == 'raw':
        return stacks.copy()
    
    if method != 'mad':
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Compute config means if not provided
    if config_means is None:
        config_means = compute_config_means(stacks)
    
    normalized = {}
    for name, stack in stacks.items():
        means = config_means[name]
        mad = np.median(np.abs(means - np.median(means))) + 1e-10
        normalized[name] = stack / mad
    
    return normalized


# =============================================================================
# PENALTY COMPUTATION
# =============================================================================

def compute_penalty(
    stacks: dict[str, np.ndarray],
    weights: dict[str, float],
    aggregation: str
) -> np.ndarray:
    """
    Compute weighted penalty stack (pixel-wise).
    
    Args:
        stacks: Normalized metric stacks {name: (n_configs, H, W)}
        weights: {metric_name: float} - only non-zero weights used
        aggregation: 'sum' or 'max'
        
    Returns:
        penalty_stack: (n_configs, H, W)
    """
    # Get shape from first stack
    first_stack = next(iter(stacks.values()))
    n_configs, H, W = first_stack.shape
    
    if aggregation == 'sum':
        penalty = np.zeros((n_configs, H, W), dtype=np.float32)
        for name, stack in stacks.items():
            w = weights.get(name, 0.0)
            if w != 0:
                penalty += w * stack
    
    elif aggregation == 'max':
        penalty = np.full((n_configs, H, W), -np.inf, dtype=np.float32)
        for name, stack in stacks.items():
            w = weights.get(name, 0.0)
            if w != 0:
                penalty = np.maximum(penalty, w * stack)
        # Handle case where all weights are 0
        penalty = np.where(np.isinf(penalty), 0, penalty)
    
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return penalty


def compute_config_penalty(
    stacks: dict[str, np.ndarray],
    weights: dict[str, float],
    aggregation: str
) -> np.ndarray:
    """
    Compute per-config penalty (mean across pixels).
    
    Args:
        stacks: {name: (n_configs, H, W)}
        weights: {metric_name: float}
        aggregation: 'sum' or 'max'
        
    Returns:
        penalty_vec: (n_configs,)
    """
    # Get config means first
    config_means = compute_config_means(stacks)
    
    n_configs = next(iter(stacks.values())).shape[0]
    
    if aggregation == 'sum':
        penalty = np.zeros(n_configs, dtype=np.float32)
        for name in stacks.keys():
            w = weights.get(name, 0.0)
            if w != 0:
                penalty += w * config_means[name]
    
    elif aggregation == 'max':
        penalty = np.full(n_configs, -np.inf, dtype=np.float32)
        for name in stacks.keys():
            w = weights.get(name, 0.0)
            if w != 0:
                penalty = np.maximum(penalty, w * config_means[name])
        penalty = np.where(np.isinf(penalty), 0, penalty)
    
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return penalty


# =============================================================================
# SELECTION
# =============================================================================

def select_ensemble(penalty_stack: np.ndarray) -> np.ndarray:
    """
    Select best config per pixel (lowest penalty).
    
    Args:
        penalty_stack: (n_configs, H, W)
        
    Returns:
        selection: (H, W) with config indices
    """
    return np.argmin(penalty_stack, axis=0)


def select_best_config(penalty_vec: np.ndarray) -> int:
    """
    Select best config overall (lowest penalty).
    
    Args:
        penalty_vec: (n_configs,)
        
    Returns:
        Best config index
    """
    return int(np.argmin(penalty_vec))


# =============================================================================
# FLOW GATHERING
# =============================================================================

def gather_flow(
    u_stack: np.ndarray,
    v_stack: np.ndarray,
    selection: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gather flow values from selected configs.
    
    Args:
        u_stack: (n_configs, H, W)
        v_stack: (n_configs, H, W)
        selection: (H, W) config indices
        
    Returns:
        (u_ensemble, v_ensemble): each (H, W)
    """
    H, W = selection.shape
    row_idx = np.arange(H)[:, None]
    col_idx = np.arange(W)[None, :]
    
    u_ensemble = u_stack[selection, row_idx, col_idx]
    v_ensemble = v_stack[selection, row_idx, col_idx]
    
    return u_ensemble, v_ensemble


# =============================================================================
# EPE COMPUTATION
# =============================================================================

def compute_epe_stats(
    u: np.ndarray,
    v: np.ndarray,
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    valid_mask: np.ndarray,
    power: float = 2.0
) -> dict[str, float]:
    """
    Compute EPE statistics over valid pixels.
    
    Args:
        u, v: Estimated flow (H, W)
        u_truth, v_truth: Ground truth flow (H, W)
        valid_mask: Boolean mask (H, W)
        power: EPE power (default 2.0 for MSE)
        
    Returns:
        {'mean': float, 'std': float, 'median': float}
    """
    epe = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
    epe_powered = epe[valid_mask] ** power
    
    return {
        'mean': float(np.mean(epe_powered)),
        'std': float(np.std(epe_powered)),
        'median': float(np.median(epe_powered)),
    }


# =============================================================================
# HIGH-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def compute_ensemble_epe(
    results_full: list,
    u_stack: np.ndarray,
    v_stack: np.ndarray,
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    valid_mask: np.ndarray,
    weights: dict[str, float],
    normalize: str = 'raw',
    aggregation: str = 'sum',
    epe_power: float = 2.0
) -> dict[str, float]:
    """
    End-to-end ensemble EPE computation.
    
    Args:
        results_full: List of result dicts with metrics
        u_stack, v_stack: Flow stacks (n_configs, H, W)
        u_truth, v_truth: Ground truth flow (H, W)
        valid_mask: Boolean mask (H, W)
        weights: {metric_name: float} - determines which metrics to use
        normalize: 'raw' or 'mad'
        aggregation: 'sum' or 'max'
        epe_power: Power for EPE (default 2.0)
        
    Returns:
        {'mean': float, 'std': float, 'median': float}
    """
    # Determine enabled metrics from non-zero weights
    enabled_metrics = [name for name, w in weights.items() if w != 0]
    
    if not enabled_metrics:
        raise ValueError("No metrics enabled (all weights are 0)")
    
    # Build and normalize stacks
    stacks = build_metric_stacks(results_full, enabled_metrics)
    stacks = normalize_stacks(stacks, normalize)
    
    # Compute penalty and select
    penalty = compute_penalty(stacks, weights, aggregation)
    selection = select_ensemble(penalty)
    
    # Gather flow and compute EPE
    u_ens, v_ens = gather_flow(u_stack, v_stack, selection)
    return compute_epe_stats(u_ens, v_ens, u_truth, v_truth, valid_mask, epe_power)


def rank_configs_by_metric(
    results_full: list,
    metric_name: str
) -> np.ndarray:
    """
    Rank configs by a single metric (for SINGLE CONFIG EPE table).
    
    Args:
        results_full: List of result dicts
        metric_name: Which metric to rank by
        
    Returns:
        Sorted config indices (best first)
    """
    stacks = build_metric_stacks(results_full, [metric_name])
    means = compute_config_means(stacks)[metric_name]
    return np.argsort(means)


# =============================================================================
# WEIGHT VALIDATION
# =============================================================================

def validate_weight_config(
    optimize_weights: dict[str, float],
    fixed_weights: dict[str, float]
) -> None:
    """
    Validate that all metrics are accounted for in weight config.
    
    Args:
        optimize_weights: Weights to optimize (from [optimization.weights])
        fixed_weights: Fixed weights (from [optimization.fixed_weights])
        
    Raises:
        AssertionError if any metric is missing or duplicated
    """
    optimize_set = set(optimize_weights.keys())
    fixed_set = set(fixed_weights.keys())
    all_metrics_set = set(ALL_METRICS)
    
    # Check for overlap
    overlap = optimize_set & fixed_set
    assert not overlap, f"Metrics in both optimize and fixed: {overlap}"
    
    # Check all metrics accounted for
    combined = optimize_set | fixed_set
    missing = all_metrics_set - combined
    assert not missing, f"Metrics not in optimize or fixed: {missing}"
    
    # Check for unknown metrics
    extra = combined - all_metrics_set
    assert not extra, f"Unknown metrics in config: {extra}"


def get_enabled_metrics(weights: dict[str, float]) -> list[str]:
    """
    Get list of metrics with non-zero weights.
    
    Args:
        weights: {metric_name: float}
        
    Returns:
        List of metric names with weight != 0
    """
    return [name for name, w in weights.items() if w != 0]


def normalize_weights_to_sum_one(weights: dict[str, float]) -> dict[str, float]:
    """
    Normalize weights so they sum to 1.
    
    Args:
        weights: {metric_name: float} (all non-negative)
        
    Returns:
        Normalized weights summing to 1
    """
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Cannot normalize: sum of weights is <= 0")
    return {k: v / total for k, v in weights.items()}


if __name__ == "__main__":
    # Quick sanity check
    print("Selection module loaded successfully")
    print(f"Known metrics: {ALL_METRICS}")
    print(f"Gain metrics (inverted): {GAIN_METRICS}")
