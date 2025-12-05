# File: src/ensemble/selection.py
"""
Unified ensemble selection module.

Single source of truth for:
- Building metric stacks from results
- Normalizing metrics (raw, MAD)
- Computing penalties (additive: sum/max, multiplicative: new loss)
- Selecting configs (pixel-wise, config-wise)
- Computing EPE statistics

Used by both optimize_weights.py and ranking_comparison.py.

Supports two loss formulations:
- "additive": Legacy weighted sum (w_photometric * photometric + w_traction * traction + ...)
- "multiplicative": New hybrid (gates × depth-scaled stability)
"""

import numpy as np
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ensemble.loss import get_loss_function, get_loss_params, LOSS_FUNCTIONS

# =============================================================================
# CONSTANTS
# =============================================================================

# Mapping from weight name to key in results['metrics'] (BOUNDED metrics)
METRIC_KEY_MAP = {
    'traction': 'traction_A',
    'perturbation_rms': 'displacements_sensitivity_A2B',
    'consistency': 'consistency_A',
    'photometric': 'photometric_A',
    'photometric_rgb': 'photometric_rgb_A',
    'photometric_rgb_log': 'photometric_rgb_log_A',
    'speed_sym': 'speed_sym_A',
}

# Mapping from loss param to key in results['metrics'] (RAW metrics)
RAW_METRIC_KEY_MAP = {
    # Photometric (raw intensity errors)
    'photo_gray_raw': 'photo_gray_raw_A',
    'photo_r_raw': 'photo_r_raw_A',
    'photo_g_raw': 'photo_g_raw_A',
    'photo_b_raw': 'photo_b_raw_A',
    'photo_log_raw': 'photo_log_raw_A',
    # Stability (raw pixel errors)
    'traction_raw': 'traction_raw_A',
    'consistency_raw': 'consistency_raw_A',
    'perturbation_raw': 'perturbation_raw_A',
}

# All known metric names (for legacy additive loss)
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
    Load metric stacks from results (BOUNDED metrics for legacy additive loss).
    
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
        key = METRIC_KEY_MAP[metric_name]
        
        # Check availability
        if key not in first_metrics:
            raise ValueError(f"Metric '{key}' not in results (requested '{metric_name}')")
        
        # Build stack
        stack = np.zeros((n_configs, H, W), dtype=np.float32)
        for i, r in enumerate(results_full):
            stack[i] = r['metrics'][key]
        
        # Invert gain metrics
        if metric_name in GAIN_METRICS:
            max_val = np.max(stack)
            if max_val > 0:
                stack = 1 - stack / max_val
        
        stacks[metric_name] = stack
    
    return stacks


def build_raw_metric_stacks(
    results_full: list,
    frame_constants: dict
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load RAW metric stacks for multiplicative loss.
    
    Args:
        results_full: List of result dicts with 'metrics' and 'params'
        frame_constants: {max_gray, max_r, max_g, max_b, max_log, perturbation_distance}
    
    Returns:
        (metrics_dict, depth_scale_stack)
        - metrics_dict: {metric_name: (n_configs, H, W)} for loss.py
        - depth_scale_stack: (n_configs,) pollution_depth + perturbation_distance per config
    """
    if not results_full:
        raise ValueError("results_full is empty")
    
    n_configs = len(results_full)
    first_metrics = results_full[0]['metrics']
    
    # Get shape from any available metric
    for key in RAW_METRIC_KEY_MAP.values():
        if key in first_metrics:
            H, W = first_metrics[key].shape
            break
    else:
        raise ValueError("No raw metrics found in results")
    
    # Build raw metric stacks
    raw_stacks = {}
    for loss_key, result_key in RAW_METRIC_KEY_MAP.items():
        if result_key in first_metrics:
            stack = np.zeros((n_configs, H, W), dtype=np.float32)
            for i, r in enumerate(results_full):
                if result_key in r['metrics']:
                    stack[i] = r['metrics'][result_key]
            raw_stacks[loss_key] = stack
    
    # Compute depth_scale per config
    pert_dist = frame_constants.get('perturbation_distance', 2.5)
    depth_scale_stack = np.zeros(n_configs, dtype=np.float32)
    
    for i, r in enumerate(results_full):
        # Parse winsize from config_name (e.g., 'FB_win9_lev4_...' -> 9)
        config_name = r['metadata'].get('config_name', '')
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        pollution_depth = winsize / 2  # Could use measured depth from cache
        depth_scale_stack[i] = pollution_depth + pert_dist
    
    return raw_stacks, depth_scale_stack


# =============================================================================
# NORMALIZATION
# =============================================================================

def compute_config_means(stacks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Compute per-config mean for each metric stack.
    
    Args:
        stacks: {metric_name: (n_configs, H, W)}
        
    Returns:
        {metric_name: (n_configs,)} mean values
    """
    return {name: np.nanmean(stack, axis=(1, 2)) for name, stack in stacks.items()}


def normalize_stacks(
    stacks: dict[str, np.ndarray],
    method: str,
    config_means: dict[str, np.ndarray] = None
) -> dict[str, np.ndarray]:
    """
    Normalize metric stacks.
    
    Args:
        stacks: {metric_name: (n_configs, H, W)}
        method: 'raw' (no normalization) or 'mad' (divide by MAD of config means)
        config_means: Optional precomputed means
        
    Returns:
        Normalized stacks (same shape)
    """
    if method == 'raw':
        return stacks
    
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
# PENALTY COMPUTATION (LEGACY ADDITIVE)
# =============================================================================

def compute_penalty(
    stacks: dict[str, np.ndarray],
    weights: dict[str, float],
    aggregation: str
) -> np.ndarray:
    """
    Compute weighted penalty stack (pixel-wise) - LEGACY ADDITIVE.
    
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
    Compute per-config penalty (mean across pixels) - LEGACY ADDITIVE.
    
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
# MULTIPLICATIVE LOSS COMPUTATION (NEW)
# =============================================================================

def compute_multiplicative_loss(
    results_full: list,
    params: dict[str, float],
    frame_constants: dict
) -> np.ndarray:
    """
    Compute multiplicative loss stack (pixel-wise).
    
    loss = (1 + c_gray * photo_gray/max_gray)
         × (1 + c_r * photo_r/max_r)
         × (1 + c_g * photo_g/max_g)
         × (1 + c_b * photo_b/max_b)
         × (1 + c_log * photo_log/max_log)
         × (1 + depth_scale * (c_traction * traction + c_consistency * consistency + c_perturbation * perturbation))
    
    Args:
        results_full: List of result dicts with 'metrics' and 'params'
        params: {c_gray, c_r, c_g, c_b, c_log, c_traction, c_consistency, c_perturbation}
        frame_constants: {max_gray, max_r, max_g, max_b, max_log, perturbation_distance}
    
    Returns:
        loss_stack: (n_configs, H, W)
    """
    n_configs = len(results_full)
    first_metrics = results_full[0]['metrics']
    
    # Get shape
    for key in RAW_METRIC_KEY_MAP.values():
        if key in first_metrics:
            H, W = first_metrics[key].shape
            break
    else:
        raise ValueError("No raw metrics found - run sweep with updated self_supervised.py")
    
    loss_stack = np.ones((n_configs, H, W), dtype=np.float32)
    
    # --- Photometric gates ---
    photo_map = [
        ('c_gray', 'photo_gray_raw_A', 'max_gray'),
        ('c_r', 'photo_r_raw_A', 'max_r'),
        ('c_g', 'photo_g_raw_A', 'max_g'),
        ('c_b', 'photo_b_raw_A', 'max_b'),
        ('c_log', 'photo_log_raw_A', 'max_log'),
    ]
    
    for c_name, metric_key, max_key in photo_map:
        c = params.get(c_name, 0)
        if c > 0 and max_key in frame_constants:
            max_val = frame_constants[max_key]
            if max_val > 0:
                for i, r in enumerate(results_full):
                    if metric_key in r['metrics']:
                        loss_stack[i] *= (1 + c * r['metrics'][metric_key] / max_val)
    
    # --- Stability term ---
    stability_map = [
        ('c_traction', 'traction_raw_A'),
        ('c_consistency', 'consistency_raw_A'),
        ('c_perturbation', 'perturbation_raw_A'),
    ]
    
    pert_dist = frame_constants.get('perturbation_distance', 2.5)
    
    for i, r in enumerate(results_full):
        # Compute depth_scale for this config (parse from config_name)
        config_name = r['metadata'].get('config_name', '')
        match = re.search(r'win(\d+)', config_name)
        winsize = int(match.group(1)) if match else 15
        pollution_depth = winsize / 2
        depth_scale = pollution_depth + pert_dist
        
        # Accumulate stability
        stability = np.zeros((H, W), dtype=np.float32)
        for c_name, metric_key in stability_map:
            c = params.get(c_name, 0)
            if c > 0 and metric_key in r['metrics']:
                stability += c * r['metrics'][metric_key]
        
        loss_stack[i] *= (1 + depth_scale * stability)
    
    return loss_stack


# =============================================================================
# SELECTION
# =============================================================================

def select_ensemble(penalty_stack: np.ndarray) -> np.ndarray:
    """
    Select best config per pixel (lowest penalty/loss).
    
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
    End-to-end ensemble EPE computation (LEGACY ADDITIVE).
    
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


def compute_ensemble_epe_multiplicative(
    results_full: list,
    u_stack: np.ndarray,
    v_stack: np.ndarray,
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    valid_mask: np.ndarray,
    params: dict[str, float],
    frame_constants: dict,
    epe_power: float = 2.0
) -> dict[str, float]:
    """
    End-to-end ensemble EPE computation (NEW MULTIPLICATIVE).
    
    Args:
        results_full: List of result dicts with metrics
        u_stack, v_stack: Flow stacks (n_configs, H, W)
        u_truth, v_truth: Ground truth flow (H, W)
        valid_mask: Boolean mask (H, W)
        params: {c_gray, c_r, c_g, c_b, c_log, c_traction, c_consistency, c_perturbation}
        frame_constants: {max_gray, max_r, max_g, max_b, max_log, perturbation_distance}
        epe_power: Power for EPE (default 2.0)
        
    Returns:
        {'mean': float, 'std': float, 'median': float}
    """
    # Compute multiplicative loss
    loss_stack = compute_multiplicative_loss(results_full, params, frame_constants)
    selection = select_ensemble(loss_stack)
    
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
    Validate that all metrics are accounted for in weight config (LEGACY ADDITIVE).
    
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


# =============================================================================
# UNIFIED LOSS INTERFACE
# =============================================================================

def compute_loss_stack(
    results_full: list,
    loss_function: str,
    params: dict[str, float],
    frame_constants: dict = None,
    normalize: str = 'raw',
    aggregation: str = 'sum'
) -> np.ndarray:
    """
    Unified interface for computing loss stacks.
    
    Args:
        results_full: List of result dicts
        loss_function: 'additive' or 'multiplicative'
        params: Loss parameters (weights for additive, c_* for multiplicative)
        frame_constants: Required for multiplicative loss
        normalize: For additive: 'raw' or 'mad'
        aggregation: For additive: 'sum' or 'max'
    
    Returns:
        loss_stack: (n_configs, H, W)
    """
    if loss_function == 'multiplicative':
        if frame_constants is None:
            raise ValueError("frame_constants required for multiplicative loss")
        return compute_multiplicative_loss(results_full, params, frame_constants)
    
    elif loss_function == 'additive':
        # Legacy additive
        enabled_metrics = [name for name, w in params.items() if w != 0 and name in METRIC_KEY_MAP]
        if not enabled_metrics:
            # Return uniform loss
            n_configs = len(results_full)
            H, W = results_full[0]['metrics']['photometric_A'].shape
            return np.ones((n_configs, H, W), dtype=np.float32)
        
        stacks = build_metric_stacks(results_full, enabled_metrics)
        stacks = normalize_stacks(stacks, normalize)
        return compute_penalty(stacks, params, aggregation)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")


if __name__ == "__main__":
    # Quick sanity check
    print("Selection module loaded successfully")
    print(f"Known metrics (additive): {ALL_METRICS}")
    print(f"Gain metrics (inverted): {GAIN_METRICS}")
    print(f"Available loss functions: {list(LOSS_FUNCTIONS.keys())}")
