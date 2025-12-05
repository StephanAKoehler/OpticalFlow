# File: src/ensemble/loss.py
"""
Loss functions for ensemble optical flow selection.

Each loss function computes per-pixel penalty scores for config ranking.
Lower loss = better config at that pixel.

Usage:
    from src.ensemble.loss import get_loss_function, get_loss_params
    
    loss_fn = get_loss_function("multiplicative")
    params = {"c_gray": 0.5, "c_traction": 0.1, ...}
    loss = loss_fn(metrics, params, constants)
"""

import numpy as np
import sys
from typing import Callable

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

# loss_fn(metrics, params, constants) -> (H, W) array
LossFn = Callable[[dict, dict, dict], np.ndarray]


# =============================================================================
# ADDITIVE LOSS (LEGACY)
# =============================================================================

ADDITIVE_PARAMS = [
    "w_photometric",
    "w_traction",
    "w_consistency",
    "w_perturbation",
]


def loss_additive(metrics: dict, params: dict, constants: dict) -> np.ndarray:
    """
    Legacy additive loss: weighted sum of bounded metrics.
    
    loss = Σ w_i * metric_i
    
    Args:
        metrics: {metric_name: (H, W) array} - bounded [0, 1) values
        params: {w_photometric, w_traction, w_consistency, w_perturbation}
        constants: unused
        
    Returns:
        loss: (H, W) array
    """
    loss = np.zeros_like(next(iter(metrics.values())), dtype=np.float32)
    
    # Photometric (use grayscale if available)
    if "photometric" in metrics:
        loss += params.get("w_photometric", 0) * metrics["photometric"]
    
    # Stability metrics
    if "traction" in metrics:
        loss += params.get("w_traction", 0) * metrics["traction"]
    
    if "consistency" in metrics:
        loss += params.get("w_consistency", 0) * metrics["consistency"]
    
    if "perturbation" in metrics:
        loss += params.get("w_perturbation", 0) * metrics["perturbation"]
    
    return loss


# =============================================================================
# MULTIPLICATIVE LOSS (NEW)
# =============================================================================

MULTIPLICATIVE_PARAMS = [
    "c_gray",
    "c_r",
    "c_g",
    "c_b",
    "c_log",
    "c_traction",
    "c_consistency",
    "c_perturbation",
]


def loss_multiplicative(metrics: dict, params: dict, constants: dict) -> np.ndarray:
    """
    Multiplicative loss with photometric gates and depth-scaled stability.
    
    loss = (1 + c_gray * photo_gray/max_gray)
         × (1 + c_r * photo_r/max_r)
         × (1 + c_g * photo_g/max_g)
         × (1 + c_b * photo_b/max_b)
         × (1 + c_log * photo_log/max_log)
         × (1 + depth_scale * (c_traction * traction + c_consistency * consistency + c_perturbation * perturbation))
    
    where depth_scale = pollution_depth + perturbation_distance
    
    Args:
        metrics: {metric_name: (H, W) array} - RAW values (not bounded)
            Required: photo_gray_raw, traction_raw, consistency_raw, perturbation_raw
            Optional: photo_r_raw, photo_g_raw, photo_b_raw, photo_log_raw
        params: {c_gray, c_r, c_g, c_b, c_log, c_traction, c_consistency, c_perturbation}
        constants: {max_gray, max_r, max_g, max_b, max_log, pollution_depth, perturbation_distance}
        
    Returns:
        loss: (H, W) array
    """
    # Get shape from first metric
    shape = next(iter(metrics.values())).shape
    loss = np.ones(shape, dtype=np.float32)
    
    # --- Photometric gates (multiplicative) ---
    
    # Grayscale
    if "photo_gray_raw" in metrics and "max_gray" in constants:
        c = params.get("c_gray", 0)
        if c > 0:
            loss *= (1 + c * metrics["photo_gray_raw"] / constants["max_gray"])
    
    # Red
    if "photo_r_raw" in metrics and "max_r" in constants:
        c = params.get("c_r", 0)
        if c > 0:
            loss *= (1 + c * metrics["photo_r_raw"] / constants["max_r"])
    
    # Green
    if "photo_g_raw" in metrics and "max_g" in constants:
        c = params.get("c_g", 0)
        if c > 0:
            loss *= (1 + c * metrics["photo_g_raw"] / constants["max_g"])
    
    # Blue
    if "photo_b_raw" in metrics and "max_b" in constants:
        c = params.get("c_b", 0)
        if c > 0:
            loss *= (1 + c * metrics["photo_b_raw"] / constants["max_b"])
    
    # Log
    if "photo_log_raw" in metrics and "max_log" in constants:
        c = params.get("c_log", 0)
        if c > 0:
            loss *= (1 + c * metrics["photo_log_raw"] / constants["max_log"])
    
    # --- Stability term (additive inside, multiplicative outside) ---
    
    depth_scale = constants.get("pollution_depth", 1) + constants.get("perturbation_distance", 0)
    
    stability = np.zeros(shape, dtype=np.float32)
    
    if "traction_raw" in metrics:
        stability += params.get("c_traction", 0) * metrics["traction_raw"]
    
    if "consistency_raw" in metrics:
        stability += params.get("c_consistency", 0) * metrics["consistency_raw"]
    
    if "perturbation_raw" in metrics:
        stability += params.get("c_perturbation", 0) * metrics["perturbation_raw"]
    
    loss *= (1 + depth_scale * stability)
    
    return loss


# =============================================================================
# REGISTRY
# =============================================================================

LOSS_FUNCTIONS: dict[str, LossFn] = {
    "additive": loss_additive,
    "multiplicative": loss_multiplicative,
}

LOSS_PARAMS: dict[str, list[str]] = {
    "additive": ADDITIVE_PARAMS,
    "multiplicative": MULTIPLICATIVE_PARAMS,
}


def get_loss_function(name: str) -> LossFn:
    """
    Get loss function by name.
    
    Args:
        name: "additive" or "multiplicative"
        
    Returns:
        Loss function callable
    """
    if name not in LOSS_FUNCTIONS:
        print(f"❌ Unknown loss function: {name}")
        print(f"   Available: {list(LOSS_FUNCTIONS.keys())}")
        sys.exit(1)
    
    return LOSS_FUNCTIONS[name]


def get_loss_params(name: str) -> list[str]:
    """
    Get parameter names for a loss function.
    
    Args:
        name: "additive" or "multiplicative"
        
    Returns:
        List of parameter names
    """
    if name not in LOSS_PARAMS:
        print(f"❌ Unknown loss function: {name}")
        print(f"   Available: {list(LOSS_PARAMS.keys())}")
        sys.exit(1)
    
    return LOSS_PARAMS[name]


def get_default_params(name: str) -> dict[str, float]:
    """
    Get default parameters (all zeros = no effect).
    
    Args:
        name: loss function name
        
    Returns:
        Dict of param_name -> 0.0
    """
    return {p: 0.0 for p in get_loss_params(name)}


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("Available loss functions:")
    for name, params in LOSS_PARAMS.items():
        print(f"  {name}: {params}")
