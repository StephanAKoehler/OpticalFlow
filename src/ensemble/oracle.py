# File: src/ensemble/oracle.py
"""
Oracle computation for optical flow ensemble.

Computes theoretical best per-pixel configuration selection based on
ground truth endpoint error.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.ground_truth import compute_epe


def compute_oracle_selection(
    results_full: list[dict],
    u_true: np.ndarray,
    v_true: np.ndarray,
    valid_mask: np.ndarray
) -> dict:
    """
    Compute oracle (theoretical best) ensemble selection.
    
    For each pixel, selects the configuration with minimum EPE.
    Computes both forward and symmetric flow oracles.
    
    Args:
        results_full: List of config result dicts, each containing:
                      - 'u_AB', 'v_AB': Forward flow
                      - 'u_sym_A', 'v_sym_A': Symmetric flow
                      - 'config_name': Configuration name
        u_true: Ground truth u flow (H, W)
        v_true: Ground truth v flow (H, W)
        valid_mask: Boolean mask (H, W) of valid pixels
    
    Returns:
        dict containing:
            - 'oracle_selection_forward': (H, W) int array, config index per pixel
            - 'oracle_selection_symmetric': (H, W) int array, config index per pixel
            - 'oracle_epe_forward': float, mean EPE of oracle forward flow
            - 'oracle_epe_symmetric': float, mean EPE of oracle symmetric flow
            - 'EPE_forward_stack': (n_configs, H, W) EPE for each config (forward)
            - 'EPE_symmetric_stack': (n_configs, H, W) EPE for each config (symmetric)
    
    Example:
        >>> oracle = compute_oracle_selection(results_full, u_true, v_true, valid_mask)
        >>> print(f"Oracle forward EPE: {oracle['oracle_epe_forward']:.4f} px")
        >>> print(f"Oracle symmetric EPE: {oracle['oracle_epe_symmetric']:.4f} px")
    """
    
    n_configs = len(results_full)
    H, W = u_true.shape
    
    # ========================================================================
    # Stack EPE for all configs
    # ========================================================================
    
    EPE_forward_stack = []
    EPE_symmetric_stack = []
    
    for i in range(n_configs):
        # Forward flow EPE
        EPE_forward = compute_epe(
            results_full[i]['u_AB'],
            results_full[i]['v_AB'],
            u_true,
            v_true
        )
        
        # Symmetric flow EPE
        EPE_symmetric = compute_epe(
            results_full[i]['u_sym_A'],
            results_full[i]['v_sym_A'],
            u_true,
            v_true
        )
        
        EPE_forward_stack.append(EPE_forward)
        EPE_symmetric_stack.append(EPE_symmetric)
    
    EPE_forward_stack = np.array(EPE_forward_stack)  # (n_configs, H, W)
    EPE_symmetric_stack = np.array(EPE_symmetric_stack)
    
    # ========================================================================
    # Oracle selection (best EPE at each pixel)
    # ========================================================================
    
    # Initialize selection maps (zeros for invalid regions)
    oracle_selection_forward = np.zeros((H, W), dtype=int)
    oracle_selection_symmetric = np.zeros((H, W), dtype=int)
    
    # For valid pixels, find config with minimum EPE
    oracle_selection_forward[valid_mask] = np.nanargmin(
        EPE_forward_stack[:, valid_mask],
        axis=0
    )
    
    oracle_selection_symmetric[valid_mask] = np.nanargmin(
        EPE_symmetric_stack[:, valid_mask],
        axis=0
    )
    
    # ========================================================================
    # Oracle EPE (mean and std of per-pixel minimum EPE)
    # ========================================================================
    
    oracle_epe_forward_pixels = np.nanmin(EPE_forward_stack, axis=0)[valid_mask]
    oracle_epe_forward = np.nanmean(oracle_epe_forward_pixels)
    oracle_epe_forward_std = np.nanstd(oracle_epe_forward_pixels)
    
    oracle_epe_symmetric_pixels = np.nanmin(EPE_symmetric_stack, axis=0)[valid_mask]
    oracle_epe_symmetric = np.nanmean(oracle_epe_symmetric_pixels)
    oracle_epe_symmetric_std = np.nanstd(oracle_epe_symmetric_pixels)
    
    # ========================================================================
    # Return results
    # ========================================================================
    
    return {
        'oracle_selection_forward': oracle_selection_forward,
        'oracle_selection_symmetric': oracle_selection_symmetric,
        'oracle_epe_forward': float(oracle_epe_forward),
        'oracle_epe_forward_std': float(oracle_epe_forward_std),
        'oracle_epe_symmetric': float(oracle_epe_symmetric),
        'oracle_epe_symmetric_std': float(oracle_epe_symmetric_std),
        'EPE_forward_stack': EPE_forward_stack,
        'EPE_symmetric_stack': EPE_symmetric_stack,
    }


def build_oracle_flows(
    results_full: list[dict],
    oracle_selection_forward: np.ndarray,
    oracle_selection_symmetric: np.ndarray
) -> dict:
    """
    Build oracle flow fields from selection maps.
    
    Args:
        results_full: List of config result dicts
        oracle_selection_forward: (H, W) int, oracle selection for forward flow
        oracle_selection_symmetric: (H, W) int, oracle selection for symmetric flow
    
    Returns:
        dict containing:
            - 'u_oracle_forward', 'v_oracle_forward': Forward oracle flows
            - 'u_oracle_backward', 'v_oracle_backward': Backward oracle flows
            - 'u_oracle_symmetric', 'v_oracle_symmetric': Symmetric oracle flows
    """
    
    n_configs = len(results_full)
    H, W = oracle_selection_forward.shape
    
    # ========================================================================
    # Build oracle flows (forward) - uses forward oracle selection
    # ========================================================================
    
    u_oracle_forward = np.zeros((H, W), dtype=np.float32)
    v_oracle_forward = np.zeros((H, W), dtype=np.float32)
    
    for i in range(n_configs):
        mask = (oracle_selection_forward == i)
        u_oracle_forward[mask] = results_full[i]['u_AB'][mask]
        v_oracle_forward[mask] = results_full[i]['v_AB'][mask]
    
    # ========================================================================
    # Build oracle flows (backward) - uses forward oracle selection for consistency
    # ========================================================================
    
    u_oracle_backward = np.zeros((H, W), dtype=np.float32)
    v_oracle_backward = np.zeros((H, W), dtype=np.float32)
    
    for i in range(n_configs):
        mask = (oracle_selection_forward == i)
        u_oracle_backward[mask] = results_full[i]['u_BA'][mask]
        v_oracle_backward[mask] = results_full[i]['v_BA'][mask]
    
    # ========================================================================
    # Build oracle flows (symmetric) - uses oracle_selection_symmetric
    # ========================================================================
    
    u_oracle_symmetric = np.zeros((H, W), dtype=np.float32)
    v_oracle_symmetric = np.zeros((H, W), dtype=np.float32)
    
    for i in range(n_configs):
        mask = (oracle_selection_symmetric == i)
        u_oracle_symmetric[mask] = results_full[i]['u_sym_A'][mask]
        v_oracle_symmetric[mask] = results_full[i]['v_sym_A'][mask]
    
    # ========================================================================
    # Return oracle flows
    # ========================================================================
    
    return {
        'u_oracle_forward': u_oracle_forward,
        'v_oracle_forward': v_oracle_forward,
        'u_oracle_backward': u_oracle_backward,
        'v_oracle_backward': v_oracle_backward,
        'u_oracle_symmetric': u_oracle_symmetric,
        'v_oracle_symmetric': v_oracle_symmetric,
    }


if __name__ == "__main__":
    print("🧪 Testing oracle computation...")
    
    # Create synthetic test data
    H, W = 100, 100
    n_configs = 3
    
    # Ground truth flow
    u_true = np.ones((H, W), dtype=np.float32) * 2.0
    v_true = np.ones((H, W), dtype=np.float32) * 1.0
    
    # Valid mask (exclude 10px border)
    valid_mask = np.ones((H, W), dtype=bool)
    valid_mask[:10, :] = False
    valid_mask[-10:, :] = False
    valid_mask[:, :10] = False
    valid_mask[:, -10:] = False
    
    # Create synthetic results with different quality configs
    results_full = []
    
    for i in range(n_configs):
        # Config 0: Perfect
        # Config 1: Off by 0.5
        # Config 2: Off by 1.0
        error = i * 0.5
        
        result = {
            'u_AB': u_true + error,
            'v_AB': v_true + error,
            'u_BA': -u_true - error,
            'v_BA': -v_true - error,
            'u_sym_A': u_true + error,
            'v_sym_A': v_true + error,
            'config_name': f'config_{i}'
        }
        results_full.append(result)
    
    print(f"✅ Created {n_configs} synthetic configs")
    print(f"   Config 0: Perfect (error = 0.0)")
    print(f"   Config 1: error = 0.5")
    print(f"   Config 2: error = 1.0")
    
    # Compute oracle
    print(f"\n📊 Computing oracle selection...")
    oracle = compute_oracle_selection(results_full, u_true, v_true, valid_mask)
    
    # Verify results
    print(f"\n✅ Oracle results:")
    print(f"   Forward EPE:   {oracle['oracle_epe_forward']:.4f} px (expected: 0.0)")
    print(f"   Symmetric EPE: {oracle['oracle_epe_symmetric']:.4f} px (expected: 0.0)")
    
    # Check selection (should all be config 0)
    selection_counts = np.bincount(
        oracle['oracle_selection_forward'][valid_mask].ravel(),
        minlength=n_configs
    )
    
    print(f"\n📈 Oracle selection distribution:")
    for i, count in enumerate(selection_counts):
        pct = 100 * count / valid_mask.sum()
        print(f"   Config {i}: {count:5d} pixels ({pct:5.1f}%)")
    
    # Verify oracle selected best config
    assert oracle['oracle_epe_forward'] < 0.01, "Oracle should select perfect config"
    assert selection_counts[0] == valid_mask.sum(), "Oracle should select config 0 everywhere"
    
    print(f"\n✅ Oracle correctly selected best config at all pixels")
    
    # Test flow building
    print(f"\n📊 Building oracle flows...")
    oracle_flows = build_oracle_flows(
        results_full,
        oracle['oracle_selection_forward'],
        oracle['oracle_selection_symmetric']
    )
    
    # Verify oracle flows match ground truth (since config 0 is perfect)
    u_diff = np.abs(oracle_flows['u_oracle_forward'] - u_true)
    v_diff = np.abs(oracle_flows['v_oracle_forward'] - v_true)
    
    print(f"✅ Oracle flows built:")
    print(f"   u difference: mean={np.nanmean(u_diff[valid_mask]):.6f} px")
    print(f"   v difference: mean={np.nanmean(v_diff[valid_mask]):.6f} px")
    
    assert np.nanmean(u_diff[valid_mask]) < 0.01, "Oracle u flow should match ground truth"
    assert np.nanmean(v_diff[valid_mask]) < 0.01, "Oracle v flow should match ground truth"
    
    print(f"\n✨ All oracle tests passed!")
