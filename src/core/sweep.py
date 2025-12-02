# File: src/core/sweep.py
"""
Optical flow parameter sweep computation.

Handles parallel computation of all OF configurations and metric evaluation.
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import hashlib
import json

from src.optical_flow.flow_computation import compute_all_flows
from src.evaluation.self_supervised import compute_metrics_from_flows
from src.optical_flow.config_naming import generate_config_name
from src.utils.resampling import downsample_metrics, upsample_metrics, compute_downsample_stride
from src.ensemble.oracle import compute_oracle_selection
from src.cache.sweep_results import build_sweep_dataframe
from src.core.data_structures import create_result_dict


def compute_rms_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute RMS difference between two frames.
    
    This is a property of the image pair, used for photometric normalization.
    Represents the "baseline" intensity change between frames.
    
    Args:
        frame1, frame2: Input frames (H, W), any range (will be converted to float32)
        
    Returns:
        RMS difference (scalar)
    """
    # Handle RGB frames - convert to grayscale for RMS diff
    if frame1.ndim == 3:
        import cv2
        f1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        f1 = frame1.astype(np.float32)
        f2 = frame2.astype(np.float32)
    
    diff = f1 - f2
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_config_worker(args):
    """
    Worker function for parallel config computation.
    
    Now returns structured result dict using create_result_dict().
    Supports optional RGB frames for RGB photometric metrics.
    """
    # Unpack args - support both old (7 args) and new (9 args) format
    if len(args) == 7:
        # Old format: no RGB frames
        frame_A, frame_B, params, deltas, config_idx, n_configs, rms_diff = args
        frame_A_rgb, frame_B_rgb = None, None
    else:
        # New format: includes RGB frames
        frame_A, frame_B, params, deltas, config_idx, n_configs, rms_diff, frame_A_rgb, frame_B_rgb = args
    
    # Generate identifiers
    config_name = generate_config_name(params)
    
    # Generate deterministic config ID from params
    params_str = json.dumps(params, sort_keys=True)
    config_id = hashlib.sha256(params_str.encode()).hexdigest()[:12]
    
    # Compute flows (always uses grayscale internally)
    all_flows = compute_all_flows(frame_A, frame_B, params, deltas, verbose=False)
    
    # Compute metrics - use RGB frames if available for RGB photometric
    # The self_supervised.py will detect RGB and compute additional metrics
    if frame_A_rgb is not None and frame_B_rgb is not None:
        result = compute_metrics_from_flows(frame_A_rgb, frame_B_rgb, all_flows, params, rms_diff, verbose=False)
    else:
        result = compute_metrics_from_flows(frame_A, frame_B, all_flows, params, rms_diff, verbose=False)
    
    # Extract separated flows and metrics
    flows = result['flows']
    metrics = result['metrics']
    
    # Downsample to native resolution
    winsize = params.get('winsize', 15)
    stride = compute_downsample_stride(winsize)
    
    flows_native = downsample_metrics(flows, stride)
    metrics_native = downsample_metrics(metrics, stride)
    
    # Create structured result using data_structures
    result_native = create_result_dict(
        params=params,
        flows=flows_native,
        metrics=metrics_native,
        config_name=config_name,
        config_id=config_id
    )
    
    # Add stride to metadata
    result_native['metadata']['stride'] = stride
    
    return result_native


def compute_sweep(frame1: np.ndarray,
                 frame2: np.ndarray,
                 configs: List[dict],
                 deltas: List[tuple],
                 n_workers: int = None,
                 frame1_original: np.ndarray = None,
                 frame2_original: np.ndarray = None) -> List[Dict]:
    """
    Compute optical flow sweep for all configurations.
    
    Args:
        frame1, frame2: Input frames (float32 in [0,1] or uint8 in [0,255])
                       These are used for optical flow computation (grayscale)
        configs: List of OF configurations
        deltas: List of perturbation vectors
        n_workers: Number of parallel workers
        frame1_original, frame2_original: Optional RGB frames for RGB photometric metrics
                                         If provided, RGB metrics will be computed
        
    Returns:
        List of results_native dicts (downsampled metrics)
    """
    print("⚙️  Computing optical flow sweep...")
    
    n_configs = len(configs)
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"   Configurations: {n_configs}")
    print(f"   Workers: {n_workers}")
    
    # Compute rms_diff once for the frame pair (use grayscale)
    rms_diff = compute_rms_diff(frame1, frame2)
    print(f"   RMS diff (frame pair): {rms_diff:.2f}")
    
    # Check if RGB frames are available
    has_rgb = frame1_original is not None and frame2_original is not None
    if has_rgb:
        is_rgb = frame1_original.ndim == 3 and frame1_original.shape[2] == 3
        if is_rgb:
            print(f"   RGB frames: Available (will compute RGB photometric)")
        else:
            print(f"   RGB frames: Provided but not RGB format")
            has_rgb = False
    else:
        print(f"   RGB frames: Not provided (grayscale photometric only)")
    print()
    
    # Prepare worker arguments
    if has_rgb:
        worker_args = [
            (frame1, frame2, configs[i], deltas, i, n_configs, rms_diff, 
             frame1_original, frame2_original)
            for i in range(n_configs)
        ]
    else:
        worker_args = [
            (frame1, frame2, configs[i], deltas, i, n_configs, rms_diff)
            for i in range(n_configs)
        ]
    
    # Compute in parallel with progress bar
    with Pool(n_workers) as pool:
        results_native = list(tqdm(
            pool.imap(compute_config_worker, worker_args),
            total=n_configs,
            desc="   Computing",
            ncols=80,
            unit="config"
        ))
    
    print(f"   ✅ Computed {n_configs} configurations")
    print()
    
    return results_native


def upsample_and_compute_oracle(results_native: List[Dict],
                                H: int, W: int,
                                u_truth: np.ndarray,
                                v_truth: np.ndarray,
                                valid_mask: np.ndarray,
                                epe_power: float = 2.0):
    """
    Upsample metrics to full resolution and compute oracle.
    
    Now reconstructs using create_result_dict() for consistent structure.
    
    Args:
        results_native: List of native resolution results
        H, W: Target dimensions for upsampling
        u_truth, v_truth: Ground truth flows
        valid_mask: Valid pixel mask
        epe_power: Power for EPE aggregation (1.0=MAE, 2.0=MSE)
    
    Returns:
        dict with:
            - results_full: List of upsampled results (structured)
            - oracle: Oracle results dict with both powered and standard EPE
    """
    print("📊 Upsampling and computing oracle...")
    print(f"   Using EPE power: {epe_power} ({'MAE' if epe_power == 1.0 else 'MSE' if epe_power == 2.0 else f'p={epe_power}'})")
    
    # Upsample to full resolution
    results_full = []
    for res_native in results_native:
        # Upsample flows and metrics separately
        flows_full = upsample_metrics(res_native['flows'], (H, W))
        metrics_full = upsample_metrics(res_native['metrics'], (H, W))
        
        # Reconstruct params dict with algorithm for create_result_dict
        # (algorithm was removed from params and stored in metadata)
        params_with_algo = {
            'algorithm': res_native['metadata']['algorithm'],
            **res_native['params']
        }
        
        # Reconstruct using data_structures
        res_full = create_result_dict(
            params=params_with_algo,
            flows=flows_full,
            metrics=metrics_full,
            config_name=res_native['metadata']['config_name'],
            config_id=res_native['metadata']['config_id']
        )
        
        # Preserve stride metadata
        res_full['metadata']['stride'] = res_native['metadata']['stride']
        
        results_full.append(res_full)
    
    print(f"   ✅ Upsampled {len(results_full)} configs")
    
    # Compute oracle using existing function
    from src.core.data_structures import flatten_for_selection
    from src.evaluation.ground_truth import compute_epe
    
    results_flat = [flatten_for_selection(r) for r in results_full]
    
    # Use the ORIGINAL oracle function (backward compatible)
    oracle_original = compute_oracle_selection(results_flat, u_truth, v_truth, valid_mask)
    
    # ========================================================================
    # Recompute with powered EPE for proper oracle selection
    # ========================================================================
    print(f"   Recomputing oracle with powered EPE (p={epe_power})...")
    
    n_configs = len(results_flat)
    
    # ========================================================================
    # FORWARD ORACLE
    # ========================================================================
    
    epe_stack_fwd_powered = np.zeros((n_configs, H, W), dtype=np.float32)
    epe_stack_fwd_standard = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, result in enumerate(results_flat):
        u, v = result['u_AB'], result['v_AB']
        
        # Powered EPE (for selection)
        epe_powered = compute_epe(u, v, u_truth, v_truth, valid_mask, power=epe_power)
        epe_stack_fwd_powered[i] = epe_powered
        
        # Standard EPE (for reporting)
        epe_standard = compute_epe(u, v, u_truth, v_truth, valid_mask, power=1.0)
        epe_stack_fwd_standard[i] = epe_standard
    
    # Select based on powered EPE
    oracle_selection_fwd = np.argmin(epe_stack_fwd_powered, axis=0)
    
    # Build oracle flow
    u_oracle_fwd = np.zeros((H, W), dtype=np.float32)
    v_oracle_fwd = np.zeros((H, W), dtype=np.float32)
    for i in range(n_configs):
        mask = oracle_selection_fwd == i
        u_oracle_fwd[mask] = results_flat[i]['u_AB'][mask]
        v_oracle_fwd[mask] = results_flat[i]['v_AB'][mask]
    
    # Compute oracle EPE (both versions) - mean, std, median over valid pixels
    oracle_epe_fwd_powered_pixels = epe_stack_fwd_powered[oracle_selection_fwd,
                              np.arange(H)[:, None],
                              np.arange(W)][valid_mask]
    oracle_epe_fwd_powered = np.nanmean(oracle_epe_fwd_powered_pixels)
    oracle_epe_fwd_powered_std = np.nanstd(oracle_epe_fwd_powered_pixels)
    oracle_epe_fwd_powered_median = np.nanmedian(oracle_epe_fwd_powered_pixels)
    
    oracle_epe_fwd_standard_pixels = epe_stack_fwd_standard[oracle_selection_fwd,
                               np.arange(H)[:, None],
                               np.arange(W)][valid_mask]
    oracle_epe_fwd_standard = np.nanmean(oracle_epe_fwd_standard_pixels)
    oracle_epe_fwd_standard_std = np.nanstd(oracle_epe_fwd_standard_pixels)
    oracle_epe_fwd_standard_median = np.nanmedian(oracle_epe_fwd_standard_pixels)
    
    # ========================================================================
    # SYMMETRIC ORACLE
    # ========================================================================
    
    epe_stack_sym_powered = np.zeros((n_configs, H, W), dtype=np.float32)
    epe_stack_sym_standard = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, result in enumerate(results_flat):
        # Handle different field name formats
        if 'u_sym_A' in result:
            u, v = result['u_sym_A'], result['v_sym_A']
        elif 'u_symmetric' in result:
            u, v = result['u_symmetric'], result['v_symmetric']
        else:
            print(f"❌ ERROR: Cannot find symmetric flow in result keys: {list(result.keys())}")
            sys.exit(1)
        
        # Powered EPE
        epe_powered = compute_epe(u, v, u_truth, v_truth, valid_mask, power=epe_power)
        epe_stack_sym_powered[i] = epe_powered
        
        # Standard EPE
        epe_standard = compute_epe(u, v, u_truth, v_truth, valid_mask, power=1.0)
        epe_stack_sym_standard[i] = epe_standard
    
    # Select based on powered EPE
    oracle_selection_sym = np.argmin(epe_stack_sym_powered, axis=0)
    
    # Build oracle flow
    u_oracle_sym = np.zeros((H, W), dtype=np.float32)
    v_oracle_sym = np.zeros((H, W), dtype=np.float32)
    for i in range(n_configs):
        mask = oracle_selection_sym == i
        # Handle different field name formats
        if 'u_sym_A' in results_flat[i]:
            u_oracle_sym[mask] = results_flat[i]['u_sym_A'][mask]
            v_oracle_sym[mask] = results_flat[i]['v_sym_A'][mask]
        elif 'u_symmetric' in results_flat[i]:
            u_oracle_sym[mask] = results_flat[i]['u_symmetric'][mask]
            v_oracle_sym[mask] = results_flat[i]['v_symmetric'][mask]
        else:
            print(f"❌ ERROR: Cannot find symmetric flow in result keys")
            sys.exit(1)
    
    # Compute oracle EPE (both versions) - mean, std, median over valid pixels
    oracle_epe_sym_powered_pixels = epe_stack_sym_powered[oracle_selection_sym,
                              np.arange(H)[:, None],
                              np.arange(W)][valid_mask]
    oracle_epe_sym_powered = np.nanmean(oracle_epe_sym_powered_pixels)
    oracle_epe_sym_powered_std = np.nanstd(oracle_epe_sym_powered_pixels)
    oracle_epe_sym_powered_median = np.nanmedian(oracle_epe_sym_powered_pixels)
    
    oracle_epe_sym_standard_pixels = epe_stack_sym_standard[oracle_selection_sym,
                               np.arange(H)[:, None],
                               np.arange(W)][valid_mask]
    oracle_epe_sym_standard = np.nanmean(oracle_epe_sym_standard_pixels)
    oracle_epe_sym_standard_std = np.nanstd(oracle_epe_sym_standard_pixels)
    oracle_epe_sym_standard_median = np.nanmedian(oracle_epe_sym_standard_pixels)
    
    print(f"   Oracle EPE^{epe_power}: {oracle_epe_sym_powered:.6f}")
    
    return {
        'results_full': results_full,
        'oracle': {
            # Forward flows
            'u_oracle_fwd': u_oracle_fwd,
            'v_oracle_fwd': v_oracle_fwd,
            'oracle_selection_fwd': oracle_selection_fwd,
            
            # Symmetric flows
            'u_oracle_sym': u_oracle_sym,
            'v_oracle_sym': v_oracle_sym,
            'oracle_selection_sym': oracle_selection_sym,
            
            # EPE values (powered - what was optimized)
            'oracle_epe_forward_powered': float(oracle_epe_fwd_powered),
            'oracle_epe_forward_powered_std': float(oracle_epe_fwd_powered_std),
            'oracle_epe_forward_powered_median': float(oracle_epe_fwd_powered_median),
            'oracle_epe_symmetric_powered': float(oracle_epe_sym_powered),
            'oracle_epe_symmetric_powered_std': float(oracle_epe_sym_powered_std),
            'oracle_epe_symmetric_powered_median': float(oracle_epe_sym_powered_median),
            
            # EPE values (standard - for reporting/comparison)
            'oracle_epe_forward_standard': float(oracle_epe_fwd_standard),
            'oracle_epe_forward_standard_std': float(oracle_epe_fwd_standard_std),
            'oracle_epe_forward_standard_median': float(oracle_epe_fwd_standard_median),
            'oracle_epe_symmetric_standard': float(oracle_epe_sym_standard),
            'oracle_epe_symmetric_standard_std': float(oracle_epe_sym_standard_std),
            'oracle_epe_symmetric_standard_median': float(oracle_epe_sym_standard_median),
            
            # Backward compatibility (use standard for old code)
            'oracle_epe_forward': float(oracle_epe_fwd_standard),
            'oracle_epe_symmetric': float(oracle_epe_sym_standard),
            
            # Metadata
            'epe_power': epe_power
        }
    }


def save_sweep_to_cache(exp_cache, results_full: List[Dict],
                       u_truth: Optional[np.ndarray], v_truth: Optional[np.ndarray],
                       valid_mask: np.ndarray,
                       oracle_epe_forward: Optional[float], oracle_epe_symmetric: Optional[float],
                       epe_power: float = 1.0):
    """Save sweep results to cache.
    
    Args:
        exp_cache: Experiment cache object
        results_full: List of result dicts from sweep
        u_truth, v_truth: Ground truth flow (can be None if no GT)
        valid_mask: Boolean mask for valid pixels
        oracle_epe_forward, oracle_epe_symmetric: Oracle EPE values (can be None)
        epe_power: Power for EPE computation (stored for cache validation)
    """
    print("   Saving sweep results to cache...")
    
    # Flatten results for backward compatibility with build_sweep_dataframe
    from src.core.data_structures import flatten_for_visualization
    results_flat = [flatten_for_visualization(r) for r in results_full]
    
    # Build and save sweep dataframe WITH epe_power
    sweep_df = build_sweep_dataframe(
        results_flat, u_truth, v_truth, valid_mask,
        oracle_epe_forward, oracle_epe_symmetric,
        epe_power=epe_power
    )
    exp_cache.save_sweep_results(sweep_df)
    
    # Save results_full for figure regeneration (keep structured version)
    import pickle
    results_full_path = exp_cache.current_dir / 'results_full.pkl'
    with open(results_full_path, 'wb') as f:
        pickle.dump(results_full, f)
    
    print(f"   ✅ Saved sweep results and full data (epe_power={epe_power})")


def load_sweep_from_cache(exp_cache, expected_epe_power: float = None):
    """
    Load sweep results from cache.
    
    Args:
        exp_cache: Experiment cache object
        expected_epe_power: If provided, validate cached data was computed with this power
    
    Returns:
        dict with:
            - sweep_df: Summary dataframe
            - results_full: Full results (if available)
            - oracle_epe_forward, oracle_epe_symmetric: Oracle EPE values
            - epe_power: EPE power used for cached computation
            - epe_power_mismatch: True if cached power differs from expected
    """
    print("📂 Loading sweep from cache...")
    
    sweep_df = exp_cache.load_sweep_results()
    print(f"   ✅ Loaded {len(sweep_df)} configs")
    
    # Load results_full if available
    results_full_path = exp_cache.current_dir / 'results_full.pkl'
    if results_full_path.exists():
        import pickle
        with open(results_full_path, 'rb') as f:
            results_full = pickle.load(f)
        print(f"   ✅ Loaded full results")
    else:
        results_full = None
        print(f"   ⚠️  Full results not in cache")
    
    # Extract oracle EPE from sweep_df
    oracle_epe_forward = sweep_df['oracle_epe_forward'].iloc[0]
    oracle_epe_symmetric = sweep_df['oracle_epe_symmetric'].iloc[0]
    
    # Check epe_power (new field, may not exist in old caches)
    if 'epe_power' in sweep_df.columns:
        cached_epe_power = sweep_df['epe_power'].iloc[0]
    else:
        # Old cache without epe_power - assume standard EPE (power=1.0)
        cached_epe_power = 1.0
        print(f"   ⚠️  Old cache format (assuming epe_power=1.0)")
    
    # Check for mismatch
    epe_power_mismatch = False
    if expected_epe_power is not None and abs(cached_epe_power - expected_epe_power) > 1e-6:
        epe_power_mismatch = True
        print(f"   ⚠️  EPE power mismatch: cached={cached_epe_power}, expected={expected_epe_power}")
    
    print()
    
    return {
        'sweep_df': sweep_df,
        'results_full': results_full,
        'oracle_epe_forward': oracle_epe_forward,
        'oracle_epe_symmetric': oracle_epe_symmetric,
        'epe_power': cached_epe_power,
        'epe_power_mismatch': epe_power_mismatch
    }


if __name__ == "__main__":
    print("✅ Sweep computation module loaded")
