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
    import cv2
    
    # Handle RGB frames - convert to grayscale for RMS diff
    if frame1.ndim == 3:
        f1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        f1 = frame1.astype(np.float32)
        f2 = frame2.astype(np.float32)
    
    diff = f1 - f2
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_frame_constants(frame1: np.ndarray, frame2: np.ndarray,
                            deltas: List[tuple],
                            frame1_rgb: np.ndarray = None,
                            frame2_rgb: np.ndarray = None) -> dict:
    """
    Compute normalization constants for the frame pair.
    
    These are used by the multiplicative loss function.
    
    Args:
        frame1, frame2: Grayscale frames for optical flow
        deltas: List of perturbation vectors [(dx, dy), ...]
        frame1_rgb, frame2_rgb: Optional RGB frames
        
    Returns:
        dict with:
            - rms_diff: RMS intensity difference
            - max_gray: Max grayscale intensity
            - max_r, max_g, max_b: Max per-channel (if RGB)
            - max_log: Max of log(1 + grayscale)
            - perturbation_distance: RMS of perturbation magnitudes
    """
    import cv2
    
    # Get grayscale versions
    if frame1.ndim == 3:
        f1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        f1_gray = frame1.astype(np.float32)
        f2_gray = frame2.astype(np.float32)
    
    # RMS diff
    diff = f1_gray - f2_gray
    rms_diff = float(np.sqrt(np.mean(diff ** 2)))
    
    # Max grayscale (use max of both frames)
    max_gray = float(max(np.max(f1_gray), np.max(f2_gray)))
    
    # Max log (for log-space photometric)
    max_log = float(max(np.max(np.log1p(f1_gray)), np.max(np.log1p(f2_gray))))
    
    # Perturbation distance (RMS of perturbation magnitudes)
    pert_magnitudes = [np.hypot(dx, dy) for dx, dy in deltas]
    perturbation_distance = float(np.sqrt(np.mean([m**2 for m in pert_magnitudes])))
    
    constants = {
        'rms_diff': rms_diff,
        'max_gray': max_gray,
        'max_log': max_log,
        'perturbation_distance': perturbation_distance,
    }
    
    # RGB per-channel max (if RGB frames provided)
    if frame1_rgb is not None and frame2_rgb is not None:
        if frame1_rgb.ndim == 3 and frame1_rgb.shape[2] == 3:
            f1_rgb = frame1_rgb.astype(np.float32)
            f2_rgb = frame2_rgb.astype(np.float32)
            
            constants['max_r'] = float(max(np.max(f1_rgb[:,:,0]), np.max(f2_rgb[:,:,0])))
            constants['max_g'] = float(max(np.max(f1_rgb[:,:,1]), np.max(f2_rgb[:,:,1])))
            constants['max_b'] = float(max(np.max(f1_rgb[:,:,2]), np.max(f2_rgb[:,:,2])))
    
    return constants


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
                 frame2_original: np.ndarray = None,
                 return_constants: bool = False) -> List[Dict]:
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
        return_constants: If True, return (results_native, frame_constants) tuple
        
    Returns:
        If return_constants=False (default): List of results_native dicts
        If return_constants=True: (results_native, frame_constants) tuple
    """
    print("⚙️  Computing optical flow sweep...")
    
    n_configs = len(configs)
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"   Configurations: {n_configs}")
    print(f"   Workers: {n_workers}")
    
    # Check if RGB frames are available
    has_rgb = frame1_original is not None and frame2_original is not None
    if has_rgb:
        is_rgb = frame1_original.ndim == 3 and frame1_original.shape[2] == 3
        if not is_rgb:
            print(f"   RGB frames: Provided but not RGB format")
            has_rgb = False
    
    # Compute frame constants (normalization values)
    if has_rgb:
        frame_constants = compute_frame_constants(
            frame1, frame2, deltas, frame1_original, frame2_original
        )
        print(f"   RGB frames: Available (will compute RGB photometric)")
    else:
        frame_constants = compute_frame_constants(frame1, frame2, deltas)
        print(f"   RGB frames: Not provided (grayscale photometric only)")
    
    rms_diff = frame_constants['rms_diff']
    print(f"   RMS diff (frame pair): {rms_diff:.2f}")
    print(f"   Max grayscale: {frame_constants['max_gray']:.1f}")
    if 'max_r' in frame_constants:
        print(f"   Max RGB: R={frame_constants['max_r']:.1f}, G={frame_constants['max_g']:.1f}, B={frame_constants['max_b']:.1f}")
    print(f"   Perturbation distance: {frame_constants['perturbation_distance']:.2f}")
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
    
    # Store frame_constants as module-level for later retrieval
    compute_sweep._last_frame_constants = frame_constants
    
    if return_constants:
        return results_native, frame_constants
    return results_native


# Initialize storage for frame_constants
compute_sweep._last_frame_constants = None


def get_last_frame_constants() -> Optional[dict]:
    """Get frame_constants from the last compute_sweep() call."""
    return compute_sweep._last_frame_constants


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
        u, v = result['u_sym_A'], result['v_sym_A']
        
        # Powered EPE (for selection)
        epe_powered = compute_epe(u, v, u_truth, v_truth, valid_mask, power=epe_power)
        epe_stack_sym_powered[i] = epe_powered
        
        # Standard EPE (for reporting)
        epe_standard = compute_epe(u, v, u_truth, v_truth, valid_mask, power=1.0)
        epe_stack_sym_standard[i] = epe_standard
    
    # Select based on powered EPE
    oracle_selection_sym = np.argmin(epe_stack_sym_powered, axis=0)
    
    # Build oracle flow
    u_oracle_sym = np.zeros((H, W), dtype=np.float32)
    v_oracle_sym = np.zeros((H, W), dtype=np.float32)
    for i in range(n_configs):
        mask = oracle_selection_sym == i
        u_oracle_sym[mask] = results_flat[i]['u_sym_A'][mask]
        v_oracle_sym[mask] = results_flat[i]['v_sym_A'][mask]
    
    # Compute oracle EPE (both versions)
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
    
    print(f"   Oracle EPE (powered, p={epe_power}):")
    print(f"      Forward:   {oracle_epe_fwd_powered:.4f} ± {oracle_epe_fwd_powered_std:.4f}")
    print(f"      Symmetric: {oracle_epe_sym_powered:.4f} ± {oracle_epe_sym_powered_std:.4f}")
    print(f"   Oracle EPE (standard, p=1.0):")
    print(f"      Forward:   {oracle_epe_fwd_standard:.4f} ± {oracle_epe_fwd_standard_std:.4f}")
    print(f"      Symmetric: {oracle_epe_sym_standard:.4f} ± {oracle_epe_sym_standard_std:.4f}")
    print()
    
    # Build comprehensive oracle dict
    oracle = {
        'epe_power': epe_power,
        
        # Forward oracle
        'forward': {
            'selection': oracle_selection_fwd,
            'u_oracle': u_oracle_fwd,
            'v_oracle': v_oracle_fwd,
            'epe_stack_powered': epe_stack_fwd_powered,
            'epe_stack_standard': epe_stack_fwd_standard,
            'epe_powered': oracle_epe_fwd_powered,
            'epe_powered_std': oracle_epe_fwd_powered_std,
            'epe_powered_median': oracle_epe_fwd_powered_median,
            'epe_standard': oracle_epe_fwd_standard,
            'epe_standard_std': oracle_epe_fwd_standard_std,
            'epe_standard_median': oracle_epe_fwd_standard_median,
        },
        
        # Symmetric oracle
        'symmetric': {
            'selection': oracle_selection_sym,
            'u_oracle': u_oracle_sym,
            'v_oracle': v_oracle_sym,
            'epe_stack_powered': epe_stack_sym_powered,
            'epe_stack_standard': epe_stack_sym_standard,
            'epe_powered': oracle_epe_sym_powered,
            'epe_powered_std': oracle_epe_sym_powered_std,
            'epe_powered_median': oracle_epe_sym_powered_median,
            'epe_standard': oracle_epe_sym_standard,
            'epe_standard_std': oracle_epe_sym_standard_std,
            'epe_standard_median': oracle_epe_sym_standard_median,
        },
        
        # Legacy compatibility (use standard EPE for backward compatibility)
        'oracle_selection_forward': oracle_selection_fwd,
        'oracle_selection_symmetric': oracle_selection_sym,
        'oracle_epe_forward': oracle_epe_fwd_standard,
        'oracle_epe_symmetric': oracle_epe_sym_standard,
        'u_oracle_forward': u_oracle_fwd,
        'v_oracle_forward': v_oracle_fwd,
        'u_oracle_symmetric': u_oracle_sym,
        'v_oracle_symmetric': v_oracle_sym,
    }
    
    return {
        'results_full': results_full,
        'oracle': oracle
    }


def run_sweep_pipeline(frame1: np.ndarray,
                      frame2: np.ndarray,
                      configs: List[dict],
                      deltas: List[tuple],
                      u_truth: np.ndarray,
                      v_truth: np.ndarray,
                      valid_mask: np.ndarray,
                      n_workers: int = None,
                      epe_power: float = 2.0,
                      frame1_original: np.ndarray = None,
                      frame2_original: np.ndarray = None) -> Dict:
    """
    Run complete sweep pipeline: compute, upsample, compute oracle.
    
    Args:
        frame1, frame2: Input frames for OF computation
        configs: List of OF configurations
        deltas: List of perturbation vectors
        u_truth, v_truth: Ground truth flows
        valid_mask: Valid pixel mask
        n_workers: Number of parallel workers
        epe_power: Power for EPE aggregation (1.0=MAE, 2.0=MSE)
        frame1_original, frame2_original: Optional RGB frames for RGB photometric metrics
        
    Returns:
        dict with:
            - results_native: Native resolution results
            - results_full: Full resolution results
            - oracle: Oracle results dict
            - frame_constants: Normalization constants for the frame pair
    """
    H, W = u_truth.shape
    
    # Compute sweep (with frame_constants)
    results_native, frame_constants = compute_sweep(
        frame1, frame2, configs, deltas, n_workers,
        frame1_original, frame2_original,
        return_constants=True
    )
    
    # Upsample and compute oracle
    oracle_result = upsample_and_compute_oracle(
        results_native, H, W, u_truth, v_truth, valid_mask, epe_power
    )
    
    return {
        'results_native': results_native,
        'results_full': oracle_result['results_full'],
        'oracle': oracle_result['oracle'],
        'frame_constants': frame_constants
    }


def save_sweep_to_cache(exp_cache, results_full: List[Dict],
                       u_truth: Optional[np.ndarray], v_truth: Optional[np.ndarray],
                       valid_mask: np.ndarray,
                       oracle_epe_forward: Optional[float], oracle_epe_symmetric: Optional[float],
                       epe_power: float = 1.0,
                       frame_constants: dict = None):
    """Save sweep results to cache.
    
    Args:
        exp_cache: Experiment cache object
        results_full: List of result dicts from sweep
        u_truth, v_truth: Ground truth flow (can be None if no GT)
        valid_mask: Boolean mask for valid pixels
        oracle_epe_forward, oracle_epe_symmetric: Oracle EPE values (can be None)
        epe_power: Power for EPE computation (stored for cache validation)
        frame_constants: Normalization constants for the frame pair (auto-retrieved if None)
    """
    print("   Saving sweep results to cache...")
    
    # Auto-retrieve frame_constants if not provided
    if frame_constants is None:
        frame_constants = get_last_frame_constants()
    
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
    
    # Save frame_constants for multiplicative loss
    if frame_constants is not None:
        frame_constants_path = exp_cache.current_dir / 'frame_constants.json'
        with open(frame_constants_path, 'w') as f:
            json.dump(frame_constants, f, indent=2)
        print(f"   ✅ Saved sweep results, full data, and frame constants (epe_power={epe_power})")
    else:
        print(f"   ✅ Saved sweep results and full data (epe_power={epe_power})")


def load_sweep_from_cache(exp_cache, expected_epe_power: float = None) -> Dict:
    """
    Load sweep results from cache.
    
    Args:
        exp_cache: ExperimentCache instance
        expected_epe_power: If provided, warn if cached EPE power differs
    
    Returns:
        dict with:
            - sweep_df: Summary dataframe
            - results_full: Full results (if available)
            - oracle_epe_forward, oracle_epe_symmetric: Oracle EPE values
            - epe_power: EPE power used for cached computation
            - epe_power_mismatch: True if cached power differs from expected
            - frame_constants: Normalization constants (if available)
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
    
    # Load frame_constants if available
    frame_constants_path = exp_cache.current_dir / 'frame_constants.json'
    if frame_constants_path.exists():
        import json
        with open(frame_constants_path, 'r') as f:
            frame_constants = json.load(f)
        print(f"   ✅ Loaded frame constants")
    else:
        frame_constants = None
        print(f"   ⚠️  Frame constants not in cache")
    
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
        'epe_power_mismatch': epe_power_mismatch,
        'frame_constants': frame_constants
    }


if __name__ == "__main__":
    print("✅ Sweep computation module loaded")
