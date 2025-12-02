# File: src/evaluation/self_supervised.py
"""
Pure metric computation from pre-computed flows.

NO optical flow calls here - only consumes flows dict.

All metrics are normalized to be dimensionless and bounded [0, 1):
- Traction: error / hypot(error, pert_scale)
- Consistency: error / hypot(error, pert_scale)  [perturbed flows only]
- Perturbation RMS: noise / hypot(noise, pert_scale)
- Photometric: residual / hypot(residual, rms_diff)  [rms_diff = RMS of frame difference]
- Photometric RGB: same but using RGB Euclidean distance
- Photometric RGB Log: same but using log-RGB distance
- Speed (sym): raw magnitude of symmetric flow (NOT normalized here - done at selection time)
"""

import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.photometric import (
    compute_photometric_raw, 
    compute_photometric_windowed,
    compute_photometric_rgb,
    compute_photometric_rgb_log
)
from src.evaluation.symmetric_flow import symmetrize_flow_to_frame_A, symmetrize_flow_to_frame_B, warp_flow


def shift_frame(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift frame (needed for photometric)."""
    H, W = frame.shape[:2]
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x, map_y = x_grid - dx, y_grid - dy
    
    if frame.ndim == 3:
        # RGB frame
        return cv2.remap(frame, map_x, map_y, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, 0)
    else:
        # Grayscale frame
        return cv2.remap(frame, map_x, map_y, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, 0)


def is_rgb(frame: np.ndarray) -> bool:
    """Check if frame is RGB (3 channels)."""
    return frame.ndim == 3 and frame.shape[2] == 3


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale if RGB."""
    if is_rgb(frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return frame.astype(np.float32)


def compute_metrics_frame_A(frame_A, frame_B, flows, config, rms_diff: float):
    """
    Compute frame A metrics from flows dict.

    Returns dimensionless, normalized metrics.
    
    Args:
        frame_A, frame_B: Input frames (grayscale or RGB)
        flows: Flows dict from compute_all_flows
        config: Algorithm config dict
        rms_diff: RMS difference between unwarped frames (for photometric normalization)
    """
    H, W = frame_A.shape[:2]
    winsize = config.get('winsize', 15)
    
    # Check if we have RGB frames
    has_rgb = is_rgb(frame_A) and is_rgb(frame_B)
    
    # Get grayscale versions for grayscale photometric
    frame_A_gray = to_grayscale(frame_A)
    frame_B_gray = to_grayscale(frame_B)

    u_AB, v_AB = flows['base']['AB']
    u_BA, v_BA = flows['base']['BA']

    # Accumulators for raw metrics
    traction_acc = np.zeros((H, W), dtype=np.float32)
    consistency_acc = np.zeros((H, W), dtype=np.float32)
    photo_acc_plus = np.zeros((H, W), dtype=np.float32)
    photo_acc_minus = np.zeros((H, W), dtype=np.float32)
    
    # RGB accumulators (only if RGB input)
    if has_rgb:
        photo_rgb_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_acc_minus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_log_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_log_acc_minus = np.zeros((H, W), dtype=np.float32)

    # Perturbations
    for pert in flows['perturbations_A']:
        dx, dy = pert['delta']
        perturbation_mag = np.hypot(dx, dy)

        u_plus, v_plus = pert['A_to_B_plus']
        u_minus, v_minus = pert['A_to_B_minus']
        u_bwd_plus, v_bwd_plus = pert['B_plus_to_A']
        u_bwd_minus, v_bwd_minus = pert['B_minus_to_A']

        # Traction (with bounded normalization per-perturbation)
        diff_u = u_plus - u_minus
        diff_v = v_plus - v_minus
        traction_raw = np.sqrt((diff_u - 2 * dx) ** 2 + (diff_v - 2 * dy) ** 2)
        # Bounded normalization: error / hypot(error, scale)
        traction_normalized = traction_raw / np.hypot(traction_raw, perturbation_mag)
        traction_acc += traction_normalized

        # Consistency +δ
        u_bwd_plus_w, v_bwd_plus_w = warp_flow(u_bwd_plus, v_bwd_plus, u_plus, v_plus)
        cons_plus = np.sqrt((u_plus + u_bwd_plus_w) ** 2 + (v_plus + v_bwd_plus_w) ** 2)
        consistency_acc += cons_plus

        # Consistency -δ
        u_bwd_minus_w, v_bwd_minus_w = warp_flow(u_bwd_minus, v_bwd_minus, u_minus, v_minus)
        cons_minus = np.sqrt((u_minus + u_bwd_minus_w) ** 2 + (v_minus + v_bwd_minus_w) ** 2)
        consistency_acc += cons_minus

        # Photometric grayscale (raw, will normalize later)
        B_plus_gray = shift_frame(frame_B_gray, dx, dy)
        B_minus_gray = shift_frame(frame_B_gray, -dx, -dy)
        photo_acc_plus += compute_photometric_raw(frame_A_gray, B_plus_gray, u_plus, v_plus)
        photo_acc_minus += compute_photometric_raw(frame_A_gray, B_minus_gray, u_minus, v_minus)
        
        # Photometric RGB variants (if RGB input)
        if has_rgb:
            B_plus_rgb = shift_frame(frame_B, dx, dy)
            B_minus_rgb = shift_frame(frame_B, -dx, -dy)
            
            # RGB Euclidean
            photo_rgb_acc_plus += compute_photometric_rgb(frame_A, B_plus_rgb, u_plus, v_plus)
            photo_rgb_acc_minus += compute_photometric_rgb(frame_A, B_minus_rgb, u_minus, v_minus)
            
            # Log-RGB
            photo_rgb_log_acc_plus += compute_photometric_rgb_log(frame_A, B_plus_rgb, u_plus, v_plus)
            photo_rgb_log_acc_minus += compute_photometric_rgb_log(frame_A, B_minus_rgb, u_minus, v_minus)

    n_deltas = len(flows['perturbations_A'])

    # ========================================================================
    # Finalize Traction (already normalized, just average)
    # ========================================================================
    traction_A = traction_acc / n_deltas  # dimensionless, [0, 1)

    # ========================================================================
    # Finalize Consistency (bounded normalization by pert_scale)
    # ========================================================================
    # Compute perturbation scale (used for both consistency and perturbation_rms)
    pert_magnitudes = [np.hypot(dx, dy) for dx, dy in [pert['delta'] for pert in flows['perturbations_A']]]
    pert_scale = np.sqrt(np.mean([m**2 for m in pert_magnitudes]))
    
    consistency_raw = consistency_acc / (2 * n_deltas)  # pixels (perturbed only, no base)

    # Bounded normalization: error / hypot(error, scale)
    consistency_A = consistency_raw / np.hypot(consistency_raw, pert_scale)  # [0, 1)

    # ========================================================================
    # Finalize Photometric Grayscale (bounded normalization by rms_diff)
    # ========================================================================
    photometric_raw_mean = (photo_acc_plus + photo_acc_minus) / (2 * n_deltas)

    # Photometric scale: RMS difference between frames, minimum 1.0 (one intensity level)
    photo_scale = max(rms_diff, 1.0)

    # Raw (unsmoothed) normalized
    photometric_A_raw = photometric_raw_mean / np.hypot(photometric_raw_mean, photo_scale)

    # Apply windowed smoothing then normalize
    photometric_smoothed = compute_photometric_windowed(photometric_raw_mean, winsize)
    photometric_A = photometric_smoothed / np.hypot(photometric_smoothed, photo_scale)

    # ========================================================================
    # Finalize Photometric RGB variants (if available)
    # ========================================================================
    if has_rgb:
        # RGB Euclidean
        photo_rgb_raw_mean = (photo_rgb_acc_plus + photo_rgb_acc_minus) / (2 * n_deltas)
        # Scale for RGB: sqrt(3) * grayscale scale (since RGB distance is larger)
        photo_rgb_scale = max(rms_diff * np.sqrt(3), 1.0)
        photo_rgb_smoothed = compute_photometric_windowed(photo_rgb_raw_mean, winsize)
        photometric_rgb_A = photo_rgb_smoothed / np.hypot(photo_rgb_smoothed, photo_rgb_scale)
        
        # Log-RGB
        photo_rgb_log_raw_mean = (photo_rgb_log_acc_plus + photo_rgb_log_acc_minus) / (2 * n_deltas)
        # Scale for log-RGB: use 0.1 as typical log difference (tunable)
        photo_rgb_log_scale = 0.1
        photo_rgb_log_smoothed = compute_photometric_windowed(photo_rgb_log_raw_mean, winsize)
        photometric_rgb_log_A = photo_rgb_log_smoothed / np.hypot(photo_rgb_log_smoothed, photo_rgb_log_scale)

    # ========================================================================
    # Compute Displacement Sensitivity (perturbation error metric)
    # ========================================================================
    # CORRECTED VERSION (2024-11-22):
    # 
    # Old (WRONG) approach:
    #   - Computed mean across base + all perturbations
    #   - Measured std relative to this mixed mean
    #   - Normalized by flow magnitude (exploded for small flows)
    #   - Result: Direction-dependent, unreliable correlations with EPE
    #
    # New (CORRECT) approach:
    #   - Base flow is the fixed reference point
    #   - Measure deviation of each perturbed flow FROM base
    #   - Normalize by perturbation scale (not flow magnitude)
    #   - Result: Measures "perturbation error" - how much algorithm fails to cancel perturbations
    #
    # Interpretation:
    #   Value = 0.0: Perfect! Algorithm perfectly cancels perturbations (ideal)
    #   Value > 0.0: Imperfect. Algorithm output changes with perturbations (error)
    #   Units: dimensionless (pixels per pixel of perturbation)
    
    # Base flow is the reference
    u_base, v_base = u_AB, v_AB
    
    # Collect deviations from base for each perturbation
    deviation_list = []
    
    for pert in flows['perturbations_A']:
        dx, dy = pert['delta']
        
        u_plus, v_plus = pert['A_to_B_plus']
        u_minus, v_minus = pert['A_to_B_minus']
        
        # Correct for perturbation shift
        u_plus_corrected = u_plus - dx
        v_plus_corrected = v_plus - dy
        u_minus_corrected = u_minus - (-dx)  # = u_minus + dx
        v_minus_corrected = v_minus - (-dy)  # = v_minus + dy
        
        # Compute deviation magnitude from base flow
        dev_plus = np.sqrt((u_plus_corrected - u_base)**2 + (v_plus_corrected - v_base)**2)
        dev_minus = np.sqrt((u_minus_corrected - u_base)**2 + (v_minus_corrected - v_base)**2)
        
        deviation_list.append(dev_plus)
        deviation_list.append(dev_minus)
    
    # Stack deviations and compute RMS (root mean square deviation)
    deviation_stack = np.stack(deviation_list, axis=0)  # Shape: (n_perturbations, H, W)
    noise = np.sqrt(np.mean(deviation_stack**2, axis=0))  # RMS deviation in pixels
    
    # Bounded normalization: noise / hypot(noise, scale) -> [0, 1)
    # (pert_scale already computed above for consistency)
    displacements_sensitivity_A2B = noise / np.hypot(noise, pert_scale)

    # ========================================================================
    # Build result dict
    # ========================================================================
    result = {
        'traction_A': traction_A,  # dimensionless, [0, 1)
        'consistency_A': consistency_A,  # dimensionless, [0, 1)
        'photometric_A_raw': photometric_A_raw,  # dimensionless (unsmoothed)
        'photometric_A': photometric_A,  # dimensionless (smoothed)
        'displacements_sensitivity_A2B': displacements_sensitivity_A2B,  # dimensionless, [0, 1)
    }
    
    # Add RGB metrics if available
    if has_rgb:
        result['photometric_rgb_A'] = photometric_rgb_A
        result['photometric_rgb_log_A'] = photometric_rgb_log_A

    return result


def compute_metrics_frame_B(frame_A, frame_B, flows, config, rms_diff: float):
    """
    Compute frame B metrics from flows dict.

    Returns dimensionless, normalized metrics.
    
    Args:
        frame_A, frame_B: Input frames (grayscale or RGB)
        flows: Flows dict from compute_all_flows
        config: Algorithm config dict
        rms_diff: RMS difference between unwarped frames (for photometric normalization)
    """
    H, W = frame_B.shape[:2]
    winsize = config.get('winsize', 15)
    
    # Check if we have RGB frames
    has_rgb = is_rgb(frame_A) and is_rgb(frame_B)
    
    # Get grayscale versions for grayscale photometric
    frame_A_gray = to_grayscale(frame_A)
    frame_B_gray = to_grayscale(frame_B)

    u_AB, v_AB = flows['base']['AB']
    u_BA, v_BA = flows['base']['BA']

    # Accumulators for raw metrics
    traction_acc = np.zeros((H, W), dtype=np.float32)
    consistency_acc = np.zeros((H, W), dtype=np.float32)
    photo_acc_plus = np.zeros((H, W), dtype=np.float32)
    photo_acc_minus = np.zeros((H, W), dtype=np.float32)
    
    # RGB accumulators (only if RGB input)
    if has_rgb:
        photo_rgb_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_acc_minus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_log_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_log_acc_minus = np.zeros((H, W), dtype=np.float32)

    # Perturbations
    for pert in flows['perturbations_B']:
        dx, dy = pert['delta']
        perturbation_mag = np.hypot(dx, dy)

        u_plus, v_plus = pert['B_to_A_plus']
        u_minus, v_minus = pert['B_to_A_minus']
        u_bwd_plus, v_bwd_plus = pert['A_plus_to_B']
        u_bwd_minus, v_bwd_minus = pert['A_minus_to_B']

        # Traction (with bounded normalization per-perturbation)
        diff_u = u_plus - u_minus
        diff_v = v_plus - v_minus
        traction_raw = np.sqrt((diff_u - 2 * dx) ** 2 + (diff_v - 2 * dy) ** 2)
        # Bounded normalization: error / hypot(error, scale)
        traction_normalized = traction_raw / np.hypot(traction_raw, perturbation_mag)
        traction_acc += traction_normalized

        # Consistency +δ
        u_bwd_plus_w, v_bwd_plus_w = warp_flow(u_bwd_plus, v_bwd_plus, u_plus, v_plus)
        cons_plus = np.sqrt((u_plus + u_bwd_plus_w) ** 2 + (v_plus + v_bwd_plus_w) ** 2)
        consistency_acc += cons_plus

        # Consistency -δ
        u_bwd_minus_w, v_bwd_minus_w = warp_flow(u_bwd_minus, v_bwd_minus, u_minus, v_minus)
        cons_minus = np.sqrt((u_minus + u_bwd_minus_w) ** 2 + (v_minus + v_bwd_minus_w) ** 2)
        consistency_acc += cons_minus

        # Photometric grayscale (raw, will normalize later)
        A_plus_gray = shift_frame(frame_A_gray, dx, dy)
        A_minus_gray = shift_frame(frame_A_gray, -dx, -dy)
        photo_acc_plus += compute_photometric_raw(frame_B_gray, A_plus_gray, u_plus, v_plus)
        photo_acc_minus += compute_photometric_raw(frame_B_gray, A_minus_gray, u_minus, v_minus)
        
        # Photometric RGB variants (if RGB input)
        if has_rgb:
            A_plus_rgb = shift_frame(frame_A, dx, dy)
            A_minus_rgb = shift_frame(frame_A, -dx, -dy)
            
            # RGB Euclidean
            photo_rgb_acc_plus += compute_photometric_rgb(frame_B, A_plus_rgb, u_plus, v_plus)
            photo_rgb_acc_minus += compute_photometric_rgb(frame_B, A_minus_rgb, u_minus, v_minus)
            
            # Log-RGB
            photo_rgb_log_acc_plus += compute_photometric_rgb_log(frame_B, A_plus_rgb, u_plus, v_plus)
            photo_rgb_log_acc_minus += compute_photometric_rgb_log(frame_B, A_minus_rgb, u_minus, v_minus)

    n_deltas = len(flows['perturbations_B'])

    # ========================================================================
    # Finalize Traction (already normalized, just average)
    # ========================================================================
    traction_B = traction_acc / n_deltas  # dimensionless, [0, 1)

    # ========================================================================
    # Finalize Consistency (bounded normalization by pert_scale)
    # ========================================================================
    # Compute perturbation scale (used for both consistency and perturbation_rms)
    pert_magnitudes = [np.hypot(dx, dy) for dx, dy in [pert['delta'] for pert in flows['perturbations_B']]]
    pert_scale = np.sqrt(np.mean([m**2 for m in pert_magnitudes]))
    
    consistency_raw = consistency_acc / (2 * n_deltas)  # pixels (perturbed only, no base)

    # Bounded normalization: error / hypot(error, scale)
    consistency_B = consistency_raw / np.hypot(consistency_raw, pert_scale)  # [0, 1)

    # ========================================================================
    # Finalize Photometric Grayscale (bounded normalization by rms_diff)
    # ========================================================================
    photometric_raw_mean = (photo_acc_plus + photo_acc_minus) / (2 * n_deltas)

    # Photometric scale: RMS difference between frames, minimum 1.0 (one intensity level)
    photo_scale = max(rms_diff, 1.0)

    # Raw (unsmoothed) normalized
    photometric_B_raw = photometric_raw_mean / np.hypot(photometric_raw_mean, photo_scale)

    # Apply windowed smoothing then normalize
    photometric_smoothed = compute_photometric_windowed(photometric_raw_mean, winsize)
    photometric_B = photometric_smoothed / np.hypot(photometric_smoothed, photo_scale)

    # ========================================================================
    # Finalize Photometric RGB variants (if available)
    # ========================================================================
    if has_rgb:
        # RGB Euclidean
        photo_rgb_raw_mean = (photo_rgb_acc_plus + photo_rgb_acc_minus) / (2 * n_deltas)
        # Scale for RGB: sqrt(3) * grayscale scale (since RGB distance is larger)
        photo_rgb_scale = max(rms_diff * np.sqrt(3), 1.0)
        photo_rgb_smoothed = compute_photometric_windowed(photo_rgb_raw_mean, winsize)
        photometric_rgb_B = photo_rgb_smoothed / np.hypot(photo_rgb_smoothed, photo_rgb_scale)
        
        # Log-RGB
        photo_rgb_log_raw_mean = (photo_rgb_log_acc_plus + photo_rgb_log_acc_minus) / (2 * n_deltas)
        # Scale for log-RGB: use 0.1 as typical log difference (tunable)
        photo_rgb_log_scale = 0.1
        photo_rgb_log_smoothed = compute_photometric_windowed(photo_rgb_log_raw_mean, winsize)
        photometric_rgb_log_B = photo_rgb_log_smoothed / np.hypot(photo_rgb_log_smoothed, photo_rgb_log_scale)

    # ========================================================================
    # Compute Displacement Sensitivity (perturbation error metric)
    # ========================================================================
    # CORRECTED VERSION (2024-11-22):
    # Same logic as frame A version - see detailed comments there
    
    # Base flow is the reference
    u_base, v_base = u_BA, v_BA
    
    # Collect deviations from base for each perturbation
    deviation_list = []
    
    for pert in flows['perturbations_B']:
        dx, dy = pert['delta']
        
        u_plus, v_plus = pert['B_to_A_plus']
        u_minus, v_minus = pert['B_to_A_minus']
        
        # Correct for perturbation shift
        u_plus_corrected = u_plus - dx
        v_plus_corrected = v_plus - dy
        u_minus_corrected = u_minus - (-dx)  # = u_minus + dx
        v_minus_corrected = v_minus - (-dy)  # = v_minus + dy
        
        # Compute deviation magnitude from base flow
        dev_plus = np.sqrt((u_plus_corrected - u_base)**2 + (v_plus_corrected - v_base)**2)
        dev_minus = np.sqrt((u_minus_corrected - u_base)**2 + (v_minus_corrected - v_base)**2)
        
        deviation_list.append(dev_plus)
        deviation_list.append(dev_minus)
    
    # Stack deviations and compute RMS
    deviation_stack = np.stack(deviation_list, axis=0)
    noise = np.sqrt(np.mean(deviation_stack**2, axis=0))
    
    # Bounded normalization: noise / hypot(noise, scale) -> [0, 1)
    # (pert_scale already computed above for consistency)
    displacements_sensitivity_B2A = noise / np.hypot(noise, pert_scale)

    # ========================================================================
    # Build result dict
    # ========================================================================
    result = {
        'traction_B': traction_B,  # dimensionless, [0, 1)
        'consistency_B': consistency_B,  # dimensionless, [0, 1)
        'photometric_B_raw': photometric_B_raw,  # dimensionless (unsmoothed)
        'photometric_B': photometric_B,  # dimensionless (smoothed)
        'displacements_sensitivity_B2A': displacements_sensitivity_B2A,  # dimensionless, [0, 1)
    }
    
    # Add RGB metrics if available
    if has_rgb:
        result['photometric_rgb_B'] = photometric_rgb_B
        result['photometric_rgb_log_B'] = photometric_rgb_log_B

    return result


def compute_metrics_from_flows(frame_A, frame_B, flows, config, rms_diff: float, verbose: bool = True):
    """
    Main entry point - compute all metrics from flows dict.

    NO optical flow calls here!

    Returns separated flows and metrics as dimensionless, normalized values.
    
    Args:
        frame_A, frame_B: Input frames (grayscale or RGB)
        flows: Flows dict from compute_all_flows
        config: Algorithm config dict
        rms_diff: RMS difference between unwarped frames (for photometric normalization)
        verbose: Print progress messages
    
    Returns:
        {
            'flows': {u_AB, v_AB, u_BA, v_BA, u_sym_A, v_sym_A, u_sym_B, v_sym_B},
            'metrics': {traction_A, consistency_A, photometric_A, photometric_rgb_A, 
                       photometric_rgb_log_A, speed_sym_A, ..., and _B variants}
        }
    """
    if verbose:
        print(f"      Computing metrics from flows...")

    u_AB, v_AB = flows['base']['AB']
    u_BA, v_BA = flows['base']['BA']

    metrics_A = compute_metrics_frame_A(frame_A, frame_B, flows, config, rms_diff)
    metrics_B = compute_metrics_frame_B(frame_A, frame_B, flows, config, rms_diff)

    u_sym_A, v_sym_A = symmetrize_flow_to_frame_A(u_AB, v_AB, u_BA, v_BA)
    u_sym_B, v_sym_B = symmetrize_flow_to_frame_B(u_AB, v_AB, u_BA, v_BA)

    # ========================================================================
    # Compute speed metrics (raw magnitude - NOT normalized here)
    # Normalization by global max across configs happens at selection time
    # ========================================================================
    speed_sym_A = np.hypot(u_sym_A, v_sym_A).astype(np.float32)
    speed_sym_B = np.hypot(u_sym_B, v_sym_B).astype(np.float32)

    # Separate flows and metrics (NO config here - handled by caller)
    flows_dict = {
        'u_AB': u_AB, 'v_AB': v_AB,
        'u_BA': u_BA, 'v_BA': v_BA,
        'u_sym_A': u_sym_A, 'v_sym_A': v_sym_A,
        'u_sym_B': u_sym_B, 'v_sym_B': v_sym_B
    }

    metrics_dict = {
        **metrics_A,
        **metrics_B,
        'speed_sym_A': speed_sym_A,  # raw pixels, NOT normalized
        'speed_sym_B': speed_sym_B,  # raw pixels, NOT normalized
    }

    return {
        'flows': flows_dict,
        'metrics': metrics_dict
    }
