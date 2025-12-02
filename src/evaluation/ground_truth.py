# File: src/evaluation/ground_truth.py
"""
Error metrics for optical flow evaluation.

Computes various error metrics between estimated and ground truth flows.
"""

import numpy as np
import sys


def compute_epe(u: np.ndarray,
               v: np.ndarray,
               u_true: np.ndarray,
               v_true: np.ndarray,
               valid_mask: np.ndarray = None,
               power: float = 1.0) -> np.ndarray:
    """
    Compute End-Point Error (EPE) with configurable power.
    
    Args:
        u, v: Estimated flow components (H, W)
        u_true, v_true: Ground truth flow components (H, W)
        valid_mask: Boolean mask (H, W) indicating valid pixels
                   If None, all pixels considered valid
        power: Exponent for error aggregation (default: 1.0)
               1.0 = linear (MAE - Mean Absolute Error)
               2.0 = squared (MSE - Mean Squared Error, standard in CV)
               >2.0 = higher penalty for outliers
    
    Returns:
        EPE^power map (H, W) as float32
        
    Note:
        - Always computes standard EPE = sqrt((u-u_true)^2 + (v-v_true)^2) first
        - Then applies power transformation: EPE^power
        - When power=1.0, returns standard EPE
        - When power=2.0, returns EPE^2 for MSE-based optimization
        
    Example:
        >>> u = np.ones((256, 256), dtype=np.float32) * 2.0
        >>> v = np.ones((256, 256), dtype=np.float32) * 1.0
        >>> u_true = np.ones((256, 256), dtype=np.float32) * 2.5
        >>> v_true = np.ones((256, 256), dtype=np.float32) * 1.0
        >>> epe = compute_epe(u, v, u_true, v_true, power=1.0)
        >>> np.allclose(epe, 0.5)
        True
        >>> epe_squared = compute_epe(u, v, u_true, v_true, power=2.0)
        >>> np.allclose(epe_squared, 0.25)
        True
    """
    if u.shape != u_true.shape or v.shape != v_true.shape:
        print(f"❌ ERROR: Flow shapes don't match")
        print(f"   Estimated: u={u.shape}, v={v.shape}")
        print(f"   Ground truth: u_true={u_true.shape}, v_true={v_true.shape}")
        sys.exit(1)
    
    # ALWAYS compute standard EPE first
    epe = np.sqrt((u - u_true)**2 + (v - v_true)**2)
    
    # Apply power transformation if requested
    if power != 1.0:
        epe = np.power(epe, power)
    
    # Set invalid pixels to NaN if mask provided
    if valid_mask is not None:
        if valid_mask.shape != u.shape:
            print(f"❌ ERROR: valid_mask shape {valid_mask.shape} doesn't match flow shape {u.shape}")
            sys.exit(1)
        epe = epe.astype(np.float32)
        epe[~valid_mask] = np.nan
    
    return epe.astype(np.float32)


# File: src/evaluation/ground_truth.py (add this function)

def compute_texture_weighted_epe(u_pred: np.ndarray,
                                 v_pred: np.ndarray,
                                 u_true: np.ndarray,
                                 v_true: np.ndarray,
                                 frame_reference: np.ndarray,
                                 valid_mask: np.ndarray,
                                 gradient_sigma: float = 1.0,
                                 texture_sigma: float = 0.0) -> tuple[float, np.ndarray]:
    """
    Compute texture-weighted endpoint error.

    Weights EPE by local gradient magnitude squared, giving more importance
    to regions with strong texture where optical flow is well-constrained.

    Args:
        u_pred, v_pred: Predicted flow fields (H, W)
        u_true, v_true: Ground truth flow fields (H, W)
        frame_reference: Reference frame for computing texture weights (H, W)
        valid_mask: Boolean mask of valid correspondence regions (H, W)
        gradient_sigma: Smoothing applied to weight map (default: 1.0)
        texture_sigma: Pre-smoothing applied to frame before gradients (default: 0.0)
                      Set > 0 for noisy images (e.g., 1.5-2.0)

    Returns:
        weighted_epe: Scalar weighted mean EPE
        weight_map: Spatial distribution of texture weights (H, W)

    Example:
        >>> weighted_epe, weights = compute_texture_weighted_epe(
        ...     u, v, u_true, v_true, frame1_gray, valid_mask,
        ...     gradient_sigma=1.0, texture_sigma=0.0
        ... )
    """
    from scipy.ndimage import gaussian_filter

    # Compute standard EPE
    EPE = np.sqrt((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2)

    # Pre-smooth frame if specified (for noisy images)
    if texture_sigma > 0:
        frame_smoothed = gaussian_filter(frame_reference.astype(np.float32), sigma=texture_sigma)
    else:
        frame_smoothed = frame_reference

    # Compute gradients
    Ix = np.gradient(frame_smoothed, axis=1)
    Iy = np.gradient(frame_smoothed, axis=0)

    # Gradient magnitude squared (emphasizes strong texture)
    gradient_mag_sq = Ix ** 2 + Iy ** 2

    # Smooth weight map to reduce single-pixel noise
    if gradient_sigma > 0:
        weight_map = gaussian_filter(gradient_mag_sq, sigma=gradient_sigma)
    else:
        weight_map = gradient_mag_sq

    # Apply valid mask to weights
    weight_map = weight_map * valid_mask.astype(np.float32)

    # Compute weighted mean EPE
    total_weight = np.sum(weight_map)

    if total_weight > 0:
        weighted_epe = np.sum(EPE * weight_map) / total_weight
    else:
        # Fallback if no texture (shouldn't happen)
        weighted_epe = np.nan

    return weighted_epe, weight_map


def compute_aee(u: np.ndarray,
               v: np.ndarray,
               u_true: np.ndarray,
               v_true: np.ndarray,
               valid_mask: np.ndarray = None) -> np.ndarray:
    """
    Compute Angular Error (AEE) between estimated and ground truth flow.
    
    Args:
        u, v: Estimated flow components
        u_true, v_true: Ground truth flow components
        valid_mask: Boolean mask indicating valid pixels
    
    Returns:
        AEE map (H, W) in degrees
        
    Note:
        Angular error measures the angle between flow vectors in 3D space (u, v, 1).
        Useful when flow magnitude varies significantly.
    """
    # Convert 2D flow to 3D (u, v, 1) for angular computation
    # This is the standard approach in optical flow literature
    norm_est = np.sqrt(u**2 + v**2 + 1)
    norm_true = np.sqrt(u_true**2 + v_true**2 + 1)
    
    # Dot product of normalized vectors
    dot_product = (u * u_true + v * v_true + 1) / (norm_est * norm_true)
    
    # Clip to handle numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Angular error in radians, then convert to degrees
    aee = np.arccos(dot_product)
    aee_degrees = np.rad2deg(aee)
    
    # Set invalid pixels to NaN if mask provided
    if valid_mask is not None:
        if valid_mask.shape != u.shape:
            print(f"❌ ERROR: valid_mask shape doesn't match flow shape")
            sys.exit(1)
        aee_degrees = aee_degrees.astype(np.float32)
        aee_degrees[~valid_mask] = np.nan
    
    return aee_degrees.astype(np.float32)


def compute_mean_epe(u: np.ndarray,
                    v: np.ndarray,
                    u_true: np.ndarray,
                    v_true: np.ndarray,
                    valid_mask: np.ndarray = None,
                    power: float = 1.0) -> float:
    """
    Compute mean End-Point Error with configurable power.
    
    Args:
        u, v: Estimated flow
        u_true, v_true: Ground truth flow
        valid_mask: Valid pixel mask
        power: Exponent for error aggregation
               1.0 = MAE (Mean Absolute Error)
               2.0 = MSE (Mean Squared Error)
    
    Returns:
        Mean EPE^power as scalar float
        
    Note:
        Returns mean(EPE^power), NOT (mean(EPE))^power
        To get RMSE from MSE: sqrt(compute_mean_epe(..., power=2.0))
        But typically we optimize mean(EPE^power) directly.
    """
    epe_powered = compute_epe(u, v, u_true, v_true, valid_mask, power=power)
    
    # Compute mean over valid (non-NaN) pixels
    mean_epe = np.nanmean(epe_powered)
    
    return float(mean_epe)


def compute_mean_aee(u: np.ndarray,
                    v: np.ndarray,
                    u_true: np.ndarray,
                    v_true: np.ndarray,
                    valid_mask: np.ndarray = None) -> float:
    """
    Compute mean Angular Error over valid pixels.
    
    Returns:
        Mean AEE in degrees as scalar float
    """
    aee = compute_aee(u, v, u_true, v_true, valid_mask)
    
    # Compute mean over valid pixels
    mean_aee = np.nanmean(aee)
    
    return float(mean_aee)


def compute_flow_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute flow magnitude.
    
    Args:
        u, v: Flow components
    
    Returns:
        Flow magnitude map
    """
    return np.sqrt(u**2 + v**2).astype(np.float32)


def compute_flow_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute flow angle in degrees.
    
    Args:
        u, v: Flow components
    
    Returns:
        Flow angle map in degrees [0, 360)
    """
    angle_rad = np.arctan2(v, u)
    angle_deg = np.rad2deg(angle_rad)
    # Convert to [0, 360) range
    angle_deg = (angle_deg + 360) % 360
    
    return angle_deg.astype(np.float32)


if __name__ == "__main__":
    # Test metrics
    print("🧪 Testing error metrics...")
    
    # Test EPE with perfect match
    u = np.ones((256, 256), dtype=np.float32) * 2.0
    v = np.ones((256, 256), dtype=np.float32) * 1.0
    
    epe_perfect = compute_epe(u, v, u, v)
    print(f"✅ EPE (perfect match): {epe_perfect.mean():.6f} (expected: 0.0)")
    assert np.allclose(epe_perfect, 0.0), "EPE should be 0 for perfect match"
    
    # Test EPE with known error
    u_est = u + 0.5  # Off by 0.5 in x
    epe_known = compute_epe(u_est, v, u, v)
    print(f"✅ EPE (known error): {epe_known.mean():.6f} (expected: 0.5)")
    assert np.allclose(epe_known, 0.5), "EPE should be 0.5"
    
    # Test EPE with power=2.0
    epe_squared = compute_epe(u_est, v, u, v, power=2.0)
    print(f"✅ EPE^2 (power=2.0): {epe_squared.mean():.6f} (expected: 0.25)")
    assert np.allclose(epe_squared, 0.25), "EPE^2 should be 0.25"
    
    # Test with valid mask
    valid_mask = np.ones((256, 256), dtype=bool)
    valid_mask[:50, :] = False  # Invalidate top 50 rows
    
    epe_masked = compute_epe(u_est, v, u, v, valid_mask)
    n_valid = np.sum(~np.isnan(epe_masked))
    print(f"✅ EPE with mask: {n_valid} valid pixels (expected: {256*206})")
    assert n_valid == 256 * 206, "Valid pixel count mismatch"
    
    # Test mean EPE with different powers
    mean_epe_p1 = compute_mean_epe(u_est, v, u, v, valid_mask, power=1.0)
    mean_epe_p2 = compute_mean_epe(u_est, v, u, v, valid_mask, power=2.0)
    print(f"✅ Mean EPE (p=1.0): {mean_epe_p1:.6f} (expected: 0.5)")
    print(f"✅ Mean EPE (p=2.0): {mean_epe_p2:.6f} (expected: 0.25)")
    
    # Test AEE
    aee = compute_aee(u_est, v, u, v)
    print(f"✅ AEE computed: mean={aee.mean():.2f} degrees")
    
    # Test flow magnitude
    mag = compute_flow_magnitude(u, v)
    expected_mag = np.sqrt(2**2 + 1**2)
    print(f"✅ Flow magnitude: {mag.mean():.6f} (expected: {expected_mag:.6f})")
    assert np.allclose(mag, expected_mag), "Magnitude calculation error"
    
    # Test flow angle
    angle = compute_flow_angle(u, v)
    expected_angle = np.rad2deg(np.arctan2(1.0, 2.0))
    print(f"✅ Flow angle: {angle.mean():.2f}° (expected: {expected_angle:.2f}°)")
    assert np.allclose(angle, expected_angle), "Angle calculation error"
    
    # Test with different flow vectors
    u_test = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)
    v_test = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    angles = compute_flow_angle(u_test, v_test)
    print(f"✅ Flow angles for test vectors: {angles[0]}")
    print(f"   Expected: [0°, 90°, 180°]")
    
    print("\n✨ All metric tests passed!")
