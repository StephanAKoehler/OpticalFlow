# File: src/evaluation/photometric.py
"""
Photometric error computation for optical flow quality assessment.

Measures alignment quality by comparing warped frames.

Supports:
- Grayscale (legacy)
- RGB with Euclidean distance
- Log-RGB for perceptually meaningful comparisons
"""

import numpy as np
import cv2
import sys
from scipy.ndimage import gaussian_filter


def warp_frame(frame: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Warp a frame using optical flow (backward warp).
    
    Args:
        frame: Source frame (H, W) or (H, W, C) to be warped
        u: Horizontal flow component (H, W)
        v: Vertical flow component (H, W)
    
    Returns:
        warped_frame: Same shape as input, warped according to flow
        
    CRITICAL: Flow direction must be REVERSED for warping!
        - Flow (u, v) describes motion: source_pixel → target_pixel
        - To warp source to target, we need: target_pixel ← source_pixel
        - Therefore: map coordinates = current_position - flow
    """
    H, W = frame.shape[:2]
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    
    # Apply BACKWARD flow to get source coordinates
    map_x = x_grid - u
    map_y = y_grid - v
    
    # Warp frame
    warped = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return warped


def compute_photometric_raw(frame_source: np.ndarray,
                            frame_target: np.ndarray,
                            u: np.ndarray,
                            v: np.ndarray) -> np.ndarray:
    """
    Compute raw per-pixel photometric error (grayscale, legacy).
    
    Args:
        frame_source: Source frame (H, W) to be warped
        frame_target: Target frame (H, W) for comparison
        u: Horizontal flow component (H, W)
        v: Vertical flow component (H, W)
    
    Returns:
        photometric_error: (H, W) array of per-pixel alignment errors
    """
    # Warp source frame
    frame_source_warped = warp_frame(frame_source, u, v)
    
    # Compute photometric error
    photometric_error = np.abs(frame_target - frame_source_warped).astype(np.float32)
    
    return photometric_error


def compute_photometric_rgb(frame_source: np.ndarray,
                            frame_target: np.ndarray,
                            u: np.ndarray,
                            v: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel photometric error using RGB Euclidean distance.
    
    Args:
        frame_source: Source frame (H, W, 3) RGB to be warped
        frame_target: Target frame (H, W, 3) RGB for comparison
        u: Horizontal flow component (H, W)
        v: Vertical flow component (H, W)
    
    Returns:
        photometric_error: (H, W) array of per-pixel RGB distances
    """
    # Ensure RGB
    if frame_source.ndim != 3 or frame_source.shape[2] != 3:
        raise ValueError(f"Expected RGB frame (H, W, 3), got shape {frame_source.shape}")
    
    # Warp source frame
    frame_source_warped = warp_frame(frame_source.astype(np.float32), u, v)
    frame_target_f = frame_target.astype(np.float32)
    
    # Compute RGB Euclidean distance
    diff = frame_target_f - frame_source_warped
    photometric_error = np.sqrt(np.sum(diff**2, axis=2))
    
    return photometric_error.astype(np.float32)


def compute_photometric_rgb_log(frame_source: np.ndarray,
                                frame_target: np.ndarray,
                                u: np.ndarray,
                                v: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel photometric error using log-RGB Euclidean distance.
    
    Log transform provides perceptually meaningful comparisons:
    - Equal weight to dark and bright regions
    - Ratio-based comparison (scale-invariant)
    
    Args:
        frame_source: Source frame (H, W, 3) RGB to be warped
        frame_target: Target frame (H, W, 3) RGB for comparison
        u: Horizontal flow component (H, W)
        v: Vertical flow component (H, W)
    
    Returns:
        photometric_error: (H, W) array of per-pixel log-RGB distances
    """
    # Ensure RGB
    if frame_source.ndim != 3 or frame_source.shape[2] != 3:
        raise ValueError(f"Expected RGB frame (H, W, 3), got shape {frame_source.shape}")
    
    # Warp source frame
    frame_source_warped = warp_frame(frame_source.astype(np.float32), u, v)
    frame_target_f = frame_target.astype(np.float32)
    
    # Compute adaptive eps = max(RGB) / 100
    # Use target frame max (always valid) as primary reference
    max_val = max(frame_target_f.max(), 1.0)  # At least 1.0 for [0,1] normalized images
    eps = max(max_val / 100.0, 1e-6)  # Floor at 1e-6 for safety
    
    # Ensure no negative values before log (warped boundaries can be 0)
    frame_source_safe = np.maximum(frame_source_warped, 0) + eps
    frame_target_safe = np.maximum(frame_target_f, 0) + eps
    
    # Log transform
    log_source = np.log(frame_source_safe)
    log_target = np.log(frame_target_safe)
    
    # Compute Euclidean distance in log-RGB space
    diff = log_target - log_source
    photometric_error = np.sqrt(np.sum(diff**2, axis=2))
    
    return photometric_error.astype(np.float32)


def compute_photometric_windowed(photometric_raw: np.ndarray,
                                 winsize: int) -> np.ndarray:
    """
    Apply Gaussian smoothing to photometric error at appropriate spatial scale.
    
    Args:
        photometric_raw: Raw per-pixel photometric error (H, W)
        winsize: Optical flow window size
    
    Returns:
        photometric_smoothed: (H, W) smoothed photometric error
    
    Method:
        Smooth with Gaussian kernel where sigma = winsize / 4
        This matches the spatial scale of the optical flow computation.
    """
    # Compute smoothing sigma based on window size
    sigma = winsize / 4.0
    
    # Apply Gaussian smoothing
    photometric_smoothed = gaussian_filter(
        photometric_raw,
        sigma=sigma,
        mode='constant',
        cval=0
    )
    
    return photometric_smoothed.astype(np.float32)


def compute_photometric(frame_source: np.ndarray,
                       frame_target: np.ndarray,
                       u: np.ndarray,
                       v: np.ndarray,
                       winsize: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute both raw and windowed photometric errors (grayscale, legacy).
    
    Args:
        frame_source: Source frame (H, W) to be warped
        frame_target: Target frame (H, W) for comparison
        u: Horizontal flow component (H, W)
        v: Vertical flow component (H, W)
        winsize: Optical flow window size
    
    Returns:
        photometric_raw: (H, W) raw per-pixel error
        photometric_windowed: (H, W) smoothed error
    """
    # Compute raw error
    photometric_raw = compute_photometric_raw(frame_source, frame_target, u, v)
    
    # Smooth to match OF spatial scale
    photometric_windowed = compute_photometric_windowed(photometric_raw, winsize)
    
    return photometric_raw, photometric_windowed


def compute_photometric_all(frame_source: np.ndarray,
                            frame_target: np.ndarray,
                            u: np.ndarray,
                            v: np.ndarray,
                            winsize: int) -> dict:
    """
    Compute all photometric variants: grayscale, RGB, and log-RGB.
    
    Handles both grayscale and RGB input frames automatically.
    
    Args:
        frame_source: Source frame (H, W) or (H, W, 3)
        frame_target: Target frame (H, W) or (H, W, 3)
        u: Horizontal flow component (H, W)
        v: Vertical flow component (H, W)
        winsize: Optical flow window size
    
    Returns:
        dict with keys:
            - photometric_gray: grayscale error (smoothed)
            - photometric_gray_raw: grayscale error (raw)
            - photometric_rgb: RGB Euclidean distance (smoothed) [if RGB input]
            - photometric_rgb_raw: RGB Euclidean distance (raw) [if RGB input]
            - photometric_rgb_log: log-RGB distance (smoothed) [if RGB input]
            - photometric_rgb_log_raw: log-RGB distance (raw) [if RGB input]
    """
    result = {}
    
    # Convert to grayscale if needed for grayscale metric
    if frame_source.ndim == 3:
        gray_source = cv2.cvtColor(frame_source, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_target = cv2.cvtColor(frame_target, cv2.COLOR_RGB2GRAY).astype(np.float32)
        is_rgb = True
    else:
        gray_source = frame_source.astype(np.float32)
        gray_target = frame_target.astype(np.float32)
        is_rgb = False
    
    # Grayscale photometric (always computed)
    photo_gray_raw = compute_photometric_raw(gray_source, gray_target, u, v)
    photo_gray = compute_photometric_windowed(photo_gray_raw, winsize)
    result['photometric_gray'] = photo_gray
    result['photometric_gray_raw'] = photo_gray_raw
    
    # RGB variants (only if input is RGB)
    if is_rgb:
        # RGB Euclidean
        photo_rgb_raw = compute_photometric_rgb(frame_source, frame_target, u, v)
        photo_rgb = compute_photometric_windowed(photo_rgb_raw, winsize)
        result['photometric_rgb'] = photo_rgb
        result['photometric_rgb_raw'] = photo_rgb_raw
        
        # Log-RGB
        photo_rgb_log_raw = compute_photometric_rgb_log(frame_source, frame_target, u, v)
        photo_rgb_log = compute_photometric_windowed(photo_rgb_log_raw, winsize)
        result['photometric_rgb_log'] = photo_rgb_log
        result['photometric_rgb_log_raw'] = photo_rgb_log_raw
    
    return result


if __name__ == "__main__":
    print("🧪 Testing photometric error computation...")
    
    # Create synthetic test data
    H, W = 64, 64
    
    # Test grayscale
    gray1 = np.random.rand(H, W).astype(np.float32) * 255
    gray2 = np.roll(gray1, 2, axis=1)  # Shift by 2 pixels
    u = np.ones((H, W), dtype=np.float32) * 2.0
    v = np.zeros((H, W), dtype=np.float32)
    
    photo_raw = compute_photometric_raw(gray1, gray2, u, v)
    print(f"✅ Grayscale raw photometric: mean={photo_raw.mean():.4f}")
    
    photo_windowed = compute_photometric_windowed(photo_raw, winsize=15)
    print(f"✅ Grayscale windowed photometric: mean={photo_windowed.mean():.4f}")
    
    # Test RGB
    rgb1 = np.random.rand(H, W, 3).astype(np.float32) * 255
    rgb2 = np.roll(rgb1, 2, axis=1)
    
    photo_rgb = compute_photometric_rgb(rgb1, rgb2, u, v)
    print(f"✅ RGB photometric: mean={photo_rgb.mean():.4f}")
    
    photo_rgb_log = compute_photometric_rgb_log(rgb1, rgb2, u, v)
    print(f"✅ Log-RGB photometric: mean={photo_rgb_log.mean():.4f}")
    
    # Test eps computation
    max_val = max(rgb1.max(), rgb2.max())
    eps = max_val / 100.0
    print(f"✅ Adaptive eps: {eps:.4f} (max_val={max_val:.1f})")
    
    # Test compute_photometric_all
    all_metrics = compute_photometric_all(rgb1, rgb2, u, v, winsize=15)
    print(f"✅ compute_photometric_all returned {len(all_metrics)} metrics:")
    for key, val in all_metrics.items():
        print(f"   {key}: mean={val.mean():.4f}")
    
    print("\n✨ All tests passed!")
