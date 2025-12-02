# File: src/optical_flow/algorithms.py
"""
Optical flow algorithm wrappers.

Provides unified interface for different optical flow algorithms.
All functions return (u, v) as float32 arrays.
"""

import numpy as np
import cv2
import sys


def compute_farneback(frame1: np.ndarray,
                      frame2: np.ndarray,
                      config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow using Farneback algorithm.

    Args:
        frame1, frame2: Input frames (grayscale)
                       Can be float32 [0,1] or uint8 [0,255]
        config: Dict with Farneback parameters:
                - pyr_scale: Pyramid scale factor
                - levels: Number of pyramid levels
                - winsize: Window size
                - iterations: Number of iterations
                - poly_n: Size of pixel neighborhood
                - poly_sigma: Std of Gaussian for derivative
                - flags: Operation flags

    Returns:
        (u, v) flow field as float32 arrays

    Example:
        >>> config = {
        ...     'algorithm': 'farneback',
        ...     'pyr_scale': 0.5,
        ...     'levels': 3,
        ...     'winsize': 15,
        ...     'iterations': 5,
        ...     'poly_n': 5,
        ...     'poly_sigma': 1.1,
        ...     'flags': 0
        ... }
        >>> u, v = compute_farneback(frame1, frame2, config)
    """
    # Extract parameters
    pyr_scale = config['pyr_scale']
    levels = config['levels']
    winsize = config['winsize']
    iterations = config['iterations']
    poly_n = config['poly_n']
    poly_sigma = config['poly_sigma']
    flags = config['flags']

    # Convert to uint8 if needed (OpenCV Farneback requires uint8)
    if frame1.dtype == np.float32:
        # Assume input is in [0, 1], convert to [0, 255]
        frame1_uint8 = (np.clip(frame1, 0, 1) * 255).astype(np.uint8)
        frame2_uint8 = (np.clip(frame2, 0, 1) * 255).astype(np.uint8)
    elif frame1.dtype == np.uint8:
        frame1_uint8 = frame1
        frame2_uint8 = frame2
    else:
        print(f"❌ ERROR: Unsupported frame dtype: {frame1.dtype}")
        print(f"   Supported: float32 [0,1] or uint8 [0,255]")
        sys.exit(1)

    # Ensure grayscale
    if len(frame1_uint8.shape) == 3:
        frame1_uint8 = cv2.cvtColor(frame1_uint8, cv2.COLOR_BGR2GRAY)
        frame2_uint8 = cv2.cvtColor(frame2_uint8, cv2.COLOR_BGR2GRAY)

    # Compute flow
    flow = cv2.calcOpticalFlowFarneback(
        frame1_uint8, frame2_uint8,
        None,  # Initial flow (None = compute from scratch)
        pyr_scale, levels, winsize,
        iterations, poly_n, poly_sigma,
        flags
    )

    # Extract components
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)

    return u, v


def compute_lucas_kanade_dense(frame1: np.ndarray,
                               frame2: np.ndarray,
                               config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute dense optical flow using Lucas-Kanade (via DIS).

    Args:
        frame1, frame2: Input frames
        config: Dict with Lucas-Kanade parameters (TBD)

    Returns:
        (u, v) flow field as float32 arrays

    Note:
        This is a placeholder for future implementation.
        OpenCV doesn't have a dense Lucas-Kanade, so we might use DIS or sparse-to-dense.
    """
    print(f"❌ ERROR: Lucas-Kanade dense not yet implemented")
    print(f"   Use 'farneback' or implement this method")
    sys.exit(1)


def compute_dis(frame1: np.ndarray,
                frame2: np.ndarray,
                config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow using DIS (Dense Inverse Search).

    Args:
        frame1, frame2: Input frames
        config: Dict with DIS parameters:
                - preset: 'ultrafast', 'fast', 'medium' (default: 'medium')

    Returns:
        (u, v) flow field as float32 arrays
    """
    # Convert to uint8 if needed
    if frame1.dtype == np.float32:
        frame1_uint8 = (np.clip(frame1, 0, 1) * 255).astype(np.uint8)
        frame2_uint8 = (np.clip(frame2, 0, 1) * 255).astype(np.uint8)
    elif frame1.dtype == np.uint8:
        frame1_uint8 = frame1
        frame2_uint8 = frame2
    else:
        print(f"❌ ERROR: Unsupported frame dtype: {frame1.dtype}")
        sys.exit(1)

    # Ensure grayscale
    if len(frame1_uint8.shape) == 3:
        frame1_uint8 = cv2.cvtColor(frame1_uint8, cv2.COLOR_BGR2GRAY)
        frame2_uint8 = cv2.cvtColor(frame2_uint8, cv2.COLOR_BGR2GRAY)

    # Create DIS instance
    preset = config.get('preset', 'medium')

    if preset == 'ultrafast':
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    elif preset == 'fast':
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    elif preset == 'medium':
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    else:
        print(f"❌ ERROR: Unknown DIS preset: {preset}")
        print(f"   Valid presets: 'ultrafast', 'fast', 'medium'")
        sys.exit(1)

    # Compute flow
    flow = dis.calc(frame1_uint8, frame2_uint8, None)

    # Extract components
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)

    return u, v


def compute_optical_flow(frame1: np.ndarray,
                         frame2: np.ndarray,
                         config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow using algorithm specified in config.

    Args:
        frame1, frame2: Input frames
        config: Dict with 'algorithm' key and algorithm-specific parameters

    Returns:
        (u, v) flow field as float32 arrays

    Example:
        >>> config = {'algorithm': 'farneback', 'winsize': 15, ...}
        >>> u, v = compute_optical_flow(frame1, frame2, config)
    """
    if 'algorithm' not in config:
        print(f"❌ ERROR: config must have 'algorithm' key")
        sys.exit(1)

    algorithm = config['algorithm']

    if algorithm == 'farneback':
        return compute_farneback(frame1, frame2, config)

    elif algorithm == 'lucas_kanade':
        return compute_lucas_kanade_dense(frame1, frame2, config)

    elif algorithm == 'dis':
        return compute_dis(frame1, frame2, config)

    else:
        print(f"❌ ERROR: Unknown algorithm: {algorithm}")
        print(f"   Supported algorithms: 'farneback', 'dis'")
        print(f"   Not yet implemented: 'lucas_kanade'")
        sys.exit(1)


if __name__ == "__main__":
    # Test optical flow computation
    print("🧪 Testing optical flow algorithms...")

    # Create test frames (checkerboard with shift)
    H, W = 256, 256
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    frame1 = ((x // 32) + (y // 32)) % 2
    frame1 = frame1.astype(np.float32)

    # Create frame2 by shifting frame1 using cv2.remap directly
    shift_x = 5.0
    shift_y = 2.0

    # Simple shift using cv2.remap
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = x_grid - shift_x
    map_y = y_grid - shift_y
    frame2 = cv2.remap(frame1, map_x, map_y,
                       interpolation=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0)

    print(f"✅ Created test frames: {frame1.shape}, shift=({shift_x}, {shift_y})")

    # Test Farneback
    config_farneback = {
        'algorithm': 'farneback',
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 5,
        'poly_n': 5,
        'poly_sigma': 1.1,
        'flags': 0
    }

    u, v = compute_farneback(frame1, frame2, config_farneback)
    print(f"✅ Farneback: u shape={u.shape}, v shape={v.shape}")
    print(f"   Mean flow: u={u.mean():.2f}, v={v.mean():.2f}")
    print(f"   Expected: u={shift_x:.2f}, v={shift_y:.2f}")

    # Test with compute_optical_flow interface
    u2, v2 = compute_optical_flow(frame1, frame2, config_farneback)
    assert np.allclose(u, u2) and np.allclose(v, v2), "Interface mismatch!"
    print(f"✅ compute_optical_flow interface works")

    # Test DIS
    config_dis = {
        'algorithm': 'dis',
        'preset': 'fast'
    }

    u_dis, v_dis = compute_dis(frame1, frame2, config_dis)
    print(f"✅ DIS: u shape={u_dis.shape}, v shape={v_dis.shape}")
    print(f"   Mean flow: u={u_dis.mean():.2f}, v={v_dis.mean():.2f}")

    # Test with uint8 input
    frame1_uint8 = (frame1 * 255).astype(np.uint8)
    frame2_uint8 = (frame2 * 255).astype(np.uint8)

    u_uint8, v_uint8 = compute_farneback(frame1_uint8, frame2_uint8, config_farneback)
    print(f"✅ uint8 input works")
    print(f"   Mean flow: u={u_uint8.mean():.2f}, v={v_uint8.mean():.2f}")

    print("\n✨ All optical flow tests passed!")