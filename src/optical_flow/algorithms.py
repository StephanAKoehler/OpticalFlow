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
    """
    # Extract parameters
    pyr_scale = config['pyr_scale']
    levels = config['levels']
    winsize = config['winsize']
    iterations = config['iterations']
    poly_n = config['poly_n']
    poly_sigma = config['poly_sigma']
    flags = config['flags']

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

    # Compute flow
    flow = cv2.calcOpticalFlowFarneback(
        frame1_uint8, frame2_uint8,
        None,
        pyr_scale, levels, winsize,
        iterations, poly_n, poly_sigma,
        flags
    )

    # Extract components
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)

    return u, v


def compute_dis(frame1: np.ndarray,
                frame2: np.ndarray,
                config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow using DIS (Dense Inverse Search).

    Args:
        frame1, frame2: Input frames (grayscale, uint8 or float32)
        config: Dict with DIS parameters:
                - preset: 'ULTRAFAST', 'FAST', or 'MEDIUM' (default: 'MEDIUM')
                - finest_scale: 0=full res, 1=half, 2=quarter (default: 1)
                - iterations: gradient descent iterations (default: 12)
                - patch_size: patch size in pixels (default: 8)
                - patch_stride: patch stride (default: 4)
                - use_spatial_propagation: bool (default: True)

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

    # Extract parameters with defaults
    preset_name = config.get('preset', 'MEDIUM')
    finest_scale = config.get('finest_scale', 1)
    iterations = config.get('iterations', 12)
    patch_size = config.get('patch_size', 8)
    patch_stride = config.get('patch_stride', 4)
    use_spatial = config.get('use_spatial_propagation', True)

    # Map preset name to OpenCV constant
    preset_map = {
        'ULTRAFAST': cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        'FAST': cv2.DISOPTICAL_FLOW_PRESET_FAST,
        'MEDIUM': cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        # Also support lowercase for backward compatibility
        'ultrafast': cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        'fast': cv2.DISOPTICAL_FLOW_PRESET_FAST,
        'medium': cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    }

    if preset_name not in preset_map:
        print(f"❌ ERROR: Unknown DIS preset: {preset_name}")
        print(f"   Valid presets: ULTRAFAST, FAST, MEDIUM")
        sys.exit(1)

    # FIXED: Removed space in class name
    dis = cv2.DISOpticalFlow_create(preset_map[preset_name])

    # Set fine-tuning parameters
    dis.setFinestScale(finest_scale)
    dis.setGradientDescentIterations(iterations)
    dis.setPatchSize(patch_size)
    dis.setPatchStride(patch_stride)
    dis.setUseSpatialPropagation(use_spatial)

    # Compute flow
    flow = dis.calc(frame1_uint8, frame2_uint8, None)

    # Extract components
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)

    return u, v


def compute_dualtvl1(frame1: np.ndarray,
                     frame2: np.ndarray,
                     config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow using DualTVL1 algorithm (variational method).

    Requires: opencv-contrib-python

    Args:
        frame1, frame2: Input frames (grayscale, float32 or uint8)
        config: Dict with DualTVL1 parameters:
                - scales_number: Number of pyramid scales (default: 5)
                                 Set to 1 for full resolution only (like levels=0)
                - scale_step: Pyramid scale factor (default: 0.8)
                - outer_iterations: Outer loop iterations (default: 10)
                - inner_iterations: Inner loop iterations (default: 30)
                - lambda_: Regularization weight (default: 0.15)
                - tau: Time step (default: 0.25)
                - theta: Tightness (default: 0.3)
                - gamma: Gradient weight (default: 0.0)

    Returns:
        (u, v) flow field as float32 arrays

    Note:
        DualTVL1 is excellent for:
        - Preserving motion discontinuities
        - Handling textureless regions
        - Global optimization
        Speed: ~7x slower than DIS
    """
    # Check if opencv-contrib is available
    try:
        tvl1_create = cv2.optflow.DualTVL1OpticalFlow_create
    except AttributeError:
        print(f"❌ ERROR: DualTVL1 requires opencv-contrib-python")
        print(f"   Install: pip install opencv-contrib-python")
        sys.exit(1)

    # Convert to float32 [0,1] (DualTVL1 prefers float32)
    if frame1.dtype == np.uint8:
        frame1_float = frame1.astype(np.float32) / 255.0
        frame2_float = frame2.astype(np.float32) / 255.0
    elif frame1.dtype == np.float32:
        frame1_float = np.clip(frame1, 0, 1)
        frame2_float = np.clip(frame2, 0, 1)
    else:
        print(f"❌ ERROR: Unsupported frame dtype: {frame1.dtype}")
        sys.exit(1)

    # Ensure grayscale
    if len(frame1_float.shape) == 3:
        frame1_float = cv2.cvtColor(frame1_float, cv2.COLOR_BGR2GRAY)
        frame2_float = cv2.cvtColor(frame2_float, cv2.COLOR_BGR2GRAY)

    # Create DualTVL1 instance
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

    # Set parameters (use config or defaults)
    scales_number = config.get('scales_number', 5)
    scale_step = config.get('scale_step', 0.8)
    outer_iterations = config.get('outer_iterations', 10)
    inner_iterations = config.get('inner_iterations', 30)
    lambda_param = config.get('lambda_', 0.15)
    tau = config.get('tau', 0.25)
    theta = config.get('theta', 0.3)
    gamma = config.get('gamma', 0.0)

    tvl1.setScalesNumber(scales_number)
    tvl1.setScaleStep(scale_step)
    tvl1.setOuterIterations(outer_iterations)
    tvl1.setInnerIterations(inner_iterations)
    tvl1.setLambda(lambda_param)
    tvl1.setTau(tau)
    tvl1.setTheta(theta)
    tvl1.setGamma(gamma)

    # Compute flow
    flow = tvl1.calc(frame1_float, frame2_float, None)

    # Extract components
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)

    return u, v


def compute_deepflow(frame1: np.ndarray,
                     frame2: np.ndarray,
                     config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow using DeepFlow algorithm.

    Requires: opencv-contrib-python

    Args:
        frame1, frame2: Input frames (grayscale, uint8 or float32)
        config: Dict with DeepFlow parameters (uses defaults if not specified)

    Returns:
        (u, v) flow field as float32 arrays

    Note:
        DeepFlow is excellent for:
        - Large displacements
        - Motion discontinuities
        - Boundary preservation
        Speed: ~3x slower than DIS, but 2x faster than DualTVL1
    """
    # Check if opencv-contrib is available
    try:
        deepflow_create = cv2.optflow.createOptFlow_DeepFlow
    except AttributeError:
        print(f"❌ ERROR: DeepFlow requires opencv-contrib-python")
        print(f"   Install: pip install opencv-contrib-python")
        sys.exit(1)

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

    # Create DeepFlow instance (uses default parameters)
    deepflow = cv2.optflow.createOptFlow_DeepFlow()

    # Compute flow
    flow = deepflow.calc(frame1_uint8, frame2_uint8, None)

    # Extract components
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)

    return u, v


def compute_brox(frame1: np.ndarray,
                 frame2: np.ndarray,
                 config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow using Brox algorithm (variational method).

    Requires: opencv-contrib-python
    Install: pip install opencv-contrib-python

    Args:
        frame1, frame2: Input frames (grayscale, uint8 or float32)
        config: Dict with Brox parameters:
                - alpha: Smoothness weight (default: 0.197)
                - gamma: Gradient constancy weight (default: 50.0)
                - scale_factor: Pyramid scale factor (default: 0.8)
                - inner_iterations: Inner fixed point iterations (default: 5)
                - outer_iterations: Outer fixed point iterations (default: 150)
                - solver_iterations: Solver iterations (default: 10)

    Returns:
        (u, v) flow field as float32 arrays

    Note:
        Brox is ~10-20x slower than Farneback/DIS but better at:
        - Preserving motion discontinuities
        - Handling textureless regions
        - Global smoothness constraints
    """
    # Check if opencv-contrib is available
    try:
        brox_create = cv2.optflow.BroxOpticalFlow_create
    except AttributeError:
        print(f"❌ ERROR: Brox requires opencv-contrib-python")
        print(f"   Install: pip install opencv-contrib-python")
        print(f"   (May need to uninstall opencv-python first)")
        sys.exit(1)

    # Convert to float32 [0,1] (Brox requires float32)
    if frame1.dtype == np.uint8:
        frame1_float = frame1.astype(np.float32) / 255.0
        frame2_float = frame2.astype(np.float32) / 255.0
    elif frame1.dtype == np.float32:
        frame1_float = np.clip(frame1, 0, 1)
        frame2_float = np.clip(frame2, 0, 1)
    else:
        print(f"❌ ERROR: Unsupported frame dtype: {frame1.dtype}")
        sys.exit(1)

    # Ensure grayscale
    if len(frame1_float.shape) == 3:
        frame1_float = cv2.cvtColor(frame1_float, cv2.COLOR_BGR2GRAY)
        frame2_float = cv2.cvtColor(frame2_float, cv2.COLOR_BGR2GRAY)

    # Extract parameters with defaults
    alpha = config.get('alpha', 0.197)
    gamma = config.get('gamma', 50.0)
    scale_factor = config.get('scale_factor', 0.8)
    inner_iterations = config.get('inner_iterations', 5)
    outer_iterations = config.get('outer_iterations', 150)
    solver_iterations = config.get('solver_iterations', 10)

    # Create Brox instance
    brox = cv2.optflow.BroxOpticalFlow_create(
        alpha=alpha,
        gamma=gamma,
        scale_factor=scale_factor,
        inner_iterations=inner_iterations,
        outer_iterations=outer_iterations,
        solver_iterations=solver_iterations
    )

    # Compute flow
    flow = brox.calc(frame1_float, frame2_float, None)

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
    """
    print(f"❌ ERROR: Lucas-Kanade dense not yet implemented")
    print(f"   Use 'farneback', 'dis', or 'brox'")
    sys.exit(1)


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

    elif algorithm == 'dualtvl1':
        return compute_dualtvl1(frame1, frame2, config)

    elif algorithm == 'deepflow':
        return compute_deepflow(frame1, frame2, config)

    elif algorithm == 'brox':
        return compute_brox(frame1, frame2, config)

    else:
        print(f"❌ ERROR: Unknown algorithm: {algorithm}")
        print(f"   Supported algorithms: 'farneback', 'dis', 'deepflow', 'brox'")
        print(f"   Not yet implemented: 'lucas_kanade'")
        sys.exit(1)


if __name__ == "__main__":
    # Test optical flow algorithms
    print("🧪 Testing optical flow algorithms...")

    # Create test frames (checkerboard with shift)
    H, W = 256, 256
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    frame1 = ((x // 32) + (y // 32)) % 2
    frame1 = frame1.astype(np.float32)

    # Create frame2 by shifting frame1
    shift_x = 5.0
    shift_y = 2.0

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

    u_fb, v_fb = compute_farneback(frame1, frame2, config_farneback)
    print(f"✅ Farneback: u shape={u_fb.shape}")
    print(f"   Mean flow: u={u_fb.mean():.2f}, v={v_fb.mean():.2f}")
    print(f"   Expected: u={shift_x:.2f}, v={shift_y:.2f}")

    # Test DIS with full parameters
    config_dis = {
        'algorithm': 'dis',
        'preset': 'FAST',
        'finest_scale': 0,
        'iterations': 12,
        'patch_size': 8,
        'patch_stride': 4,
        'use_spatial_propagation': True
    }

    u_dis, v_dis = compute_dis(frame1, frame2, config_dis)
    print(f"✅ DIS: u shape={u_dis.shape}")
    print(f"   Mean flow: u={u_dis.mean():.2f}, v={v_dis.mean():.2f}")

    # Test with compute_optical_flow interface
    u2, v2 = compute_optical_flow(frame1, frame2, config_farneback)
    assert np.allclose(u_fb, u2) and np.allclose(v_fb, v2)
    print(f"✅ compute_optical_flow interface works")

    u3, v3 = compute_optical_flow(frame1, frame2, config_dis)
    assert np.allclose(u_dis, u3) and np.allclose(v_dis, v3)
    print(f"✅ DIS through compute_optical_flow works")

    # Test Brox (if opencv-contrib available)
    print("\n🧪 Testing Brox (requires opencv-contrib)...")
    try:
        config_brox = {
            'algorithm': 'brox',
            'alpha': 0.197,
            'gamma': 50.0,
            'scale_factor': 0.8,
            'inner_iterations': 5,
            'outer_iterations': 150,
            'solver_iterations': 10
        }

        u_brox, v_brox = compute_brox(frame1, frame2, config_brox)
        print(f"✅ Brox: u shape={u_brox.shape}")
        print(f"   Mean flow: u={u_brox.mean():.2f}, v={v_brox.mean():.2f}")

        u4, v4 = compute_optical_flow(frame1, frame2, config_brox)
        assert np.allclose(u_brox, u4) and np.allclose(v_brox, v4)
        print(f"✅ Brox through compute_optical_flow works")
    except SystemExit:
        print(f"⚠️  Brox not available (opencv-contrib not installed)")

    print("\n✨ All optical flow tests passed!")