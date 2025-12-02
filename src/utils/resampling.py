# File: src/utils/resampling.py
"""
Resampling utilities for optical flow metrics.

Handles downsampling to native resolution and upsampling to full resolution.
"""

import numpy as np
import cv2
import sys


def downsample_metrics(metrics: dict, stride: int) -> dict:
    """
    Downsample all metric arrays by the given stride using area averaging.
    
    Args:
        metrics: Dictionary of metric arrays (H, W)
        stride: Downsampling stride (sample every stride-th pixel)
    
    Returns:
        Dictionary of downsampled arrays as float16
    
    Example:
        >>> metrics = {'u': np.random.rand(288, 288)}
        >>> downsampled = downsample_metrics(metrics, stride=8)
        >>> downsampled['u'].shape
        (36, 36)
        >>> downsampled['u'].dtype
        dtype('float16')
    """
    if stride <= 0:
        print("❌ Error: stride must be positive")
        sys.exit(1)
    
    downsampled = {}
    
    for key, array in metrics.items():
        if array is None:
            downsampled[key] = None
            continue
            
        if not isinstance(array, np.ndarray):
            # Keep non-array values as-is
            downsampled[key] = array
            continue
        
        if len(array.shape) != 2:
            # Non-2D arrays: keep as-is or skip
            downsampled[key] = array
            continue
        
        H, W = array.shape
        h_down = H // stride
        w_down = W // stride
        
        # Determine downsampling method based on dtype
        if array.dtype == bool:
            # Boolean: use majority voting (OR operation for occupancy)
            # Reshape and take max over blocks
            trimmed = array[:h_down*stride, :w_down*stride]
            reshaped = trimmed.reshape(h_down, stride, w_down, stride)
            downsampled_array = reshaped.any(axis=(1, 3))  # OR operation
            downsampled[key] = downsampled_array
            
        elif np.issubdtype(array.dtype, np.floating):
            # Floating point: use area averaging with cv2.resize
            # cv2.INTER_AREA does proper anti-aliased averaging
            if array.dtype == np.float16:
                array_for_resize = array.astype(np.float32)
            else:
                array_for_resize = array
            
            downsampled_array = cv2.resize(
                array_for_resize,
                (w_down, h_down),
                interpolation=cv2.INTER_AREA  # Area averaging - critical!
            )
            # downsampled[key] = downsampled_array.astype(np.float16)
            if np.any(np.abs(downsampled_array) > 65000):
                print(
                    f"⚠️  Warning: {key} has values > 65000 (max={np.max(np.abs(downsampled_array)):.1f}), will overflow float16")
            downsampled[key] = downsampled_array.astype(np.float16)
            
        elif np.issubdtype(array.dtype, np.integer):
            # Integer arrays: use nearest neighbor
            downsampled_array = cv2.resize(
                array.astype(np.float32),
                (w_down, h_down),
                interpolation=cv2.INTER_NEAREST
            )
            downsampled[key] = downsampled_array.astype(array.dtype)
        else:
            # Unknown dtype: fall back to striding
            downsampled[key] = array[::stride, ::stride]
    
    return downsampled


def upsample_metrics(metrics: dict, target_shape: tuple, interpolation: int = cv2.INTER_LINEAR) -> dict:
    """
    Upsample all metric arrays to target shape using bilinear interpolation.
    
    Args:
        metrics: Dictionary of metric arrays (h, w) at native resolution
        target_shape: Target shape (H, W) for upsampling
        interpolation: OpenCV interpolation method (default: bilinear)
    
    Returns:
        Dictionary of upsampled arrays as float32
    
    Example:
        >>> metrics = {'u': np.random.rand(36, 36).astype(np.float16)}
        >>> upsampled = upsample_metrics(metrics, target_shape=(288, 288))
        >>> upsampled['u'].shape
        (288, 288)
        >>> upsampled['u'].dtype
        dtype('float32')
    """
    H, W = target_shape
    upsampled = {}
    
    for key, array in metrics.items():
        if array is None:
            upsampled[key] = None
            continue
            
        if not isinstance(array, np.ndarray):
            # Keep non-array values as-is
            upsampled[key] = array
            continue
        
        # Skip if already at target shape
        if array.shape == target_shape:
            upsampled[key] = array.astype(np.float32) if np.issubdtype(array.dtype, np.floating) else array
            continue
        
        # Upsample using cv2.resize
        if len(array.shape) == 2:
            # 2D array
            # Convert float16 to float32 for cv2.resize (doesn't support float16)
            if array.dtype == np.float16:
                array_for_resize = array.astype(np.float32)
            elif array.dtype == bool:
                # Convert bool to uint8 for resize, then back to bool
                array_for_resize = array.astype(np.uint8)
                upsampled_array = cv2.resize(array_for_resize, (W, H), interpolation=cv2.INTER_NEAREST)
                upsampled[key] = upsampled_array.astype(bool)
                continue
            else:
                array_for_resize = array
            
            upsampled_array = cv2.resize(array_for_resize, (W, H), interpolation=interpolation)
        else:
            print(f"❌ Error: Cannot upsample array with shape {array.shape}")
            sys.exit(1)
        
        # Convert to float32 for computation
        if np.issubdtype(array.dtype, np.floating):
            upsampled[key] = upsampled_array.astype(np.float32)
        else:
            upsampled[key] = upsampled_array
    
    return upsampled


def compute_downsample_stride(winsize: int) -> int:
    """
    Compute appropriate downsampling stride for given window size.
    
    Rule: stride = winsize // 2 (Nyquist sampling at half the window size)
    
    TEMPORARILY DISABLED: Always returns 1 for full-resolution processing.
    
    Args:
        winsize: Optical flow window size
    
    Returns:
        Downsampling stride (currently always 1)
    
    Example:
        >>> compute_downsample_stride(7)
        1
        >>> compute_downsample_stride(41)
        1
    """
    # TEMPORARY: Disable downsampling for debugging
    return 1
    
    # Original code (disabled):
    # stride = max(1, winsize // 2)
    # return stride


if __name__ == "__main__":
    print("🧪 Testing resampling utilities...")
    
    # Test data
    H, W = 288, 288
    metrics_full = {
        'u': np.random.randn(H, W).astype(np.float32),
        'v': np.random.randn(H, W).astype(np.float32),
        'traction': np.abs(np.random.randn(H, W).astype(np.float32)),
        'mask': np.ones((H, W), dtype=bool),
        'scalar': 42.0
    }
    
    print(f"✅ Created test metrics: {H}×{W}")
    
    # Test downsampling
    stride = 8
    metrics_down = downsample_metrics(metrics_full, stride=stride)
    
    h_down, w_down = H // stride, W // stride
    print(f"\n📉 Downsampling with stride={stride}")
    print(f"   Expected shape: ({h_down}, {w_down})")
    print(f"   u shape: {metrics_down['u'].shape}, dtype: {metrics_down['u'].dtype}")
    print(f"   v shape: {metrics_down['v'].shape}, dtype: {metrics_down['v'].dtype}")
    print(f"   traction shape: {metrics_down['traction'].shape}, dtype: {metrics_down['traction'].dtype}")
    print(f"   mask shape: {metrics_down['mask'].shape}, dtype: {metrics_down['mask'].dtype}")
    print(f"   scalar: {metrics_down['scalar']}")
    
    assert metrics_down['u'].shape == (h_down, w_down), "Downsampled shape mismatch"
    assert metrics_down['u'].dtype == np.float16, "Should be float16"
    assert metrics_down['mask'].dtype == bool, "Mask should stay bool"
    
    # Test upsampling
    metrics_up = upsample_metrics(metrics_down, target_shape=(H, W))
    
    print(f"\n📈 Upsampling to ({H}, {W})")
    print(f"   u shape: {metrics_up['u'].shape}, dtype: {metrics_up['u'].dtype}")
    print(f"   v shape: {metrics_up['v'].shape}, dtype: {metrics_up['v'].dtype}")
    print(f"   traction shape: {metrics_up['traction'].shape}, dtype: {metrics_up['traction'].dtype}")
    
    assert metrics_up['u'].shape == (H, W), "Upsampled shape mismatch"
    assert metrics_up['u'].dtype == np.float32, "Should be float32 after upsampling"
    
    # Test stride computation
    print(f"\n🔢 Stride computation:")
    for winsize in [7, 11, 15, 21, 31, 41]:
        stride = compute_downsample_stride(winsize)
        native_size = H // stride
        print(f"   winsize={winsize:2d} → stride={stride:2d} → native size={native_size:3d}×{native_size:3d}")
    
    # Test memory savings
    mem_full = sum(arr.nbytes for arr in metrics_full.values() if isinstance(arr, np.ndarray))
    mem_down = sum(arr.nbytes for arr in metrics_down.values() if isinstance(arr, np.ndarray))
    savings = mem_full / mem_down
    
    print(f"\n💾 Memory usage:")
    print(f"   Full resolution: {mem_full / 1024:.1f} KB")
    print(f"   Downsampled:     {mem_down / 1024:.1f} KB")
    print(f"   Savings:         {savings:.1f}×")
    
    print("\n✨ All resampling tests passed!")
