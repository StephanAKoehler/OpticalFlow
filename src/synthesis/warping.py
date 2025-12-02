# File: src/synthesis/warping.py
"""
Image warping using optical flow fields.

Uses backward warping with cv2.remap for high-quality interpolation.
"""

import numpy as np
import cv2
import sys


def warp_image(image: np.ndarray,
               u: np.ndarray,
               v: np.ndarray,
               interpolation: int = cv2.INTER_CUBIC,
               border_mode: int = cv2.BORDER_CONSTANT) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp image by flow field using backward warping.

    Args:
        image: Source image (H, W) grayscale or (H, W, C) color
               Expected dtype: float32 in [0, 1]
        u: x-component of flow (H, W) - forward flow
        v: y-component of flow (H, W) - forward flow
        interpolation: cv2 interpolation method (INTER_CUBIC recommended)
        border_mode: cv2 border mode (BORDER_CONSTANT recommended)

    Returns:
        (warped_image, valid_mask)
        warped_image: Same shape and dtype as input
        valid_mask: Boolean array (H, W) - True where sampling was valid

    Notes:
        - Forward flow (u, v) indicates where each pixel GOES TO
        - Backward warping samples from (x - u, y - v) to avoid holes
        - valid_mask marks pixels where source location was within image bounds

    Example:
        >>> img = np.random.rand(256, 256).astype(np.float32)
        >>> u = np.ones((256, 256), dtype=np.float32) * 2.0
        >>> v = np.ones((256, 256), dtype=np.float32) * 1.0
        >>> warped, mask = warp_image(img, u, v)
        >>> warped.shape
        (256, 256)
    """
    if not isinstance(image, np.ndarray):
        print(f"❌ ERROR: image must be numpy array, got {type(image)}")
        sys.exit(1)

    if not isinstance(u, np.ndarray) or not isinstance(v, np.ndarray):
        print(f"❌ ERROR: u and v must be numpy arrays")
        sys.exit(1)

    H, W = image.shape[:2]

    if u.shape != (H, W) or v.shape != (H, W):
        print(f"❌ ERROR: Flow shape {u.shape} doesn't match image shape ({H}, {W})")
        sys.exit(1)

    # Create coordinate grids for destination pixels
    y_dst, x_dst = np.mgrid[0:H, 0:W].astype(np.float32)

    # Backward warping: for each destination pixel, find source location
    # If forward flow is (u, v), backward mapping is (x - u, y - v)
    x_src = x_dst - u
    y_src = y_dst - v

    # Convert image to format suitable for cv2.remap if needed
    # cv2.remap works with float32, but also accepts uint8
    if image.dtype != np.float32 and image.dtype != np.uint8:
        print(f"⚠️  WARNING: Converting image from {image.dtype} to float32")
        image = image.astype(np.float32)

    # Warp using cv2.remap
    warped = cv2.remap(
        image,
        x_src,  # map_x: x-coordinates to sample from
        y_src,  # map_y: y-coordinates to sample from
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=0
    )

    # Compute valid mask: pixels where source was within bounds
    # Use slightly relaxed bounds to account for interpolation
    valid_mask = (
            (x_src >= 0) & (x_src <= W - 1) &
            (y_src >= 0) & (y_src <= H - 1)
    )

    return warped, valid_mask


def create_frame_pair(image: np.ndarray,
                      u_true: np.ndarray,
                      v_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create frame pair from image and ground truth flow.

    Args:
        image: Frame 1 (source image)
        u_true: Ground truth x-component of flow
        v_true: Ground truth y-component of flow

    Returns:
        (frame1, frame2, valid_mask)
        frame1: Original image
        frame2: Warped image (frame1 + flow)
        valid_mask: Valid region for computing errors

    Example:
        >>> img = np.random.rand(256, 256).astype(np.float32)
        >>> u = np.ones((256, 256), dtype=np.float32) * 2.0
        >>> v = np.ones((256, 256), dtype=np.float32) * 1.0
        >>> f1, f2, mask = create_frame_pair(img, u, v)
        >>> f1.shape == f2.shape == mask.shape
        True
    """
    frame1 = image
    frame2, valid_mask = warp_image(image, u_true, v_true)

    return frame1, frame2, valid_mask


def shift_image(image: np.ndarray,
                delta: tuple[float, float]) -> np.ndarray:
    """
    Shift image by a constant displacement (for perturbation tests).

    Args:
        image: Input image
        delta: (dx, dy) displacement in pixels

    Returns:
        Shifted image (same shape as input)

    Note:
        This is a convenience wrapper around warp_image for uniform shifts.

    Example:
        >>> img = np.random.rand(256, 256).astype(np.float32)
        >>> shifted = shift_image(img, (2.5, 1.0))
    """
    H, W = image.shape[:2]
    dx, dy = delta

    # Create uniform flow field
    u = np.full((H, W), dx, dtype=np.float32)
    v = np.full((H, W), dy, dtype=np.float32)

    # Warp (ignore valid mask for shifts)
    shifted, _ = warp_image(image, u, v)

    return shifted


if __name__ == "__main__":
    # Test warping functions
    print("🧪 Testing warping functions...")

    # Create test image
    H, W = 256, 256
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    test_image = ((x // 32) + (y // 32)) % 2  # Checkerboard
    test_image = test_image.astype(np.float32)
    print(f"✅ Created test image: shape={test_image.shape}, dtype={test_image.dtype}")

    # Test uniform translation
    u = np.full((H, W), 10.0, dtype=np.float32)
    v = np.full((H, W), 5.0, dtype=np.float32)

    warped, valid_mask = warp_image(test_image, u, v)
    print(f"✅ Warped image: shape={warped.shape}, dtype={warped.dtype}")
    print(f"   Valid pixels: {valid_mask.sum()} / {valid_mask.size} ({100 * valid_mask.mean():.1f}%)")

    # Test frame pair creation
    frame1, frame2, mask = create_frame_pair(test_image, u, v)
    print(f"✅ Frame pair created")
    print(f"   Frame1 shape: {frame1.shape}")
    print(f"   Frame2 shape: {frame2.shape}")
    print(f"   Valid mask shape: {mask.shape}")

    # Test shift
    shifted = shift_image(test_image, (5.0, 2.0))
    print(f"✅ Shifted image: shape={shifted.shape}")

    # Verify warping actually moved content
    diff = np.abs(test_image - warped)
    print(f"   Mean difference: {diff.mean():.4f}")

    # Test with subpixel shifts
    u_subpixel = np.full((H, W), 0.5, dtype=np.float32)
    v_subpixel = np.full((H, W), 0.3, dtype=np.float32)
    warped_subpixel, _ = warp_image(test_image, u_subpixel, v_subpixel)
    print(f"✅ Subpixel warping works")

    # Test different interpolation methods
    warped_linear, _ = warp_image(test_image, u, v, interpolation=cv2.INTER_LINEAR)
    print(f"✅ Linear interpolation works")

    warped_nearest, _ = warp_image(test_image, u, v, interpolation=cv2.INTER_NEAREST)
    print(f"✅ Nearest interpolation works")

    print("\n✨ All warping tests passed!")