#!/usr/bin/env python3
# File: src_sprites/motion.py
"""
Apply motion transformations to sprites.

Handles translation and rotation around sprite midpoint.
"""

import numpy as np
import cv2


def apply_motion(
    sprite_image: np.ndarray,
    dx: float,
    dy: float,
    rot_deg: float,
    midpoint: tuple[float, float]
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Apply translation and rotation to sprite image.
    
    Args:
        sprite_image: RGB image [H, W, 3] uint8
        dx: Horizontal translation (pixels)
        dy: Vertical translation (pixels)
        rot_deg: Rotation angle (degrees, counter-clockwise)
        midpoint: Current sprite midpoint (x, y) in image coordinates
    
    Returns:
        Transformed image [H, W, 3] uint8
        New midpoint (x, y) after translation
    """
    H, W = sprite_image.shape[:2]
    
    # Build transformation matrix: rotate around midpoint, then translate
    # OpenCV uses (x, y) = (col, row) convention
    cx, cy = midpoint
    
    # Rotation matrix around midpoint
    M_rot = cv2.getRotationMatrix2D((cx, cy), rot_deg, scale=1.0)
    
    # Add translation
    M_rot[0, 2] += dx
    M_rot[1, 2] += dy
    
    # Apply transformation
    transformed = cv2.warpAffine(
        sprite_image,
        M_rot,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # Update midpoint
    new_midpoint = (cx + dx, cy + dy)
    
    return transformed, new_midpoint


def compute_flow_field(
    dx: float,
    dy: float,
    rot_deg: float,
    midpoint: tuple[float, float],
    shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ground truth flow field for sprite motion.
    
    For each pixel (x, y), compute its displacement due to translation + rotation.
    
    Flow at pixel (x, y):
        u = dx - ω*(y - cy)
        v = dy + ω*(x - cx)
    
    where ω = rot_deg * π/180
    
    Args:
        dx: Horizontal translation
        dy: Vertical translation
        rot_deg: Rotation angle (degrees)
        midpoint: Sprite midpoint (x, y)
        shape: Image shape (height, width)
    
    Returns:
        u: Horizontal flow [H, W]
        v: Vertical flow [H, W]
    """
    H, W = shape
    cx, cy = midpoint
    
    # Convert rotation to radians
    omega = np.deg2rad(rot_deg)
    
    # Create coordinate grids (x=col, y=row)
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    
    # Compute flow
    u = dx - omega * (y_grid - cy)
    v = dy + omega * (x_grid - cx)
    
    return u.astype(np.float32), v.astype(np.float32)
