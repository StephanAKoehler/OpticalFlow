#!/usr/bin/env python3
# File: src_sprites/compositor.py
"""
Composite sprites with z-ordering onto background.
"""

import numpy as np


def composite_frame(
    background: np.ndarray,
    sprites: list[dict],
    shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Composite sprites onto background with z-ordering.
    
    Args:
        background: Background image [H, W, 3] uint8
        sprites: List of sprite dicts, each containing:
            - 'image': RGB image [H, W, 3] uint8
            - 'mask': Boolean mask [H, W] (optional, explicit shape mask)
            - 'z_order': int (higher = on top)
            - 'id': sprite ID for mask
            - 'u': flow field [H, W] float32
            - 'v': flow field [H, W] float32
        shape: Output shape (height, width)
    
    Returns:
        composite_image: Final RGB image [H, W, 3] uint8
        sprite_mask: Sprite ID at each pixel [H, W] int32 (0 = background)
        u_field: Combined flow field [H, W] float32
        v_field: Combined flow field [H, W] float32
    """
    H, W = shape
    
    # Initialize outputs
    composite = background.copy()
    sprite_mask = np.zeros((H, W), dtype=np.int32)
    u_field = np.zeros((H, W), dtype=np.float32)
    v_field = np.zeros((H, W), dtype=np.float32)
    
    # Sort sprites by z_order (lowest first, so highest renders last)
    sorted_sprites = sorted(sprites, key=lambda s: s['z_order'])
    
    # Composite each sprite
    for sprite in sorted_sprites:
        sprite_img = sprite['image']
        sprite_id = sprite['id']
        sprite_u = sprite['u']
        sprite_v = sprite['v']
        
        # Use explicit mask if provided, otherwise fall back to non-zero pixel detection
        if 'mask' in sprite:
            is_sprite = sprite['mask']
        else:
            # Legacy fallback: non-black pixels (alpha channel proxy)
            is_sprite = (sprite_img[:, :, 0] > 0) | (sprite_img[:, :, 1] > 0) | (sprite_img[:, :, 2] > 0)
        
        # Update composite where sprite is visible
        composite[is_sprite] = sprite_img[is_sprite]
        sprite_mask[is_sprite] = sprite_id
        u_field[is_sprite] = sprite_u[is_sprite]
        v_field[is_sprite] = sprite_v[is_sprite]
    
    return composite, sprite_mask, u_field, v_field
