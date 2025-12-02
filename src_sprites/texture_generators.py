#!/usr/bin/env python3
# File: src_sprites/texture_generators.py
"""
Texture generation functions for sprites.

Supports:
- Checkerboard patterns
- Solid colors  
- Perlin noise
- Gaussian noise
"""

import numpy as np


def generate_checkerboard(
    size: tuple[int, int],
    square_size: int,
    color_rgb: list[int],
    line_width: int = 0,
    line_color: list[int] = None
) -> np.ndarray:
    """
    Generate checkerboard pattern.
    
    Args:
        size: (height, width)
        square_size: Size of each square in pixels
        color_rgb: Base color [r, g, b] for alternating squares
        line_width: Width of grid lines (0 for no lines)
        line_color: Color of grid lines [r, g, b]
    
    Returns:
        RGB image [H, W, 3] uint8
    """
    H, W = size
    image = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Create checkerboard pattern
    for i in range(0, H, square_size):
        for j in range(0, W, square_size):
            # Determine if this square is colored or white
            row_idx = i // square_size
            col_idx = j // square_size
            is_colored = (row_idx + col_idx) % 2 == 0
            
            # Calculate actual square bounds
            i_end = min(i + square_size, H)
            j_end = min(j + square_size, W)
            
            if is_colored:
                image[i:i_end, j:j_end] = color_rgb
            else:
                image[i:i_end, j:j_end] = [255, 255, 255]
    
    # Add grid lines if requested
    if line_width > 0 and line_color is not None:
        for i in range(0, H, square_size):
            i_start = max(0, i - line_width // 2)
            i_end = min(H, i + line_width // 2 + 1)
            image[i_start:i_end, :] = line_color
        
        for j in range(0, W, square_size):
            j_start = max(0, j - line_width // 2)
            j_end = min(W, j + line_width // 2 + 1)
            image[:, j_start:j_end] = line_color
    
    return image


def generate_solid(size: tuple[int, int], color_rgb: list[int]) -> np.ndarray:
    """
    Generate solid color texture.
    
    Args:
        size: (height, width)
        color_rgb: Color [r, g, b]
    
    Returns:
        RGB image [H, W, 3] uint8
    """
    H, W = size
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[:, :] = color_rgb
    return image


def generate_texture(texture_type: str, size: tuple[int, int], **params) -> np.ndarray:
    """
    Generate texture based on type string.
    
    Args:
        texture_type: One of "checkerboard", "solid"
        size: (height, width)
        **params: Texture-specific parameters
    
    Returns:
        RGB image [H, W, 3] uint8
    """
    if texture_type == "checkerboard":
        return generate_checkerboard(size, **params)
    elif texture_type == "solid":
        return generate_solid(size, **params)
    else:
        raise ValueError(f"Unknown texture type: {texture_type}")
