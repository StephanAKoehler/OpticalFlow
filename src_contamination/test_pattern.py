# File: src_contamination/test_pattern.py
"""
Test pattern generation for boundary contamination measurement.

Multi-angle, multi-octave sinusoid pattern that is isotropic and provides
texture at all scales relevant to optical flow window sizes 5-21.
"""

import numpy as np
import sys


# Wavelengths covering window sizes 5-21
# - 8px:  small windows (5-9)
# - 16px: medium windows (11-15)  
# - 32px: large windows (17-21+)
WAVELENGTHS = [8, 16, 32]

# 3 angles for isotropy: 0°, 60°, 120°
N_ANGLES = 3


def create_template(size: int, shift_x: float = 0.0, shift_y: float = 0.0) -> np.ndarray:
    """
    Create multi-octave, multi-angle sinusoid pattern.
    
    Args:
        size: Template size in pixels (square)
        shift_x: Phase shift in x direction (pixels)
        shift_y: Phase shift in y direction (pixels)
        
    Returns:
        Float32 array normalized to [0, 1]
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    template = np.zeros((size, size), dtype=np.float32)
    
    for wavelength in WAVELENGTHS:
        k = 2 * np.pi / wavelength
        for i in range(N_ANGLES):
            theta = i * np.pi / N_ANGLES  # 0°, 60°, 120°
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            phase = k * ((x - shift_x) * cos_t + (y - shift_y) * sin_t)
            template += np.sin(phase)
    
    # Normalize to [0, 1]
    template = (template - template.min()) / (template.max() - template.min())
    return template.astype(np.float32)


def create_test_frames(size: int, shift: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Create 3x3 tiled test frames for boundary contamination measurement.
    
    Center tile has known motion (shift in x direction).
    Border tiles are static (flipped reflections of center).
    This means any detected flow at boundaries is pure algorithmic contamination.
    
    Args:
        size: Size of center tile in pixels (full image is 3*size)
        shift: Motion to apply in x direction (pixels)
        
    Returns:
        (frame1, frame2) as float32 arrays, each 3*size x 3*size
        
    Ground truth flow:
        ┌───────┬───────┬───────┐
        │ (0,0) │ (0,0) │ (0,0) │
        ├───────┼───────┼───────┤
        │ (0,0) │(s, 0) │ (0,0) │  ← only center has flow
        ├───────┼───────┼───────┤
        │ (0,0) │ (0,0) │ (0,0) │
        └───────┴───────┴───────┘
    """
    # Center templates
    template_A = create_template(size, shift_x=0.0, shift_y=0.0)
    template_B = create_template(size, shift_x=shift, shift_y=0.0)
    
    # Flipped versions (no inversion - just geometric flips for C0 continuity)
    flip_x = np.flip(template_A, axis=1)    # Mirror horizontally
    flip_y = np.flip(template_A, axis=0)    # Mirror vertically
    flip_xy = np.flip(template_A, axis=(0, 1))  # Both axes
    
    # Frame 1: A in center, flipped A around border
    frame1 = np.block([
        [flip_xy, flip_y,     flip_xy],
        [flip_x,  template_A, flip_x ],
        [flip_xy, flip_y,     flip_xy]
    ])
    
    # Frame 2: B in center, same flipped A around border (static!)
    frame2 = np.block([
        [flip_xy, flip_y,     flip_xy],
        [flip_x,  template_B, flip_x ],
        [flip_xy, flip_y,     flip_xy]
    ])
    
    return frame1.astype(np.float32), frame2.astype(np.float32)


def create_ground_truth(size: int, shift: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Create ground truth flow fields for test frames.
    
    Args:
        size: Size of center tile in pixels
        shift: Motion applied in x direction
        
    Returns:
        (u_truth, v_truth) flow fields, each 3*size x 3*size
    """
    full_size = 3 * size
    u_truth = np.zeros((full_size, full_size), dtype=np.float32)
    v_truth = np.zeros((full_size, full_size), dtype=np.float32)
    
    # Only center tile has motion
    u_truth[size:2*size, size:2*size] = shift
    # v remains zero everywhere
    
    return u_truth, v_truth


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("🔬 Test pattern generation demo")
    print("=" * 40)
    
    size = 100
    shift = 1.0
    
    # Create frames
    frame1, frame2 = create_test_frames(size, shift)
    u_truth, v_truth = create_ground_truth(size, shift)
    
    print(f"Center tile size: {size}x{size}")
    print(f"Full frame size:  {frame1.shape}")
    print(f"Applied shift:    {shift} px")
    print(f"Wavelengths:      {WAVELENGTHS}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(frame1, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Frame 1')
    axes[0].axhline(size, color='cyan', linewidth=0.5, linestyle='--')
    axes[0].axhline(2*size, color='cyan', linewidth=0.5, linestyle='--')
    axes[0].axvline(size, color='cyan', linewidth=0.5, linestyle='--')
    axes[0].axvline(2*size, color='cyan', linewidth=0.5, linestyle='--')
    
    axes[1].imshow(frame2, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Frame 2')
    axes[1].axhline(size, color='cyan', linewidth=0.5, linestyle='--')
    axes[1].axhline(2*size, color='cyan', linewidth=0.5, linestyle='--')
    axes[1].axvline(size, color='cyan', linewidth=0.5, linestyle='--')
    axes[1].axvline(2*size, color='cyan', linewidth=0.5, linestyle='--')
    
    axes[2].imshow(u_truth, cmap='RdBu_r', vmin=-shift, vmax=shift)
    axes[2].set_title(f'Ground Truth u (shift={shift})')
    axes[2].axhline(size, color='black', linewidth=0.5, linestyle='--')
    axes[2].axhline(2*size, color='black', linewidth=0.5, linestyle='--')
    axes[2].axvline(size, color='black', linewidth=0.5, linestyle='--')
    axes[2].axvline(2*size, color='black', linewidth=0.5, linestyle='--')
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('results_contamination/test_pattern_demo.png', dpi=150)
    print(f"\n✅ Saved: results_contamination/test_pattern_demo.png")
