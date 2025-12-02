# File: src/evaluation/flow_deformation.py
"""
Flow deformation analysis at a consistent spatial scale.

Computes:
- Destination density (pixel counting with circular smoothing)
- Jacobian (spatial finite differences)
- Occlusion metrics (log-density)

Naming convention:
    flow_AB_* → computed from A→B flow → lives in Frame B
    flow_BA_* → computed from B→A flow → lives in Frame A

Scale convention:
    magnitude is an INTEGER
    Perturbations: ±magnitude pixel shifts
    Jacobian: central difference spanning 2*magnitude
    Density: circular smoothing with radius=magnitude
"""

import numpy as np
from scipy.ndimage import convolve
import sys


def disk_kernel(radius: int) -> np.ndarray:
    """
    Create a circular (disk) kernel with given radius.
    
    Args:
        radius: Integer radius of disk
        
    Returns:
        Normalized kernel where pixels within radius are equal weight
    """
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def compute_density(u: np.ndarray, v: np.ndarray, magnitude: int) -> np.ndarray:
    """
    Compute destination density with circular smoothing.
    
    Counts how many source pixels map to each destination, then smooths
    with a circular kernel of radius=magnitude.
    
    Args:
        u, v: Flow field (H, W)
        magnitude: Smoothing radius (integer)
        
    Returns:
        density: Sources per destination pixel (H, W)
                 1.0 = normal one-to-one mapping
                 >1.0 = compression (many-to-one, occlusion)
                 <1.0 = expansion (one-to-many, disocclusion)
    """
    H, W = u.shape
    
    # Compute destinations
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    x_dst = np.round(x_grid + u).astype(np.int32)
    y_dst = np.round(y_grid + v).astype(np.int32)
    
    # Clip to valid range
    x_dst = np.clip(x_dst, 0, W - 1)
    y_dst = np.clip(y_dst, 0, H - 1)
    
    # Count at pixel level
    flat_indices = y_dst.ravel() * W + x_dst.ravel()
    counts = np.bincount(flat_indices, minlength=H * W)
    density_pixel = counts.reshape(H, W).astype(np.float32)
    
    # Circular smoothing
    kernel = disk_kernel(magnitude)
    density = convolve(density_pixel, kernel, mode='nearest')
    
    return density


def compute_jacobian(u: np.ndarray, v: np.ndarray, magnitude: int) -> dict:
    """
    Compute spatial Jacobian via central differences at ±magnitude.
    
    Args:
        u, v: Flow field (H, W)
        magnitude: Finite difference step (integer)
        
    Returns:
        Dictionary with:
            du_dx, du_dy, dv_dx, dv_dy: Jacobian components
            divergence: ∂u/∂x + ∂v/∂y (compression/expansion)
            curl: ∂v/∂x - ∂u/∂y (rotation)
            det: det(I + J) (area ratio of deformation)
    
    Note:
        Output arrays are smaller than input by 2*magnitude in each dimension
    """
    mag = magnitude
    
    # Central differences: sample at x±mag, y±mag
    du_dx = (u[:, 2*mag:] - u[:, :-2*mag]) / (2 * mag)
    du_dy = (u[2*mag:, :] - u[:-2*mag, :]) / (2 * mag)
    dv_dx = (v[:, 2*mag:] - v[:, :-2*mag]) / (2 * mag)
    dv_dy = (v[2*mag:, :] - v[:-2*mag, :]) / (2 * mag)
    
    # Crop to common size
    min_h = min(du_dx.shape[0], du_dy.shape[0])
    min_w = min(du_dx.shape[1], du_dy.shape[1])
    
    du_dx = du_dx[:min_h, :min_w]
    du_dy = du_dy[:min_h, :min_w]
    dv_dx = dv_dx[:min_h, :min_w]
    dv_dy = dv_dy[:min_h, :min_w]
    
    # Kinematic descriptors
    divergence = du_dx + dv_dy
    curl = dv_dx - du_dy
    
    # Deformation determinant: det(I + J)
    # This is the area ratio: destination_area / source_area
    det_deformation = (1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx
    
    return {
        'du_dx': du_dx.astype(np.float32),
        'du_dy': du_dy.astype(np.float32),
        'dv_dx': dv_dx.astype(np.float32),
        'dv_dy': dv_dy.astype(np.float32),
        'divergence': divergence.astype(np.float32),
        'curl': curl.astype(np.float32),
        'det': det_deformation.astype(np.float32),
    }


def compute_occlusion(density: np.ndarray, direction: str = 'forward') -> np.ndarray:
    """
    Compute occlusion metric from density.
    
    Uses log-density for symmetric scaling:
        density=2.0 (2x compression) → +0.69
        density=0.5 (2x expansion)   → -0.69
    
    Args:
        density: Destination density field
        direction: 'forward' (A→B) or 'backward' (B→A)
                  Affects sign convention
    
    Returns:
        occlusion: Log-density score
                   For forward: >0 = occlusion in dest frame
                   For backward: >0 = occluded in source frame (sign flipped)
    """
    eps = 1e-6
    log_density = np.log(np.maximum(density, eps))
    
    if direction == 'forward':
        # Forward flow: high density = occlusion in destination
        return log_density.astype(np.float32)
    else:
        # Backward flow: LOW density = occluded in source frame
        # Flip sign so positive = occluded
        return -log_density.astype(np.float32)


def compute_flow_deformation(u: np.ndarray, v: np.ndarray, magnitude: int) -> dict:
    """
    Compute all deformation metrics for a flow field.
    
    Args:
        u, v: Base flow field (H, W)
        magnitude: Integer scale for analysis
        
    Returns:
        Dictionary with:
            density: Sources per destination (H, W)
            divergence: Compression/expansion (H-2*mag, W-2*mag)
            curl: Rotation (H-2*mag, W-2*mag)
            det: Area ratio (H-2*mag, W-2*mag)
            occlusion: Log-density (H, W)
            magnitude_used: The magnitude parameter
    """
    if not isinstance(magnitude, int) or magnitude < 1:
        print(f"❌ ERROR: magnitude must be integer >= 1, got {magnitude}")
        sys.exit(1)
    
    # Density with circular smoothing
    density = compute_density(u, v, magnitude)
    
    # Jacobian from spatial finite differences
    jacobian = compute_jacobian(u, v, magnitude)
    
    # Occlusion from log-density
    occlusion = compute_occlusion(density, direction='forward')
    
    return {
        'density': density,
        'occlusion': occlusion,
        **jacobian,
        'magnitude_used': magnitude,
    }


def plot_flow_decomposition(u: np.ndarray, v: np.ndarray, 
                            curl: np.ndarray, occlusion: np.ndarray,
                            title: str = "Flow Decomposition",
                            quiver_step: int = None,
                            output_path: str = None):
    """
    Three-panel flow visualization: magnitude+quivers | curl | occlusion
    
    Args:
        u, v: Flow components (full size)
        curl: Rotation field (may be smaller due to Jacobian cropping)
        occlusion: Log-density field (full size)
        title: Figure title
        quiver_step: Subsample interval for arrows (auto if None)
        output_path: If provided, save figure to this path
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    H, W = u.shape
    
    # Auto quiver step
    if quiver_step is None:
        quiver_step = max(1, min(H, W) // 25)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # === Panel 1: Magnitude + Quivers ===
    magnitude = np.sqrt(u**2 + v**2)
    mag_max = np.percentile(magnitude, 98)
    
    im0 = axes[0].imshow(magnitude, cmap='viridis', vmin=0, vmax=mag_max)
    
    # Quivers (subsampled)
    y, x = np.mgrid[0:H:quiver_step, 0:W:quiver_step]
    axes[0].quiver(x, y,
                   u[::quiver_step, ::quiver_step],
                   -v[::quiver_step, ::quiver_step],  # Flip v for image coords
                   color='white', alpha=0.8, scale_units='xy')
    
    axes[0].set_title('magnitude + direction')
    plt.colorbar(im0, ax=axes[0], label='pixels')
    
    # === Panel 2: Curl ===
    curl_max = np.percentile(np.abs(curl), 98) + 1e-6
    im1 = axes[1].imshow(curl, cmap='PuOr', vmin=-curl_max, vmax=curl_max)
    axes[1].set_title('curl (rotation)')
    plt.colorbar(im1, ax=axes[1], label='1/pixel')
    
    # === Panel 3: Occlusion ===
    occ_max = np.percentile(np.abs(occlusion), 98) + 1e-6
    im2 = axes[2].imshow(occlusion, cmap='BrBG', vmin=-occ_max, vmax=occ_max)
    axes[2].set_title('occlusion (log-density)')
    plt.colorbar(im2, ax=axes[2], label='log(ρ)')
    
    # Clean up axes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved figure to {output_path}")
    
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("🧪 Testing flow deformation analysis...")
    
    H, W = 200, 200
    magnitude = 3
    
    # ========================================================================
    # Test 1: Uniform Translation
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 1: Uniform Translation")
    print("="*70)
    
    u_trans = np.ones((H, W), dtype=np.float32) * 5.0
    v_trans = np.zeros((H, W), dtype=np.float32)
    
    result = compute_flow_deformation(u_trans, v_trans, magnitude)
    
    print(f"✓ Uniform translation: u=5.0, v=0.0, magnitude={magnitude}")
    
    density_interior = result['density'][20:-20, 20:-20]
    print(f"  Density: mean={density_interior.mean():.3f} (expect ≈1.0)")
    
    print(f"  Divergence: mean={result['divergence'].mean():.6f} (expect ≈0)")
    print(f"  Curl: mean={result['curl'].mean():.6f} (expect ≈0)")
    print(f"  Occlusion: mean={result['occlusion'][20:-20, 20:-20].mean():.3f} (expect ≈0)")
    
    assert 0.9 < density_interior.mean() < 1.1, "Density should be ≈1.0"
    assert abs(result['divergence'].mean()) < 0.01, "Divergence should be ≈0"
    print("✅ Test 1 passed")
    
    # ========================================================================
    # Test 2: Compression (Occlusion)
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 2: Compression (Occlusion)")
    print("="*70)
    
    u_compress = np.zeros((H, W), dtype=np.float32)
    u_compress[:, W//2:] = -30.0  # Right half moves left
    v_compress = np.zeros((H, W), dtype=np.float32)
    
    result_comp = compute_flow_deformation(u_compress, v_compress, magnitude)
    
    print(f"✓ Split motion: left static, right moves left by 30px")
    
    # Check overlap zone
    overlap_zone = result_comp['density'][:, W//2-35:W//2-5]
    left_zone = result_comp['density'][:, 10:W//2-40]
    
    print(f"  Left (static): density={left_zone.mean():.3f}")
    print(f"  Overlap zone:  density={overlap_zone.mean():.3f} (expect >1.0)")
    print(f"  Occlusion at overlap: {result_comp['occlusion'][:, W//2-35:W//2-5].mean():.3f} (expect >0)")
    
    assert overlap_zone.mean() > 1.2, "Overlap should have high density"
    print("✅ Test 2 passed")
    
    # ========================================================================
    # Test 3: Rotation
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 3: Rotation")
    print("="*70)
    
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W/2, H/2
    
    omega = 0.1  # Rotation rate
    u_rot = -omega * (y_grid - cy)
    v_rot = omega * (x_grid - cx)
    
    result_rot = compute_flow_deformation(u_rot, v_rot, magnitude)
    
    print(f"✓ Solid rotation: ω={omega}")
    print(f"  Curl: mean={result_rot['curl'].mean():.3f} (expect ≈{2*omega:.3f})")
    print(f"  Divergence: mean={result_rot['divergence'].mean():.6f} (expect ≈0)")
    print(f"  Density: mean={result_rot['density'][20:-20, 20:-20].mean():.3f} (expect ≈1.0)")
    
    assert abs(result_rot['curl'].mean() - 2*omega) < 0.05, "Curl should be 2*omega"
    assert abs(result_rot['divergence'].mean()) < 0.01, "Divergence should be ≈0"
    print("✅ Test 3 passed")
    
    # ========================================================================
    # Test 4: Expansion (Disocclusion)
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 4: Expansion (Disocclusion)")
    print("="*70)
    
    alpha = 0.3  # Expansion rate
    u_expand = alpha * (x_grid - cx)
    v_expand = alpha * (y_grid - cy)
    
    result_exp = compute_flow_deformation(u_expand, v_expand, magnitude)
    
    print(f"✓ Radial expansion: α={alpha}")
    print(f"  Divergence: mean={result_exp['divergence'].mean():.3f} (expect ≈{2*alpha:.3f})")
    print(f"  Density center: {result_exp['density'][H//2-10:H//2+10, W//2-10:W//2+10].mean():.3f} (expect <1.0)")
    print(f"  Occlusion center: {result_exp['occlusion'][H//2-10:H//2+10, W//2-10:W//2+10].mean():.3f} (expect <0)")
    
    assert result_exp['divergence'].mean() > 0.4, "Divergence should be positive"
    print("✅ Test 4 passed")
    
    # ========================================================================
    # Test 5: Visualization
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 5: Visualization")
    print("="*70)
    
    # Create interesting flow: rotation + compression on right
    u_test = -0.05 * (y_grid - cy)
    v_test = 0.05 * (x_grid - cx)
    u_test[:, W//2:] += -20.0  # Add leftward motion on right
    
    result_test = compute_flow_deformation(u_test, v_test, magnitude)
    
    # Pad curl to match full size for visualization
    curl_padded = np.zeros((H, W), dtype=np.float32)
    ch, cw = result_test['curl'].shape
    offset_h = (H - ch) // 2
    offset_w = (W - cw) // 2
    curl_padded[offset_h:offset_h+ch, offset_w:offset_w+cw] = result_test['curl']
    
    fig = plot_flow_decomposition(
        u_test, v_test, curl_padded, result_test['occlusion'],
        title="Flow AB: Rotation + Occlusion Test",
        output_path="/mnt/user-data/outputs/flow_decomposition_test.png"
    )
    plt.close(fig)
    
    print("✅ Test 5 passed")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("✨ ALL FLOW DEFORMATION TESTS PASSED!")
    print("="*70)
    print("\nModule provides:")
    print("  • compute_density(u, v, magnitude) - circular-smoothed pixel counting")
    print("  • compute_jacobian(u, v, magnitude) - spatial finite differences")
    print("  • compute_occlusion(density, direction) - log-density metric")
    print("  • compute_flow_deformation(u, v, magnitude) - unified interface")
    print("  • plot_flow_decomposition(...) - 1×3 visualization")
    print("\nNaming convention:")
    print("  flow_AB_* → from A→B flow → lives in Frame B")
    print("  flow_BA_* → from B→A flow → lives in Frame A")
    print("\nScale convention:")
    print("  magnitude = INTEGER")
    print("  Perturbations: ±magnitude")
    print("  Jacobian: span = 2*magnitude")
    print("  Density smoothing: radius = magnitude (circular kernel)")
    print("\nOcclusion interpretation:")
    print("  > 0: occlusion (compression, many-to-one)")
    print("  = 0: normal (one-to-one)")
    print("  < 0: disocclusion (expansion, one-to-many)")
