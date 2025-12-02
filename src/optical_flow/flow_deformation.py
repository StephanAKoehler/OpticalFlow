# File: src/optical_flow/flow_deformation.py
"""
Compute geometric deformation metrics from optical flow.

Metrics computed at consistent spatial scale defined by `magnitude`:
- density: Count of sources per destination (circular smoothing)
- divergence: ∂u/∂x + ∂v/∂y (expansion/compression)
- curl: ∂v/∂x - ∂u/∂y (rotation)
- det: det(I + J) (area ratio of deformation)
- consistency: 1/det (should approximate density)

Naming convention:
    flow_AB_* → computed from A→B flow → lives in Frame B
    flow_BA_* → computed from B→A flow → lives in Frame A
"""

import numpy as np
from scipy.ndimage import convolve
import sys


def disk_kernel(radius: int) -> np.ndarray:
    """
    Circular kernel with given radius.
    
    Args:
        radius: Integer radius of disk
        
    Returns:
        Normalized circular kernel of shape (2*radius+1, 2*radius+1)
    """
    assert isinstance(radius, int) and radius >= 1, f"radius must be int >= 1, got {radius}"
    
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def compute_density(u: np.ndarray, v: np.ndarray, magnitude: int) -> np.ndarray:
    """
    Destination density with circular smoothing.
    
    Steps:
    1. Round destinations to nearest pixel
    2. Count arrivals at each pixel
    3. Smooth with circular kernel (radius=magnitude)
    
    Args:
        u, v: Flow field components (H, W)
        magnitude: Integer scale for smoothing radius
        
    Returns:
        density: (H, W) smoothed destination density
    """
    assert isinstance(magnitude, int) and magnitude >= 1, f"magnitude must be int >= 1, got {magnitude}"
    
    H, W = u.shape
    
    # Source coordinates
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    
    # Destination coordinates (rounded to nearest pixel)
    x_dst = np.round(x_grid + u).astype(np.int32)
    y_dst = np.round(y_grid + v).astype(np.int32)
    
    # Clip to image bounds
    x_dst = np.clip(x_dst, 0, W - 1)
    y_dst = np.clip(y_dst, 0, H - 1)
    
    # Count arrivals at each destination
    flat_indices = y_dst.ravel() * W + x_dst.ravel()
    counts = np.bincount(flat_indices, minlength=H * W)
    density_pixel = counts.reshape(H, W).astype(np.float32)
    
    # Circular smoothing
    kernel = disk_kernel(magnitude)
    density = convolve(density_pixel, kernel, mode='nearest')
    
    return density


def compute_jacobian(u: np.ndarray, v: np.ndarray, magnitude: int) -> dict:
    """
    Spatial Jacobian via central differences at ±magnitude.
    
    Central difference formula: df/dx = (f[x+mag] - f[x-mag]) / (2*mag)
    Output is NaN-padded to maintain original (H, W) shape.
    
    Args:
        u, v: Flow field components (H, W)
        magnitude: Integer scale for differentiation
        
    Returns:
        Dictionary with:
        - du_dx, du_dy, dv_dx, dv_dy: Jacobian components (H, W)
        - divergence: du_dx + dv_dy (H, W)
        - curl: dv_dx - du_dy (H, W)
        - det: det(I + J) (H, W)
    """
    assert isinstance(magnitude, int) and magnitude >= 1, f"magnitude must be int >= 1, got {magnitude}"
    
    H, W = u.shape
    mag = magnitude
    
    # Initialize with NaN (marks invalid boundary regions)
    du_dx = np.full((H, W), np.nan, dtype=np.float32)
    du_dy = np.full((H, W), np.nan, dtype=np.float32)
    dv_dx = np.full((H, W), np.nan, dtype=np.float32)
    dv_dy = np.full((H, W), np.nan, dtype=np.float32)
    
    # Central differences - valid region is [mag:-mag] in each dimension
    # x-derivatives: valid for columns [mag:-mag]
    du_dx[:, mag:-mag] = (u[:, 2*mag:] - u[:, :-2*mag]) / (2 * mag)
    dv_dx[:, mag:-mag] = (v[:, 2*mag:] - v[:, :-2*mag]) / (2 * mag)
    
    # y-derivatives: valid for rows [mag:-mag]
    du_dy[mag:-mag, :] = (u[2*mag:, :] - u[:-2*mag, :]) / (2 * mag)
    dv_dy[mag:-mag, :] = (v[2*mag:, :] - v[:-2*mag, :]) / (2 * mag)
    
    # Derived quantities (NaN propagates correctly)
    divergence = du_dx + dv_dy
    curl = dv_dx - du_dy
    
    # Deformation determinant: det(I + J) = (1 + du_dx)(1 + dv_dy) - du_dy * dv_dx
    det = (1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx
    
    return {
        'du_dx': du_dx,
        'du_dy': du_dy,
        'dv_dx': dv_dx,
        'dv_dy': dv_dy,
        'divergence': divergence,
        'curl': curl,
        'det': det,
    }


def compute_flow_deformation(u: np.ndarray, v: np.ndarray, magnitude: int) -> dict:
    """
    Compute all deformation metrics for a flow field.
    
    Args:
        u, v: Flow field components (H, W)
        magnitude: Integer scale for analysis (must be >= 1)
        
    Returns:
        Dictionary with all metrics, each (H, W):
        - density: destination density (circular smoothed)
        - divergence: ∂u/∂x + ∂v/∂y
        - curl: ∂v/∂x - ∂u/∂y  
        - det: det(I + J)
        - consistency: 1/det (should approximate density)
        - du_dx, du_dy, dv_dx, dv_dy: Jacobian components
        
        Boundary regions (within `magnitude` of edges) contain NaN
        for Jacobian-derived quantities.
    """
    assert isinstance(magnitude, int) and magnitude >= 1, f"magnitude must be int >= 1, got {magnitude}"
    assert u.shape == v.shape, f"Shape mismatch: u={u.shape}, v={v.shape}"
    assert u.ndim == 2, f"Expected 2D arrays, got {u.ndim}D"
    
    # Compute components
    density = compute_density(u, v, magnitude)
    jacobian = compute_jacobian(u, v, magnitude)
    
    # Consistency metric: 1/det should approximate density
    # Use np.where to avoid divide-by-zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        consistency = np.where(jacobian['det'] != 0, 1.0 / jacobian['det'], np.nan)
    
    return {
        'density': density,
        'divergence': jacobian['divergence'],
        'curl': jacobian['curl'],
        'det': jacobian['det'],
        'consistency': consistency,
        'du_dx': jacobian['du_dx'],
        'du_dy': jacobian['du_dy'],
        'dv_dx': jacobian['dv_dx'],
        'dv_dy': jacobian['dv_dy'],
    }


def compute_bidirectional_deformation(
    u_AB: np.ndarray, v_AB: np.ndarray,
    u_BA: np.ndarray, v_BA: np.ndarray,
    magnitude: int
) -> dict:
    """
    Compute deformation metrics for both flow directions.
    
    Args:
        u_AB, v_AB: Forward flow A→B (H, W)
        u_BA, v_BA: Backward flow B→A (H, W)
        magnitude: Integer scale for analysis
        
    Returns:
        Dictionary with prefixed keys:
        - flow_AB_density, flow_AB_divergence, flow_AB_curl, flow_AB_det, ...
        - flow_BA_density, flow_BA_divergence, flow_BA_curl, flow_BA_det, ...
        
    Naming convention:
        flow_AB_* → computed from A→B flow → lives in Frame B
        flow_BA_* → computed from B→A flow → lives in Frame A
    """
    assert isinstance(magnitude, int) and magnitude >= 1, f"magnitude must be int >= 1, got {magnitude}"
    
    deform_AB = compute_flow_deformation(u_AB, v_AB, magnitude)
    deform_BA = compute_flow_deformation(u_BA, v_BA, magnitude)
    
    result = {}
    for key, value in deform_AB.items():
        result[f'flow_AB_{key}'] = value
    for key, value in deform_BA.items():
        result[f'flow_BA_{key}'] = value
    
    return result


if __name__ == "__main__":
    print("🧪 Testing flow deformation module...")
    
    # ========================================================================
    # Test 1: disk_kernel
    # ========================================================================
    print("\n--- Test 1: disk_kernel ---")
    k = disk_kernel(3)
    assert k.shape == (7, 7), f"Wrong kernel shape: {k.shape}"
    assert abs(k.sum() - 1.0) < 1e-6, f"Kernel not normalized: {k.sum()}"
    assert k[3, 3] > 0, "Center should be non-zero"
    assert k[0, 0] == 0, "Corner should be zero (outside disk)"
    print(f"✅ disk_kernel(3): shape={k.shape}, sum={k.sum():.6f}")
    
    # ========================================================================
    # Test 2: Uniform flow (zero motion)
    # ========================================================================
    print("\n--- Test 2: Uniform zero flow ---")
    H, W = 64, 64
    magnitude = 3
    u_zero = np.zeros((H, W), dtype=np.float32)
    v_zero = np.zeros((H, W), dtype=np.float32)
    
    result = compute_flow_deformation(u_zero, v_zero, magnitude)
    
    # Check shapes
    for key, arr in result.items():
        assert arr.shape == (H, W), f"{key} wrong shape: {arr.shape}"
    print(f"✅ All outputs have shape ({H}, {W})")
    
    # Check density (should be ~1.0 everywhere for zero flow)
    density_center = result['density'][H//4:-H//4, W//4:-W//4]
    assert np.allclose(density_center, 1.0, atol=0.1), f"Density not ~1.0: {density_center.mean()}"
    print(f"✅ Zero flow density ≈ 1.0 (mean={density_center.mean():.4f})")
    
    # Check Jacobian (should be ~0 for zero flow)
    div_center = result['divergence'][magnitude:-magnitude, magnitude:-magnitude]
    assert np.allclose(div_center, 0.0, atol=1e-6), f"Divergence not zero: {div_center.mean()}"
    print(f"✅ Zero flow divergence ≈ 0.0 (mean={div_center.mean():.6f})")
    
    # Check det (should be ~1 for zero flow: det(I + 0) = 1)
    det_center = result['det'][magnitude:-magnitude, magnitude:-magnitude]
    assert np.allclose(det_center, 1.0, atol=1e-6), f"Det not 1.0: {det_center.mean()}"
    print(f"✅ Zero flow det ≈ 1.0 (mean={det_center.mean():.6f})")
    
    # ========================================================================
    # Test 3: NaN boundaries
    # ========================================================================
    print("\n--- Test 3: NaN boundary regions ---")
    # Jacobian-derived quantities should have NaN in boundary regions
    div = result['divergence']
    
    # Check that boundaries are NaN
    assert np.all(np.isnan(div[:magnitude, :])), "Top boundary should be NaN"
    assert np.all(np.isnan(div[-magnitude:, :])), "Bottom boundary should be NaN"
    assert np.all(np.isnan(div[:, :magnitude])), "Left boundary should be NaN"
    assert np.all(np.isnan(div[:, -magnitude:])), "Right boundary should be NaN"
    
    # Check that interior is valid
    interior = div[magnitude:-magnitude, magnitude:-magnitude]
    assert not np.any(np.isnan(interior)), "Interior should not have NaN"
    print(f"✅ NaN boundaries correct (magnitude={magnitude})")
    
    # ========================================================================
    # Test 4: Consistency check on expanding flow
    # ========================================================================
    print("\n--- Test 4: Expanding flow consistency ---")
    H, W = 128, 128
    magnitude = 5
    
    # Radial expansion: u = α*x, v = α*y (from center)
    alpha = 0.1
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    x_centered = x_grid - W/2
    y_centered = y_grid - H/2
    u_expand = (alpha * x_centered).astype(np.float32)
    v_expand = (alpha * y_centered).astype(np.float32)
    
    result = compute_flow_deformation(u_expand, v_expand, magnitude)
    
    # For radial expansion: divergence = 2*alpha
    div_interior = result['divergence'][2*magnitude:-2*magnitude, 2*magnitude:-2*magnitude]
    expected_div = 2 * alpha
    div_mean = np.nanmean(div_interior)
    assert abs(div_mean - expected_div) < 0.01, f"Divergence wrong: {div_mean} vs {expected_div}"
    print(f"✅ Expanding flow divergence: {div_mean:.4f} (expected {expected_div})")
    
    # Curl should be ~0 for radial flow
    curl_interior = result['curl'][2*magnitude:-2*magnitude, 2*magnitude:-2*magnitude]
    curl_mean = np.nanmean(curl_interior)
    assert abs(curl_mean) < 0.01, f"Curl should be ~0: {curl_mean}"
    print(f"✅ Expanding flow curl: {curl_mean:.6f} (expected ~0)")
    
    # det = (1 + α)(1 + α) = (1 + α)² ≈ 1 + 2α for small α
    det_interior = result['det'][2*magnitude:-2*magnitude, 2*magnitude:-2*magnitude]
    expected_det = (1 + alpha) ** 2
    det_mean = np.nanmean(det_interior)
    assert abs(det_mean - expected_det) < 0.01, f"Det wrong: {det_mean} vs {expected_det}"
    print(f"✅ Expanding flow det: {det_mean:.4f} (expected {expected_det:.4f})")
    
    # Consistency: 1/det should approximate density for small flows
    consistency_interior = result['consistency'][2*magnitude:-2*magnitude, 2*magnitude:-2*magnitude]
    consistency_mean = np.nanmean(consistency_interior)
    print(f"ℹ️ Consistency (1/det): {consistency_mean:.4f}")
    
    # ========================================================================
    # Test 5: Integer magnitude assertion
    # ========================================================================
    print("\n--- Test 5: Integer magnitude assertion ---")
    try:
        compute_flow_deformation(u_zero, v_zero, 2.5)
        print("❌ Should have failed for float magnitude")
        sys.exit(1)
    except AssertionError as e:
        print(f"✅ Correctly rejected float magnitude: {e}")
    
    try:
        compute_flow_deformation(u_zero, v_zero, 0)
        print("❌ Should have failed for magnitude=0")
        sys.exit(1)
    except AssertionError as e:
        print(f"✅ Correctly rejected magnitude=0: {e}")
    
    # ========================================================================
    # Test 6: Bidirectional deformation
    # ========================================================================
    print("\n--- Test 6: Bidirectional deformation ---")
    H, W = 64, 64
    magnitude = 3
    
    # Create simple flows
    u_AB = np.ones((H, W), dtype=np.float32) * 2.0
    v_AB = np.ones((H, W), dtype=np.float32) * 1.0
    u_BA = -np.ones((H, W), dtype=np.float32) * 2.0
    v_BA = -np.ones((H, W), dtype=np.float32) * 1.0
    
    bidir = compute_bidirectional_deformation(u_AB, v_AB, u_BA, v_BA, magnitude)
    
    # Check keys are prefixed correctly
    assert 'flow_AB_density' in bidir, "Missing flow_AB_density"
    assert 'flow_AB_divergence' in bidir, "Missing flow_AB_divergence"
    assert 'flow_AB_curl' in bidir, "Missing flow_AB_curl"
    assert 'flow_AB_det' in bidir, "Missing flow_AB_det"
    assert 'flow_AB_consistency' in bidir, "Missing flow_AB_consistency"
    
    assert 'flow_BA_density' in bidir, "Missing flow_BA_density"
    assert 'flow_BA_divergence' in bidir, "Missing flow_BA_divergence"
    assert 'flow_BA_curl' in bidir, "Missing flow_BA_curl"
    assert 'flow_BA_det' in bidir, "Missing flow_BA_det"
    assert 'flow_BA_consistency' in bidir, "Missing flow_BA_consistency"
    
    # Check shapes
    for key, arr in bidir.items():
        assert arr.shape == (H, W), f"{key} wrong shape: {arr.shape}"
    
    print(f"✅ Bidirectional deformation: {len(bidir)} metrics")
    print(f"   AB keys: {[k for k in bidir.keys() if k.startswith('flow_AB_')]}")
    print(f"   BA keys: {[k for k in bidir.keys() if k.startswith('flow_BA_')]}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*60)
    print("✨ All flow deformation tests passed!")
    print("="*60)
