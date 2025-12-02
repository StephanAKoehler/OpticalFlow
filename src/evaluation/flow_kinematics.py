# File: src/evaluation/flow_kinematics.py
"""
Kinematic characterization of optical flow fields.

Computes geometric/kinematic descriptors (divergence, curl, shear, strain)
to characterize the type of motion represented by a flow field.

Complements quality metrics:
- Quality metrics: "Is this flow reliable?"
- Kinematics: "What type of motion does this flow represent?"
"""

import numpy as np
from typing import Dict


def compute_flow_kinematics(u: np.ndarray, v: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute kinematic descriptors of flow field.
    
    Decomposes flow into fundamental motion components:
    - Divergence: compression/expansion (source/sink)
    - Curl: rotation/circulation
    - Shear: distortion without volume change
    - Strain: pure stretching/compression along principal axes
    - Determinant: area scaling factor
    
    Args:
        u, v: Flow fields (H, W)
        
    Returns:
        Dictionary containing:
            - divergence: ∂u/∂x + ∂v/∂y (compression/expansion)
            - curl: ∂v/∂x - ∂u/∂y (rotation)
            - determinant: det(J) (area scaling)
            - shear_magnitude: Total shear distortion
            - shear_xy, shear_yx: Shear components
            - lambda1, lambda2: Principal strain rates
            - max_strain, min_strain: Magnitude of principal strains
            - flow_type: Classification (0=saddle, 1=source, 2=sink, 3=spiral)
            - du_dx, du_dy, dv_dx, dv_dy: Jacobian components
    
    Physical Interpretation:
        - divergence > 0: Expansion (disocclusion)
        - divergence < 0: Compression (occlusion)
        - curl > 0: Counter-clockwise rotation
        - curl < 0: Clockwise rotation
        - det > 1: Area expansion
        - det < 1: Area compression
        - det < 0: FOLDING (physically impossible, algorithm error)
    
    Example:
        >>> kinematics = compute_flow_kinematics(u, v)
        >>> occlusion_regions = kinematics['divergence'] < -0.2
        >>> rotation_regions = np.abs(kinematics['curl']) > 0.5
    """
    # Spatial derivatives (Jacobian components)
    du_dx = np.gradient(u, axis=1)  # ∂u/∂x
    du_dy = np.gradient(u, axis=0)  # ∂u/∂y
    dv_dx = np.gradient(v, axis=1)  # ∂v/∂x
    dv_dy = np.gradient(v, axis=0)  # ∂v/∂y
    
    # Basic descriptors
    divergence = du_dx + dv_dy  # Trace of Jacobian
    curl = dv_dx - du_dy         # Rotation (scalar in 2D)
    det_J = du_dx * dv_dy - du_dy * dv_dx  # Determinant
    
    # Shear components
    shear_xy = du_dy  # ∂u/∂y (horizontal flow changing vertically)
    shear_yx = dv_dx  # ∂v/∂x (vertical flow changing horizontally)
    shear_mag = np.sqrt(shear_xy**2 + shear_yx**2)
    
    # Symmetric strain rate tensor (for principal strains)
    E11 = du_dx
    E22 = dv_dy
    E12 = 0.5 * (du_dy + dv_dx)  # Symmetric part
    
    # Principal strains (eigenvalues of symmetric part)
    # For 2x2 matrix: λ = (tr ± √(tr² - 4det)) / 2
    trace_E = E11 + E22
    det_E = E11 * E22 - E12**2
    discriminant = trace_E**2 - 4*det_E
    discriminant = np.maximum(discriminant, 0)  # Ensure non-negative
    
    lambda1 = 0.5 * (trace_E + np.sqrt(discriminant))
    lambda2 = 0.5 * (trace_E - np.sqrt(discriminant))
    
    max_strain = np.maximum(np.abs(lambda1), np.abs(lambda2))
    min_strain = np.minimum(np.abs(lambda1), np.abs(lambda2))
    
    # Flow type classification based on eigenvalues of full Jacobian
    # (not symmetric part - includes rotation)
    trace_J = du_dx + dv_dy
    det_J_for_class = det_J
    discriminant_J = trace_J**2 - 4*det_J_for_class
    
    # Compute eigenvalues of full Jacobian
    lambda1_J = 0.5 * (trace_J + np.sqrt(np.maximum(discriminant_J, 0)))
    lambda2_J = 0.5 * (trace_J - np.sqrt(np.maximum(discriminant_J, 0)))
    
    flow_type = np.zeros_like(u, dtype=np.int32)
    
    # Classification:
    # 0: Saddle (one positive, one negative eigenvalue)
    flow_type[(lambda1_J * lambda2_J < 0)] = 0
    
    # 1: Source (both positive)
    flow_type[(lambda1_J > 0) & (lambda2_J > 0)] = 1
    
    # 2: Sink (both negative)
    flow_type[(lambda1_J < 0) & (lambda2_J < 0)] = 2
    
    # 3: Center/Spiral (complex eigenvalues - discriminant < 0)
    flow_type[discriminant_J < 0] = 3
    
    return {
        'divergence': divergence.astype(np.float32),
        'curl': curl.astype(np.float32),
        'determinant': det_J.astype(np.float32),
        'shear_magnitude': shear_mag.astype(np.float32),
        'shear_xy': shear_xy.astype(np.float32),
        'shear_yx': shear_yx.astype(np.float32),
        'lambda1': lambda1.astype(np.float32),
        'lambda2': lambda2.astype(np.float32),
        'max_strain': max_strain.astype(np.float32),
        'min_strain': min_strain.astype(np.float32),
        'flow_type': flow_type,
        # Raw derivatives for further analysis
        'du_dx': du_dx.astype(np.float32),
        'du_dy': du_dy.astype(np.float32),
        'dv_dx': dv_dx.astype(np.float32),
        'dv_dy': dv_dy.astype(np.float32),
    }


if __name__ == "__main__":
    print("🧪 Testing flow kinematics computation...")
    
    H, W = 100, 100
    
    # ========================================================================
    # Test 1: Uniform Translation
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: Uniform Translation")
    print("="*80)
    
    u_trans = np.ones((H, W), dtype=np.float32) * 5.0
    v_trans = np.ones((H, W), dtype=np.float32) * 2.0
    
    kin_trans = compute_flow_kinematics(u_trans, v_trans)
    
    print(f"✓ Uniform translation: u=5.0, v=2.0")
    print(f"  divergence: {kin_trans['divergence'].mean():.6f} (expected ≈0)")
    print(f"  curl:       {kin_trans['curl'].mean():.6f} (expected ≈0)")
    print(f"  det:        {kin_trans['determinant'].mean():.6f} (expected ≈0)")
    print(f"  shear:      {kin_trans['shear_magnitude'].mean():.6f} (expected ≈0)")
    
    assert abs(kin_trans['divergence'].mean()) < 0.01, "Translation should have zero divergence"
    assert abs(kin_trans['curl'].mean()) < 0.01, "Translation should have zero curl"
    print("✅ Test 1 passed")
    
    # ========================================================================
    # Test 2: Radial Expansion
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: Radial Expansion")
    print("="*80)
    
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W/2, H/2
    
    # Radial flow: u = α(x - cx), v = α(y - cy)
    alpha = 0.1
    u_radial = alpha * (x_grid - cx)
    v_radial = alpha * (y_grid - cy)
    
    kin_radial = compute_flow_kinematics(u_radial, v_radial)
    
    print(f"✓ Radial expansion: α={alpha}")
    print(f"  divergence: {kin_radial['divergence'].mean():.6f} (expected 2α={2*alpha})")
    print(f"  curl:       {kin_radial['curl'].mean():.6f} (expected ≈0)")
    print(f"  det:        {kin_radial['determinant'].mean():.6f} (expected α²={alpha**2})")
    
    # Interior region to avoid boundary effects
    interior = kin_radial['divergence'][10:-10, 10:-10]
    assert abs(interior.mean() - 2*alpha) < 0.01, f"Expected divergence={2*alpha}"
    
    # Check flow type classification (should be mostly "source")
    interior_type = kin_radial['flow_type'][10:-10, 10:-10]
    source_fraction = np.sum(interior_type == 1) / interior_type.size
    print(f"  flow_type:  {source_fraction*100:.1f}% classified as 'source' (expected >80%)")
    assert source_fraction > 0.8, "Most should be classified as source"
    
    print("✅ Test 2 passed")
    
    # ========================================================================
    # Test 3: Rotation
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 3: Rotation")
    print("="*80)
    
    # Rotation: u = -ω(y - cy), v = ω(x - cx)
    omega = 0.05
    u_rot = -omega * (y_grid - cy)
    v_rot = omega * (x_grid - cx)
    
    kin_rot = compute_flow_kinematics(u_rot, v_rot)
    
    print(f"✓ Counter-clockwise rotation: ω={omega}")
    print(f"  divergence: {kin_rot['divergence'].mean():.6f} (expected ≈0)")
    print(f"  curl:       {kin_rot['curl'].mean():.6f} (expected 2ω={2*omega})")
    print(f"  det:        {kin_rot['determinant'].mean():.6f} (expected ω²={omega**2})")
    
    interior = kin_rot['curl'][10:-10, 10:-10]
    assert abs(interior.mean() - 2*omega) < 0.01, f"Expected curl={2*omega}"
    print("✅ Test 3 passed")
    
    # ========================================================================
    # Test 4: Shear Flow
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 4: Shear Flow")
    print("="*80)
    
    # Pure shear: u = ky, v = 0
    k = 0.1
    u_shear = k * y_grid
    v_shear = np.zeros_like(y_grid)
    
    kin_shear = compute_flow_kinematics(u_shear, v_shear)
    
    print(f"✓ Horizontal shear: k={k}")
    print(f"  divergence:     {kin_shear['divergence'].mean():.6f} (expected ≈0)")
    print(f"  curl:           {kin_shear['curl'].mean():.6f} (expected -k={-k})")
    print(f"  shear_xy:       {kin_shear['shear_xy'].mean():.6f} (expected k={k})")
    print(f"  det:            {kin_shear['determinant'].mean():.6f} (expected ≈0)")
    
    interior = kin_shear['shear_xy'][10:-10, 10:-10]
    assert abs(interior.mean() - k) < 0.01, f"Expected shear_xy={k}"
    print("✅ Test 4 passed")
    
    # ========================================================================
    # Test 5: Compression (Occlusion Signature)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 5: Compression (Occlusion Signature)")
    print("="*80)
    
    # Compression: u = -α(x - cx), v = -α(y - cy)
    alpha_comp = 0.1
    u_comp = -alpha_comp * (x_grid - cx)
    v_comp = -alpha_comp * (y_grid - cy)
    
    kin_comp = compute_flow_kinematics(u_comp, v_comp)
    
    print(f"✓ Radial compression: α={-alpha_comp}")
    print(f"  divergence: {kin_comp['divergence'].mean():.6f} (expected -2α={-2*alpha_comp})")
    print(f"  det:        {kin_comp['determinant'].mean():.6f} (expected α²={alpha_comp**2})")
    
    interior = kin_comp['divergence'][10:-10, 10:-10]
    assert interior.mean() < 0, "Compression should have negative divergence"
    
    # Check flow type classification (should be mostly "sink")
    interior_type = kin_comp['flow_type'][10:-10, 10:-10]
    sink_fraction = np.sum(interior_type == 2) / interior_type.size
    print(f"  flow_type:  {sink_fraction*100:.1f}% classified as 'sink' (expected >80%)")
    assert sink_fraction > 0.8, "Most should be classified as sink"
    
    print("✅ Test 5 passed")
    
    # ========================================================================
    # Test 6: Motion Boundary (Saddle Point)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 6: Motion Boundary (Saddle Point)")
    print("="*80)
    
    # Left half static, right half moving
    u_boundary = np.zeros((H, W), dtype=np.float32)
    u_boundary[:, W//2:] = 5.0
    v_boundary = np.zeros((H, W), dtype=np.float32)
    
    kin_boundary = compute_flow_kinematics(u_boundary, v_boundary)
    
    print(f"✓ Motion boundary at x={W//2}")
    
    # Check near boundary
    boundary_region = kin_boundary['du_dx'][:, W//2-2:W//2+2]
    print(f"  ∂u/∂x at boundary: mean={boundary_region.mean():.2f}, max={boundary_region.max():.2f}")
    print(f"  Expected: Large gradient at boundary")
    
    assert boundary_region.max() > 1.0, "Should have large gradient at boundary"
    print("✅ Test 6 passed")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("✨ ALL FLOW KINEMATICS TESTS PASSED!")
    print("="*80)
    print("\nModule provides:")
    print("  • compute_flow_kinematics() - complete kinematic analysis")
    print("\nKey descriptors:")
    print("  • divergence - compression/expansion (occlusion marker)")
    print("  • curl - rotation/circulation")
    print("  • shear_magnitude - distortion")
    print("  • determinant - area scaling (det<0 = algorithm error)")
    print("  • principal strains (lambda1, lambda2)")
    print("  • flow_type classification")
    print("\nUse cases:")
    print("  • Occlusion detection (divergence < 0)")
    print("  • Rotation detection (high curl)")
    print("  • Algorithm validation (det < 0 = folding error)")
    print("  • Motion characterization (source/sink/saddle/spiral)")
    print("  • Ensemble selection (prefer physically plausible flows)")
