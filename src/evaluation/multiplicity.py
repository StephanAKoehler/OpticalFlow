# File: src/evaluation/multiplicity.py
"""
Unified deformation analysis at perturbation scale.

Computes destination density and Jacobian-based kinematics using the
already-computed perturbation flows. Everything is consistent with the
perturbation scale used for uncertainty analysis.

Key principle: Use the perturbation flows directly - no recomputation!
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Dict


def compute_destination_density(u: np.ndarray, 
                                v: np.ndarray,
                                perturbation_scale: float) -> Dict[str, np.ndarray]:
    """
    Compute destination density: sources per destination pixel.
    
    For flow from Frame S (source) to Frame D (destination):
    - Counts how many Frame S pixels map to each Frame D bin
    - Bins at perturbation scale for consistency with uncertainty analysis
    - Returns density normalized so 1.0 = one-to-one mapping
    
    Args:
        u, v: Flow field S→D (H, W)
        perturbation_scale: RMS perturbation magnitude (pixels)
                           Determines binning resolution
    
    Returns:
        Dictionary containing:
            - density: Sources per destination pixel (H, W)
                      1.0 = normal one-to-one mapping
                      >1.0 = compression (many sources → one destination)
                      <1.0 = expansion (few sources → one destination)
            - bin_size: Actual bin size used (pixels)
            - perturbation_scale: Input scale (for reference)
    
    Interpretation (depends on flow direction):
        Forward flow (A→B):
            density_B > 1.0 → occlusion in Frame B (multiple Frame A sources)
            density_B < 1.0 → disocclusion in Frame B (no Frame A source)
        
        Backward flow (B→A):
            density_A > 1.0 → disocclusion source in Frame A (multiple Frame B pixels trace here)
            density_A < 1.0 → occlusion in Frame A (few Frame B pixels trace here)
    
    Note:
        - Boundary artifacts expected (clipping at frame edges)
        - Interior region more reliable
        - Scale-invariant: density=1.0 means normal regardless of bin_size
    """
    H, W = u.shape
    
    # Bin size from perturbation scale
    bin_size = max(1, int(np.round(perturbation_scale)))
    
    # Compute destination coordinates
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    x_dst = x_grid + u
    y_dst = y_grid + v
    
    # Bin destinations
    x_binned = (x_dst / bin_size).astype(np.int32)
    y_binned = (y_dst / bin_size).astype(np.int32)
    
    # Grid dimensions
    H_bins = (H + bin_size - 1) // bin_size
    W_bins = (W + bin_size - 1) // bin_size
    
    # Clip to valid range (creates boundary artifacts)
    x_binned = np.clip(x_binned, 0, W_bins - 1)
    y_binned = np.clip(y_binned, 0, H_bins - 1)
    
    # Count sources per destination bin (vectorized)
    flat_indices = y_binned.ravel() * W_bins + x_binned.ravel()
    counts = np.bincount(flat_indices, minlength=H_bins * W_bins)
    counts_binned = counts.reshape(H_bins, W_bins).astype(np.float32)
    
    # Normalize by bin area to get sources per destination pixel
    # This makes density=1.0 mean "normal" regardless of bin_size
    density_binned = counts_binned / (bin_size ** 2)
    
    # Upsample to original resolution
    if bin_size > 1:
        density = zoom(density_binned, bin_size, order=1)
        density = density[:H, :W]  # Crop to exact size
    else:
        density = density_binned
    
    return {
        'density': density.astype(np.float32),
        'bin_size': bin_size,
        'perturbation_scale': perturbation_scale,
    }


def compute_jacobian_from_perturbations(flows: dict,
                                        base_key: str = 'AB',
                                        pert_key: str = 'perturbations_A') -> Dict[str, np.ndarray]:
    """
    Compute Jacobian directly from perturbation flow responses.
    
    Uses the already-computed perturbed flows - no additional computation!
    Only uses pure horizontal (dx,0) and vertical (0,dy) perturbations
    for clean ∂/∂x and ∂/∂y estimates.
    
    Args:
        flows: Dict from flow_computation.compute_all_flows()
        base_key: 'AB' or 'BA'
        pert_key: 'perturbations_A' or 'perturbations_B'
    
    Returns:
        Dictionary with Jacobian components and derived quantities:
            - du_dx, du_dy, dv_dx, dv_dy: Jacobian components
            - divergence: ∂u/∂x + ∂v/∂y (compression/expansion)
            - curl: ∂v/∂x - ∂u/∂y (rotation)
            - determinant: det(J) (area scaling of flow gradient)
            - n_horizontal, n_vertical: Number of perturbations used
    
    Note:
        Requires pure horizontal and vertical perturbations in flows.
        Diagonal perturbations are skipped (they give mixed derivatives).
    """
    u_base, v_base = flows['base'][base_key]
    H, W = u_base.shape
    
    du_dx_list = []
    dv_dx_list = []
    du_dy_list = []
    dv_dy_list = []
    
    # Determine which flow keys to use
    if pert_key == 'perturbations_A':
        plus_key = 'A_to_B_plus'
        minus_key = 'A_to_B_minus'
    else:
        plus_key = 'B_to_A_plus'
        minus_key = 'B_to_A_minus'
    
    for pert in flows[pert_key]:
        dx, dy = pert['delta']
        
        u_plus, v_plus = pert[plus_key]
        u_minus, v_minus = pert[minus_key]
        
        # Pure horizontal perturbation (dy = 0) → measure ∂/∂x
        if abs(dy) < 0.01 and abs(dx) > 0.01:
            # Central difference: (u(+dx) - u(-dx)) / (2*dx)
            du_dx = (u_plus - u_minus) / (2 * dx)
            dv_dx = (v_plus - v_minus) / (2 * dx)
            du_dx_list.append(du_dx)
            dv_dx_list.append(dv_dx)
        
        # Pure vertical perturbation (dx = 0) → measure ∂/∂y
        elif abs(dx) < 0.01 and abs(dy) > 0.01:
            # Central difference: (u(+dy) - u(-dy)) / (2*dy)
            du_dy = (u_plus - u_minus) / (2 * dy)
            dv_dy = (v_plus - v_minus) / (2 * dy)
            du_dy_list.append(du_dy)
            dv_dy_list.append(dv_dy)
        
        # Skip diagonal perturbations (they give mixed derivatives)
    
    # Average across perturbations of each type
    if du_dx_list:
        du_dx = np.mean(du_dx_list, axis=0)
        dv_dx = np.mean(dv_dx_list, axis=0)
    else:
        import sys
        print("⚠️  WARNING: No horizontal perturbations found for Jacobian!")
        print("    Add perturbations like (±1,0), (±2,0), (±3,0) to your delta grid.")
        du_dx = np.zeros((H, W), dtype=np.float32)
        dv_dx = np.zeros((H, W), dtype=np.float32)
    
    if du_dy_list:
        du_dy = np.mean(du_dy_list, axis=0)
        dv_dy = np.mean(dv_dy_list, axis=0)
    else:
        import sys
        print("⚠️  WARNING: No vertical perturbations found for Jacobian!")
        print("    Add perturbations like (0,±1), (0,±2), (0,±3) to your delta grid.")
        du_dy = np.zeros((H, W), dtype=np.float32)
        dv_dy = np.zeros((H, W), dtype=np.float32)
    
    # Compute kinematic descriptors
    divergence = du_dx + dv_dy
    curl = dv_dx - du_dy
    det_J = du_dx * dv_dy - du_dy * dv_dx
    
    return {
        'du_dx': du_dx.astype(np.float32),
        'du_dy': du_dy.astype(np.float32),
        'dv_dx': dv_dx.astype(np.float32),
        'dv_dy': dv_dy.astype(np.float32),
        'divergence': divergence.astype(np.float32),
        'curl': curl.astype(np.float32),
        'determinant': det_J.astype(np.float32),
        'n_horizontal': len(du_dx_list),
        'n_vertical': len(du_dy_list),
    }


def compute_unified_deformation(flows: dict,
                                base_key: str = 'AB',
                                pert_key: str = 'perturbations_A') -> Dict[str, np.ndarray]:
    """
    Compute all deformation metrics at perturbation scale.
    
    Combines:
    1. Destination density (direct pixel counting)
    2. Jacobian (from perturbation responses)
    3. Consistency checks between them
    
    All use the same perturbation flows - perfectly consistent!
    
    Args:
        flows: Dict from flow_computation.compute_all_flows()
        base_key: 'AB' or 'BA'
        pert_key: 'perturbations_A' or 'perturbations_B'
    
    Returns:
        Dictionary with:
            - density: Direct count (discrete, sources per destination pixel)
            - divergence: From Jacobian (differential, ∂u/∂x + ∂v/∂y)
            - curl: From Jacobian (∂v/∂x - ∂u/∂y)
            - determinant: From Jacobian (det of flow gradient)
            - du_dx, du_dy, dv_dx, dv_dy: Jacobian components
            - density_from_jacobian: Theoretical density from det(J_deformation)
            - consistency_error: |density - density_from_jacobian|
            - perturbation_scale, bin_size: Metadata
    
    Note:
        density vs density_from_jacobian should agree in well-behaved regions.
        Large consistency_error indicates numerical issues or complex flow.
    """
    u_base, v_base = flows['base'][base_key]
    
    # 1. Compute density (direct counting)
    deltas = [pert['delta'] for pert in flows[pert_key]]
    pert_scale = np.sqrt(np.mean([np.hypot(dx, dy)**2 for dx, dy in deltas]))
    
    density_result = compute_destination_density(u_base, v_base, pert_scale)
    density = density_result['density']
    
    # 2. Compute Jacobian (from perturbations)
    jacobian = compute_jacobian_from_perturbations(flows, base_key, pert_key)
    
    # 3. Theoretical density from Jacobian
    # For deformation x' = x + u(x):
    # Jacobian of deformation: J_def = I + ∇u
    # J_def = [[1 + ∂u/∂x,  ∂u/∂y    ],
    #          [∂v/∂x,       1 + ∂v/∂y]]
    # det(J_def) = (1 + ∂u/∂x)(1 + ∂v/∂y) - ∂u/∂y · ∂v/∂x
    
    det_J_def = (1 + jacobian['du_dx']) * (1 + jacobian['dv_dy']) - \
                jacobian['du_dy'] * jacobian['dv_dx']
    
    # Density ≈ 1 / det(J_def)
    # (sources per unit destination area)
    density_from_jacobian = 1.0 / (np.abs(det_J_def) + 1e-6)
    
    # 4. Consistency check
    consistency_error = np.abs(density - density_from_jacobian)
    
    return {
        # Direct measurements
        'density': density,
        'divergence': jacobian['divergence'],
        'curl': jacobian['curl'],
        'determinant': jacobian['determinant'],
        
        # Jacobian components
        'du_dx': jacobian['du_dx'],
        'du_dy': jacobian['du_dy'],
        'dv_dx': jacobian['dv_dx'],
        'dv_dy': jacobian['dv_dy'],
        
        # Consistency
        'deformation_determinant': det_J_def,
        'density_from_jacobian': density_from_jacobian,
        'consistency_error': consistency_error,
        
        # Metadata
        'perturbation_scale': pert_scale,
        'bin_size': density_result['bin_size'],
        'n_horizontal_perturbations': jacobian['n_horizontal'],
        'n_vertical_perturbations': jacobian['n_vertical'],
    }


if __name__ == "__main__":
    print("🧪 Testing unified deformation analysis...")
    
    # ========================================================================
    # Test 1: Mock flows structure with pure perturbations
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: Uniform Translation with Pure Perturbations")
    print("="*80)
    
    H, W = 100, 100
    
    # Base flow: uniform translation
    u_base = np.ones((H, W), dtype=np.float32) * 5.0
    v_base = np.zeros((H, W), dtype=np.float32)
    
    # Mock flows structure
    flows = {
        'base': {
            'AB': (u_base, v_base)
        },
        'perturbations_A': []
    }
    
    # Add pure horizontal perturbations
    for dx in [1, 2, 3, -1, -2, -3]:
        flows['perturbations_A'].append({
            'delta': (dx, 0),
            'A_to_B_plus': (u_base + 0.0, v_base + 0.0),  # Uniform flow unchanged
            'A_to_B_minus': (u_base + 0.0, v_base + 0.0),
        })
    
    # Add pure vertical perturbations
    for dy in [1, 2, 3]:
        flows['perturbations_A'].append({
            'delta': (0, dy),
            'A_to_B_plus': (u_base + 0.0, v_base + 0.0),
            'A_to_B_minus': (u_base + 0.0, v_base + 0.0),
        })
    
    # Compute unified deformation
    result = compute_unified_deformation(flows, 'AB', 'perturbations_A')
    
    print(f"✓ Uniform translation: u=5.0, v=0.0")
    print(f"  Perturbation scale: {result['perturbation_scale']:.2f} px")
    print(f"  Bin size: {result['bin_size']} px")
    print(f"  Horizontal perturbations used: {result['n_horizontal_perturbations']}")
    print(f"  Vertical perturbations used: {result['n_vertical_perturbations']}")
    
    # Check interior region
    interior = result['density'][10:-10, 10:-10]
    div_interior = result['divergence'][10:-10, 10:-10]
    
    print(f"\n  Density: mean={interior.mean():.3f}, std={interior.std():.3f}")
    print(f"  Expected: ≈1.0 (one-to-one mapping)")
    
    print(f"\n  Divergence: mean={div_interior.mean():.6f}, std={div_interior.std():.6f}")
    print(f"  Expected: ≈0.0 (no compression/expansion)")
    
    print(f"\n  Curl: mean={result['curl'][10:-10, 10:-10].mean():.6f}")
    print(f"  Expected: ≈0.0 (no rotation)")
    
    assert 0.9 < interior.mean() < 1.1, "Density should be ≈1.0"
    assert abs(div_interior.mean()) < 0.01, "Divergence should be ≈0.0"
    
    print("✅ Test 1 passed")
    
    # ========================================================================
    # Test 2: Compression (Occlusion)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: Compression (Occlusion Simulation)")
    print("="*80)
    
    # Left half static, right half moves left
    u_occlude = np.zeros((H, W), dtype=np.float32)
    u_occlude[:, W//2:] = -20.0
    v_occlude = np.zeros((H, W), dtype=np.float32)
    
    # Mock perturbed flows (simplified - just use base)
    flows_occ = {
        'base': {'AB': (u_occlude, v_occlude)},
        'perturbations_A': []
    }
    
    for dx in [1, 2, 3, -1, -2, -3]:
        flows_occ['perturbations_A'].append({
            'delta': (dx, 0),
            'A_to_B_plus': (u_occlude, v_occlude),
            'A_to_B_minus': (u_occlude, v_occlude),
        })
    
    for dy in [1, 2, 3]:
        flows_occ['perturbations_A'].append({
            'delta': (0, dy),
            'A_to_B_plus': (u_occlude, v_occlude),
            'A_to_B_minus': (u_occlude, v_occlude),
        })
    
    result_occ = compute_unified_deformation(flows_occ, 'AB', 'perturbations_A')
    
    print(f"✓ Split motion: left static, right moves left by 20px")
    
    # Check different regions
    left_region = result_occ['density'][:, :W//2-25]
    overlap_region = result_occ['density'][:, W//2-25:W//2-5]
    
    print(f"  Left (static):    density={left_region.mean():.3f}")
    print(f"  Overlap zone:     density={overlap_region.mean():.3f} (expect >1.0)")
    
    # Divergence at boundary
    div_boundary = result_occ['divergence'][:, W//2-5:W//2+5]
    print(f"  Boundary div:     mean={div_boundary.mean():.3f} (expect <0, compression)")
    
    assert overlap_region.mean() > 1.2, "Overlap should have high density"
    
    print("✅ Test 2 passed")
    
    # ========================================================================
    # Test 3: Consistency Check
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 3: Density-Jacobian Consistency")
    print("="*80)
    
    # For uniform flow, density and density_from_jacobian should agree
    consistency = result['consistency_error'][10:-10, 10:-10]
    
    print(f"✓ Consistency error for uniform flow:")
    print(f"  Mean: {consistency.mean():.4f}")
    print(f"  Max:  {consistency.max():.4f}")
    print(f"  Expected: Low (density ≈ density_from_jacobian)")
    
    assert consistency.mean() < 0.2, "Should have low consistency error"
    
    print("✅ Test 3 passed")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("✨ ALL MULTIPLICITY TESTS PASSED!")
    print("="*80)
    print("\nModule provides:")
    print("  • compute_destination_density() - direct pixel counting")
    print("  • compute_jacobian_from_perturbations() - derivatives from perturbed flows")
    print("  • compute_unified_deformation() - combined analysis with consistency check")
    print("\nKey features:")
    print("  • Uses already-computed perturbation flows (zero additional OF calls)")
    print("  • All metrics at perturbation scale (consistent with uncertainty analysis)")
    print("  • Consistency check between discrete (density) and differential (Jacobian)")
    print("\nMetrics returned:")
    print("  • density: Sources per destination pixel (discrete count)")
    print("  • divergence: Compression/expansion (∂u/∂x + ∂v/∂y)")
    print("  • curl: Rotation (∂v/∂x - ∂u/∂y)")
    print("  • determinant: Area scaling of flow gradient")
    print("\nInterpretation:")
    print("  Forward (A→B):  density_B > 1.0 = occlusion, divergence < 0 = compression")
    print("  Backward (B→A): density_A < 1.0 = occlusion, divergence > 0 = expansion")
