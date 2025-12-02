# File: src/evaluation/symmetric_flow.py
"""
Symmetric flow computation from bidirectional optical flow.

Averages forward and backward flows for improved robustness.
"""

import numpy as np
import cv2
import sys


def warp_flow(flow_u: np.ndarray,
              flow_v: np.ndarray,
              warp_u: np.ndarray,
              warp_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp a flow field using another flow field.
    
    Args:
        flow_u, flow_v: Flow field to be warped (H, W)
        warp_u, warp_v: Flow field defining the warping (H, W)
    
    Returns:
        warped_u, warped_v: Warped flow components (H, W)
    
    Method:
        Apply warp_flow to sample flow_field at displaced locations
    """
    H, W = flow_u.shape
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    
    # Apply warp
    map_x = x_grid + warp_u
    map_y = y_grid + warp_v
    
    # Warp both flow components
    warped_u = cv2.remap(
        flow_u,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    warped_v = cv2.remap(
        flow_v,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return warped_u.astype(np.float32), warped_v.astype(np.float32)


def symmetrize_flow_to_frame_A(u_AB: np.ndarray,
                               v_AB: np.ndarray,
                               u_BA: np.ndarray,
                               v_BA: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute symmetrized flow in frame A coordinates.
    
    Args:
        u_AB, v_AB: Forward flow A→B (in frame A coordinates)
        u_BA, v_BA: Backward flow B→A (in frame B coordinates)
    
    Returns:
        u_sym_A, v_sym_A: Symmetrized flow (in frame A coordinates)
    
    Method:
        1. Warp backward flow to frame A coordinates
        2. Reverse direction (B→A becomes A→B)
        3. Average with forward flow
    """
    # Warp backward flow to frame A coordinates using FORWARD flow
    # We start at frame A and follow u_AB, v_AB to sample u_BA, v_BA from frame B
    u_BA_in_A, v_BA_in_A = warp_flow(u_BA, v_BA, u_AB, v_AB)
    
    # Reverse direction: B→A becomes A→B
    u_BA_reversed = -u_BA_in_A
    v_BA_reversed = -v_BA_in_A
    
    # Average forward and reversed backward
    u_sym_A = (u_AB + u_BA_reversed) / 2.0
    v_sym_A = (v_AB + v_BA_reversed) / 2.0
    
    return u_sym_A.astype(np.float32), v_sym_A.astype(np.float32)


def symmetrize_flow_to_frame_B(u_AB: np.ndarray,
                               v_AB: np.ndarray,
                               u_BA: np.ndarray,
                               v_BA: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute symmetrized flow in frame B coordinates.
    
    Args:
        u_AB, v_AB: Forward flow A→B (in frame A coordinates)
        u_BA, v_BA: Backward flow B→A (in frame B coordinates)
    
    Returns:
        u_sym_B, v_sym_B: Symmetrized flow (in frame B coordinates)
    
    Method:
        1. Warp forward flow to frame B coordinates
        2. Reverse direction (A→B becomes B→A)
        3. Average with backward flow
    """
    # Warp forward flow to frame B coordinates using BACKWARD flow
    # We start at frame B and follow u_BA, v_BA to sample u_AB, v_AB from frame A
    u_AB_in_B, v_AB_in_B = warp_flow(u_AB, v_AB, u_BA, v_BA)
    
    # Reverse direction: A→B becomes B→A
    u_AB_reversed = -u_AB_in_B
    v_AB_reversed = -v_AB_in_B
    
    # Average backward and reversed forward
    u_sym_B = (u_BA + u_AB_reversed) / 2.0
    v_sym_B = (v_BA + v_AB_reversed) / 2.0
    
    return u_sym_B.astype(np.float32), v_sym_B.astype(np.float32)


if __name__ == "__main__":
    print("🧪 Testing symmetric flow computation...")
    
    # Create test flows
    H, W = 100, 100
    
    # Forward flow: uniform rightward motion
    u_AB = np.ones((H, W), dtype=np.float32) * 2.0
    v_AB = np.zeros((H, W), dtype=np.float32)
    
    # Backward flow: uniform leftward motion (opposite)
    u_BA = np.ones((H, W), dtype=np.float32) * -2.0
    v_BA = np.zeros((H, W), dtype=np.float32)
    
    print(f"✅ Created test flows: {H}×{W}")
    print(f"   Forward (A→B): u={u_AB[0,0]:.1f}, v={v_AB[0,0]:.1f}")
    print(f"   Backward (B→A): u={u_BA[0,0]:.1f}, v={v_BA[0,0]:.1f}")
    
    # Test symmetrization to frame A
    u_sym_A, v_sym_A = symmetrize_flow_to_frame_A(u_AB, v_AB, u_BA, v_BA)
    
    print(f"\n🔄 Symmetrized to frame A:")
    print(f"   u_sym_A: mean={u_sym_A.mean():.2f}, std={u_sym_A.std():.4f}")
    print(f"   v_sym_A: mean={v_sym_A.mean():.2f}, std={v_sym_A.std():.4f}")
    print(f"   Expected: u≈2.0 (forward and backward should agree after reversal)")
    
    # Test symmetrization to frame B
    u_sym_B, v_sym_B = symmetrize_flow_to_frame_B(u_AB, v_AB, u_BA, v_BA)
    
    print(f"\n🔄 Symmetrized to frame B:")
    print(f"   u_sym_B: mean={u_sym_B.mean():.2f}, std={u_sym_B.std():.4f}")
    print(f"   v_sym_B: mean={v_sym_B.mean():.2f}, std={v_sym_B.std():.4f}")
    print(f"   Expected: u≈-2.0 (B→A perspective)")
    
    # Test with mismatched flows (simulating algorithm error)
    print(f"\n🔄 Test with mismatched flows:")
    u_AB_noisy = u_AB + np.random.randn(H, W) * 0.5
    v_AB_noisy = v_AB + np.random.randn(H, W) * 0.5
    
    u_sym_noisy, v_sym_noisy = symmetrize_flow_to_frame_A(
        u_AB_noisy, v_AB_noisy, u_BA, v_BA
    )
    
    print(f"   Noisy forward: u_std={u_AB_noisy.std():.3f}")
    print(f"   Symmetrized: u_std={u_sym_noisy.std():.3f}")
    print(f"   Noise reduction: {(1 - u_sym_noisy.std() / u_AB_noisy.std()) * 100:.1f}%")
    
    # Verify shapes
    assert u_sym_A.shape == (H, W), "Symmetrized flow shape mismatch"
    assert u_sym_A.dtype == np.float32, "Should be float32"
    
    # Verify that perfect opposite flows give correct result
    assert np.abs(u_sym_A.mean() - 2.0) < 0.1, "Symmetrized flow should average correctly"
    
    print("\n✨ All symmetric flow tests passed!")
