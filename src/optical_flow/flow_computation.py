# File: src/optical_flow/flow_computation.py
"""
Optical flow computation with organized caching.

Computes all optical flows needed for metric evaluation in a single pass.
Returns organized dictionary of flows - no metrics computation here.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.optical_flow.algorithms import compute_optical_flow


def shift_frame(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Shift frame by (dx, dy) pixels using cubic interpolation.
    
    Args:
        frame: Input frame (H, W)
        dx, dy: Shift in pixels
    
    Returns:
        shifted_frame: (H, W) shifted frame
    """
    H, W = frame.shape[:2]
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    
    map_x = x_grid - dx
    map_y = y_grid - dy
    
    shifted = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return shifted


def compute_all_flows(frame_A: np.ndarray,
                     frame_B: np.ndarray,
                     config: dict,
                     deltas: list[tuple[float, float]],
                     verbose: bool = True) -> dict:
    """
    Compute all optical flows needed for metrics computation.
    
    This function computes:
    - Base bidirectional flows (A↔B)
    - Perturbation flows from frame A perspective (shifting frame B)
    - Perturbation flows from frame B perspective (shifting frame A)
    
    Args:
        frame_A: First frame (H, W)
        frame_B: Second frame (H, W)
        config: Optical flow algorithm configuration
        deltas: List of (dx, dy) perturbation vectors
    
    Returns:
        Dictionary with structure:
        {
            'base': {
                'AB': (u, v),  # frame_A → frame_B
                'BA': (u, v)   # frame_B → frame_A
            },
            'perturbations_A': [
                {
                    'delta': (dx, dy),
                    'A_to_B_plus': (u, v),    # frame_A → shift(frame_B, +dx, +dy)
                    'A_to_B_minus': (u, v),   # frame_A → shift(frame_B, -dx, -dy)
                    'B_plus_to_A': (u, v),    # shift(frame_B, +dx, +dy) → frame_A
                    'B_minus_to_A': (u, v)    # shift(frame_B, -dx, -dy) → frame_A
                },
                ... one dict per delta
            ],
            'perturbations_B': [
                {
                    'delta': (dx, dy),
                    'B_to_A_plus': (u, v),    # frame_B → shift(frame_A, +dx, +dy)
                    'B_to_A_minus': (u, v),   # frame_B → shift(frame_A, -dx, -dy)
                    'A_plus_to_B': (u, v),    # shift(frame_A, +dx, +dy) → frame_B
                    'A_minus_to_B': (u, v)    # shift(frame_A, -dx, -dy) → frame_B
                },
                ... one dict per delta
            ],
            'n_of_calls': int  # Total number of OF calls made
        }
    """
    
    # ========================================================================
    # PHASE 1: Base Flows
    # ========================================================================
    
    if verbose:
        print(f"      Computing base flows...")
    u_AB, v_AB = compute_optical_flow(frame_A, frame_B, config)
    u_BA, v_BA = compute_optical_flow(frame_B, frame_A, config)
    
    n_of_calls = 2  # Base flows
    
    flows = {
        'base': {
            'AB': (u_AB, v_AB),
            'BA': (u_BA, v_BA)
        },
        'perturbations_A': [],
        'perturbations_B': []
    }
    
    # ========================================================================
    # PHASE 2: Frame A Perturbations (shift frame B)
    # ========================================================================
    
    if verbose:
        print(f"      Computing frame A perturbations...")
    for dx, dy in deltas:
        # Shift frame B by ±δ
        B_plus = shift_frame(frame_B, dx, dy)
        B_minus = shift_frame(frame_B, -dx, -dy)
        
        # Compute flows: A → B±δ and B±δ → A
        u_A_to_B_plus, v_A_to_B_plus = compute_optical_flow(frame_A, B_plus, config)
        u_A_to_B_minus, v_A_to_B_minus = compute_optical_flow(frame_A, B_minus, config)
        u_B_plus_to_A, v_B_plus_to_A = compute_optical_flow(B_plus, frame_A, config)
        u_B_minus_to_A, v_B_minus_to_A = compute_optical_flow(B_minus, frame_A, config)
        
        n_of_calls += 4
        
        flows['perturbations_A'].append({
            'delta': (dx, dy),
            'A_to_B_plus': (u_A_to_B_plus, v_A_to_B_plus),
            'A_to_B_minus': (u_A_to_B_minus, v_A_to_B_minus),
            'B_plus_to_A': (u_B_plus_to_A, v_B_plus_to_A),
            'B_minus_to_A': (u_B_minus_to_A, v_B_minus_to_A)
        })
    
    # ========================================================================
    # PHASE 3: Frame B Perturbations (shift frame A)
    # ========================================================================
    
    if verbose:
        print(f"      Computing frame B perturbations...")
    for dx, dy in deltas:
        # Shift frame A by ±δ
        A_plus = shift_frame(frame_A, dx, dy)
        A_minus = shift_frame(frame_A, -dx, -dy)
        
        # Compute flows: B → A±δ and A±δ → B
        u_B_to_A_plus, v_B_to_A_plus = compute_optical_flow(frame_B, A_plus, config)
        u_B_to_A_minus, v_B_to_A_minus = compute_optical_flow(frame_B, A_minus, config)
        u_A_plus_to_B, v_A_plus_to_B = compute_optical_flow(A_plus, frame_B, config)
        u_A_minus_to_B, v_A_minus_to_B = compute_optical_flow(A_minus, frame_B, config)
        
        n_of_calls += 4
        
        flows['perturbations_B'].append({
            'delta': (dx, dy),
            'B_to_A_plus': (u_B_to_A_plus, v_B_to_A_plus),
            'B_to_A_minus': (u_B_to_A_minus, v_B_to_A_minus),
            'A_plus_to_B': (u_A_plus_to_B, v_A_plus_to_B),
            'A_minus_to_B': (u_A_minus_to_B, v_A_minus_to_B)
        })
    
    # Store metadata
    flows['n_of_calls'] = n_of_calls
    
    if verbose:
        print(f"      Total OF calls: {n_of_calls}")
    
    return flows


if __name__ == "__main__":
    print("🧪 Testing flow computation...")
    
    # Create test frames
    H, W = 128, 128
    frame_A = np.random.rand(H, W).astype(np.float32)
    frame_B = np.random.rand(H, W).astype(np.float32)
    
    # Test config
    config = {
        'algorithm': 'farneback',
        'pyr_scale': 0.5,
        'levels': 1,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0
    }
    
    # Test deltas
    deltas = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    
    print(f"\nComputing flows for {len(deltas)} deltas...")
    flows = compute_all_flows(frame_A, frame_B, config, deltas)
    
    # Verify structure
    assert 'base' in flows, "Missing base flows"
    assert 'AB' in flows['base'], "Missing AB base flow"
    assert 'BA' in flows['base'], "Missing BA base flow"
    
    assert 'perturbations_A' in flows, "Missing frame A perturbations"
    assert len(flows['perturbations_A']) == len(deltas), "Wrong number of A perturbations"
    
    assert 'perturbations_B' in flows, "Missing frame B perturbations"
    assert len(flows['perturbations_B']) == len(deltas), "Wrong number of B perturbations"
    
    # Verify first perturbation structure
    pert_A = flows['perturbations_A'][0]
    assert 'delta' in pert_A, "Missing delta in perturbation"
    assert 'A_to_B_plus' in pert_A, "Missing A_to_B_plus"
    assert 'A_to_B_minus' in pert_A, "Missing A_to_B_minus"
    assert 'B_plus_to_A' in pert_A, "Missing B_plus_to_A"
    assert 'B_minus_to_A' in pert_A, "Missing B_minus_to_A"
    
    # Verify flow shapes
    u_AB, v_AB = flows['base']['AB']
    assert u_AB.shape == (H, W), f"Wrong shape: {u_AB.shape}"
    assert v_AB.shape == (H, W), f"Wrong shape: {v_AB.shape}"
    
    # Verify call count
    expected_calls = 2 + len(deltas) * 4 + len(deltas) * 4
    assert flows['n_of_calls'] == expected_calls, f"Wrong call count: {flows['n_of_calls']} vs {expected_calls}"
    
    print(f"\n✅ Flow structure validated")
    print(f"✅ Base flows: AB, BA")
    print(f"✅ Frame A perturbations: {len(flows['perturbations_A'])} deltas")
    print(f"✅ Frame B perturbations: {len(flows['perturbations_B'])} deltas")
    print(f"✅ Total OF calls: {flows['n_of_calls']}")
    print("\n✨ All flow computation tests passed!")
