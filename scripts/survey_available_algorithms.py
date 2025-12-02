#!/usr/bin/env python3
# File: scripts/survey_available_algorithms.py
"""
Survey what optical flow algorithms are actually available in your OpenCV.
"""

import cv2
import numpy as np
import time

print("="*80)
print("OPTICAL FLOW ALGORITHM SURVEY")
print("="*80)
print(f"OpenCV version: {cv2.__version__}")
print()

# Test image pair for benchmarking
H, W = 256, 256
frame1 = np.random.rand(H, W).astype(np.float32)
frame2 = np.random.rand(H, W).astype(np.float32)

algorithms = []

# ============================================================================
# Standard OpenCV algorithms (opencv-python)
# ============================================================================

print("STANDARD ALGORITHMS (opencv-python)")
print("-" * 80)

# 1. Farneback
try:
    start = time.time()
    flow = cv2.calcOpticalFlowFarneback(
        (frame1 * 255).astype(np.uint8),
        (frame2 * 255).astype(np.uint8),
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    elapsed = time.time() - start
    algorithms.append({
        'name': 'Farneback',
        'module': 'cv2',
        'speed': elapsed,
        'available': True,
        'quality': 'Good',
        'notes': 'Fast, pyramid-based, good for smooth regions'
    })
    print(f"✅ Farneback - {elapsed:.3f}s")
except Exception as e:
    print(f"❌ Farneback - {e}")

# 2. DIS (Dense Inverse Search)
try:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    start = time.time()
    flow = dis.calc(
        (frame1 * 255).astype(np.uint8),
        (frame2 * 255).astype(np.uint8),
        None
    )
    elapsed = time.time() - start
    algorithms.append({
        'name': 'DIS',
        'module': 'cv2',
        'speed': elapsed,
        'available': True,
        'quality': 'Good',
        'notes': 'Very fast, patch-based, good general purpose'
    })
    print(f"✅ DIS - {elapsed:.3f}s")
except Exception as e:
    print(f"❌ DIS - {e}")

print()

# ============================================================================
# Contrib algorithms (opencv-contrib-python)
# ============================================================================

print("CONTRIB ALGORITHMS (opencv-contrib-python)")
print("-" * 80)

if not hasattr(cv2, 'optflow'):
    print("❌ cv2.optflow not available - need opencv-contrib-python")
    print()
else:
    # List all creators in optflow
    creators = [attr for attr in dir(cv2.optflow) if attr.endswith('_create')]
    
    print(f"Found {len(creators)} algorithm creators in cv2.optflow:")
    for creator in sorted(creators):
        print(f"  - {creator}")
    print()
    
    # Test each one
    print("Testing available algorithms:")
    print()
    
    # 3. DualTVL1
    if hasattr(cv2.optflow, 'DualTVL1OpticalFlow_create'):
        try:
            tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            start = time.time()
            flow = tvl1.calc(frame1, frame2, None)
            elapsed = time.time() - start
            algorithms.append({
                'name': 'DualTVL1',
                'module': 'cv2.optflow',
                'speed': elapsed,
                'available': True,
                'quality': 'Excellent',
                'notes': 'Variational method, preserves boundaries, slow but high quality'
            })
            print(f"✅ DualTVL1 - {elapsed:.3f}s")
            print(f"   Quality: Excellent for boundaries")
            print(f"   Speed: ~{elapsed/0.05:.1f}x slower than DIS")
        except Exception as e:
            print(f"❌ DualTVL1 - {e}")
    else:
        print(f"❌ DualTVL1 - Creator not found")
    
    print()
    
    # 4. DeepFlow
    if hasattr(cv2.optflow, 'createOptFlow_DeepFlow'):
        try:
            deepflow = cv2.optflow.createOptFlow_DeepFlow()
            start = time.time()
            flow = deepflow.calc(
                (frame1 * 255).astype(np.uint8),
                (frame2 * 255).astype(np.uint8),
                None
            )
            elapsed = time.time() - start
            algorithms.append({
                'name': 'DeepFlow',
                'module': 'cv2.optflow',
                'speed': elapsed,
                'available': True,
                'quality': 'Excellent',
                'notes': 'Large displacement, good for boundaries, slow'
            })
            print(f"✅ DeepFlow - {elapsed:.3f}s")
            print(f"   Quality: Excellent for large motions and boundaries")
            print(f"   Speed: ~{elapsed/0.05:.1f}x slower than DIS")
        except Exception as e:
            print(f"❌ DeepFlow - {e}")
    else:
        print(f"❌ DeepFlow - Creator not found")
    
    print()
    
    # 5. PCAFlow
    if hasattr(cv2.optflow, 'createOptFlow_PCAFlow'):
        try:
            pcaflow = cv2.optflow.createOptFlow_PCAFlow()
            start = time.time()
            flow = pcaflow.calc(
                (frame1 * 255).astype(np.uint8),
                (frame2 * 255).astype(np.uint8),
                None
            )
            elapsed = time.time() - start
            algorithms.append({
                'name': 'PCAFlow',
                'module': 'cv2.optflow',
                'speed': elapsed,
                'available': True,
                'quality': 'Good',
                'notes': 'PCA-based, fast, good for rigid motion'
            })
            print(f"✅ PCAFlow - {elapsed:.3f}s")
            print(f"   Quality: Good for rigid/piecewise motion")
            print(f"   Speed: ~{elapsed/0.05:.1f}x slower than DIS")
        except Exception as e:
            print(f"❌ PCAFlow - {e}")
    else:
        print(f"❌ PCAFlow - Creator not found")
    
    print()
    
    # 6. SimpleFlow
    if hasattr(cv2.optflow, 'createOptFlow_SimpleFlow'):
        try:
            simpleflow = cv2.optflow.createOptFlow_SimpleFlow()
            start = time.time()
            flow = simpleflow.calc(
                (frame1 * 255).astype(np.uint8),
                (frame2 * 255).astype(np.uint8),
                None
            )
            elapsed = time.time() - start
            algorithms.append({
                'name': 'SimpleFlow',
                'module': 'cv2.optflow',
                'speed': elapsed,
                'available': True,
                'quality': 'Good',
                'notes': 'Fast approximation'
            })
            print(f"✅ SimpleFlow - {elapsed:.3f}s")
            print(f"   Quality: Good, faster approximation")
            print(f"   Speed: ~{elapsed/0.05:.1f}x slower than DIS")
        except Exception as e:
            print(f"❌ SimpleFlow - {e}")
    else:
        print(f"❌ SimpleFlow - Creator not found")
    
    print()
    
    # 7. SparseToDense
    if hasattr(cv2.optflow, 'createOptFlow_SparseToDense'):
        try:
            sparse2dense = cv2.optflow.createOptFlow_SparseToDense()
            start = time.time()
            flow = sparse2dense.calc(
                (frame1 * 255).astype(np.uint8),
                (frame2 * 255).astype(np.uint8),
                None
            )
            elapsed = time.time() - start
            algorithms.append({
                'name': 'SparseToDense',
                'module': 'cv2.optflow',
                'speed': elapsed,
                'available': True,
                'quality': 'Good',
                'notes': 'Sparse-to-dense interpolation'
            })
            print(f"✅ SparseToDense - {elapsed:.3f}s")
            print(f"   Quality: Good for well-textured regions")
            print(f"   Speed: ~{elapsed/0.05:.1f}x slower than DIS")
        except Exception as e:
            print(f"❌ SparseToDense - {e}")
    else:
        print(f"❌ SparseToDense - Creator not found")
    
    print()
    
    # 8. Brox (the troublemaker)
    if hasattr(cv2.optflow, 'BroxOpticalFlow_create'):
        try:
            brox = cv2.optflow.BroxOpticalFlow_create()
            start = time.time()
            flow = brox.calc(frame1, frame2, None)
            elapsed = time.time() - start
            algorithms.append({
                'name': 'Brox',
                'module': 'cv2.optflow',
                'speed': elapsed,
                'available': True,
                'quality': 'Excellent',
                'notes': 'Variational, preserves boundaries, very slow'
            })
            print(f"✅ Brox - {elapsed:.3f}s")
            print(f"   Quality: Excellent for boundaries")
            print(f"   Speed: ~{elapsed/0.05:.1f}x slower than DIS")
        except Exception as e:
            print(f"❌ Brox - {e}")
    else:
        print(f"❌ Brox - Creator not found (expected in older versions)")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print()

available = [a for a in algorithms if a['available']]
print(f"Available algorithms: {len(available)}")
print()

if available:
    # Sort by speed
    available_sorted = sorted(available, key=lambda x: x['speed'])
    
    print("Ranked by speed (fastest first):")
    for i, algo in enumerate(available_sorted, 1):
        print(f"{i}. {algo['name']:15s} - {algo['speed']:.3f}s - {algo['quality']:10s} - {algo['notes']}")
    
    print()
    print("RECOMMENDATIONS FOR YOUR ENSEMBLE:")
    print("-" * 80)
    
    # Find best complementary algorithm
    has_dis = any(a['name'] == 'DIS' for a in available)
    has_farneback = any(a['name'] == 'Farneback' for a in available)
    
    boundary_preserving = [a for a in available if 'boundaries' in a['notes'].lower() or 'variational' in a['notes'].lower()]
    
    if boundary_preserving:
        best = boundary_preserving[0]
        print(f"\n✅ BEST ADDITION: {best['name']}")
        print(f"   Why: {best['notes']}")
        print(f"   Speed impact: ~{best['speed']/0.05:.1f}x slower than DIS")
        print(f"   Complements: DIS/Farneback (smooth) + {best['name']} (boundaries)")
    else:
        print("\n⚠️  No boundary-preserving algorithms available")
        print("   Your current DIS + Farneback ensemble is optimal")

print()
print("="*80)
