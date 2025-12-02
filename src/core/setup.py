# File: src/core/setup.py
"""
Core setup utilities for ensemble flow pipeline.

Handles:
- Image loading/generation
- Hash computation
- Experiment cache setup
- Valid mask computation (separate from ground truth)
"""

import hashlib
import numpy as np
import cv2
import toml
from pathlib import Path
from typing import Tuple, Optional

from src.synthesis import (
    generate_pattern_from_config,
    generate_flow_from_config,
    create_frame_pair
)
from src.cache.experiment_cache import (
    ExperimentCache,
    compute_image_hash,
    compute_config_hash
)


def compute_weight_hash(weights: dict) -> str:
    """
    Compute hash of weight vector.
    
    Args:
        weights: Dict with weight values
        
    Returns:
        6-character hash
    """
    hasher = hashlib.sha256()
    
    # Sort keys for consistency
    weight_keys = sorted(weights.keys())
    for key in weight_keys:
        val = weights.get(key, 0.0)
        hasher.update(f"{key}:{val:.6f}".encode())
    
    return hasher.hexdigest()[:6]


def get_next_optim_number(exp_dir: Path) -> int:
    """
    Get next sequential number for weights_optim_NNN directory.
    
    Args:
        exp_dir: Experiment directory path
        
    Returns:
        Next available number (1-based)
    """
    import re
    
    pattern = re.compile(r'weights_optim_(\d+)')
    max_num = 0
    
    if exp_dir.exists():
        for item in exp_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)
    
    return max_num + 1


def get_latest_optim_weights(exp_dir: Path) -> Optional[dict]:
    """
    Get weights from the latest weights_optim_NNN directory.
    
    Args:
        exp_dir: Experiment directory path
        
    Returns:
        Dict of weights, or None if no previous optimization
    """
    import re
    
    pattern = re.compile(r'weights_optim_(\d+)')
    max_num = 0
    latest_dir = None
    
    if exp_dir.exists():
        for item in exp_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
                        latest_dir = item
    
    if latest_dir is None:
        return None
    
    config_path = latest_dir / 'config.toml'
    if not config_path.exists():
        return None
    
    try:
        config = toml.load(config_path)
        return config.get('ensemble', {}).get('weights', None)
    except Exception:
        return None


def weights_are_equal(w1: dict, w2: dict, tolerance: float = 1e-6) -> bool:
    """
    Check if two weight dictionaries are equal within tolerance.
    
    Args:
        w1, w2: Weight dictionaries
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if weights are essentially identical
    """
    if set(w1.keys()) != set(w2.keys()):
        return False
    
    for key in w1.keys():
        if abs(w1[key] - w2[key]) > tolerance:
            return False
    
    return True


def get_image_identifier(config: dict, frame1: np.ndarray, frame2: np.ndarray,
                         u_truth: np.ndarray, v_truth: np.ndarray) -> str:
    """
    Get image identifier based on input type.
    
    For videos: frame00100_frame00101
    For others: <image_hash>
    """
    input_config = config.get('input', {})
    input_type = input_config.get('type', 'synthetic')
    
    if input_type == 'video':
        frame_a = input_config.get('frame_a', 0)
        frame_b = input_config.get('frame_b', 1)
        return f"frame{frame_a:05d}_frame{frame_b:05d}"
    else:
        return compute_image_hash(frame1, frame2, u_truth, v_truth)


def compute_valid_mask(H: int, W: int, boundary_margin: int) -> np.ndarray:
    """
    Compute valid pixel mask based on boundary margin.
    
    Args:
        H, W: Image dimensions
        boundary_margin: Margin in pixels to exclude from edges
        
    Returns:
        Boolean mask (True = valid pixel)
    """
    valid_mask = np.ones((H, W), dtype=bool)
    valid_mask[:boundary_margin, :] = False
    valid_mask[-boundary_margin:, :] = False
    valid_mask[:, :boundary_margin] = False
    valid_mask[:, -boundary_margin:] = False
    return valid_mask


def setup_test_data(config: dict, eval_config: dict, configs: list):
    """
    Generate or load test images and ground truth.
    
    NOTE: Ground truth (u_truth, v_truth) is returned UNMASKED.
    Valid mask is computed separately and should be applied during evaluation.
    
    Args:
        config: Full TOML config dict
        eval_config: The [evaluation] section from config
        configs: List of expanded OF configs from parameter sweep
    
    Returns:
        dict with:
            - frame1, frame2: Input frames (grayscale float32)
            - frame1_original, frame2_original: Original frames (RGB if applicable)
            - u_truth, v_truth: Ground truth flow (UNMASKED)
            - valid_mask: Valid pixel mask (algorithm-dependent)
            - H, W: Image dimensions
            - boundary_margin: Applied boundary margin in pixels
    """
    print("🎨 Generating test data...")
    
    image_config = config['image']
    flow_config = config['flow']
    
    # Set random seed for reproducible test data
    seed = image_config.get('seed', None)
    if seed is None:
        import json
        config_str = json.dumps(image_config, sort_keys=True)
        seed = int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16) % (2**31)
    
    np.random.seed(seed)
    print(f"   Random seed: {seed}")
    
    # Generate pattern
    pattern = generate_pattern_from_config(image_config)
    
    # Generate flow
    flow_shape = pattern.shape[:2]
    u_truth, v_truth = generate_flow_from_config(flow_config, flow_shape)
    
    # Compute boundary margin
    boundary_margin = eval_config.get('boundary_margin', None)
    if boundary_margin is None:
        from src_contamination import get_margin
        perturbation_config = config.get('perturbations', {})
        magnitude = float(perturbation_config.get('magnitude', 1))
        
        print(f"   🔍 Auto-computing boundary margins (magnitude={magnitude}px)...")
        margins = []
        for c in configs:
            m = get_margin(c, magnitude)
            margins.append(m)
        
        boundary_margin = max(margins)
        print(f"   ✅ Boundary margin: {boundary_margin} px (max of {len(configs)} configs)")
    else:
        print(f"   Boundary margin: {boundary_margin} px (from config)")
    
    H, W = pattern.shape[:2]
    
    # Compute valid mask (algorithm-dependent due to window sizes)
    valid_mask = compute_valid_mask(H, W, boundary_margin)
    
    # Create frame pair
    frame1_original, frame2_original, warp_valid_mask = create_frame_pair(pattern, u_truth, v_truth)
    
    # Convert to grayscale if needed
    if len(frame1_original.shape) == 3:
        frame1 = cv2.cvtColor(frame1_original, cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(frame2_original, cv2.COLOR_RGB2GRAY)
    else:
        frame1 = frame1_original.copy()
        frame2 = frame2_original.copy()
    
    # Ensure float32
    if frame1.dtype != np.float32:
        frame1 = frame1.astype(np.float32) / 255.0
        frame2 = frame2.astype(np.float32) / 255.0
    
    # NOTE: Ground truth is NOT masked here
    # Masking is applied during evaluation via valid_mask
    
    print(f"   Image size: {H}×{W}")
    print(f"   Valid pixels: {valid_mask.sum()} ({100 * valid_mask.sum() / (H*W):.1f}%)")
    print()
    
    return {
        'frame1': frame1,
        'frame2': frame2,
        'frame1_original': frame1_original,
        'frame2_original': frame2_original,
        'u_truth': u_truth,
        'v_truth': v_truth,
        'valid_mask': valid_mask,
        'H': H,
        'W': W,
        'boundary_margin': boundary_margin
    }


def setup_experiment_cache(config: dict, test_data: dict, no_cache: bool = False):
    """
    Setup experiment cache with two-phase directory structure.
    
    Phase 1: Image-level data (shared across algorithms)
        results/<image_hash>/frame1.npy, frame2.npy, u_truth.npy, v_truth.npy, config_image.toml
        
    Phase 2: Algorithm-level data (per sweep configuration)
        results/<image_hash>/<of_type>_<config_hash>/config.toml, valid_mask.npy, ...
    
    Returns:
        dict with:
            - exp_cache: ExperimentCache instance
            - exp_dir: Algorithm-level experiment directory path
            - image_dir: Image-level directory path
            - image_id: Image identifier (hash or frame numbers)
            - of_config_hash: OF config hash
            - should_compute: Whether to compute sweep
    """
    print("📂 Setting up experiment cache...")
    
    # Get results directory
    results_dir = Path(config.get('paths', {}).get('results_dir', 'results'))
    
    # Create experiment cache
    exp_cache = ExperimentCache(base_dir=results_dir)
    
    # Phase 1: Setup image-level data
    image_dir, image_hash, is_new_image = exp_cache.setup_image_data(
        config,
        test_data['frame1'],
        test_data['frame2'],
        test_data['u_truth'],
        test_data['v_truth']
    )
    
    # Get OF type for directory naming
    sweep_config = config['parameter_sweep']
    of_type = sweep_config['algorithm']
    if isinstance(of_type, list):
        of_type = of_type[0]
    
    # Phase 2: Setup algorithm-level experiment
    exp_dir, should_compute = exp_cache.setup_experiment(
        of_type,
        config,
        test_data['valid_mask']
    )
    
    # Override if --no-cache
    if no_cache:
        should_compute = True
        print("   ⚠️  Forcing recomputation (--no-cache)")
    
    # Compute config hash for reporting
    of_config_hash = compute_config_hash(config)
    
    print(f"   Image ID: {image_hash}")
    print(f"   OF config hash: {of_config_hash}")
    print(f"   Image dir: {image_dir}")
    print(f"   Experiment dir: {exp_dir}")
    print()
    
    return {
        'exp_cache': exp_cache,
        'exp_dir': exp_dir,
        'image_dir': image_dir,
        'image_id': image_hash,
        'of_config_hash': of_config_hash,
        'should_compute': should_compute
    }


if __name__ == "__main__":
    print("✅ Core setup module loaded")
