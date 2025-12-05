# File: src/evaluation/self_supervised.py
"""
Pure metric computation from pre-computed flows.

NO optical flow calls here - only consumes flows dict.

Outputs both:
- Bounded metrics [0, 1): for legacy additive loss
- Raw metrics (pixels, intensity): for new multiplicative loss

Bounded metrics use: normalized = raw / hypot(raw, scale)
Raw metrics are in physical units (pixels for flow errors, intensity for photometric).

Normalization:
- Spatial metrics (perturbation, traction, consistency, kinematics) are normalized
  by pollution_depth to enable fair cross-config comparison.
- Photometric metrics are NOT normalized (intensity-based, not spatial).
- Ratio metrics (relative_asym) are NOT normalized (already dimensionless).
"""

import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Pollution Depth / Normalization
# =============================================================================

def parse_config_name(config_name: str) -> dict:
    """
    Parse config name into dict.
    
    Handles:
    - Farneback: 'FB_win7_...' or 'farneback_w21_p5_i3_s0.5_e0.6_fl1_fs1'
    - DIS: 'F_sc0_it12_p12_st3' or 'DIS_...'
    """
    parts = config_name.split('_')
    algorithm = parts[0].lower()
    
    # Normalize algorithm name
    if algorithm == 'fb':
        algorithm = 'farneback'
    elif algorithm == 'f':
        # DIS uses 'F_sc...' format
        algorithm = 'dis'
    
    config = {'algorithm': algorithm}
    
    if algorithm == 'dis':
        # Parse DIS formats:
        # - dis_iter8_fine0_patch6_stride2 (new format from experiment)
        # - F_sc0_it12_p12_st3 (old format)
        for part in parts[1:]:
            part_lower = part.lower()
            # New format: fine0, patch6, stride2, iter8
            if part_lower.startswith('fine'):
                try:
                    config['finest_scale'] = int(part[4:])
                except ValueError:
                    pass
            elif part_lower.startswith('patch'):
                try:
                    config['patch_size'] = int(part[5:])
                except ValueError:
                    pass
            elif part_lower.startswith('stride'):
                try:
                    config['stride'] = int(part[6:])
                except ValueError:
                    pass
            elif part_lower.startswith('iter'):
                try:
                    config['iterations'] = int(part[4:])
                except ValueError:
                    pass
            # Old format: sc0, it12, p12, st3
            elif part_lower.startswith('sc'):
                try:
                    config['finest_scale'] = int(part[2:])
                except ValueError:
                    pass
            elif part_lower.startswith('it'):
                try:
                    config['iterations'] = int(part[2:])
                except ValueError:
                    pass
            elif part_lower.startswith('p') and not part_lower.startswith('poly') and not part_lower.startswith('pyr') and not part_lower.startswith('patch'):
                try:
                    config['patch_size'] = int(part[1:])
                except ValueError:
                    pass
            elif part_lower.startswith('st') and not part_lower.startswith('stride'):
                try:
                    config['stride'] = int(part[2:])
                except ValueError:
                    pass
    elif algorithm == 'farneback':
        for part in parts[1:]:
            part_lower = part.lower()
            if part_lower.startswith('win'):
                # Handle 'win7' format
                config['winsize'] = int(part[3:])
            elif part_lower.startswith('w') and not part_lower.startswith('win'):
                # Handle 'w7' format
                config['winsize'] = int(part[1:])
            elif part_lower.startswith('pyr'):
                # Handle 'pyr5' format
                try:
                    config['pyr_scale'] = int(part[3:]) / 10.0
                except ValueError:
                    pass
            elif part_lower.startswith('p') and not part_lower.startswith('poly') and not part_lower.startswith('pyr'):
                # Handle 'p5' format
                try:
                    config['pyr_scale'] = int(part[1:]) / 10.0
                except ValueError:
                    pass
            elif part_lower.startswith('iter'):
                # Handle 'iter3' format
                try:
                    config['iterations'] = int(part[4:])
                except ValueError:
                    pass
            elif part_lower.startswith('i') and not part_lower.startswith('iter'):
                # Handle 'i3' format
                try:
                    config['iterations'] = int(part[1:])
                except ValueError:
                    pass
            elif part_lower.startswith('sig'):
                # Handle 'sig1.5' format
                try:
                    config['poly_sigma'] = float(part[3:])
                except ValueError:
                    pass
            elif part_lower.startswith('s') and not part_lower.startswith('sig'):
                # Handle 's1.5' format
                try:
                    config['poly_sigma'] = float(part[1:])
                except ValueError:
                    pass
            elif part_lower.startswith('polyn'):
                # Handle 'polyN5' or 'polyn5' format
                try:
                    config['poly_n'] = int(part[5:])
                except ValueError:
                    pass
            elif part_lower.startswith('e'):
                try:
                    config['poly_n_encoded'] = float(part[1:])
                except ValueError:
                    pass
            elif part_lower.startswith('fl'):
                try:
                    config['flags'] = int(part[2:])
                except ValueError:
                    pass
            elif part_lower.startswith('fast'):
                try:
                    config['fast_pyramids'] = int(part[4:])
                except ValueError:
                    pass
            elif part_lower.startswith('fs') and algorithm != 'dis':
                try:
                    config['fast_pyramids'] = int(part[2:])
                except ValueError:
                    pass
    elif algorithm == 'dis':
        for part in parts[1:]:
            if part.startswith('fs'):
                config['finest_scale'] = int(part[2:])
            elif part.startswith('vd'):
                config['variational_refinement_delta'] = int(part[2:])
            elif part.startswith('gd'):
                config['gradient_descent_iterations'] = int(part[2:])
            elif part.startswith('gs'):
                config['patch_stride'] = int(part[2:])
            elif part.startswith('gi'):
                config['patch_size'] = int(part[2:])
    
    return config


def get_pollution_depth(config_name: str) -> float:
    """
    Get empirically measured pollution depth for a config.
    
    Falls back to winsize-based estimate if src_contamination not available.
    """
    config = parse_config_name(config_name)
    
    try:
        import io
        import sys
        from src_contamination import get_margin
        
        # Suppress stdout from get_margin
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            depth = float(get_margin(config, magnitude=1.0))
        finally:
            sys.stdout = old_stdout
        return depth
    except (ImportError, Exception):
        # Fallback: estimate from winsize
        if config.get('algorithm') == 'farneback':
            winsize = config.get('winsize', 15)
            # Empirical: depth ≈ 0.6 * winsize for Farneback
            return 0.6 * winsize
        elif config.get('algorithm') == 'dis':
            # DIS: pollution depth depends on patch_size AND finest_scale
            # finest_scale=0: works at full resolution, margin ≈ patch_size
            # finest_scale=1: works at half resolution, margin roughly 4x larger
            # finest_scale=2: even coarser, margin grows exponentially
            patch_size = config.get('patch_size', 8)
            finest_scale = config.get('finest_scale', 0)
            
            # Base depth similar to patch_size
            base_depth = patch_size
            
            # Each pyramid level roughly doubles the effective footprint
            # Empirically: fine0 ~patch_size, fine1 ~4*patch_size or more
            scale_factor = 4 ** finest_scale  # Exponential growth
            
            return base_depth * scale_factor
        else:
            return 10.0  # Conservative default


# Metrics that need pollution depth normalization (spatial flow metrics)
METRICS_TO_NORMALIZE = {
    'perturbation_raw_A', 'perturbation_raw_B',
    'traction_raw_A', 'traction_raw_B',
    'consistency_raw_A', 'consistency_raw_B',
    'divergence_A', 'divergence_B',
    'curl_A', 'curl_B',
    'shear_magnitude_A', 'shear_magnitude_B',
    'folding_A', 'folding_B',
}


def compute_traction_fidelity(traction_raw: np.ndarray, delta_mag: float = 1.0) -> np.ndarray:
    """
    Convert traction error to fidelity (0-1 scale).
    
    Traction measures |Δu - 2δ| where Δu is flow response to known shift δ.
    - Perfect tracking: Δu = 2δ → traction_error = 0 → fidelity = 1.0
    - No tracking: Δu = 0 → traction_error = 2δ → fidelity = 0.0
    - 50% fidelity: traction_error = δ (half the expected response)
    
    Args:
        traction_raw: Traction error in pixels (RMS across perturbations)
        delta_mag: Perturbation magnitude used (default 1.0 pixel)
    
    Returns:
        Fidelity array in [0, 1] range (clipped)
    """
    expected_response = 2.0 * delta_mag
    fidelity = 1.0 - traction_raw / expected_response
    return np.clip(fidelity, 0.0, 1.0)


def normalize_metric(metric: np.ndarray, depth: float) -> np.ndarray:
    """
    Normalize spatial metric by pollution depth.
    
    Large windows average over more pixels → artificially low raw values.
    Multiplying by depth compensates for this averaging effect.
    """
    return metric * depth


# =============================================================================
# Traction-Gated Ensemble Selection
# =============================================================================

def traction_gated_ensemble(results: list, delta_mag: float = 1.0, 
                            fidelity_threshold: float = 0.5) -> dict:
    """
    Build ensemble flow using traction-gated metric selection.
    
    For each pixel:
        1. Find best config by photo_log (lowest photometric error)
        2. Check if that config's traction fidelity >= threshold at this pixel
        3. If yes: use photo-selected config's flow
        4. If no: use perturbation-selected config's flow (most stable)
    
    Fully self-supervised - no ground truth required.
    
    Args:
        results: List of per-config result dicts from sweep
                 Each has 'flows', 'metrics', 'metadata'
        delta_mag: Perturbation magnitude used in sweep (default 1.0)
        fidelity_threshold: Traction fidelity gate (default 0.5 = 50%)
    
    Returns:
        {
            'u': ensemble u flow,
            'v': ensemble v flow,
            'selection_idx': which config was selected per pixel,
            'used_photo': boolean mask where photo selection was used,
            'photo_idx': photo selection (for diagnostics),
            'pert_idx': perturbation selection (for diagnostics),
            'traction_fidelity': fidelity of photo-selected config per pixel,
        }
    """
    n_configs = len(results)
    
    # Get shape from first result
    sample_u = results[0]['flows']['u_AB']
    H, W = sample_u.shape
    
    # Build stacks
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    photo_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    pert_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    traction_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, r in enumerate(results):
        u_stack[i] = r['flows']['u_AB']
        v_stack[i] = r['flows']['v_AB']
        photo_stack[i] = r['metrics']['photo_log_raw_A']
        
        # Get pollution depth for normalization
        config_name = r['metadata'].get('config_name', f'config_{i}')
        depth = get_pollution_depth(config_name)
        
        # Normalize perturbation by depth for fair cross-config comparison
        pert_stack[i] = r['metrics']['perturbation_raw_A'] * depth
        
        # Traction raw (not normalized - we want per-config behavior)
        traction_stack[i] = r['metrics']['traction_raw_A']
    
    # Index grids for advanced indexing
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    
    # Photo selection: argmin photo_log per pixel
    photo_idx = np.argmin(photo_stack, axis=0)
    
    # Perturbation selection: argmin perturbation per pixel
    pert_idx = np.argmin(pert_stack, axis=0)
    
    # Get traction of photo-selected config at each pixel
    traction_at_photo = traction_stack[photo_idx, y_idx, x_idx]
    
    # Compute fidelity
    traction_fidelity = compute_traction_fidelity(traction_at_photo, delta_mag)
    
    # Gate: use photo where fidelity is high enough, else use pert
    use_photo = traction_fidelity >= fidelity_threshold
    
    # Final selection
    selection_idx = np.where(use_photo, photo_idx, pert_idx)
    
    # Extract flows
    u_ensemble = u_stack[selection_idx, y_idx, x_idx]
    v_ensemble = v_stack[selection_idx, y_idx, x_idx]
    
    return {
        'u': u_ensemble,
        'v': v_ensemble,
        'selection_idx': selection_idx,
        'used_photo': use_photo,
        'photo_idx': photo_idx,
        'pert_idx': pert_idx,
        'traction_fidelity': traction_fidelity,
    }


def traction_gated_fallback_ensemble(results: list, delta_mag: float = 1.0,
                                      fidelity_threshold: float = 0.5,
                                      fallback_metric: str = 'photo',
                                      traction_exponent: float = 2.0) -> dict:
    """
    Build ensemble with traction-weighted global fallback config.
    
    1. Compute mean traction fidelity per pixel (across configs)
    2. Select global fallback config using traction²-weighted metric
    3. Per-pixel: if traction >= threshold, use photo selection; else use fallback
    
    The key insight: in textureless regions, per-pixel metrics are unreliable,
    so use a single global config chosen by the textured regions.
    
    Args:
        results: List of per-config result dicts from sweep
        delta_mag: Perturbation magnitude used in sweep (default 1.0)
        fidelity_threshold: Traction fidelity gate (default 0.5 = 50%)
        fallback_metric: 'photo' or 'pert' - which metric to use for fallback selection
        traction_exponent: Power for traction weighting (default 2.0, use 1.0 for linear)
    
    Returns:
        {
            'u': ensemble u flow,
            'v': ensemble v flow,
            'selection_idx': which config was selected per pixel,
            'used_photo': boolean mask where photo selection was used,
            'photo_idx': photo selection per pixel,
            'fallback_idx': global fallback config index,
            'traction_fidelity': mean traction fidelity per pixel,
        }
    """
    n_configs = len(results)
    
    # Get shape from first result
    sample_u = results[0]['flows']['u_AB']
    H, W = sample_u.shape
    
    # Build stacks
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    photo_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    pert_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    traction_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, r in enumerate(results):
        u_stack[i] = r['flows']['u_AB']
        v_stack[i] = r['flows']['v_AB']
        photo_stack[i] = r['metrics']['photo_log_raw_A']
        traction_stack[i] = r['metrics']['traction_raw_A']
        
        # Normalize perturbation by depth
        config_name = r['metadata'].get('config_name', f'config_{i}')
        depth = get_pollution_depth(config_name)
        pert_stack[i] = r['metrics']['perturbation_raw_A'] * depth
    
    # Index grids
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    
    # Mean traction fidelity per pixel (across configs)
    mean_traction_raw = np.mean(traction_stack, axis=0)
    traction_fidelity = compute_traction_fidelity(mean_traction_raw, delta_mag)
    
    # Traction weights for fallback selection
    w = traction_fidelity ** traction_exponent
    w_sum = w.sum()
    
    # Select global fallback using chosen metric
    if fallback_metric == 'photo':
        metric_stack = photo_stack
    elif fallback_metric == 'pert':
        metric_stack = pert_stack
    else:
        raise ValueError(f"Unknown fallback_metric: {fallback_metric}")
    
    weighted_metric_per_config = np.zeros(n_configs, dtype=np.float32)
    for i in range(n_configs):
        weighted_metric_per_config[i] = (metric_stack[i] * w).sum() / w_sum
    
    fallback_idx = int(np.argmin(weighted_metric_per_config))
    
    # Per-pixel photo selection
    photo_idx = np.argmin(photo_stack, axis=0)
    
    # Get traction of photo-selected config at each pixel
    traction_at_photo = traction_stack[photo_idx, y_idx, x_idx]
    traction_fidelity_at_photo = compute_traction_fidelity(traction_at_photo, delta_mag)
    
    # Gate: use photo where fidelity is high enough, else use fallback
    use_photo = traction_fidelity_at_photo >= fidelity_threshold
    
    # Final selection
    selection_idx = np.where(use_photo, photo_idx, fallback_idx)
    
    # Extract flows
    u_ensemble = u_stack[selection_idx, y_idx, x_idx]
    v_ensemble = v_stack[selection_idx, y_idx, x_idx]
    
    return {
        'u': u_ensemble,
        'v': v_ensemble,
        'selection_idx': selection_idx,
        'used_photo': use_photo,
        'photo_idx': photo_idx,
        'fallback_idx': fallback_idx,
        'traction_fidelity': traction_fidelity_at_photo,
        'weighted_metric_per_config': weighted_metric_per_config,
    }


def traction_weighted_ensemble(results: list, delta_mag: float = 1.0,
                                fallback_metric: str = 'photo',
                                traction_exponent: float = 2.0) -> dict:
    """
    Build ensemble with soft traction weighting (no threshold).
    
    1. Compute mean traction fidelity per pixel
    2. Select global fallback config using traction-weighted metric
    3. Per-pixel: blend photo-selected flow with fallback using traction weight
    
    u_final = w * u_photo + (1-w) * u_fallback
    where w = traction^exponent
    
    No threshold parameter. Smooth transition from photo to fallback.
    
    Args:
        results: List of per-config result dicts from sweep
        delta_mag: Perturbation magnitude used in sweep (default 1.0)
        fallback_metric: 'photo' or 'pert' - which metric to use for fallback selection
        traction_exponent: Power for traction weighting (default 2.0, use 1.0 for linear)
    
    Returns:
        {
            'u': ensemble u flow,
            'v': ensemble v flow,
            'photo_idx': photo selection per pixel,
            'fallback_idx': global fallback config index,
            'traction_weight': traction weight per pixel (0-1),
        }
    """
    n_configs = len(results)
    
    # Get shape from first result
    sample_u = results[0]['flows']['u_AB']
    H, W = sample_u.shape
    
    # Build stacks
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    photo_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    pert_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    traction_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, r in enumerate(results):
        u_stack[i] = r['flows']['u_AB']
        v_stack[i] = r['flows']['v_AB']
        photo_stack[i] = r['metrics']['photo_log_raw_A']
        traction_stack[i] = r['metrics']['traction_raw_A']
        
        # Normalize perturbation by depth
        config_name = r['metadata'].get('config_name', f'config_{i}')
        depth = get_pollution_depth(config_name)
        pert_stack[i] = r['metrics']['perturbation_raw_A'] * depth
    
    # Index grids
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    
    # Mean traction fidelity per pixel (across configs)
    mean_traction_raw = np.mean(traction_stack, axis=0)
    traction_fidelity = compute_traction_fidelity(mean_traction_raw, delta_mag)
    
    # Traction weights for fallback selection
    w = traction_fidelity ** traction_exponent
    w_sum = w.sum()
    
    # Select global fallback using chosen metric
    if fallback_metric == 'photo':
        metric_stack = photo_stack
    elif fallback_metric == 'pert':
        metric_stack = pert_stack
    else:
        raise ValueError(f"Unknown fallback_metric: {fallback_metric}")
    
    weighted_metric_per_config = np.zeros(n_configs, dtype=np.float32)
    for i in range(n_configs):
        weighted_metric_per_config[i] = (metric_stack[i] * w).sum() / w_sum
    
    fallback_idx = int(np.argmin(weighted_metric_per_config))
    
    # Per-pixel photo selection
    photo_idx = np.argmin(photo_stack, axis=0)
    
    # Get traction of photo-selected config at each pixel
    traction_at_photo = traction_stack[photo_idx, y_idx, x_idx]
    traction_fidelity_at_photo = compute_traction_fidelity(traction_at_photo, delta_mag)
    
    # Traction weight for blending
    w_blend = traction_fidelity_at_photo ** traction_exponent
    
    # Get photo-selected and fallback flows
    u_photo = u_stack[photo_idx, y_idx, x_idx]
    v_photo = v_stack[photo_idx, y_idx, x_idx]
    u_fallback = u_stack[fallback_idx]
    v_fallback = v_stack[fallback_idx]
    
    # Blend: high traction → photo, low traction → fallback
    u_ensemble = w_blend * u_photo + (1 - w_blend) * u_fallback
    v_ensemble = w_blend * v_photo + (1 - w_blend) * v_fallback
    
    return {
        'u': u_ensemble,
        'v': v_ensemble,
        'photo_idx': photo_idx,
        'fallback_idx': fallback_idx,
        'traction_weight': w_blend,
        'weighted_metric_per_config': weighted_metric_per_config,
    }


def smooth_traction_ensemble(results: list, delta_mag: float = 1.0,
                              traction_exponent: float = 1.0,
                              sigma_fraction: float = 0.25,
                              temperature: float = None) -> dict:
    """
    Fully smooth ensemble with spatially-varying soft fallback blend.
    
    1. Per-pixel photo selection (hard argmin)
    2. Spatially-varying fallback: smooth pert scores, softmax blend across configs
    3. Smooth gate: traction blends between photo and fallback
    
    u_final = traction * u_photo + (1 - traction) * u_fallback
    
    Args:
        results: List of per-config result dicts from sweep
        delta_mag: Perturbation magnitude used in sweep (default 1.0)
        traction_exponent: Power for traction weighting (default 1.0)
        sigma_fraction: Gaussian sigma as fraction of image size (default 0.25)
        temperature: Softmax temperature. If None, uses median of smooth scores.
    
    Returns:
        {
            'u': ensemble u flow,
            'v': ensemble v flow,
            'photo_idx': photo selection per pixel,
            'traction_weight': traction weight used for blending,
            'fallback_weights': softmax weights per config (n_configs, H, W),
            'temperature': temperature used,
        }
    """
    from scipy.ndimage import gaussian_filter
    
    n_configs = len(results)
    
    # Get shape from first result
    sample_u = results[0]['flows']['u_AB']
    H, W = sample_u.shape
    
    # Build stacks
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    photo_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    pert_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    traction_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, r in enumerate(results):
        u_stack[i] = r['flows']['u_AB']
        v_stack[i] = r['flows']['v_AB']
        photo_stack[i] = r['metrics']['photo_log_raw_A']
        traction_stack[i] = r['metrics']['traction_raw_A']
        
        # Normalize perturbation by depth
        config_name = r['metadata'].get('config_name', f'config_{i}')
        depth = get_pollution_depth(config_name)
        pert_stack[i] = r['metrics']['perturbation_raw_A'] * depth
    
    # Index grids
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    
    # Mean traction fidelity per pixel (across configs)
    mean_traction_raw = np.mean(traction_stack, axis=0)
    traction_fidelity = compute_traction_fidelity(mean_traction_raw, delta_mag)
    
    # Traction weight for blending and scoring
    traction_weight = traction_fidelity ** traction_exponent
    
    # === Per-pixel photo selection (hard argmin) ===
    photo_idx = np.argmin(photo_stack, axis=0)
    u_photo = u_stack[photo_idx, y_idx, x_idx]
    v_photo = v_stack[photo_idx, y_idx, x_idx]
    
    # === Spatially-varying soft fallback blend ===
    sigma = sigma_fraction * max(H, W)
    
    # Pre-compute weighted pert for each config
    weighted_pert_stack = pert_stack * traction_weight[np.newaxis, :, :]
    
    # Downsample for faster filtering (sigma=160 on full res is slow)
    # Use factor of 4 for ~40x speedup
    downsample = 4
    H_ds, W_ds = H // downsample, W // downsample
    sigma_ds = sigma / downsample
    
    from skimage.transform import resize
    
    # Downsample traction weight and filter once
    traction_weight_ds = resize(traction_weight, (H_ds, W_ds), order=1, preserve_range=True)
    traction_smooth_ds = gaussian_filter(traction_weight_ds.astype(np.float32), sigma=sigma_ds, mode='reflect')
    traction_smooth = resize(traction_smooth_ds, (H, W), order=1, preserve_range=True).astype(np.float32)
    
    # Downsample, filter, upsample each weighted pert
    smooth_score = np.zeros((n_configs, H, W), dtype=np.float32)
    for i in range(n_configs):
        weighted_ds = resize(weighted_pert_stack[i], (H_ds, W_ds), order=1, preserve_range=True)
        weighted_smooth_ds = gaussian_filter(weighted_ds.astype(np.float32), sigma=sigma_ds, mode='reflect')
        weighted_smooth = resize(weighted_smooth_ds, (H, W), order=1, preserve_range=True)
        smooth_score[i] = weighted_smooth / (traction_smooth + 1e-10)
    
    # Determine temperature if not provided
    if temperature is None:
        # Use IQR of smooth scores as temperature
        # If configs are tightly clustered → small temp → sharper selection
        # If widely spread → larger temp → softer blending
        q25 = np.percentile(smooth_score, 25)
        q75 = np.percentile(smooth_score, 75)
        temperature = float(q75 - q25)
        if temperature < 1e-10:
            temperature = 1.0  # Fallback if scores are all identical
    
    # Softmax weights for fallback blend
    # w[i] = exp(-score[i] / T) / sum(exp(-score[j] / T))
    neg_scaled = -smooth_score / temperature
    # Subtract max for numerical stability
    neg_scaled_stable = neg_scaled - np.max(neg_scaled, axis=0, keepdims=True)
    exp_scores = np.exp(neg_scaled_stable)
    fallback_weights = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    
    # Blend fallback flow (vectorized)
    u_fallback = np.sum(fallback_weights * u_stack, axis=0)
    v_fallback = np.sum(fallback_weights * v_stack, axis=0)
    
    # === Smooth gate ===
    # Get traction of photo-selected config at each pixel
    traction_at_photo = traction_stack[photo_idx, y_idx, x_idx]
    traction_fidelity_at_photo = compute_traction_fidelity(traction_at_photo, delta_mag)
    gate_weight = traction_fidelity_at_photo ** traction_exponent
    
    # Blend: high traction → photo, low traction → fallback
    u_ensemble = gate_weight * u_photo + (1 - gate_weight) * u_fallback
    v_ensemble = gate_weight * v_photo + (1 - gate_weight) * v_fallback
    
    return {
        'u': u_ensemble,
        'v': v_ensemble,
        'photo_idx': photo_idx,
        'traction_weight': gate_weight,
        'fallback_weights': fallback_weights,
        'temperature': temperature,
    }


from src.evaluation.photometric import (
    compute_photometric_raw, 
    compute_photometric_windowed,
    compute_photometric_rgb,
    compute_photometric_rgb_log
)
from src.evaluation.symmetric_flow import symmetrize_flow_to_frame_A, symmetrize_flow_to_frame_B, warp_flow
from src.evaluation.asymmetric_flow import compute_asymmetry_in_frame_A, compute_asymmetry_in_frame_B
from src.evaluation.flow_kinematics import compute_flow_kinematics


def shift_frame(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift frame (needed for photometric)."""
    H, W = frame.shape[:2]
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x, map_y = x_grid - dx, y_grid - dy
    
    if frame.ndim == 3:
        return cv2.remap(frame, map_x, map_y, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, 0)
    else:
        return cv2.remap(frame, map_x, map_y, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, 0)


def is_rgb(frame: np.ndarray) -> bool:
    """Check if frame is RGB (3 channels)."""
    return frame.ndim == 3 and frame.shape[2] == 3


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale if RGB."""
    if is_rgb(frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return frame.astype(np.float32)


def compute_metrics_frame_A(frame_A, frame_B, flows, config, rms_diff: float):
    """
    Compute frame A metrics from flows dict.

    Returns both bounded (legacy) and raw metrics.
    
    Args:
        frame_A, frame_B: Input frames (grayscale or RGB)
        flows: Flows dict from compute_all_flows
        config: Algorithm config dict
        rms_diff: RMS difference between unwarped frames (for photometric normalization)
    """
    H, W = frame_A.shape[:2]
    winsize = config.get('winsize', 15)
    
    has_rgb = is_rgb(frame_A) and is_rgb(frame_B)
    
    frame_A_gray = to_grayscale(frame_A)
    frame_B_gray = to_grayscale(frame_B)

    u_AB, v_AB = flows['base']['AB']
    u_BA, v_BA = flows['base']['BA']

    # Accumulators - both bounded and raw for traction
    traction_acc_bounded = np.zeros((H, W), dtype=np.float32)
    traction_acc_raw = np.zeros((H, W), dtype=np.float32)
    consistency_acc = np.zeros((H, W), dtype=np.float32)
    photo_gray_acc_plus = np.zeros((H, W), dtype=np.float32)
    photo_gray_acc_minus = np.zeros((H, W), dtype=np.float32)
    
    # RGB per-channel accumulators
    if has_rgb:
        photo_r_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_r_acc_minus = np.zeros((H, W), dtype=np.float32)
        photo_g_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_g_acc_minus = np.zeros((H, W), dtype=np.float32)
        photo_b_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_b_acc_minus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_acc_minus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_log_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_log_acc_minus = np.zeros((H, W), dtype=np.float32)

    # Perturbations
    for pert in flows['perturbations_A']:
        dx, dy = pert['delta']
        perturbation_mag = np.hypot(dx, dy)

        u_plus, v_plus = pert['A_to_B_plus']
        u_minus, v_minus = pert['A_to_B_minus']
        u_bwd_plus, v_bwd_plus = pert['B_plus_to_A']
        u_bwd_minus, v_bwd_minus = pert['B_minus_to_A']

        # Traction - accumulate both raw and bounded
        diff_u = u_plus - u_minus
        diff_v = v_plus - v_minus
        traction_raw_pert = np.sqrt((diff_u - 2 * dx) ** 2 + (diff_v - 2 * dy) ** 2)
        traction_bounded_pert = traction_raw_pert / np.hypot(traction_raw_pert, perturbation_mag)
        traction_acc_bounded += traction_bounded_pert
        traction_acc_raw += traction_raw_pert

        # Consistency
        u_bwd_plus_w, v_bwd_plus_w = warp_flow(u_bwd_plus, v_bwd_plus, u_plus, v_plus)
        cons_plus = np.sqrt((u_plus + u_bwd_plus_w) ** 2 + (v_plus + v_bwd_plus_w) ** 2)
        consistency_acc += cons_plus

        u_bwd_minus_w, v_bwd_minus_w = warp_flow(u_bwd_minus, v_bwd_minus, u_minus, v_minus)
        cons_minus = np.sqrt((u_minus + u_bwd_minus_w) ** 2 + (v_minus + v_bwd_minus_w) ** 2)
        consistency_acc += cons_minus

        # Photometric grayscale
        B_plus_gray = shift_frame(frame_B_gray, dx, dy)
        B_minus_gray = shift_frame(frame_B_gray, -dx, -dy)
        photo_gray_acc_plus += compute_photometric_raw(frame_A_gray, B_plus_gray, u_plus, v_plus)
        photo_gray_acc_minus += compute_photometric_raw(frame_A_gray, B_minus_gray, u_minus, v_minus)
        
        # Photometric RGB variants
        if has_rgb:
            B_plus_rgb = shift_frame(frame_B, dx, dy)
            B_minus_rgb = shift_frame(frame_B, -dx, -dy)
            
            # Per-channel RGB (for new multiplicative loss)
            photo_r_acc_plus += compute_photometric_raw(frame_A[:,:,0].astype(np.float32), 
                                                         B_plus_rgb[:,:,0].astype(np.float32), u_plus, v_plus)
            photo_r_acc_minus += compute_photometric_raw(frame_A[:,:,0].astype(np.float32), 
                                                          B_minus_rgb[:,:,0].astype(np.float32), u_minus, v_minus)
            photo_g_acc_plus += compute_photometric_raw(frame_A[:,:,1].astype(np.float32), 
                                                         B_plus_rgb[:,:,1].astype(np.float32), u_plus, v_plus)
            photo_g_acc_minus += compute_photometric_raw(frame_A[:,:,1].astype(np.float32), 
                                                          B_minus_rgb[:,:,1].astype(np.float32), u_minus, v_minus)
            photo_b_acc_plus += compute_photometric_raw(frame_A[:,:,2].astype(np.float32), 
                                                         B_plus_rgb[:,:,2].astype(np.float32), u_plus, v_plus)
            photo_b_acc_minus += compute_photometric_raw(frame_A[:,:,2].astype(np.float32), 
                                                          B_minus_rgb[:,:,2].astype(np.float32), u_minus, v_minus)
            
            # RGB Euclidean (legacy)
            photo_rgb_acc_plus += compute_photometric_rgb(frame_A, B_plus_rgb, u_plus, v_plus)
            photo_rgb_acc_minus += compute_photometric_rgb(frame_A, B_minus_rgb, u_minus, v_minus)
            
            # Log-RGB
            photo_rgb_log_acc_plus += compute_photometric_rgb_log(frame_A, B_plus_rgb, u_plus, v_plus)
            photo_rgb_log_acc_minus += compute_photometric_rgb_log(frame_A, B_minus_rgb, u_minus, v_minus)

    n_deltas = len(flows['perturbations_A'])

    # ========================================================================
    # Finalize Traction
    # ========================================================================
    traction_A = traction_acc_bounded / n_deltas  # bounded [0, 1)
    traction_raw_A = traction_acc_raw / n_deltas  # raw (pixels)

    # ========================================================================
    # Finalize Consistency
    # ========================================================================
    pert_magnitudes = [np.hypot(dx, dy) for dx, dy in [pert['delta'] for pert in flows['perturbations_A']]]
    pert_scale = np.sqrt(np.mean([m**2 for m in pert_magnitudes]))
    
    consistency_raw_A = consistency_acc / (2 * n_deltas)  # raw (pixels)
    consistency_A = consistency_raw_A / np.hypot(consistency_raw_A, pert_scale)  # bounded [0, 1)

    # ========================================================================
    # Finalize Photometric Grayscale
    # ========================================================================
    photo_gray_raw_A = (photo_gray_acc_plus + photo_gray_acc_minus) / (2 * n_deltas)  # raw (intensity)
    
    photo_scale = max(rms_diff, 1.0)
    photometric_A_raw = photo_gray_raw_A / np.hypot(photo_gray_raw_A, photo_scale)  # bounded, unsmoothed
    
    photometric_smoothed = compute_photometric_windowed(photo_gray_raw_A, winsize)
    photometric_A = photometric_smoothed / np.hypot(photometric_smoothed, photo_scale)  # bounded, smoothed

    # ========================================================================
    # Finalize Photometric RGB variants
    # ========================================================================
    if has_rgb:
        # Per-channel raw (for new multiplicative loss)
        photo_r_raw_A = (photo_r_acc_plus + photo_r_acc_minus) / (2 * n_deltas)
        photo_g_raw_A = (photo_g_acc_plus + photo_g_acc_minus) / (2 * n_deltas)
        photo_b_raw_A = (photo_b_acc_plus + photo_b_acc_minus) / (2 * n_deltas)
        
        # Log raw
        photo_log_raw_A = (photo_rgb_log_acc_plus + photo_rgb_log_acc_minus) / (2 * n_deltas)
        
        # Bounded RGB Euclidean (legacy)
        photo_rgb_raw_mean = (photo_rgb_acc_plus + photo_rgb_acc_minus) / (2 * n_deltas)
        photo_rgb_scale = max(rms_diff * np.sqrt(3), 1.0)
        photo_rgb_smoothed = compute_photometric_windowed(photo_rgb_raw_mean, winsize)
        photometric_rgb_A = photo_rgb_smoothed / np.hypot(photo_rgb_smoothed, photo_rgb_scale)
        
        # Bounded log-RGB (legacy)
        photo_rgb_log_scale = 0.1
        photo_rgb_log_smoothed = compute_photometric_windowed(photo_log_raw_A, winsize)
        photometric_rgb_log_A = photo_rgb_log_smoothed / np.hypot(photo_rgb_log_smoothed, photo_rgb_log_scale)

    # ========================================================================
    # Compute Displacement Sensitivity (perturbation error metric)
    # ========================================================================
    u_base, v_base = u_AB, v_AB
    
    deviation_list = []
    for pert in flows['perturbations_A']:
        dx, dy = pert['delta']
        
        u_plus, v_plus = pert['A_to_B_plus']
        u_minus, v_minus = pert['A_to_B_minus']
        
        u_plus_corrected = u_plus - dx
        v_plus_corrected = v_plus - dy
        u_minus_corrected = u_minus - (-dx)
        v_minus_corrected = v_minus - (-dy)
        
        dev_plus = np.sqrt((u_plus_corrected - u_base)**2 + (v_plus_corrected - v_base)**2)
        dev_minus = np.sqrt((u_minus_corrected - u_base)**2 + (v_minus_corrected - v_base)**2)
        
        deviation_list.append(dev_plus)
        deviation_list.append(dev_minus)
    
    deviation_stack = np.stack(deviation_list, axis=0)
    perturbation_raw_A = np.sqrt(np.mean(deviation_stack**2, axis=0))  # raw (pixels)
    
    # Bounded normalization
    displacements_sensitivity_A2B = perturbation_raw_A / np.hypot(perturbation_raw_A, pert_scale)  # bounded [0, 1)

    # ========================================================================
    # Build result dict
    # ========================================================================
    result = {
        # Bounded metrics (legacy)
        'traction_A': traction_A,
        'consistency_A': consistency_A,
        'photometric_A_raw': photometric_A_raw,  # bounded, unsmoothed
        'photometric_A': photometric_A,  # bounded, smoothed
        'displacements_sensitivity_A2B': displacements_sensitivity_A2B,
        
        # Raw metrics (for new multiplicative loss)
        'traction_raw_A': traction_raw_A,
        'consistency_raw_A': consistency_raw_A,
        'perturbation_raw_A': perturbation_raw_A,
        'photo_gray_raw_A': photo_gray_raw_A,
    }
    
    # Add RGB metrics if available
    if has_rgb:
        # Bounded (legacy)
        result['photometric_rgb_A'] = photometric_rgb_A
        result['photometric_rgb_log_A'] = photometric_rgb_log_A
        
        # Raw (new)
        result['photo_r_raw_A'] = photo_r_raw_A
        result['photo_g_raw_A'] = photo_g_raw_A
        result['photo_b_raw_A'] = photo_b_raw_A
        result['photo_log_raw_A'] = photo_log_raw_A

    return result


def compute_metrics_frame_B(frame_A, frame_B, flows, config, rms_diff: float):
    """
    Compute frame B metrics from flows dict.

    Returns both bounded (legacy) and raw metrics.
    
    Args:
        frame_A, frame_B: Input frames (grayscale or RGB)
        flows: Flows dict from compute_all_flows
        config: Algorithm config dict
        rms_diff: RMS difference between unwarped frames (for photometric normalization)
    """
    H, W = frame_B.shape[:2]
    winsize = config.get('winsize', 15)
    
    has_rgb = is_rgb(frame_A) and is_rgb(frame_B)
    
    frame_A_gray = to_grayscale(frame_A)
    frame_B_gray = to_grayscale(frame_B)

    u_AB, v_AB = flows['base']['AB']
    u_BA, v_BA = flows['base']['BA']

    # Accumulators - both bounded and raw for traction
    traction_acc_bounded = np.zeros((H, W), dtype=np.float32)
    traction_acc_raw = np.zeros((H, W), dtype=np.float32)
    consistency_acc = np.zeros((H, W), dtype=np.float32)
    photo_gray_acc_plus = np.zeros((H, W), dtype=np.float32)
    photo_gray_acc_minus = np.zeros((H, W), dtype=np.float32)
    
    # RGB per-channel accumulators
    if has_rgb:
        photo_r_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_r_acc_minus = np.zeros((H, W), dtype=np.float32)
        photo_g_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_g_acc_minus = np.zeros((H, W), dtype=np.float32)
        photo_b_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_b_acc_minus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_acc_minus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_log_acc_plus = np.zeros((H, W), dtype=np.float32)
        photo_rgb_log_acc_minus = np.zeros((H, W), dtype=np.float32)

    # Perturbations
    for pert in flows['perturbations_B']:
        dx, dy = pert['delta']
        perturbation_mag = np.hypot(dx, dy)

        u_plus, v_plus = pert['B_to_A_plus']
        u_minus, v_minus = pert['B_to_A_minus']
        u_bwd_plus, v_bwd_plus = pert['A_plus_to_B']
        u_bwd_minus, v_bwd_minus = pert['A_minus_to_B']

        # Traction - accumulate both raw and bounded
        diff_u = u_plus - u_minus
        diff_v = v_plus - v_minus
        traction_raw_pert = np.sqrt((diff_u - 2 * dx) ** 2 + (diff_v - 2 * dy) ** 2)
        traction_bounded_pert = traction_raw_pert / np.hypot(traction_raw_pert, perturbation_mag)
        traction_acc_bounded += traction_bounded_pert
        traction_acc_raw += traction_raw_pert

        # Consistency
        u_bwd_plus_w, v_bwd_plus_w = warp_flow(u_bwd_plus, v_bwd_plus, u_plus, v_plus)
        cons_plus = np.sqrt((u_plus + u_bwd_plus_w) ** 2 + (v_plus + v_bwd_plus_w) ** 2)
        consistency_acc += cons_plus

        u_bwd_minus_w, v_bwd_minus_w = warp_flow(u_bwd_minus, v_bwd_minus, u_minus, v_minus)
        cons_minus = np.sqrt((u_minus + u_bwd_minus_w) ** 2 + (v_minus + v_bwd_minus_w) ** 2)
        consistency_acc += cons_minus

        # Photometric grayscale
        A_plus_gray = shift_frame(frame_A_gray, dx, dy)
        A_minus_gray = shift_frame(frame_A_gray, -dx, -dy)
        photo_gray_acc_plus += compute_photometric_raw(frame_B_gray, A_plus_gray, u_plus, v_plus)
        photo_gray_acc_minus += compute_photometric_raw(frame_B_gray, A_minus_gray, u_minus, v_minus)
        
        # Photometric RGB variants
        if has_rgb:
            A_plus_rgb = shift_frame(frame_A, dx, dy)
            A_minus_rgb = shift_frame(frame_A, -dx, -dy)
            
            # Per-channel RGB (for new multiplicative loss)
            photo_r_acc_plus += compute_photometric_raw(frame_B[:,:,0].astype(np.float32), 
                                                         A_plus_rgb[:,:,0].astype(np.float32), u_plus, v_plus)
            photo_r_acc_minus += compute_photometric_raw(frame_B[:,:,0].astype(np.float32), 
                                                          A_minus_rgb[:,:,0].astype(np.float32), u_minus, v_minus)
            photo_g_acc_plus += compute_photometric_raw(frame_B[:,:,1].astype(np.float32), 
                                                         A_plus_rgb[:,:,1].astype(np.float32), u_plus, v_plus)
            photo_g_acc_minus += compute_photometric_raw(frame_B[:,:,1].astype(np.float32), 
                                                          A_minus_rgb[:,:,1].astype(np.float32), u_minus, v_minus)
            photo_b_acc_plus += compute_photometric_raw(frame_B[:,:,2].astype(np.float32), 
                                                         A_plus_rgb[:,:,2].astype(np.float32), u_plus, v_plus)
            photo_b_acc_minus += compute_photometric_raw(frame_B[:,:,2].astype(np.float32), 
                                                          A_minus_rgb[:,:,2].astype(np.float32), u_minus, v_minus)
            
            # RGB Euclidean (legacy)
            photo_rgb_acc_plus += compute_photometric_rgb(frame_B, A_plus_rgb, u_plus, v_plus)
            photo_rgb_acc_minus += compute_photometric_rgb(frame_B, A_minus_rgb, u_minus, v_minus)
            
            # Log-RGB
            photo_rgb_log_acc_plus += compute_photometric_rgb_log(frame_B, A_plus_rgb, u_plus, v_plus)
            photo_rgb_log_acc_minus += compute_photometric_rgb_log(frame_B, A_minus_rgb, u_minus, v_minus)

    n_deltas = len(flows['perturbations_B'])

    # ========================================================================
    # Finalize Traction
    # ========================================================================
    traction_B = traction_acc_bounded / n_deltas  # bounded [0, 1)
    traction_raw_B = traction_acc_raw / n_deltas  # raw (pixels)

    # ========================================================================
    # Finalize Consistency
    # ========================================================================
    pert_magnitudes = [np.hypot(dx, dy) for dx, dy in [pert['delta'] for pert in flows['perturbations_B']]]
    pert_scale = np.sqrt(np.mean([m**2 for m in pert_magnitudes]))
    
    consistency_raw_B = consistency_acc / (2 * n_deltas)  # raw (pixels)
    consistency_B = consistency_raw_B / np.hypot(consistency_raw_B, pert_scale)  # bounded [0, 1)

    # ========================================================================
    # Finalize Photometric Grayscale
    # ========================================================================
    photo_gray_raw_B = (photo_gray_acc_plus + photo_gray_acc_minus) / (2 * n_deltas)  # raw (intensity)
    
    photo_scale = max(rms_diff, 1.0)
    photometric_B_raw = photo_gray_raw_B / np.hypot(photo_gray_raw_B, photo_scale)  # bounded, unsmoothed
    
    photometric_smoothed = compute_photometric_windowed(photo_gray_raw_B, winsize)
    photometric_B = photometric_smoothed / np.hypot(photometric_smoothed, photo_scale)  # bounded, smoothed

    # ========================================================================
    # Finalize Photometric RGB variants
    # ========================================================================
    if has_rgb:
        # Per-channel raw (for new multiplicative loss)
        photo_r_raw_B = (photo_r_acc_plus + photo_r_acc_minus) / (2 * n_deltas)
        photo_g_raw_B = (photo_g_acc_plus + photo_g_acc_minus) / (2 * n_deltas)
        photo_b_raw_B = (photo_b_acc_plus + photo_b_acc_minus) / (2 * n_deltas)
        
        # Log raw
        photo_log_raw_B = (photo_rgb_log_acc_plus + photo_rgb_log_acc_minus) / (2 * n_deltas)
        
        # Bounded RGB Euclidean (legacy)
        photo_rgb_raw_mean = (photo_rgb_acc_plus + photo_rgb_acc_minus) / (2 * n_deltas)
        photo_rgb_scale = max(rms_diff * np.sqrt(3), 1.0)
        photo_rgb_smoothed = compute_photometric_windowed(photo_rgb_raw_mean, winsize)
        photometric_rgb_B = photo_rgb_smoothed / np.hypot(photo_rgb_smoothed, photo_rgb_scale)
        
        # Bounded log-RGB (legacy)
        photo_rgb_log_scale = 0.1
        photo_rgb_log_smoothed = compute_photometric_windowed(photo_log_raw_B, winsize)
        photometric_rgb_log_B = photo_rgb_log_smoothed / np.hypot(photo_rgb_log_smoothed, photo_rgb_log_scale)

    # ========================================================================
    # Compute Displacement Sensitivity (perturbation error metric)
    # ========================================================================
    u_base, v_base = u_BA, v_BA
    
    deviation_list = []
    for pert in flows['perturbations_B']:
        dx, dy = pert['delta']
        
        u_plus, v_plus = pert['B_to_A_plus']
        u_minus, v_minus = pert['B_to_A_minus']
        
        u_plus_corrected = u_plus - dx
        v_plus_corrected = v_plus - dy
        u_minus_corrected = u_minus - (-dx)
        v_minus_corrected = v_minus - (-dy)
        
        dev_plus = np.sqrt((u_plus_corrected - u_base)**2 + (v_plus_corrected - v_base)**2)
        dev_minus = np.sqrt((u_minus_corrected - u_base)**2 + (v_minus_corrected - v_base)**2)
        
        deviation_list.append(dev_plus)
        deviation_list.append(dev_minus)
    
    deviation_stack = np.stack(deviation_list, axis=0)
    perturbation_raw_B = np.sqrt(np.mean(deviation_stack**2, axis=0))  # raw (pixels)
    
    # Bounded normalization
    displacements_sensitivity_B2A = perturbation_raw_B / np.hypot(perturbation_raw_B, pert_scale)  # bounded [0, 1)

    # ========================================================================
    # Build result dict
    # ========================================================================
    result = {
        # Bounded metrics (legacy)
        'traction_B': traction_B,
        'consistency_B': consistency_B,
        'photometric_B_raw': photometric_B_raw,  # bounded, unsmoothed
        'photometric_B': photometric_B,  # bounded, smoothed
        'displacements_sensitivity_B2A': displacements_sensitivity_B2A,
        
        # Raw metrics (for new multiplicative loss)
        'traction_raw_B': traction_raw_B,
        'consistency_raw_B': consistency_raw_B,
        'perturbation_raw_B': perturbation_raw_B,
        'photo_gray_raw_B': photo_gray_raw_B,
    }
    
    # Add RGB metrics if available
    if has_rgb:
        # Bounded (legacy)
        result['photometric_rgb_B'] = photometric_rgb_B
        result['photometric_rgb_log_B'] = photometric_rgb_log_B
        
        # Raw (new)
        result['photo_r_raw_B'] = photo_r_raw_B
        result['photo_g_raw_B'] = photo_g_raw_B
        result['photo_b_raw_B'] = photo_b_raw_B
        result['photo_log_raw_B'] = photo_log_raw_B

    return result


def compute_metrics_from_flows(frame_A, frame_B, flows, config, rms_diff: float, verbose: bool = True):
    """
    Main entry point - compute all metrics from flows dict.

    NO optical flow calls here!

    Returns separated flows and metrics (both bounded and raw).
    
    Args:
        frame_A, frame_B: Input frames (grayscale or RGB)
        flows: Flows dict from compute_all_flows
        config: Algorithm config dict
        rms_diff: RMS difference between unwarped frames (for photometric normalization)
        verbose: Print progress messages
    
    Returns:
        {
            'flows': {u_AB, v_AB, u_BA, v_BA, u_sym_A, v_sym_A, u_sym_B, v_sym_B},
            'metrics': {
                # Bounded (legacy): traction_A, consistency_A, photometric_A, ...
                # Raw (new): traction_raw_A, consistency_raw_A, photo_gray_raw_A, ...
                # Asymmetry: asym_magnitude_A, relative_asym_A, ...
                # Kinematics: divergence_A, curl_A, determinant_A, folding_A, ...
            }
        }
    """
    if verbose:
        print(f"      Computing metrics from flows...")

    u_AB, v_AB = flows['base']['AB']
    u_BA, v_BA = flows['base']['BA']

    metrics_A = compute_metrics_frame_A(frame_A, frame_B, flows, config, rms_diff)
    metrics_B = compute_metrics_frame_B(frame_A, frame_B, flows, config, rms_diff)

    u_sym_A, v_sym_A = symmetrize_flow_to_frame_A(u_AB, v_AB, u_BA, v_BA)
    u_sym_B, v_sym_B = symmetrize_flow_to_frame_B(u_AB, v_AB, u_BA, v_BA)

    # ========================================================================
    # Compute speed metrics (raw magnitude - NOT normalized here)
    # ========================================================================
    speed_sym_A = np.hypot(u_sym_A, v_sym_A).astype(np.float32)
    speed_sym_B = np.hypot(u_sym_B, v_sym_B).astype(np.float32)

    # ========================================================================
    # Compute asymmetry metrics (forward-backward disagreement)
    # ========================================================================
    asym_A = compute_asymmetry_in_frame_A(u_AB, v_AB, u_BA, v_BA)
    asym_B = compute_asymmetry_in_frame_B(u_AB, v_AB, u_BA, v_BA)

    # ========================================================================
    # Compute flow kinematics (divergence, curl, folding, etc.)
    # ========================================================================
    kin_A = compute_flow_kinematics(u_AB, v_AB)
    kin_B = compute_flow_kinematics(u_BA, v_BA)
    
    # Compute folding fraction (det < 0 is physically impossible)
    folding_A = (kin_A['determinant'] < 0).astype(np.float32)
    folding_B = (kin_B['determinant'] < 0).astype(np.float32)

    # Separate flows and metrics
    flows_dict = {
        'u_AB': u_AB, 'v_AB': v_AB,
        'u_BA': u_BA, 'v_BA': v_BA,
        'u_sym_A': u_sym_A, 'v_sym_A': v_sym_A,
        'u_sym_B': u_sym_B, 'v_sym_B': v_sym_B
    }

    metrics_dict = {
        **metrics_A,
        **metrics_B,
        'speed_sym_A': speed_sym_A,
        'speed_sym_B': speed_sym_B,
        # Asymmetry metrics
        'asym_magnitude_A': asym_A['asym_magnitude'],
        'relative_asym_A': asym_A['relative_asym'],
        'asym_parallel_A': asym_A['asym_parallel'],
        'asym_perpendicular_A': asym_A['asym_perpendicular'],
        'asym_magnitude_B': asym_B['asym_magnitude'],
        'relative_asym_B': asym_B['relative_asym'],
        'asym_parallel_B': asym_B['asym_parallel'],
        'asym_perpendicular_B': asym_B['asym_perpendicular'],
        # Kinematics metrics
        'divergence_A': kin_A['divergence'],
        'curl_A': kin_A['curl'],
        'determinant_A': kin_A['determinant'],
        'shear_magnitude_A': kin_A['shear_magnitude'],
        'max_strain_A': kin_A['max_strain'],
        'folding_A': folding_A,
        'divergence_B': kin_B['divergence'],
        'curl_B': kin_B['curl'],
        'determinant_B': kin_B['determinant'],
        'shear_magnitude_B': kin_B['shear_magnitude'],
        'max_strain_B': kin_B['max_strain'],
        'folding_B': folding_B,
    }

    return {
        'flows': flows_dict,
        'metrics': metrics_dict
    }
