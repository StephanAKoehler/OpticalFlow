#!/usr/bin/env python3
# File: src/ensemble/analysis.py
"""
Ensemble Analysis

Evaluates different ensemble methods on optical flow results:
- Single config selection (by various metrics)
- Per-pixel ensemble selection (photometric, perturbation)
- Smooth + fallback combination

Usage:
    python -m src.ensemble.analysis --config configs/rubberwhale.toml
    python -m src.ensemble.analysis --crawl
    python -m src.ensemble.analysis --crawl --data-dir data/

Requires [ensemble] section in config:
    [ensemble]
    sigma = 2.5

Output:
    Saves ensemble_analysis.json next to results_full.pkl
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import tomli
from scipy.ndimage import gaussian_filter

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src_sprites.generate_sprites import extract_generation_config, compute_generation_hash
from scripts.optical_flow_track import extract_of_config, compute_of_hash


OUTPUT_FILENAME = 'ensemble_analysis.json'


# =============================================================================
# Pollution Depth Lookup
# =============================================================================

def parse_config_name(config_name: str) -> dict:
    """
    Parse config name string back into config dict.
    
    Farneback format: FB_win9_lev4_iter3_poly5_s1.1_pyr0.50
    DIS format: DIS_F_sc0_it12_p8_st4
    """
    config = {}
    
    # Detect algorithm from prefix
    if config_name.startswith('FB_'):
        config['algorithm'] = 'farneback'
        
        # winsize
        match = re.search(r'win(\d+)', config_name)
        if match:
            config['winsize'] = int(match.group(1))
        
        # levels
        match = re.search(r'lev(\d+)', config_name)
        if match:
            config['levels'] = int(match.group(1))
        
        # iterations
        match = re.search(r'iter(\d+)', config_name)
        if match:
            config['iterations'] = int(match.group(1))
        
        # poly_n
        match = re.search(r'poly(\d+)', config_name)
        if match:
            config['poly_n'] = int(match.group(1))
        
        # poly_sigma (s1.1)
        match = re.search(r's([\d.]+)', config_name)
        if match:
            config['poly_sigma'] = float(match.group(1))
        
        # pyr_scale
        match = re.search(r'pyr([\d.]+)', config_name)
        if match:
            config['pyr_scale'] = float(match.group(1))
        
        # flags - default to 0
        config['flags'] = 0
    
    elif config_name.startswith('DIS_'):
        config['algorithm'] = 'dis'
        
        # preset (F/UF/M)
        if '_UF_' in config_name or config_name.startswith('DIS_UF'):
            config['preset'] = 'ULTRAFAST'
        elif '_F_' in config_name or config_name.startswith('DIS_F'):
            config['preset'] = 'FAST'
        elif '_M_' in config_name or config_name.startswith('DIS_M'):
            config['preset'] = 'MEDIUM'
        
        # finest_scale (sc0)
        match = re.search(r'sc(\d+)', config_name)
        if match:
            config['finest_scale'] = int(match.group(1))
        
        # iterations (it12)
        match = re.search(r'it(\d+)', config_name)
        if match:
            config['iterations'] = int(match.group(1))
        
        # patch_size (p8)
        match = re.search(r'p(\d+)', config_name)
        if match:
            config['patch_size'] = int(match.group(1))
        
        # patch_stride (st4)
        match = re.search(r'st(\d+)', config_name)
        if match:
            config['patch_stride'] = int(match.group(1))
    
    else:
        # Unknown algorithm - try to extract what we can
        config['algorithm'] = 'unknown'
        print(f"⚠️  Unknown config name format: {config_name}")
    
    return config


def get_pollution_depth(config_name: str) -> float:
    """
    Get empirically measured pollution depth for a config.
    
    Uses src_contamination module - hard fails if not available.
    """
    from src_contamination import get_margin
    
    config = parse_config_name(config_name)
    return float(get_margin(config, magnitude=1.0))


# =============================================================================
# Config Helpers
# =============================================================================

def get_ensemble_sigma(config: dict) -> float:
    """
    Get sigma from config. Raises ValueError if not specified.
    
    Requires:
        [ensemble]
        sigma = 2.5
    """
    if 'ensemble' not in config:
        raise ValueError("Config missing [ensemble] section")
    
    if 'sigma' not in config['ensemble']:
        raise ValueError("Config [ensemble] missing sigma")
    
    sigma = config['ensemble']['sigma']
    
    # Handle list (for sweeps) vs scalar
    if isinstance(sigma, list):
        return [float(s) for s in sigma]
    return float(sigma)


def sigma_sweep(results_path: Path, sigmas: list, quiet: bool = False) -> dict:
    """
    Run ensemble analysis for multiple sigma values.
    
    Args:
        results_path: Path to results_full.pkl
        sigmas: List of sigma values to test
        quiet: Suppress per-sigma output
    
    Returns:
        Dict with 'sigmas' list and 'results' list of analysis dicts
    """
    results = []
    
    for sigma in sigmas:
        analysis = analyze_results(results_path, sigma=sigma, quiet=True)
        if analysis is None:
            return None
        results.append(analysis)
    
    return {'sigmas': sigmas, 'results': results}


def print_sigma_sweep_table(sweep_results: dict):
    """Print comparison table for sigma sweep."""
    sigmas = sweep_results['sigmas']
    results = sweep_results['results']
    
    # Header
    print("\n" + "=" * 100)
    print("SIGMA SWEEP RESULTS")
    print("=" * 100)
    
    # Get reference values from first result
    best_single = results[0]['best_single']['mean']
    oracle = results[0]['oracle']['mean']
    
    print(f"   Best single: {best_single:.4f}")
    print(f"   Oracle:      {oracle:.4f}")
    print(f"   Headroom:    {best_single - oracle:.4f}")
    print()
    
    # Table header
    print(f"{'σ':>6} | {'Photo':>8} | {'Smooth':>8} | {'Pert':>8} | {'Smooth vs Photo':>15} | {'Photo %':>8} | {'Headroom %':>10}")
    print("-" * 100)
    
    for sigma, analysis in zip(sigmas, results):
        photo = analysis['photo_ensemble']['mean']
        smooth = analysis['smooth_fallback']['mean']
        pert = analysis['pert_ensemble']['mean']
        photo_pct = analysis['smooth_fallback']['photo_pct']
        smooth_vs_photo = analysis['improvements']['smooth_vs_photo_pct']
        headroom_captured = analysis['improvements']['smooth_headroom_captured_pct']
        
        # Color code: positive = good (green thinking), negative = bad
        sign = "+" if smooth_vs_photo > 0 else ""
        
        print(f"{sigma:>6.1f} | {photo:>8.4f} | {smooth:>8.4f} | {pert:>8.4f} | {sign}{smooth_vs_photo:>14.1f}% | {photo_pct:>7.1f}% | {headroom_captured:>9.1f}%")
    
    print("=" * 100)
    
    # Find optimal sigma
    best_idx = min(range(len(results)), key=lambda i: results[i]['smooth_fallback']['mean'])
    best_sigma = sigmas[best_idx]
    best_smooth = results[best_idx]['smooth_fallback']['mean']
    best_improvement = results[best_idx]['improvements']['smooth_vs_photo_pct']
    
    print(f"\n   Optimal σ = {best_sigma:.1f} (smooth EPE = {best_smooth:.4f}, {'+' if best_improvement > 0 else ''}{best_improvement:.1f}% vs photo)")
    print()


# =============================================================================
# Config → Path Resolution
# =============================================================================

def get_source_type(config: dict) -> str:
    """Determine source type from config."""
    has_sprites = 'sprites' in config
    has_source = 'source' in config
    
    if has_sprites and has_source:
        print("❌ Config has both [sprites.*] and [source] sections")
        sys.exit(1)
    
    if not has_sprites and not has_source:
        print("❌ Config missing data source")
        sys.exit(1)
    
    return 'sprites' if has_sprites else 'external'


def extract_source_config(config: dict) -> dict:
    """Extract source config for hashing (matches run_experiment.py)."""
    source = config['source']
    source_type = source['type']
    
    base = {
        'type': source_type,
        'sequence': source.get('sequence'),
        'frames': source.get('frames'),
    }
    
    if source_type == 'sintel':
        base['pass'] = source.get('pass')
        base['root'] = source.get('sintel_root', '')
    elif source_type == 'middlebury':
        base['root'] = source.get('middlebury_root', '')
    else:
        base['root'] = source.get('root', '')
    
    return base


def compute_source_hash(source_config: dict) -> str:
    """Compute hash for external source config."""
    config_str = json.dumps(source_config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def compute_movie_hash(config: dict) -> str:
    """Compute movie hash from config."""
    source_type = get_source_type(config)
    
    if source_type == 'sprites':
        gen_config = extract_generation_config(config)
        return compute_generation_hash(gen_config)
    else:
        source_config = extract_source_config(config)
        return compute_source_hash(source_config)


def config_to_results_path(config: dict, data_dir: Path) -> Path:
    """Derive results_full.pkl path from config."""
    movie_hash = compute_movie_hash(config)
    of_config = extract_of_config(config)
    of_hash = compute_of_hash(of_config)
    
    # For now, just pair_000 - could extend to multiple pairs
    return data_dir / movie_hash / 'analysis' / of_hash / 'sweep' / 'pair_000' / 'results_full.pkl'


# =============================================================================
# Data Loading
# =============================================================================

def load_ground_truth(results_path: Path):
    """Load ground truth from frames directory."""
    # Navigate up from results_full.pkl to movie_dir
    pair_dir = results_path.parent
    sweep_dir = pair_dir.parent
    of_dir = sweep_dir.parent
    analysis_dir = of_dir.parent
    movie_dir = analysis_dir.parent
    frames_dir = movie_dir / 'frames'
    
    u_path = frames_dir / 'u_000.npz'
    v_path = frames_dir / 'v_000.npz'
    
    if not u_path.exists():
        return None, None, None
    
    u_data = np.load(u_path)
    v_data = np.load(v_path)
    u_truth = u_data[list(u_data.keys())[0]]
    v_truth = v_data[list(v_data.keys())[0]]
    
    valid_mask = (
        ~np.isnan(u_truth) & ~np.isnan(v_truth) &
        (np.abs(u_truth) < 1e8) & (np.abs(v_truth) < 1e8)
    )
    
    return u_truth, v_truth, valid_mask


def load_results(results_path: Path):
    """Load results_full.pkl and return structured data."""
    import pickle
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results


# =============================================================================
# Analysis Core
# =============================================================================

def compute_epe_stats(epe: np.ndarray, mask: np.ndarray) -> dict:
    """Compute EPE statistics over valid pixels."""
    vals = epe[mask]
    return {
        'mean': float(vals.mean()),
        'std': float(vals.std()),
        'median': float(np.median(vals)),
        'p90': float(np.percentile(vals, 90)),
        'p95': float(np.percentile(vals, 95)),
        'p99': float(np.percentile(vals, 99)),
    }


def analyze_results(results_path: Path, sigma: float, quiet: bool = False) -> dict:
    """
    Run ensemble analysis on results_full.pkl.
    
    Args:
        results_path: Path to results_full.pkl
        sigma: Gaussian sigma for smooth+fallback (required)
        quiet: Suppress progress output
    
    Returns dict with all key metrics, or None if no ground truth.
    """
    results = load_results(results_path)
    n_configs = len(results)
    
    if not quiet:
        print(f"   {n_configs} configurations")
    
    # Load ground truth
    u_truth, v_truth, valid_mask = load_ground_truth(results_path)
    if u_truth is None:
        print(f"   ⚠️  No ground truth found, skipping")
        return None
    
    H, W = u_truth.shape
    n_valid = int(valid_mask.sum())
    
    if not quiet:
        print(f"   Shape: {H}×{W}, valid: {n_valid:,}")
        print(f"   Building stacks...")
    
    # Build stacks
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    photo_log_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    pert_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    config_names = []
    depths = []
    
    for i, r in enumerate(results):
        config_name = r['metadata'].get('config_name', f'config_{i}')
        config_names.append(config_name)
        
        # Get empirically measured pollution depth
        depth = get_pollution_depth(config_name)
        depths.append(depth)
        
        u = r['flows']['u_AB']
        v = r['flows']['v_AB']
        
        u_stack[i] = u
        v_stack[i] = v
        epe_stack[i] = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        photo_log_stack[i] = r['metrics']['photo_log_raw_A']
        pert_stack[i] = r['metrics']['perturbation_raw_A'] * depth
    
    # Index arrays for gathering
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    
    if not quiet:
        print(f"   Computing ensemble selections...")
    
    # =========================================================================
    # Oracle (per-pixel best possible)
    # =========================================================================
    oracle_idx = np.argmin(epe_stack, axis=0)
    epe_oracle = epe_stack[oracle_idx, y_idx, x_idx]
    
    # =========================================================================
    # Best single config (by mean EPE)
    # =========================================================================
    mean_epes = [epe_stack[i][valid_mask].mean() for i in range(n_configs)]
    best_single_idx = int(np.argmin(mean_epes))
    worst_single_idx = int(np.argmax(mean_epes))
    
    # =========================================================================
    # Photometric ensemble (per-pixel selection by photo_log)
    # =========================================================================
    photo_idx = np.argmin(photo_log_stack, axis=0)
    u_photo = u_stack[photo_idx, y_idx, x_idx]
    v_photo = v_stack[photo_idx, y_idx, x_idx]
    epe_photo = epe_stack[photo_idx, y_idx, x_idx]
    
    # =========================================================================
    # Perturbation ensemble (per-pixel selection by pert×depth)
    # =========================================================================
    pert_idx = np.argmin(pert_stack, axis=0)
    u_pert = u_stack[pert_idx, y_idx, x_idx]
    v_pert = v_stack[pert_idx, y_idx, x_idx]
    epe_pert = epe_stack[pert_idx, y_idx, x_idx]
    
    # =========================================================================
    # Smooth + fallback (Gaussian smooth photo, fallback to pert if closer)
    # =========================================================================
    u_smooth = gaussian_filter(u_photo, sigma=sigma, mode='nearest')
    v_smooth = gaussian_filter(v_photo, sigma=sigma, mode='nearest')
    
    dist_photo = np.sqrt((u_photo - u_smooth)**2 + (v_photo - v_smooth)**2)
    dist_pert = np.sqrt((u_pert - u_smooth)**2 + (v_pert - v_smooth)**2)
    
    use_photo = dist_photo <= dist_pert
    u_combined = np.where(use_photo, u_photo, u_pert)
    v_combined = np.where(use_photo, v_photo, v_pert)
    epe_combined = np.sqrt((u_combined - u_truth)**2 + (v_combined - v_truth)**2)
    
    photo_pct = float(use_photo[valid_mask].mean() * 100)
    
    # =========================================================================
    # Build results dict
    # =========================================================================
    analysis = {
        'metadata': {
            'n_configs': n_configs,
            'n_valid_pixels': n_valid,
            'shape': [H, W],
            'sigma': sigma,
        },
        'oracle': compute_epe_stats(epe_oracle, valid_mask),
        'best_single': {
            'config': config_names[best_single_idx],
            'idx': best_single_idx,
            **compute_epe_stats(epe_stack[best_single_idx], valid_mask)
        },
        'worst_single': {
            'config': config_names[worst_single_idx],
            'idx': worst_single_idx,
            **compute_epe_stats(epe_stack[worst_single_idx], valid_mask)
        },
        'photo_ensemble': compute_epe_stats(epe_photo, valid_mask),
        'pert_ensemble': compute_epe_stats(epe_pert, valid_mask),
        'smooth_fallback': {
            'photo_pct': photo_pct,
            **compute_epe_stats(epe_combined, valid_mask)
        },
    }
    
    # =========================================================================
    # Compute improvement metrics
    # =========================================================================
    oracle_mean = analysis['oracle']['mean']
    best_single_mean = analysis['best_single']['mean']
    photo_mean = analysis['photo_ensemble']['mean']
    pert_mean = analysis['pert_ensemble']['mean']
    smooth_mean = analysis['smooth_fallback']['mean']
    
    # Headroom = gap between current method and oracle
    photo_headroom = photo_mean - oracle_mean
    
    analysis['improvements'] = {
        # Photo vs best single
        'photo_vs_single_pct': (best_single_mean - photo_mean) / best_single_mean * 100,
        # Smooth vs photo
        'smooth_vs_photo_pct': (photo_mean - smooth_mean) / photo_mean * 100 if photo_mean > 0 else 0,
        # Smooth vs best single
        'smooth_vs_single_pct': (best_single_mean - smooth_mean) / best_single_mean * 100,
        # Pert vs photo (expected negative)
        'pert_vs_photo_pct': (photo_mean - pert_mean) / photo_mean * 100 if photo_mean > 0 else 0,
        # Headroom analysis
        'photo_headroom': photo_headroom,
        'smooth_headroom_captured_pct': (photo_mean - smooth_mean) / photo_headroom * 100 if photo_headroom > 0 else 0,
    }
    
    return analysis


# =============================================================================
# High-level API for run_experiment integration
# =============================================================================

def analyze_and_save(results_path: Path, config: dict, quiet: bool = False) -> dict:
    """
    Run ensemble analysis and save to JSON.
    
    Called by run_experiment.py after sweep stage.
    
    Args:
        results_path: Path to results_full.pkl
        config: Full experiment config dict (must have [ensemble] sigma)
        quiet: Suppress output
    
    Returns:
        Analysis dict, or None if no ground truth
        For sigma sweeps, returns sweep results dict
    """
    sigma = get_ensemble_sigma(config)
    output_path = results_path.parent / OUTPUT_FILENAME
    
    # Handle sigma sweep (list of values)
    if isinstance(sigma, list):
        if not quiet:
            print(f"📊 Sigma sweep ({len(sigma)} values: {sigma})...")
        
        sweep_results = sigma_sweep(results_path, sigma, quiet=True)
        
        if sweep_results is None:
            if not quiet:
                print("   ⚠️  Skipped (no ground truth)")
            return None
        
        # Save sweep results
        sweep_output = results_path.parent / 'sigma_sweep.json'
        with open(sweep_output, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        
        if not quiet:
            print(f"   ✓ Saved: {sweep_output.name}")
            print_sigma_sweep_table(sweep_results)
        
        return sweep_results
    
    # Single sigma value
    if not quiet:
        print(f"📊 Ensemble analysis (σ={sigma})...")
    
    analysis = analyze_results(results_path, sigma=sigma, quiet=quiet)
    
    if analysis is None:
        if not quiet:
            print("   ⚠️  Skipped (no ground truth)")
        return None
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    if not quiet:
        print(f"   ✓ Saved: {output_path.name}")
        print_summary(analysis, verbose=True)
    
    return analysis


# =============================================================================
# Smart Regeneration
# =============================================================================

def should_regenerate(results_path: Path, output_path: Path, script_path: Path) -> bool:
    """
    Check if analysis should be regenerated.
    
    Regenerate if:
    - Output doesn't exist
    - results_full.pkl is newer than output
    - This script is newer than output
    """
    if not output_path.exists():
        return True
    
    output_mtime = output_path.stat().st_mtime
    results_mtime = results_path.stat().st_mtime
    script_mtime = script_path.stat().st_mtime
    
    return results_mtime > output_mtime or script_mtime > output_mtime


# =============================================================================
# Crawl Mode
# =============================================================================

def crawl_data_dir(data_dir: Path):
    """Find all results_full.pkl files in data directory."""
    return sorted(data_dir.glob('*/analysis/*/sweep/pair_*/results_full.pkl'))


def get_sequence_name(results_path: Path) -> str:
    """Extract human-readable sequence name from path."""
    # Path: data/{movie_hash}/analysis/{of_hash}/sweep/pair_000/results_full.pkl
    movie_hash = results_path.parent.parent.parent.parent.parent.name
    pair = results_path.parent.name
    return f"{movie_hash}/{pair}"


def get_movie_config(results_path: Path) -> dict:
    """Load config.toml from movie directory."""
    # Navigate: results_full.pkl -> pair_000 -> sweep -> of_hash -> analysis -> movie_hash
    pair_dir = results_path.parent
    sweep_dir = pair_dir.parent
    of_dir = sweep_dir.parent
    analysis_dir = of_dir.parent
    movie_dir = analysis_dir.parent
    
    config_path = movie_dir / 'config.toml'
    if not config_path.exists():
        return None
    
    with open(config_path, 'rb') as f:
        return tomli.load(f)


# =============================================================================
# Output Formatting
# =============================================================================

def print_summary(analysis: dict, name: str = "", verbose: bool = True):
    """Print summary of analysis results."""
    
    if not verbose:
        # Compact one-liner
        oracle = analysis['oracle']['mean']
        best = analysis['best_single']['mean']
        photo = analysis['photo_ensemble']['mean']
        smooth = analysis['smooth_fallback']['mean']
        imp = analysis['improvements']
        print(f"   Oracle={oracle:.4f} | Best={best:.4f} | Photo={photo:.4f} | Smooth={smooth:.4f}")
        return
    
    # Full informative output
    meta = analysis['metadata']
    oracle = analysis['oracle']
    best = analysis['best_single']
    worst = analysis['worst_single']
    photo = analysis['photo_ensemble']
    pert = analysis['pert_ensemble']
    smooth = analysis['smooth_fallback']
    imp = analysis['improvements']
    
    print()
    print("=" * 80)
    print("ENSEMBLE ANALYSIS RESULTS")
    print("=" * 80)
    print(f"   Configs: {meta['n_configs']}")
    print(f"   Shape: {meta['shape'][0]}×{meta['shape'][1]}, valid pixels: {meta['n_valid_pixels']:,}")
    print(f"   Smoothing σ: {meta['sigma']}")
    
    print()
    print("-" * 80)
    print("EPE COMPARISON")
    print("-" * 80)
    print(f"{'Method':<30} | {'mean':>8} | {'std':>8} | {'p50':>8} | {'p90':>8} | {'p95':>8}")
    print("-" * 80)
    
    def print_row(name, stats, config=None):
        suffix = f"  [{config}]" if config else ""
        print(f"{name:<30} | {stats['mean']:>8.4f} | {stats['std']:>8.4f} | "
              f"{stats['median']:>8.4f} | {stats['p90']:>8.4f} | {stats['p95']:>8.4f}{suffix}")
    
    print_row("Oracle (per-pixel best)", oracle)
    print_row("Best single config", best, best['config'])
    print_row("Worst single config", worst, worst['config'])
    print("-" * 80)
    print_row("Photo ensemble", photo)
    print_row("Pert ensemble", pert)
    print_row(f"Smooth+fallback (σ={meta['sigma']})", smooth)
    print("-" * 80)
    
    print()
    print("-" * 80)
    print("IMPROVEMENTS")
    print("-" * 80)
    print(f"   Photo vs best single:     {imp['photo_vs_single_pct']:>+6.1f}%")
    print(f"   Pert vs photo:            {imp['pert_vs_photo_pct']:>+6.1f}%  (expected negative)")
    print(f"   Smooth vs photo:          {imp['smooth_vs_photo_pct']:>+6.1f}%")
    print(f"   Smooth vs best single:    {imp['smooth_vs_single_pct']:>+6.1f}%")
    print()
    print(f"   Smooth+fallback stats:")
    print(f"      Photo pixels kept:     {smooth['photo_pct']:>6.1f}%")
    print(f"      Pert pixels used:      {100 - smooth['photo_pct']:>6.1f}%")
    print()
    print(f"   Headroom analysis:")
    print(f"      Photo gap to oracle:   {imp['photo_headroom']:.4f}")
    print(f"      Smooth captures:       {imp['smooth_headroom_captured_pct']:>6.1f}% of remaining headroom")
    print("=" * 80)


def print_table(analyses: list):
    """Print comparison table for multiple analyses."""
    if not analyses:
        return
    
    print("\n" + "=" * 100)
    print("ENSEMBLE ANALYSIS SUMMARY")
    print("=" * 100)
    print(f"{'Sequence':<25} | {'Oracle':>8} | {'Best':>8} | {'Photo':>8} | {'Smooth':>8} | "
          f"{'P/S %':>7} | {'Sm/P %':>7} | {'Hroom%':>7}")
    print("-" * 100)
    
    for name, analysis in analyses:
        oracle = analysis['oracle']['mean']
        best = analysis['best_single']['mean']
        photo = analysis['photo_ensemble']['mean']
        smooth = analysis['smooth_fallback']['mean']
        imp = analysis['improvements']
        
        print(f"{name:<25} | {oracle:>8.4f} | {best:>8.4f} | {photo:>8.4f} | {smooth:>8.4f} | "
              f"{imp['photo_vs_single_pct']:>+6.1f}% | {imp['smooth_vs_photo_pct']:>+6.1f}% | "
              f"{imp['smooth_headroom_captured_pct']:>6.1f}%")
    
    print("=" * 100)
    
    # Averages
    if len(analyses) > 1:
        avg_photo_vs_single = np.mean([a['improvements']['photo_vs_single_pct'] for _, a in analyses])
        avg_smooth_vs_photo = np.mean([a['improvements']['smooth_vs_photo_pct'] for _, a in analyses])
        avg_headroom = np.mean([a['improvements']['smooth_headroom_captured_pct'] for _, a in analyses])
        print(f"{'AVERAGE':<25} | {'':>8} | {'':>8} | {'':>8} | {'':>8} | "
              f"{avg_photo_vs_single:>+6.1f}% | {avg_smooth_vs_photo:>+6.1f}% | {avg_headroom:>6.1f}%")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ensemble analysis for optical flow results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.ensemble.analysis --config configs/rubberwhale.toml
    python -m src.ensemble.analysis --crawl
    python -m src.ensemble.analysis --crawl --force

Requires [ensemble] section in config:
    [ensemble]
    sigma = 2.5
        """
    )
    parser.add_argument('--config', type=Path, help='TOML config file')
    parser.add_argument('--crawl', action='store_true', help='Crawl data directory for all results')
    parser.add_argument('--data-dir', type=Path, default=Path('data'), help='Data directory (default: data/)')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if up to date')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output (full details for each sequence in crawl mode)')
    
    args = parser.parse_args()
    
    if not args.config and not args.crawl:
        print("❌ Must specify --config or --crawl")
        parser.print_help()
        sys.exit(1)
    
    script_path = Path(__file__).resolve()
    
    # =========================================================================
    # Config mode: analyze single experiment
    # =========================================================================
    if args.config:
        if not args.config.exists():
            print(f"❌ Config not found: {args.config}")
            sys.exit(1)
        
        with open(args.config, 'rb') as f:
            config = tomli.load(f)
        
        try:
            sigma = get_ensemble_sigma(config)
        except ValueError as e:
            print(f"❌ {e}")
            print("   Add to config:")
            print("   [ensemble]")
            print("   sigma = 2.5")
            sys.exit(1)
        
        results_path = config_to_results_path(config, args.data_dir)
        
        if not results_path.exists():
            print(f"❌ Results not found: {results_path}")
            print(f"   Run sweep first: python scripts/run_experiment.py {args.config}")
            sys.exit(1)
        
        output_path = results_path.parent / OUTPUT_FILENAME
        
        if not args.force and not should_regenerate(results_path, output_path, script_path):
            print(f"✓ Up to date: {output_path}")
            # Load and print existing
            with open(output_path) as f:
                analysis = json.load(f)
            print_summary(analysis, verbose=True)
        else:
            print(f"📊 Analyzing: {results_path}")
            print(f"   σ = {sigma}")
            analysis = analyze_results(results_path, sigma=sigma)
            if analysis:
                with open(output_path, 'w') as f:
                    json.dump(analysis, f, indent=2)
                print(f"✓ Saved: {output_path}")
                print_summary(analysis, verbose=True)
    
    # =========================================================================
    # Crawl mode: find and analyze all results
    # =========================================================================
    elif args.crawl:
        results_files = crawl_data_dir(args.data_dir)
        
        if not results_files:
            print(f"❌ No results found in {args.data_dir}")
            sys.exit(1)
        
        print(f"📂 Found {len(results_files)} results files\n")
        
        analyses = []
        
        for results_path in results_files:
            output_path = results_path.parent / OUTPUT_FILENAME
            name = get_sequence_name(results_path)
            
            # Load config to get sigma
            config = get_movie_config(results_path)
            if config is None:
                print(f"⚠️  {name}: no config.toml found, skipping")
                continue
            
            try:
                sigma = get_ensemble_sigma(config)
            except ValueError as e:
                print(f"⚠️  {name}: {e}, skipping")
                continue
            
            if not args.force and not should_regenerate(results_path, output_path, script_path):
                # Load existing
                with open(output_path) as f:
                    analysis = json.load(f)
                print(f"✓ {name}: up to date")
                if args.verbose:
                    print_summary(analysis, verbose=True)
            else:
                print(f"📊 {name} (σ={sigma})...")
                analysis = analyze_results(results_path, sigma=sigma, quiet=not args.verbose)
                if analysis:
                    with open(output_path, 'w') as f:
                        json.dump(analysis, f, indent=2)
                    print(f"   ✓ Saved")
                    if args.verbose:
                        print_summary(analysis, verbose=True)
            
            if analysis:
                analyses.append((name, analysis))
        
        # Print summary table
        print_table(analyses)


if __name__ == "__main__":
    main()
