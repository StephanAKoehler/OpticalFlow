#!/usr/bin/env python3
# File: scripts/run_experiment.py
"""
Experiment Orchestrator

Runs the full optical flow pipeline:
1. load/generate  - Load external frames OR generate sprites + ground truth
2. sweep          - Run OF parameter sweep
3. optimize       - Find optimal weights across sequence (requires ground_truth=true)

Supports series mode with SEQUENCES list for batch processing.

Usage:
    python scripts/run_experiment.py config.toml                    # default: flow (load+sweep)
    python scripts/run_experiment.py config.toml --stage full       # load+sweep+optimize
    python scripts/run_experiment.py config.toml --stage load
    python scripts/run_experiment.py config.toml --stage sweep
    python scripts/run_experiment.py config.toml --stage optimize
    
Series mode (config with SEQUENCES list):
    python scripts/run_experiment.py configs/middlebury_farneback.toml
"""

import copy
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import tomli
import tomli_w

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the main functions from each module
from src_sprites.generate_sprites import generate_movie, extract_generation_config, compute_generation_hash
from scripts.optical_flow_track import run_sweep, extract_of_config, compute_of_hash
from scripts.optimize_weights import optimize_sequence
from src.analysis.ranking_comparison import save_ranking_comparison, print_ranking_comparison


# =============================================================================
# Stage Definitions
# =============================================================================

STAGES = ['load', 'sweep', 'optimize', 'flow', 'full']


# =============================================================================
# Prerequisite Checking
# =============================================================================

def check_movie_exists(data_dir: Path, movie_hash: str) -> bool:
    """Check if movie data exists and is complete."""
    movie_dir = data_dir / movie_hash
    if not movie_dir.exists():
        return False
    if not (movie_dir / 'frames').exists():
        return False
    if not (movie_dir / 'config.toml').exists():
        return False
    return True


def check_sweep_exists(data_dir: Path, movie_hash: str, of_hash: str) -> bool:
    """Check if sweep results exist."""
    sweep_dir = data_dir / movie_hash / 'analysis' / of_hash / 'sweep'
    if not sweep_dir.exists():
        return False
    # Check for at least one pair directory with results
    pair_dirs = list(sweep_dir.glob('pair_*'))
    if not pair_dirs:
        return False
    return (pair_dirs[0] / 'results_full.pkl').exists()


# =============================================================================
# Source Type Detection
# =============================================================================

def get_source_type(config: dict) -> str:
    """
    Determine source type from config.
    Returns 'sprites' or 'external'.
    Errors if both or neither present.
    """
    has_sprites = 'sprites' in config
    has_source = 'source' in config
    
    if has_sprites and has_source:
        print("❌ ERROR: Config has both [sprites.*] and [source] sections")
        print("   Use one or the other, not both")
        sys.exit(1)
    
    if not has_sprites and not has_source:
        print("❌ ERROR: Config missing data source")
        print("   Add [sprites.*] sections for generation, or [source] for external data")
        sys.exit(1)
    
    return 'sprites' if has_sprites else 'external'


def uses_ground_truth(config: dict) -> bool:
    """Check if config requests ground truth usage."""
    return config.get('evaluation', {}).get('ground_truth', True)


# =============================================================================
# External Source Loading (Sintel, etc.)
# =============================================================================

def extract_source_config(config: dict) -> dict:
    """Extract source config for hashing."""
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


def load_sintel_sequence(
    config: dict,
    data_dir: Path,
    force: bool = False
) -> str:
    """
    Load Sintel sequence into data directory structure.
    
    Returns:
        movie_hash for the loaded sequence
    """
    source = config['source']
    source_config = extract_source_config(config)
    movie_hash = compute_source_hash(source_config)
    
    # Expand paths
    sintel_root = Path(source['sintel_root']).expanduser()
    sequence = source['sequence']
    pass_type = source.get('pass', 'clean')
    frames = source.get('frames')  # None means all frames
    use_gt = uses_ground_truth(config)
    
    # Validate sintel root exists
    if not sintel_root.exists():
        print(f"❌ ERROR: Sintel root not found: {sintel_root}")
        sys.exit(1)
    
    # Paths within Sintel structure
    frames_dir = sintel_root / 'training' / pass_type / sequence
    flow_dir = sintel_root / 'training' / 'flow' / sequence
    
    if not frames_dir.exists():
        print(f"❌ ERROR: Sintel sequence not found: {frames_dir}")
        sys.exit(1)
    
    if use_gt and not flow_dir.exists():
        print(f"❌ ERROR: Sintel flow directory not found: {flow_dir}")
        print("   Set [evaluation] ground_truth = false to skip GT loading")
        sys.exit(1)
    
    # Determine frame indices
    # Config uses 0-indexed frames, Sintel uses 1-indexed filenames
    available_frames = sorted(frames_dir.glob('frame_*.png'))
    if not available_frames:
        print(f"❌ ERROR: No frames found in {frames_dir}")
        sys.exit(1)
    
    # Extract frame numbers from filenames (frame_0001.png -> 0 in our 0-indexed system)
    available_indices_0 = [int(f.stem.split('_')[1]) - 1 for f in available_frames]
    num_available = len(available_indices_0)
    
    if frames is None:
        frame_indices_0 = available_indices_0
    else:
        # Validate requested frames exist (0-indexed)
        for f in frames:
            if f < 0 or f >= num_available:
                print(f"❌ ERROR: Frame {f} not found in sequence")
                print(f"   Available: 0-{num_available - 1} ({num_available} frames)")
                sys.exit(1)
        frame_indices_0 = frames
    
    # Output directory
    movie_dir = data_dir / movie_hash
    
    # Check if already loaded AND complete
    is_complete = (
        movie_dir.exists() and 
        (movie_dir / 'frames').exists() and 
        (movie_dir / 'config.toml').exists()
    )
    
    if is_complete and not force:
        print(f"✓ Sintel sequence already loaded: {movie_hash}")
        print(f"   Sequence: {sequence} ({pass_type})")
        print(f"   Frames: {frame_indices_0} (0-indexed)")
        return movie_hash
    
    if movie_dir.exists() and not is_complete:
        print(f"⚠️  Incomplete data found, reloading: {movie_hash}")
        shutil.rmtree(movie_dir)
    
    print(f"📂 Loading Sintel sequence: {sequence} ({pass_type})")
    print(f"   Source: {sintel_root}")
    print(f"   Frames: {frame_indices_0} (0-indexed)")
    print(f"   Ground truth: {use_gt}")
    
    # Create directory structure
    movie_dir.mkdir(parents=True, exist_ok=True)
    frames_out_dir = movie_dir / 'frames'
    frames_out_dir.mkdir(exist_ok=True)
    
    # Copy frames (translate 0-indexed config to 1-indexed Sintel filenames)
    # Use image_XXX.png naming to match sprite generator
    for i, frame_idx_0 in enumerate(frame_indices_0):
        frame_idx_1 = frame_idx_0 + 1  # Sintel uses 1-indexed filenames
        src_frame = frames_dir / f'frame_{frame_idx_1:04d}.png'
        dst_frame = frames_out_dir / f'image_{i:03d}.png'
        shutil.copy2(src_frame, dst_frame)
        print(f"   Copied frame_{frame_idx_1:04d}.png -> frames/image_{i:03d}.png")
    
    # Copy ground truth flow files (if using GT)
    # Convert .flo format to u_XXX.npz and v_XXX.npz to match sprite generator
    if use_gt:
        import numpy as np
        
        def read_flo(path):
            """Read Sintel .flo file format."""
            with open(path, 'rb') as f:
                magic = np.fromfile(f, np.float32, count=1)[0]
                if magic != 202021.25:
                    print(f"❌ ERROR: Invalid .flo file: {path}")
                    sys.exit(1)
                w = np.fromfile(f, np.int32, count=1)[0]
                h = np.fromfile(f, np.int32, count=1)[0]
                flow = np.fromfile(f, np.float32, count=2*w*h)
                flow = flow.reshape((h, w, 2))
                return flow[:, :, 0], flow[:, :, 1]  # u, v
        
        # Flow files: frame_0001.flo is flow from frame 1 to frame 2
        # So for N frames, we have N-1 flow files
        for i in range(len(frame_indices_0) - 1):
            frame_idx_0 = frame_indices_0[i]
            frame_idx_1 = frame_idx_0 + 1  # Sintel uses 1-indexed filenames
            src_flow = flow_dir / f'frame_{frame_idx_1:04d}.flo'
            
            if not src_flow.exists():
                print(f"❌ ERROR: Flow file not found: {src_flow}")
                sys.exit(1)
            
            u, v = read_flo(src_flow)
            np.savez_compressed(frames_out_dir / f'u_{i:03d}.npz', u=u)
            np.savez_compressed(frames_out_dir / f'v_{i:03d}.npz', v=v)
            print(f"   Converted frame_{frame_idx_1:04d}.flo -> frames/u_{i:03d}.npz, v_{i:03d}.npz")
    
    # Save source info for reference
    info = {
        'source_type': 'sintel',
        'sintel_root': str(sintel_root),
        'sequence': sequence,
        'pass': pass_type,
        'original_frames_0indexed': frame_indices_0,
        'ground_truth': use_gt,
    }
    with open(movie_dir / 'source_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    # Create config.toml for compatibility with load_movie_sequence
    # Read first frame to get dimensions
    import cv2
    first_frame = cv2.imread(str(frames_out_dir / 'image_000.png'))
    if first_frame is None:
        print(f"❌ ERROR: Failed to read first frame")
        sys.exit(1)
    H, W = first_frame.shape[:2]
    
    movie_config = {
        'image': {
            'size': [W, H],  # [width, height] to match sprite config convention
        },
        'temporal': {
            'num_frames': len(frame_indices_0),
        },
        'source': {
            'type': 'sintel',
            'sequence': sequence,
            'pass': pass_type,
        },
    }
    with open(movie_dir / 'config.toml', 'wb') as f:
        tomli_w.dump(movie_config, f)
    
    print(f"✓ Loaded {len(frame_indices_0)} frames -> {movie_hash}")
    
    return movie_hash


def load_middlebury_sequence(
    config: dict,
    data_dir: Path,
    force: bool = False
) -> str:
    """
    Load Middlebury sequence into data directory structure.
    
    Middlebury structure:
        other-color-twoframes/
            RubberWhale/
                frame10.png
                frame11.png
        other-gt-flow/
            RubberWhale/
                flow10.flo
    
    Returns:
        movie_hash for the loaded sequence
    """
    source = config['source']
    source_config = extract_source_config(config)
    movie_hash = compute_source_hash(source_config)
    
    # Expand paths
    middlebury_root = Path(source['middlebury_root']).expanduser()
    sequence = source['sequence']
    use_gt = uses_ground_truth(config)
    
    # Validate middlebury root exists
    if not middlebury_root.exists():
        print(f"❌ ERROR: Middlebury root not found: {middlebury_root}")
        sys.exit(1)
    
    # Paths within Middlebury structure
    # Try both possible structures
    frames_dir = middlebury_root / 'other-color-twoframes' / sequence
    if not frames_dir.exists():
        frames_dir = middlebury_root / sequence
    
    flow_dir = middlebury_root / 'other-gt-flow' / sequence
    if not flow_dir.exists():
        flow_dir = middlebury_root / sequence
    
    if not frames_dir.exists():
        print(f"❌ ERROR: Middlebury sequence not found: {frames_dir}")
        print(f"   Expected structure: {middlebury_root}/other-color-twoframes/{sequence}/")
        print(f"   Or: {middlebury_root}/{sequence}/")
        sys.exit(1)
    
    # Check for frame files
    frame10 = frames_dir / 'frame10.png'
    frame11 = frames_dir / 'frame11.png'
    
    if not frame10.exists() or not frame11.exists():
        print(f"❌ ERROR: Frame files not found in {frames_dir}")
        print(f"   Expected: frame10.png, frame11.png")
        sys.exit(1)
    
    if use_gt:
        flow_file = flow_dir / 'flow10.flo'
        if not flow_file.exists():
            print(f"❌ ERROR: Flow file not found: {flow_file}")
            print("   Set [evaluation] ground_truth = false to skip GT loading")
            sys.exit(1)
    
    # Output directory
    movie_dir = data_dir / movie_hash
    
    # Check if already loaded AND complete
    is_complete = (
        movie_dir.exists() and 
        (movie_dir / 'frames').exists() and 
        (movie_dir / 'config.toml').exists()
    )
    
    if is_complete and not force:
        print(f"✓ Middlebury sequence already loaded: {movie_hash}")
        print(f"   Sequence: {sequence}")
        return movie_hash
    
    if movie_dir.exists() and not is_complete:
        print(f"⚠️  Incomplete data found, reloading: {movie_hash}")
        shutil.rmtree(movie_dir)
    
    print(f"📂 Loading Middlebury sequence: {sequence}")
    print(f"   Source: {middlebury_root}")
    print(f"   Ground truth: {use_gt}")
    
    # Create directory structure
    movie_dir.mkdir(parents=True, exist_ok=True)
    frames_out_dir = movie_dir / 'frames'
    frames_out_dir.mkdir(exist_ok=True)
    
    # Copy frames - use image_XXX.png naming to match sprite generator
    shutil.copy2(frame10, frames_out_dir / 'image_000.png')
    shutil.copy2(frame11, frames_out_dir / 'image_001.png')
    print(f"   Copied frame10.png -> frames/image_000.png")
    print(f"   Copied frame11.png -> frames/image_001.png")
    
    # Copy ground truth flow file
    if use_gt:
        import numpy as np
        
        # Middlebury uses this value to mark unknown flow
        UNKNOWN_FLOW_THRESH = 1e9
        
        def read_flo(path):
            """Read Middlebury .flo file format (same as Sintel)."""
            with open(path, 'rb') as f:
                magic = np.fromfile(f, np.float32, count=1)[0]
                if magic != 202021.25:
                    print(f"❌ ERROR: Invalid .flo file: {path}")
                    sys.exit(1)
                w = np.fromfile(f, np.int32, count=1)[0]
                h = np.fromfile(f, np.int32, count=1)[0]
                flow = np.fromfile(f, np.float32, count=2*w*h)
                flow = flow.reshape((h, w, 2))
                u, v = flow[:, :, 0], flow[:, :, 1]
                
                # Convert unknown flow markers to NaN
                unknown_mask = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH)
                u = u.astype(np.float64)
                v = v.astype(np.float64)
                u[unknown_mask] = np.nan
                v[unknown_mask] = np.nan
                
                n_unknown = unknown_mask.sum()
                if n_unknown > 0:
                    pct = 100 * n_unknown / unknown_mask.size
                    print(f"   Note: {n_unknown} pixels ({pct:.1f}%) have unknown GT flow")
                
                return u, v
        
        flow_file = flow_dir / 'flow10.flo'
        u, v = read_flo(flow_file)
        np.savez_compressed(frames_out_dir / 'u_000.npz', u=u)
        np.savez_compressed(frames_out_dir / 'v_000.npz', v=v)
        print(f"   Converted flow10.flo -> frames/u_000.npz, v_000.npz")
    
    # Save source info for reference
    info = {
        'source_type': 'middlebury',
        'middlebury_root': str(middlebury_root),
        'sequence': sequence,
        'ground_truth': use_gt,
    }
    with open(movie_dir / 'source_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    # Create config.toml for compatibility with load_movie_sequence
    import cv2
    first_frame = cv2.imread(str(frames_out_dir / 'image_000.png'))
    if first_frame is None:
        print(f"❌ ERROR: Failed to read first frame")
        sys.exit(1)
    H, W = first_frame.shape[:2]
    
    movie_config = {
        'image': {
            'size': [W, H],
        },
        'temporal': {
            'num_frames': 2,
        },
        'source': {
            'type': 'middlebury',
            'sequence': sequence,
        },
    }
    with open(movie_dir / 'config.toml', 'wb') as f:
        tomli_w.dump(movie_config, f)
    
    print(f"✓ Loaded 2 frames -> {movie_hash}")
    
    return movie_hash


def load_external_source(
    config: dict,
    data_dir: Path,
    force: bool = False
) -> str:
    """
    Load external source based on type.
    
    Returns:
        movie_hash for the loaded data
    """
    source = config['source']
    source_type = source['type']
    
    if source_type == 'sintel':
        return load_sintel_sequence(config, data_dir, force)
    elif source_type == 'middlebury':
        return load_middlebury_sequence(config, data_dir, force)
    else:
        print(f"❌ ERROR: Unknown source type: {source_type}")
        print("   Supported types: sintel, middlebury")
        sys.exit(1)


# =============================================================================
# Series Mode Support
# =============================================================================

def compute_series_hash(config: dict) -> str:
    """
    Compute hash for series config, excluding SEQUENCES list.
    
    This ensures adding/removing sequences doesn't change the hash,
    but changing parameters does.
    """
    config_copy = copy.deepcopy(config)
    
    # Remove SEQUENCES from source (we hash the template, not the list)
    if 'source' in config_copy and 'SEQUENCES' in config_copy['source']:
        del config_copy['source']['SEQUENCES']
    
    # Also remove 'sequence' if present (will be set per-sequence)
    if 'source' in config_copy and 'sequence' in config_copy['source']:
        del config_copy['source']['sequence']
    
    config_str = json.dumps(config_copy, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def extract_sequence_summary(sigma_sweep_path: Path) -> dict:
    """
    Extract key metrics from a sigma_sweep.json file.
    
    Returns dict with:
        - best_single, oracle, headroom_pct
        - photo_vs_single_pct, pert_vs_single_pct
        - optimal_sigma, smooth_vs_photo_pct at optimal
    """
    with open(sigma_sweep_path) as f:
        sweep = json.load(f)
    
    sigmas = sweep['sigmas']
    results = sweep['results']
    
    # Get reference values
    best_single = results[0]['best_single']['mean']
    oracle = results[0]['oracle']['mean']
    headroom = best_single - oracle
    headroom_pct = (headroom / best_single) * 100 if best_single > 0 else 0
    
    photo_mean = results[0]['photo_ensemble']['mean']
    photo_vs_single_pct = (best_single - photo_mean) / best_single * 100 if best_single > 0 else 0
    
    pert_mean = results[0]['pert_ensemble']['mean']
    pert_vs_single_pct = results[0]['improvements'].get('pert_vs_single_pct', 
                          (best_single - pert_mean) / best_single * 100 if best_single > 0 else 0)
    
    # Find optimal sigma (minimum smooth_fallback EPE)
    best_idx = min(range(len(results)), key=lambda i: results[i]['smooth_fallback']['mean'])
    optimal_sigma = sigmas[best_idx]
    smooth_mean = results[best_idx]['smooth_fallback']['mean']
    smooth_vs_photo_pct = results[best_idx]['improvements']['smooth_vs_photo_pct']
    
    return {
        'best_single': best_single,
        'oracle': oracle,
        'headroom': headroom,
        'headroom_pct': headroom_pct,
        'photo_mean': photo_mean,
        'photo_vs_single_pct': photo_vs_single_pct,
        'pert_mean': pert_mean,
        'pert_vs_single_pct': pert_vs_single_pct,
        'optimal_sigma': optimal_sigma,
        'smooth_mean': smooth_mean,
        'smooth_vs_photo_pct': smooth_vs_photo_pct,
    }


def collect_series_summary(
    sequences: list,
    data_dir: Path,
    config: dict,
) -> dict:
    """
    Collect summary from all sequences' sigma_sweep.json files.
    
    Args:
        sequences: List of sequence names
        data_dir: Base data directory
        config: Config dict (used to compute hashes)
    
    Returns:
        Dict mapping sequence name -> summary metrics
    """
    summaries = {}
    
    # Get OF hash (same for all sequences since params are identical)
    of_config = extract_of_config(config)
    of_hash = compute_of_hash(of_config)
    
    for seq in sequences:
        # Compute movie hash for this sequence
        seq_config = copy.deepcopy(config)
        seq_config['source']['sequence'] = seq
        if 'SEQUENCES' in seq_config['source']:
            del seq_config['source']['SEQUENCES']
        
        source_config = extract_source_config(seq_config)
        movie_hash = compute_source_hash(source_config)
        
        # Find sigma_sweep.json
        sweep_dir = data_dir / movie_hash / 'analysis' / of_hash / 'sweep'
        pair_dirs = sorted(sweep_dir.glob('pair_*'))
        
        if not pair_dirs:
            print(f"⚠️  No pair directories found for {seq}, skipping")
            continue
        
        # Use first pair (most sequences have only one)
        sigma_sweep_path = pair_dirs[0] / 'sigma_sweep.json'
        
        if not sigma_sweep_path.exists():
            print(f"⚠️  sigma_sweep.json not found for {seq}, skipping")
            continue
        
        summary = extract_sequence_summary(sigma_sweep_path)
        summary['movie_hash'] = movie_hash
        summaries[seq] = summary
    
    return summaries


def save_series_summary(
    summaries: dict,
    config: dict,
    config_path: Path,
    series_dir: Path,
):
    """
    Save series summary as JSON and CSV.
    
    Creates:
        - series_dir/summary.json
        - series_dir/summary.csv
        - series_dir/config.toml (copy of original)
    """
    series_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config
    shutil.copy(config_path, series_dir / 'config.toml')
    
    # Build full summary dict
    algorithm = config.get('parameter_sweep', {}).get('algorithm', 'unknown')
    
    full_summary = {
        'config_file': str(config_path),
        'algorithm': algorithm,
        'sequences': summaries,
    }
    
    # Save JSON
    json_path = series_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(full_summary, f, indent=2)
    
    # Save CSV
    csv_path = series_dir / 'summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Sequence',
            'Best Single',
            'Oracle',
            'Headroom',
            'Headroom %',
            'Photo Mean',
            'Photo vs Single %',
            'Pert Mean',
            'Pert vs Single %',
            'Optimal σ',
            'Smooth Mean', 
            'Smooth vs Photo %',
            'Movie Hash',
        ])
        
        # Data rows
        for seq, s in summaries.items():
            writer.writerow([
                seq,
                f"{s['best_single']:.4f}",
                f"{s['oracle']:.4f}",
                f"{s['headroom']:.4f}",
                f"{s['headroom_pct']:.1f}",
                f"{s['photo_mean']:.4f}",
                f"{s['photo_vs_single_pct']:.1f}",
                f"{s['pert_mean']:.4f}",
                f"{s['pert_vs_single_pct']:.1f}",
                f"{s['optimal_sigma']:.1f}",
                f"{s['smooth_mean']:.4f}",
                f"{s['smooth_vs_photo_pct']:.1f}",
                s['movie_hash'],
            ])
    
    return json_path, csv_path


def print_series_summary(summaries: dict, config: dict):
    """Print formatted summary table to stdout."""
    algorithm = config.get('parameter_sweep', {}).get('algorithm', 'unknown')
    n_configs = 1
    
    # Count configs from parameter_sweep
    param_sweep = config.get('parameter_sweep', {})
    for key, value in param_sweep.items():
        if isinstance(value, list):
            n_configs *= len(value)
    
    print()
    print("=" * 140)
    print(f"SERIES SUMMARY ({algorithm.upper()}, {n_configs} configs)")
    print("=" * 140)
    
    # Header
    print(f"{'Sequence':<14} | {'Best Single':>11} | {'Oracle':>8} | {'Headroom':>8} | {'Photo vs Single':>15} | {'Pert vs Single':>14} | {'σ_opt':>5} | {'Smooth vs Photo':>15}")
    print("-" * 140)
    
    # Data rows
    for seq, s in summaries.items():
        photo_sign = "+" if s['photo_vs_single_pct'] > 0 else ""
        pert_sign = "+" if s['pert_vs_single_pct'] > 0 else ""
        smooth_sign = "+" if s['smooth_vs_photo_pct'] > 0 else ""
        
        print(f"{seq:<14} | {s['best_single']:>11.4f} | {s['oracle']:>8.4f} | {s['headroom_pct']:>7.1f}% | {photo_sign}{s['photo_vs_single_pct']:>14.1f}% | {pert_sign}{s['pert_vs_single_pct']:>13.1f}% | {s['optimal_sigma']:>5.1f} | {smooth_sign}{s['smooth_vs_photo_pct']:>14.1f}%")
    
    print("-" * 140)
    
    # Averages
    n = len(summaries)
    if n > 0:
        avg_headroom = sum(s['headroom_pct'] for s in summaries.values()) / n
        avg_photo = sum(s['photo_vs_single_pct'] for s in summaries.values()) / n
        avg_pert = sum(s['pert_vs_single_pct'] for s in summaries.values()) / n
        avg_smooth = sum(s['smooth_vs_photo_pct'] for s in summaries.values()) / n
        
        photo_sign = "+" if avg_photo > 0 else ""
        pert_sign = "+" if avg_pert > 0 else ""
        smooth_sign = "+" if avg_smooth > 0 else ""
        
        print(f"{'AVERAGE':<14} | {'-':>11} | {'-':>8} | {avg_headroom:>7.1f}% | {photo_sign}{avg_photo:>14.1f}% | {pert_sign}{avg_pert:>13.1f}% | {'-':>5} | {smooth_sign}{avg_smooth:>14.1f}%")
    print("=" * 140)
    print()


def run_series(
    config_path: Path,
    stage: str,
    data_dir: Path,
    force: bool = False,
    no_cache: bool = False,
    n_workers: int = None,
    n_trials: int = None,
) -> dict:
    """
    Run experiment for multiple sequences defined in SEQUENCES.
    
    Args:
        config_path: Path to config file with SEQUENCES list
        stage: Stage to run
        data_dir: Base data directory
        force: Force regeneration
        no_cache: Skip cache
        n_workers: Number of workers
        n_trials: Optuna trials
    
    Returns:
        Series summary dict
    """
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
    
    sequences = config['source']['SEQUENCES']
    series_hash = compute_series_hash(config)
    series_name = f"{config_path.stem}_{series_hash}"
    series_dir = data_dir / 'series' / series_name
    
    print()
    print("=" * 80)
    print("🎬 SERIES MODE")
    print("=" * 80)
    print(f"   Config: {config_path}")
    print(f"   Sequences: {len(sequences)} - {sequences}")
    print(f"   Series dir: {series_dir}")
    print()
    
    # Check if summary already exists and is up-to-date
    summary_path = series_dir / 'summary.json'
    if summary_path.exists() and not force:
        config_mtime = config_path.stat().st_mtime
        summary_mtime = summary_path.stat().st_mtime
        
        if summary_mtime > config_mtime:
            print(f"✓ Series summary up-to-date: {summary_path}")
            print("   Use --force to regenerate")
            print()
            
            # Load and print existing summary
            with open(summary_path) as f:
                existing = json.load(f)
            print_series_summary(existing['sequences'], config)
            return existing
    
    # Run each sequence
    print(f"Running {len(sequences)} sequences...")
    print()
    
    for i, seq in enumerate(sequences, 1):
        print()
        print("=" * 80)
        print(f"📍 SEQUENCE {i}/{len(sequences)}: {seq}")
        print("=" * 80)
        
        # Create sequence-specific config
        seq_config = copy.deepcopy(config)
        seq_config['source']['sequence'] = seq
        del seq_config['source']['SEQUENCES']
        
        # Write temp config
        series_dir.mkdir(parents=True, exist_ok=True)
        temp_config_path = series_dir / f'temp_{seq}.toml'
        
        with open(temp_config_path, 'wb') as f:
            tomli_w.dump(seq_config, f)
        
        # Run experiment for this sequence
        run_experiment(
            config_path=temp_config_path,
            stage=stage,
            data_dir=data_dir,
            force=force,
            no_cache=no_cache,
            n_workers=n_workers,
            n_trials=n_trials,
        )
        
        # Clean up temp config
        temp_config_path.unlink()
    
    # Collect and save summary
    print()
    print("=" * 80)
    print("📊 COLLECTING SERIES SUMMARY")
    print("=" * 80)
    print()
    
    summaries = collect_series_summary(sequences, data_dir, config)
    json_path, csv_path = save_series_summary(summaries, config, config_path, series_dir)
    
    print(f"   ✓ Saved: {json_path}")
    print(f"   ✓ Saved: {csv_path}")
    
    # Print summary table
    print_series_summary(summaries, config)
    
    return {'sequences': summaries, 'series_dir': str(series_dir)}


# =============================================================================
# Config Validation
# =============================================================================

def validate_config_for_stage(config: dict, stage: str, source_type: str):
    """
    Validate config has required sections for given stage.
    """
    # Load stage validation
    if stage in ['load', 'flow', 'full']:
        if source_type == 'sprites':
            if 'image' not in config:
                print("❌ ERROR: Config missing [image] section for sprite generation")
                sys.exit(1)
        # External source validation happens in load_external_source
    
    # Sweep stage validation
    if stage in ['sweep', 'flow', 'full']:
        if 'parameter_sweep' not in config:
            print("❌ ERROR: Config missing [parameter_sweep] section for sweep")
            sys.exit(1)
    
    # Optimize stage validation
    if stage in ['optimize', 'full']:
        if not uses_ground_truth(config):
            print("❌ ERROR: Optimization requires [evaluation] ground_truth = true")
            sys.exit(1)
        if 'epe_power' not in config.get('evaluation', {}):
            print("❌ ERROR: Config missing [evaluation] epe_power for optimization")
            sys.exit(1)


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_experiment(
    config_path: Path,
    stage: str = 'flow',
    data_dir: Path = Path('data'),
    force: bool = False,
    no_cache: bool = False,
    n_workers: int = None,
    n_trials: int = None
) -> dict:
    """
    Run optical flow experiment pipeline.
    
    Args:
        config_path: Path to TOML config file
        stage: Which stage(s) to run: 'load', 'sweep', 'optimize', 'flow', or 'full'
        data_dir: Base output directory
        force: Force reload/regeneration of data (for 'load' stage)
        no_cache: Force recomputation of sweep (for 'sweep' stage)
        n_workers: Number of parallel workers for sweep
        n_trials: Number of Optuna trials for optimization
        
    Returns:
        Dict with movie_hash, of_hash, and results
    """
    # Load config
    if not config_path.exists():
        print(f"❌ ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
    
    # Determine source type
    source_type = get_source_type(config)
    
    # Validate config
    validate_config_for_stage(config, stage, source_type)
    
    print("=" * 80)
    print("🚀 OPTICAL FLOW EXPERIMENT")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Source: {source_type}")
    print(f"Stage: {stage}")
    print(f"Data dir: {data_dir}")
    print(f"Ground truth: {uses_ground_truth(config)}")
    print("=" * 80)
    print()
    
    results = {
        'config_path': str(config_path),
        'data_dir': str(data_dir),
        'stage': stage,
        'source_type': source_type,
        'ground_truth': uses_ground_truth(config),
    }
    
    # Compute hashes to determine paths
    movie_hash = None
    of_hash = None
    
    if source_type == 'sprites':
        gen_config = extract_generation_config(config)
        movie_hash = compute_generation_hash(gen_config)
    else:
        source_config = extract_source_config(config)
        movie_hash = compute_source_hash(source_config)
    results['movie_hash'] = movie_hash
    
    if 'parameter_sweep' in config:
        of_config = extract_of_config(config)
        of_hash = compute_of_hash(of_config)
        results['of_hash'] = of_hash
    
    # ========================================================================
    # Stage 1: Load / Generate
    # ========================================================================
    
    if stage in ['load', 'flow', 'full']:
        print()
        print("=" * 80)
        if source_type == 'sprites':
            print("STAGE 1: GENERATE MOVIE")
        else:
            print("STAGE 1: LOAD EXTERNAL DATA")
        print("=" * 80)
        print()
        
        if source_type == 'sprites':
            movie_hash = generate_movie(config, data_dir, force=force)
        else:
            movie_hash = load_external_source(config, data_dir, force=force)
        results['movie_hash'] = movie_hash
    
    # ========================================================================
    # Stage 2: Sweep
    # ========================================================================
    
    if stage in ['sweep', 'flow', 'full']:
        print()
        print("=" * 80)
        print("STAGE 2: OPTICAL FLOW SWEEP")
        print("=" * 80)
        print()
        
        if movie_hash is None:
            print("❌ ERROR: movie_hash not available")
            print("   Run 'load' stage first, or provide --movie-hash")
            sys.exit(1)
        
        of_hash = run_sweep(
            config=config,
            movie_hash=movie_hash,
            data_dir=data_dir,
            no_cache=no_cache,
            n_workers=n_workers
        )
        results['of_hash'] = of_hash
    
    # ========================================================================
    # Stage 3: Optimize (with automatic prerequisite handling)
    # ========================================================================
    
    if stage in ['optimize', 'full']:
        print()
        print("=" * 80)
        print("STAGE 3: WEIGHT OPTIMIZATION")
        print("=" * 80)
        print()
        
        # Smart prerequisite handling for 'optimize' stage
        # If prerequisites are missing, run them automatically
        if stage == 'optimize':
            # Check if movie exists
            if not check_movie_exists(data_dir, movie_hash):
                print(f"📦 Movie data not found, running load stage first...")
                print()
                if source_type == 'sprites':
                    movie_hash = generate_movie(config, data_dir, force=force)
                else:
                    movie_hash = load_external_source(config, data_dir, force=force)
                results['movie_hash'] = movie_hash
                print()
            
            # Check if sweep exists
            if not check_sweep_exists(data_dir, movie_hash, of_hash):
                print(f"📦 Sweep results not found, running sweep stage first...")
                print()
                of_hash = run_sweep(
                    config=config,
                    movie_hash=movie_hash,
                    data_dir=data_dir,
                    no_cache=no_cache,
                    n_workers=n_workers
                )
                results['of_hash'] = of_hash
                print()
        
        if movie_hash is None or of_hash is None:
            print("❌ ERROR: movie_hash and of_hash required for optimization")
            print("   This should not happen - please report this bug")
            sys.exit(1)
        
        # Get n_trials from config or argument
        trials = n_trials
        if trials is None:
            trials = config.get('optimization', {}).get('n_trials', 100)
        
        optimization_results = optimize_sequence(
            config=config,
            movie_hash=movie_hash,
            of_hash=of_hash,
            data_dir=data_dir,
            n_trials=trials
        )
        results['optimization'] = {
            m: {
                'best_epe': r['best_epe'], 
                'best_params': r.get('best_params', r.get('best_weights', {}))
            }
            for m, r in optimization_results.items()
        }
        
        # Get opt_hash from results for ranking comparison
        opt_hash = None
        if optimization_results:
            first_result = next(iter(optimization_results.values()))
            opt_hash = first_result.get('opt_hash')
        results['opt_hash'] = opt_hash
    
    # ========================================================================
    # Ranking Comparison (after sweep or optimize, if ground truth available)
    # ========================================================================
    
    if stage in ['sweep', 'flow', 'optimize', 'full'] and uses_ground_truth(config):
        analysis_dir = data_dir / movie_hash / 'analysis' / of_hash
        sweep_dir = analysis_dir / 'sweep'
        
        # Determine optimization directory
        optimization_dir = None
        if 'opt_hash' in results and results['opt_hash']:
            optimization_dir = analysis_dir / 'optimization' / results['opt_hash']
        elif (analysis_dir / 'optimization').exists():
            # Find most recent optimization subdirectory
            opt_subdirs = list((analysis_dir / 'optimization').glob('*/optimization_summary.json'))
            if opt_subdirs:
                optimization_dir = max(opt_subdirs, key=lambda p: p.stat().st_mtime).parent
        
        # Save and print ranking comparison
        save_ranking_comparison(
            analysis_dir=analysis_dir,
            sweep_dir=sweep_dir,
            optimization_dir=optimization_dir,
            config=config
        )
        print_ranking_comparison(analysis_dir)
        
        # Ensemble analysis (sigma sweep)
        from src.ensemble.analysis import analyze_and_save
        for pair_dir in sorted(sweep_dir.glob('pair_*')):
            results_path = pair_dir / 'results_full.pkl'
            if results_path.exists():
                analyze_and_save(results_path, config)
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print()
    print("=" * 80)
    print("🎉 EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"   Data:         {data_dir / movie_hash}" if movie_hash else "")
    print(f"   Analysis:     {data_dir / movie_hash / 'analysis' / of_hash}" if of_hash else "")
    if 'optimization' in results:
        best = min(results['optimization'].items(), key=lambda x: x[1]['best_epe'])
        print(f"   Best method:  {best[0]} (EPE = {best[1]['best_epe']:.6f})")
    print("=" * 80)
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run optical flow experiment pipeline'
    )
    parser.add_argument('config', type=Path, help='TOML configuration file')
    parser.add_argument('--stage', type=str, default='flow',
                       choices=STAGES,
                       help='Stage to run (default: flow = load+sweep)')
    parser.add_argument('--data-dir', type=Path, default=Path('data'),
                       help='Base data directory (default: data/)')
    parser.add_argument('--force', action='store_true',
                       help='Force reload/regeneration of data')
    parser.add_argument('--no-cache', action='store_true',
                       help='Force recomputation of sweep')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers for sweep')
    parser.add_argument('--trials', type=int, default=None,
                       help='Number of Optuna trials (overrides config)')
    
    # Additional arguments for running individual stages with explicit hashes
    parser.add_argument('--movie-hash', type=str, default=None,
                       help='Explicit movie hash (for sweep/optimize without load)')
    parser.add_argument('--of-hash', type=str, default=None,
                       help='Explicit OF hash (for optimize without sweep)')
    
    args = parser.parse_args()
    
    # Check for series mode (SEQUENCES list in config)
    if args.config.exists():
        with open(args.config, 'rb') as f:
            config_check = tomli.load(f)
        
        if 'source' in config_check and 'SEQUENCES' in config_check['source']:
            run_series(
                config_path=args.config,
                stage=args.stage,
                data_dir=args.data_dir,
                force=args.force,
                no_cache=args.no_cache,
                n_workers=args.workers,
                n_trials=args.trials,
            )
            return
    
    # Handle explicit hashes for partial runs
    if args.movie_hash or args.of_hash:
        # Custom partial run with explicit hashes
        if not args.config.exists():
            print(f"❌ ERROR: Config file not found: {args.config}")
            sys.exit(1)
        
        with open(args.config, 'rb') as f:
            config = tomli.load(f)
        
        movie_hash = args.movie_hash
        of_hash = args.of_hash
        
        if args.stage == 'sweep':
            if movie_hash is None:
                print("❌ ERROR: --movie-hash required for sweep stage")
                sys.exit(1)
            run_sweep(
                config=config,
                movie_hash=movie_hash,
                data_dir=args.data_dir,
                no_cache=args.no_cache,
                n_workers=args.workers
            )
        
        elif args.stage == 'optimize':
            if movie_hash is None or of_hash is None:
                print("❌ ERROR: --movie-hash and --of-hash required for optimize stage")
                sys.exit(1)
            
            if not uses_ground_truth(config):
                print("❌ ERROR: Optimization requires [evaluation] ground_truth = true")
                sys.exit(1)
            
            trials = args.trials or config.get('optimization', {}).get('n_trials', 100)
            optimize_sequence(
                config=config,
                movie_hash=movie_hash,
                of_hash=of_hash,
                data_dir=args.data_dir,
                n_trials=trials
            )
        
        else:
            print(f"❌ ERROR: Explicit hashes only supported for 'sweep' and 'optimize' stages")
            sys.exit(1)
    
    else:
        # Normal run
        run_experiment(
            config_path=args.config,
            stage=args.stage,
            data_dir=args.data_dir,
            force=args.force,
            no_cache=args.no_cache,
            n_workers=args.workers,
            n_trials=args.trials
        )


if __name__ == "__main__":
    main()
