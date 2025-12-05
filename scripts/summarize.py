#!/usr/bin/env python3
"""
File: scripts/summarize.py

Summarize optical flow ensemble performance across multiple algorithm configurations.

This script evaluates self-supervised optical flow selection methods by comparing
various ensemble strategies (per-pixel selection) and single-config strategies
(whole-image selection) against ground truth.

ENSEMBLE METHODS (per-pixel selection):
  - Oracle: Per-pixel best config using GT EPE (theoretical limit)
  - Photometric: Per-pixel best config using photometric error
  - Perturbation: Per-pixel best config using perturbation sensitivity
  - Traction-gated: Photo in textured regions, global pert fallback elsewhere

SINGLE CONFIG METHODS (whole-image selection):
  - GT Best: Config with lowest mean EPE (cheating baseline)
  - GT Worst: Config with highest mean EPE (worst case)
  - Photometric: Config with lowest mean photometric error
  - Perturbation: Config with lowest mean perturbation sensitivity  
  - Traction: Config with lowest combined score (photo*traction + pert*(1-traction))

OUTPUT:
  data/summary/{hash}/
    summary.txt  - Terminal-formatted table
    summary.csv  - Flat CSV for analysis
    summary.md   - Markdown table for reports

Usage:
    python scripts/summarize.py configs/middlebury_dis.toml configs/middlebury_farneback.toml

Author: Stephan
"""

import sys
import json
import hashlib
import pickle
import tomllib
from pathlib import Path
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tomli_w
except ImportError:
    tomli_w = None

# Import from project modules
from src.evaluation.self_supervised import (
    traction_gated_fallback_ensemble,
    get_pollution_depth
)

# ==============================================================================
# Configuration & Path Handling
# ==============================================================================

def load_config(config_path: Path) -> dict:
    """Load and validate TOML config file."""
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    # Validate required sections
    required = ["source", "parameter_sweep", "perturbations"]
    for section in required:
        if section not in config:
            print(f"❌ Missing required section [{section}] in {config_path}")
            sys.exit(1)
    
    return config


def get_sequences(config: dict) -> list[str]:
    """Get list of sequences from config."""
    return config["source"].get("SEQUENCES", [])


def get_algorithm(config: dict) -> str:
    """Get algorithm name from config."""
    return config["parameter_sweep"].get("algorithm", "farneback")


def get_perturbation_magnitude(config: dict) -> float:
    """Get perturbation magnitude for traction computation."""
    return config["perturbations"].get("magnitude", 2.0)


def compute_source_hash(config: dict, sequence: str) -> str:
    """Compute movie_hash for a specific sequence."""
    source = config["source"]
    source_type = source.get("type", "middlebury")
    
    source_config = {
        "type": source_type,
        "sequence": sequence,
        "frames": source.get("frames"),
    }
    
    if source_type == "sintel":
        source_config["pass"] = source.get("pass")
        source_config["root"] = source.get("sintel_root", "")
    elif source_type == "middlebury":
        source_config["root"] = source.get("middlebury_root", "")
    else:
        source_config["root"] = source.get("root", "")
    
    config_str = json.dumps(source_config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def compute_of_hash(config: dict) -> str:
    """Compute of_hash from parameter_sweep and perturbations."""
    of_config = {}
    
    if "parameter_sweep" in config:
        of_config["parameter_sweep"] = config["parameter_sweep"]
    if "perturbations" in config:
        of_config["perturbations"] = config["perturbations"]
    
    if tomli_w:
        config_str = tomli_w.dumps(of_config)
    else:
        # Fallback to json if tomli_w not available
        config_str = json.dumps(of_config, sort_keys=True)
    
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def get_results_path(data_dir: Path, movie_hash: str, of_hash: str) -> Path:
    """Get path to results_full.pkl for a sequence."""
    return data_dir / movie_hash / "analysis" / of_hash / "sweep" / "pair_000" / "results_full.pkl"


def get_frames_dir(data_dir: Path, movie_hash: str) -> Path:
    """Get path to frames directory for a sequence."""
    return data_dir / movie_hash / "frames"


# ==============================================================================
# Data Loading
# ==============================================================================

def load_results(results_path: Path) -> list[dict]:
    """Load results_full.pkl containing all config results."""
    if not results_path.exists():
        print(f"❌ Results not found: {results_path}")
        sys.exit(1)
    
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    
    return results


def load_ground_truth_from_frames(frames_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ground truth flow and valid mask from frames directory."""
    u_path = frames_dir / "u_000.npz"
    v_path = frames_dir / "v_000.npz"
    
    if not u_path.exists() or not v_path.exists():
        print(f"❌ Ground truth not found in {frames_dir}")
        sys.exit(1)
    
    u_data = np.load(u_path)
    v_data = np.load(v_path)
    
    # Get first (only) array from npz
    u_gt = u_data[list(u_data.keys())[0]]
    v_gt = v_data[list(v_data.keys())[0]]
    
    # Valid mask: not NaN and not huge values
    valid = (
        ~np.isnan(u_gt) & ~np.isnan(v_gt) &
        (np.abs(u_gt) < 1e8) & (np.abs(v_gt) < 1e8)
    )
    
    return u_gt, v_gt, valid


# ==============================================================================
# Metric Computation
# ==============================================================================

def compute_epe(u: np.ndarray, v: np.ndarray, u_gt: np.ndarray, v_gt: np.ndarray, 
                valid: np.ndarray) -> float:
    """Compute mean endpoint error over valid pixels."""
    epe = np.sqrt((u - u_gt)**2 + (v - v_gt)**2)
    return float(np.mean(epe[valid]))


def compute_epe_map(u: np.ndarray, v: np.ndarray, u_gt: np.ndarray, v_gt: np.ndarray) -> np.ndarray:
    """Compute per-pixel endpoint error."""
    return np.sqrt((u - u_gt)**2 + (v - v_gt)**2)


def build_stacks_from_results(
    results: list[dict],
    u_gt: np.ndarray,
    v_gt: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Build metric stacks from results_full.pkl data.
    
    Returns dict with:
      - u_stack, v_stack: flow components (n_configs, H, W)
      - epe_stack: endpoint error (n_configs, H, W)
      - photo_stack: photometric error (n_configs, H, W)
      - pert_stack: perturbation sensitivity * depth (n_configs, H, W)
      - traction_stack: traction (n_configs, H, W)
    """
    n_configs = len(results)
    
    # Get shape from first result
    sample_u = results[0]["flows"]["u_AB"]
    H, W = sample_u.shape
    
    u_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    v_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    epe_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    photo_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    pert_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    traction_stack = np.zeros((n_configs, H, W), dtype=np.float32)
    
    for i, r in enumerate(results):
        u = r["flows"]["u_AB"]
        v = r["flows"]["v_AB"]
        
        u_stack[i] = u
        v_stack[i] = v
        epe_stack[i] = compute_epe_map(u, v, u_gt, v_gt)
        
        # Photometric - prefer log, fall back to gray
        metrics = r["metrics"]
        if "photo_log_raw_A" in metrics:
            photo_stack[i] = metrics["photo_log_raw_A"]
        elif "photo_gray_raw_A" in metrics:
            photo_stack[i] = metrics["photo_gray_raw_A"]
        elif "photometric_A_raw" in metrics:
            photo_stack[i] = metrics["photometric_A_raw"]
        elif "photometric_A" in metrics:
            photo_stack[i] = metrics["photometric_A"]
        else:
            print(f"⚠️  No photometric metric found for config {i}")
            photo_stack[i] = np.nan
        
        # Perturbation sensitivity - NORMALIZED BY POLLUTION DEPTH
        config_name = r["metadata"].get("config_name", f"config_{i}")
        depth = get_pollution_depth(config_name)
        
        if "perturbation_raw_A" in metrics:
            pert_stack[i] = metrics["perturbation_raw_A"] * depth
        elif "displacements_sensitivity_A2B" in metrics:
            pert_stack[i] = metrics["displacements_sensitivity_A2B"] * depth
        else:
            print(f"⚠️  No perturbation metric found for config {i}")
            pert_stack[i] = np.nan
        
        # Traction
        if "traction_raw_A" in metrics:
            traction_stack[i] = metrics["traction_raw_A"]
        elif "traction_A" in metrics:
            traction_stack[i] = metrics["traction_A"]
        else:
            print(f"⚠️  No traction metric found for config {i}")
            traction_stack[i] = np.nan
    
    return {
        "u_stack": u_stack,
        "v_stack": v_stack,
        "epe_stack": epe_stack,
        "photo_stack": photo_stack,
        "pert_stack": pert_stack,
        "traction_stack": traction_stack,
    }


# ==============================================================================
# Analysis Pipeline
# ==============================================================================

def analyze_sequence(
    data_dir: Path, 
    sequence: str, 
    config: dict,
    movie_hash: str,
    of_hash: str
) -> dict:
    """
    Run full analysis for a single sequence.
    Returns dict with all metrics.
    """
    print(f"  📊 {sequence}...", end=" ", flush=True)
    
    # Load results
    results_path = get_results_path(data_dir, movie_hash, of_hash)
    results = load_results(results_path)
    n_configs = len(results)
    
    # Load ground truth
    frames_dir = get_frames_dir(data_dir, movie_hash)
    u_gt, v_gt, valid = load_ground_truth_from_frames(frames_dir)
    H, W = u_gt.shape
    
    # Build stacks (with pollution depth normalization for perturbation)
    stacks = build_stacks_from_results(results, u_gt, v_gt)
    u_stack = stacks["u_stack"]
    v_stack = stacks["v_stack"]
    epe_stack = stacks["epe_stack"]
    photo_stack = stacks["photo_stack"]
    pert_stack = stacks["pert_stack"]
    
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    
    # --- Ensemble methods ---
    
    # Oracle (best possible with GT)
    oracle_idx = np.argmin(epe_stack, axis=0)
    epe_oracle = epe_stack[oracle_idx, y_idx, x_idx]
    oracle_epe = float(np.mean(epe_oracle[valid]))
    
    # Photometric per-pixel selection
    photo_masked = np.where(np.isnan(photo_stack), np.inf, photo_stack)
    photo_idx = np.argmin(photo_masked, axis=0)
    epe_photo_ens = epe_stack[photo_idx, y_idx, x_idx]
    photo_ens_epe = float(np.mean(epe_photo_ens[valid]))
    
    # Perturbation per-pixel selection
    pert_masked = np.where(np.isnan(pert_stack), np.inf, pert_stack)
    pert_idx = np.argmin(pert_masked, axis=0)
    epe_pert_ens = epe_stack[pert_idx, y_idx, x_idx]
    pert_ens_epe = float(np.mean(epe_pert_ens[valid]))
    
    # Traction-gated with global pert fallback (using project function)
    fb_pert = traction_gated_fallback_ensemble(
        results, 
        delta_mag=1.0,
        fidelity_threshold=0.5,
        fallback_metric='pert',
        traction_exponent=1.0
    )
    epe_traction = np.sqrt((fb_pert['u'] - u_gt)**2 + (fb_pert['v'] - v_gt)**2)
    traction_ens_epe = float(np.mean(epe_traction[valid]))
    
    # --- Single config methods ---
    
    # Mean EPE per config for ranking
    mean_epes = [float(np.mean(epe_stack[i][valid])) for i in range(n_configs)]
    sorted_epes = sorted(enumerate(mean_epes), key=lambda x: x[1])
    
    # GT Best
    gt_best_idx = int(np.argmin(mean_epes))
    gt_best_epe = mean_epes[gt_best_idx]
    
    # GT Worst
    gt_worst_idx = int(np.argmax(mean_epes))
    gt_worst_epe = mean_epes[gt_worst_idx]
    
    # Photometric: config with lowest mean photo error
    mean_photo = [float(np.nanmean(photo_stack[i][valid])) for i in range(n_configs)]
    photo_single_idx = int(np.argmin(mean_photo))
    photo_single_epe = mean_epes[photo_single_idx]
    photo_rank = next(i+1 for i, (idx, _) in enumerate(sorted_epes) if idx == photo_single_idx)
    
    # Perturbation: config with lowest mean perturbation (depth-normalized)
    mean_pert = [float(np.nanmean(pert_stack[i][valid])) for i in range(n_configs)]
    pert_single_idx = int(np.argmin(mean_pert))
    pert_single_epe = mean_epes[pert_single_idx]
    pert_rank = next(i+1 for i, (idx, _) in enumerate(sorted_epes) if idx == pert_single_idx)
    
    # Traction: config with lowest combined (photo*traction + pert*(1-traction))
    # Use mean traction across configs as weight
    traction_stack = stacks["traction_stack"]
    mean_traction = np.nanmean(traction_stack, axis=0)  # (H, W)
    
    traction_scores = []
    for i in range(n_configs):
        # Normalize to similar scales
        photo_norm = photo_stack[i] / (np.nanmean(photo_stack[i][valid]) + 1e-8)
        pert_norm = pert_stack[i] / (np.nanmean(pert_stack[i][valid]) + 1e-8)
        combined = photo_norm * mean_traction + pert_norm * (1 - mean_traction)
        traction_scores.append(float(np.nanmean(combined[valid])))
    
    traction_single_idx = int(np.argmin(traction_scores))
    traction_single_epe = mean_epes[traction_single_idx]
    traction_rank = next(i+1 for i, (idx, _) in enumerate(sorted_epes) if idx == traction_single_idx)
    
    print(f"✓ (oracle={oracle_epe:.3f}, best={gt_best_epe:.3f})")
    
    return {
        "sequence": sequence,
        "n_configs": n_configs,
        # Ensemble EPEs
        "oracle_epe": oracle_epe,
        "photo_ens_epe": photo_ens_epe,
        "pert_ens_epe": pert_ens_epe,
        "traction_ens_epe": traction_ens_epe,
        # Ensemble ratios (vs oracle)
        "photo_ens_ratio": photo_ens_epe / oracle_epe,
        "pert_ens_ratio": pert_ens_epe / oracle_epe,
        "traction_ens_ratio": traction_ens_epe / oracle_epe,
        # Single config EPEs
        "gt_best_epe": gt_best_epe,
        "gt_worst_epe": gt_worst_epe,
        "gt_worst_ratio": gt_worst_epe / gt_best_epe,
        "gt_worst_rank": n_configs,
        "photo_single_epe": photo_single_epe,
        "pert_single_epe": pert_single_epe,
        "traction_single_epe": traction_single_epe,
        # Single config ratios (vs GT best)
        "photo_single_ratio": photo_single_epe / gt_best_epe,
        "pert_single_ratio": pert_single_epe / gt_best_epe,
        "traction_single_ratio": traction_single_epe / gt_best_epe,
        # Single config ranks
        "photo_rank": photo_rank,
        "pert_rank": pert_rank,
        "traction_rank": traction_rank,
    }


def analyze_config(config_path: Path, data_dir: Path = Path("data")) -> dict:
    """Analyze all sequences for a config file."""
    print(f"\n🔧 Processing: {config_path}")
    
    config = load_config(config_path)
    sequences = get_sequences(config)
    algorithm = get_algorithm(config)
    of_hash = compute_of_hash(config)
    
    # Count configs from first sequence
    first_seq = sequences[0]
    first_movie_hash = compute_source_hash(config, first_seq)
    first_results_path = get_results_path(data_dir, first_movie_hash, of_hash)
    first_results = load_results(first_results_path)
    n_configs = len(first_results)
    
    print(f"   Algorithm: {algorithm}")
    print(f"   Configs: {n_configs}")
    print(f"   Sequences: {len(sequences)}")
    print(f"   OF hash: {of_hash}")
    
    results = []
    for seq in sequences:
        movie_hash = compute_source_hash(config, seq)
        result = analyze_sequence(data_dir, seq, config, movie_hash, of_hash)
        results.append(result)
    
    return {
        "config_path": str(config_path),
        "algorithm": algorithm,
        "n_configs": n_configs,
        "sequences": results,
    }


# ==============================================================================
# Statistics & Formatting
# ==============================================================================

def format_range_median(values: list[float], fmt: str = ".2f") -> str:
    """Format as 'min - max (median)'."""
    arr = np.array(values)
    return f"{arr.min():{fmt}} - {arr.max():{fmt}} ({np.median(arr):{fmt}})"


def format_range_median_int(values: list[int]) -> str:
    """Format integers as 'min - max (median)'."""
    arr = np.array(values)
    return f"{int(arr.min())} - {int(arr.max())} ({int(np.median(arr))})"


# ==============================================================================
# Output Generation
# ==============================================================================

def generate_terminal_output(all_results: list[dict]) -> str:
    """Generate terminal-formatted table."""
    lines = []
    
    # Header
    lines.append("=" * 110)
    lines.append("OPTICAL FLOW ENSEMBLE ANALYSIS")
    lines.append("=" * 110)
    
    # List sequences
    if all_results:
        sequences = [s["sequence"] for s in all_results[0]["sequences"]]
        lines.append(f"Sequences ({len(sequences)}): {', '.join(sequences)}")
    
    # --- Ensembles section ---
    lines.append("")
    lines.append("Ensembles (per-pixel selection)")
    lines.append("-" * 110)
    
    # Build header row
    header = f"{'Method':<20}"
    for res in all_results:
        alg = res["algorithm"].capitalize()
        header += f"  {alg + ' EPE':<24}  {alg + ' ×Oracle':<20}"
    lines.append(header)
    lines.append("-" * 110)
    
    # Data rows
    ensemble_methods = [
        ("Oracle", "oracle_epe", None),
        ("Photometric", "photo_ens_epe", "photo_ens_ratio"),
        ("Perturbation", "pert_ens_epe", "pert_ens_ratio"),
        ("Traction-gated", "traction_ens_epe", "traction_ens_ratio"),
    ]
    
    for method_name, epe_key, ratio_key in ensemble_methods:
        row = f"  {method_name:<18}"
        for res in all_results:
            epes = [s[epe_key] for s in res["sequences"]]
            epe_str = format_range_median(epes)
            
            if ratio_key:
                ratios = [s[ratio_key] for s in res["sequences"]]
                ratio_str = format_range_median(ratios)
            else:
                ratio_str = ""  # Oracle - leave blank
            
            row += f"  {epe_str:<24}  {ratio_str:<20}"
        lines.append(row)
    
    # --- Single Config section ---
    lines.append("")
    lines.append("=" * 110)
    lines.append("Single Config (whole-image selection)")
    lines.append("-" * 110)
    
    for res in all_results:
        alg = res["algorithm"].capitalize()
        n_cfg = res["n_configs"]
        
        lines.append("")
        lines.append(f"{alg} ({n_cfg} configs)")
        lines.append(f"{'Method':<20}  {'EPE (range, median)':<24}  {'×GT Best (range, med)':<22}  {'Rank (range, median)':<20}")
        lines.append("-" * 90)
        
        single_methods = [
            ("GT Best", "gt_best_epe", None, None),
            ("GT Worst", "gt_worst_epe", "gt_worst_ratio", "gt_worst_rank"),
            ("Photometric", "photo_single_epe", "photo_single_ratio", "photo_rank"),
            ("Perturbation", "pert_single_epe", "pert_single_ratio", "pert_rank"),
            ("Traction", "traction_single_epe", "traction_single_ratio", "traction_rank"),
        ]
        
        for method_name, epe_key, ratio_key, rank_key in single_methods:
            epes = [s[epe_key] for s in res["sequences"]]
            epe_str = format_range_median(epes)
            
            if ratio_key:
                ratios = [s[ratio_key] for s in res["sequences"]]
                ratio_str = format_range_median(ratios)
            else:
                ratio_str = ""  # GT Best - leave blank
            
            if rank_key:
                ranks = [s[rank_key] for s in res["sequences"]]
                rank_str = format_range_median_int(ranks)
            else:
                rank_str = ""  # GT Best - leave blank
            
            lines.append(f"  {method_name:<18}  {epe_str:<24}  {ratio_str:<22}  {rank_str:<20}")
    
    lines.append("")
    lines.append("=" * 110)
    
    return "\n".join(lines)


def generate_csv_output(all_results: list[dict]) -> str:
    """Generate flat CSV output."""
    rows = []
    
    # Header
    header = [
        "config_file", "algorithm", "n_configs", "section", "method", "sequence",
        "epe", "ratio", "rank"
    ]
    rows.append(",".join(header))
    
    for res in all_results:
        config_file = Path(res["config_path"]).name
        algorithm = res["algorithm"]
        n_configs = res["n_configs"]
        
        for seq_data in res["sequences"]:
            seq = seq_data["sequence"]
            
            # Ensemble rows
            ensemble_data = [
                ("Oracle", seq_data["oracle_epe"], 1.0),
                ("Photometric", seq_data["photo_ens_epe"], seq_data["photo_ens_ratio"]),
                ("Perturbation", seq_data["pert_ens_epe"], seq_data["pert_ens_ratio"]),
                ("Traction-gated", seq_data["traction_ens_epe"], seq_data["traction_ens_ratio"]),
            ]
            for method, epe, ratio in ensemble_data:
                rows.append(f"{config_file},{algorithm},{n_configs},ensemble,{method},{seq},{epe:.4f},{ratio:.4f},")
            
            # Single config rows
            single_data = [
                ("GT Best", seq_data["gt_best_epe"], 1.0, 1),
                ("GT Worst", seq_data["gt_worst_epe"], seq_data["gt_worst_ratio"], seq_data["gt_worst_rank"]),
                ("Photometric", seq_data["photo_single_epe"], seq_data["photo_single_ratio"], seq_data["photo_rank"]),
                ("Perturbation", seq_data["pert_single_epe"], seq_data["pert_single_ratio"], seq_data["pert_rank"]),
                ("Traction", seq_data["traction_single_epe"], seq_data["traction_single_ratio"], seq_data["traction_rank"]),
            ]
            for method, epe, ratio, rank in single_data:
                rows.append(f"{config_file},{algorithm},{n_configs},single,{method},{seq},{epe:.4f},{ratio:.4f},{rank}")
    
    return "\n".join(rows)


def generate_markdown_output(all_results: list[dict]) -> str:
    """Generate markdown table output."""
    lines = []
    
    lines.append("# Optical Flow Ensemble Analysis")
    lines.append("")
    
    # List sequences
    if all_results:
        sequences = [s["sequence"] for s in all_results[0]["sequences"]]
        lines.append(f"**Sequences ({len(sequences)}):** {', '.join(sequences)}")
        lines.append("")
    
    # --- Ensembles section ---
    lines.append("## Ensembles (per-pixel selection)")
    lines.append("")
    
    # Build header
    header = "| Method |"
    sep = "|--------|"
    for res in all_results:
        alg = res["algorithm"].capitalize()
        header += f" {alg} EPE | {alg} ×Oracle |"
        sep += "---------|---------|"
    lines.append(header)
    lines.append(sep)
    
    ensemble_methods = [
        ("Oracle", "oracle_epe", None),
        ("Photometric", "photo_ens_epe", "photo_ens_ratio"),
        ("Perturbation", "pert_ens_epe", "pert_ens_ratio"),
        ("Traction-gated", "traction_ens_epe", "traction_ens_ratio"),
    ]
    
    for method_name, epe_key, ratio_key in ensemble_methods:
        row = f"| {method_name} |"
        for res in all_results:
            epes = [s[epe_key] for s in res["sequences"]]
            epe_str = format_range_median(epes)
            
            if ratio_key:
                ratios = [s[ratio_key] for s in res["sequences"]]
                ratio_str = format_range_median(ratios)
            else:
                ratio_str = ""  # Oracle - leave blank
            
            row += f" {epe_str} | {ratio_str} |"
        lines.append(row)
    
    # --- Single Config sections ---
    lines.append("")
    lines.append("## Single Config (whole-image selection)")
    
    for res in all_results:
        alg = res["algorithm"].capitalize()
        n_cfg = res["n_configs"]
        
        lines.append("")
        lines.append(f"### {alg} ({n_cfg} configs)")
        lines.append("")
        lines.append("| Method | EPE (range, median) | ×GT Best (range, med) | Rank (range, median) |")
        lines.append("|--------|---------------------|----------------------|---------------------|")
        
        single_methods = [
            ("GT Best", "gt_best_epe", None, None),
            ("GT Worst", "gt_worst_epe", "gt_worst_ratio", "gt_worst_rank"),
            ("Photometric", "photo_single_epe", "photo_single_ratio", "photo_rank"),
            ("Perturbation", "pert_single_epe", "pert_single_ratio", "pert_rank"),
            ("Traction", "traction_single_epe", "traction_single_ratio", "traction_rank"),
        ]
        
        for method_name, epe_key, ratio_key, rank_key in single_methods:
            epes = [s[epe_key] for s in res["sequences"]]
            epe_str = format_range_median(epes)
            
            if ratio_key:
                ratios = [s[ratio_key] for s in res["sequences"]]
                ratio_str = format_range_median(ratios)
            else:
                ratio_str = ""  # GT Best - leave blank
            
            if rank_key:
                ranks = [s[rank_key] for s in res["sequences"]]
                rank_str = format_range_median_int(ranks)
            else:
                rank_str = ""  # GT Best - leave blank
            
            lines.append(f"| {method_name} | {epe_str} | {ratio_str} | {rank_str} |")
    
    return "\n".join(lines)


# ==============================================================================
# Main
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Summarize optical flow ensemble performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/summarize.py configs/middlebury_dis.toml
    python scripts/summarize.py configs/middlebury_dis.toml configs/middlebury_farneback.toml
    python scripts/summarize.py configs/middlebury_dis.toml --data-dir /path/to/data
        """
    )
    parser.add_argument("configs", type=Path, nargs="+", 
                        help="TOML configuration file(s)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Base data directory (default: data/)")
    
    args = parser.parse_args()
    
    config_paths = args.configs
    data_dir = args.data_dir
    
    # Validate all configs exist
    for p in config_paths:
        if not p.exists():
            print(f"❌ Config file not found: {p}")
            sys.exit(1)
    
    # Generate output hash from sorted config names
    sorted_names = sorted([p.name for p in config_paths])
    hash_input = "|".join(sorted_names)
    output_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    
    # Create output directory
    output_dir = Path("data/summary") / output_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔬 OPTICAL FLOW ENSEMBLE ANALYZER")
    print("=" * 60)
    print(f"📁 Data dir: {data_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Analyze each config
    all_results = []
    for config_path in config_paths:
        result = analyze_config(config_path, data_dir)
        all_results.append(result)
    
    # Generate outputs
    print("\n📝 Generating outputs...")
    
    # Terminal
    terminal_output = generate_terminal_output(all_results)
    print("\n" + terminal_output)
    
    # Save files
    (output_dir / "summary.txt").write_text(terminal_output)
    print(f"   ✓ summary.txt")
    
    csv_output = generate_csv_output(all_results)
    (output_dir / "summary.csv").write_text(csv_output)
    print(f"   ✓ summary.csv")
    
    md_output = generate_markdown_output(all_results)
    (output_dir / "summary.md").write_text(md_output)
    print(f"   ✓ summary.md")
    
    print(f"\n✅ Done! Results in {output_dir}")


if __name__ == "__main__":
    main()
