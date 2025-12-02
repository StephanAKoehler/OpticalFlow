# File: scripts/selection_pipeline.py
"""
Main pipeline for optical flow config selection.

Runs all selection methods defined in config and optionally optimizes weights.
"""

import argparse
import json
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add src to path (adjust as needed for your project structure)
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# These imports assume the modules are in the Python path
# Adjust imports based on your actual project structure
# from config.config_parser import load_config, validate_config, print_config_summary
# from evaluation.selection import (
#     SelectionParams, select_best_config, select_oracle_per_pixel,
#     build_oracle_flow, select_best_config_epe, select_oracle_per_pixel_epe,
#     compute_epe
# )
# from optimization.weight_optimizer import optimize_weights, save_optimization_result


def get_output_dir(
    results_base: Path,
    image_hash: str,
    algorithm: str,
    sweep_hash: str
) -> Path:
    """Get output directory for sweep results."""
    return results_base / image_hash / f"{algorithm}_{sweep_hash}"


def get_selection_dir(
    sweep_dir: Path,
    selection_name: str,
    param_hash: str
) -> Path:
    """Get output directory for selection results."""
    return sweep_dir / f"selection_{selection_name}_{param_hash}"


def get_optimization_dir(
    sweep_dir: Path,
    normalize: str,
    aggregation: str
) -> Path:
    """Get output directory for optimization results."""
    return sweep_dir / f"optimization_{normalize}_{aggregation}"


def save_selection_results(
    output_dir: Path,
    selection_name: str,
    params: dict,
    oracle_u: np.ndarray,
    oracle_v: np.ndarray,
    config_map: np.ndarray,
    best_config_hash: str,
    best_u: np.ndarray,
    best_v: np.ndarray,
    config_hashes: list[str],
    epe_stats: dict = None
) -> None:
    """
    Save selection results to directory.
    
    Args:
        output_dir: Directory to save to
        selection_name: Name of selection method
        params: Selection parameters dict
        oracle_u, oracle_v: Oracle flow arrays
        config_map: Per-pixel config index
        best_config_hash: Hash of best single config
        best_u, best_v: Best single config flow
        config_hashes: List of config hashes (for mapping indices)
        epe_stats: Optional EPE statistics if GT available
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    params_path = output_dir / "config.toml"
    with open(params_path, "w") as f:
        f.write(f"# Selection: {selection_name}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        f.write(f'normalize = "{params["normalize"]}"\n')
        f.write(f'aggregation = "{params["aggregation"]}"\n')
        f.write(f'power = {params["power"]}\n\n')
        f.write("# Weights\n")
        for metric, weight in params["weights"].items():
            f.write(f"{metric} = {weight}\n")
    
    # Save oracle flow
    np.save(output_dir / "oracle_u.npy", oracle_u)
    np.save(output_dir / "oracle_v.npy", oracle_v)
    np.save(output_dir / "oracle_config_map.npy", config_map)
    
    # Save config hash mapping
    with open(output_dir / "config_hashes.json", "w") as f:
        json.dump(config_hashes, f)
    
    # Save best single config
    with open(output_dir / "best_config_hash.txt", "w") as f:
        f.write(best_config_hash)
    np.save(output_dir / "best_u.npy", best_u)
    np.save(output_dir / "best_v.npy", best_v)
    
    # Save EPE stats if available
    if epe_stats is not None:
        with open(output_dir / "epe_stats.json", "w") as f:
            json.dump(epe_stats, f, indent=2)
    
    print(f"   ✓ Saved to {output_dir}")


def run_selection(
    selection_name: str,
    params: "SelectionParams",
    all_metrics: dict,
    all_flows: dict,
    valid_mask: np.ndarray,
    u_truth: np.ndarray = None,
    v_truth: np.ndarray = None,
    epe_power: float = 2.0
) -> dict:
    """
    Run a single selection method.
    
    Args:
        selection_name: Name of selection method
        params: SelectionParams instance
        all_metrics: Dict of config_hash -> metrics
        all_flows: Dict of config_hash -> (u, v)
        valid_mask: Boolean mask of valid pixels
        u_truth, v_truth: Ground truth (optional)
        epe_power: Power for EPE statistics
        
    Returns:
        Dict with all results
    """
    from evaluation.selection import (
        select_best_config, select_oracle_per_pixel, build_oracle_flow, compute_epe
    )
    
    print(f"\n🎯 Running selection: {selection_name}")
    print(f"   normalize={params.normalize}, aggregation={params.aggregation}")
    
    config_hashes = list(all_flows.keys())
    
    # Select best single config
    best_hash, scores = select_best_config(all_metrics, params, valid_mask)
    best_u, best_v = all_flows[best_hash]
    
    print(f"   Best config: {best_hash} (score={scores[best_hash]:.4f})")
    
    # Select oracle (per-pixel best)
    config_map, penalties = select_oracle_per_pixel(all_metrics, params, valid_mask)
    oracle_u, oracle_v = build_oracle_flow(all_flows, config_map, config_hashes)
    
    # Config distribution
    for i, h in enumerate(config_hashes):
        count = np.sum(config_map == i)
        pct = 100 * count / config_map.size
        print(f"   Config {h[:8]}: {pct:.1f}%")
    
    # Compute EPE if GT available
    epe_stats = None
    if u_truth is not None and v_truth is not None:
        # Oracle EPE
        oracle_epe = compute_epe(oracle_u, oracle_v, u_truth, v_truth)
        oracle_mean = np.mean(oracle_epe[valid_mask] ** epe_power)
        
        # Best single EPE
        best_epe = compute_epe(best_u, best_v, u_truth, v_truth)
        best_mean = np.mean(best_epe[valid_mask] ** epe_power)
        
        epe_stats = {
            "oracle_mean_epe_power": float(oracle_mean),
            "best_mean_epe_power": float(best_mean),
            "epe_power": epe_power,
            "oracle_mean_epe": float(np.mean(oracle_epe[valid_mask])),
            "best_mean_epe": float(np.mean(best_epe[valid_mask])),
        }
        
        print(f"   Oracle EPE^{epe_power}: {oracle_mean:.4f}")
        print(f"   Best single EPE^{epe_power}: {best_mean:.4f}")
    
    return {
        "oracle_u": oracle_u,
        "oracle_v": oracle_v,
        "config_map": config_map,
        "best_config_hash": best_hash,
        "best_u": best_u,
        "best_v": best_v,
        "config_hashes": config_hashes,
        "scores": scores,
        "epe_stats": epe_stats
    }


def run_epe_selection(
    all_flows: dict,
    u_truth: np.ndarray,
    v_truth: np.ndarray,
    valid_mask: np.ndarray,
    epe_power: float = 2.0
) -> dict:
    """
    Run EPE-based selection (requires ground truth).
    
    Returns dict with oracle and best single config results.
    """
    from evaluation.selection import (
        select_best_config_epe, select_oracle_per_pixel_epe, build_oracle_flow, compute_epe
    )
    
    print("\n📏 Running EPE selection (ground truth)")
    
    config_hashes = list(all_flows.keys())
    
    # Best single config by EPE
    best_hash, scores = select_best_config_epe(
        all_flows, u_truth, v_truth, epe_power, valid_mask
    )
    best_u, best_v = all_flows[best_hash]
    
    print(f"   Best config: {best_hash} (EPE^{epe_power}={scores[best_hash]:.4f})")
    
    # Oracle (per-pixel best by EPE)
    config_map, all_epe = select_oracle_per_pixel_epe(all_flows, u_truth, v_truth)
    oracle_u, oracle_v = build_oracle_flow(all_flows, config_map, config_hashes)
    
    # Oracle EPE
    oracle_epe = compute_epe(oracle_u, oracle_v, u_truth, v_truth)
    oracle_mean = np.mean(oracle_epe[valid_mask] ** epe_power)
    
    print(f"   Oracle EPE^{epe_power}: {oracle_mean:.4f}")
    
    # Config distribution
    for i, h in enumerate(config_hashes):
        count = np.sum(config_map == i)
        pct = 100 * count / config_map.size
        print(f"   Config {h[:8]}: {pct:.1f}%")
    
    return {
        "oracle_u": oracle_u,
        "oracle_v": oracle_v,
        "config_map": config_map,
        "best_config_hash": best_hash,
        "best_u": best_u,
        "best_v": best_v,
        "config_hashes": config_hashes,
        "scores": scores,
        "oracle_mean_epe_power": oracle_mean
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optical flow config selection pipeline"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to TOML config file"
    )
    parser.add_argument(
        "--optimize",
        type=str,
        nargs="?",
        const="all",
        help="Optimize weights. Specify selection name or 'all'"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials (default: 100)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base results directory (default: results)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OPTICAL FLOW CONFIG SELECTION PIPELINE")
    print("=" * 70)
    
    # Load config
    from config.config_parser import load_config, validate_config, print_config_summary
    
    config = load_config(args.config)
    validate_config(config)
    print_config_summary(config)
    
    # TODO: Load precomputed flows and metrics from sweep results
    # This would typically load from:
    #   {results_dir}/{image_hash}/{algorithm}_{sweep_hash}/results_full.pkl
    #
    # For now, we show the structure:
    
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Placeholder - replace with actual data loading
    print("⚠️  Data loading not implemented - this is a template")
    print("   Would load from: {results_dir}/{image_hash}/{algorithm}_{sweep_hash}/")
    
    # Example structure expected:
    # all_metrics = {
    #     "config_hash_1": {"perturbation_rms": arr, "bidirectional": arr, ...},
    #     "config_hash_2": {...},
    # }
    # all_flows = {
    #     "config_hash_1": (u, v),
    #     "config_hash_2": (u, v),
    # }
    # valid_mask = np.ndarray (H, W) bool
    # u_truth, v_truth = np.ndarray or None
    
    # --- DEMO MODE: Create synthetic data for testing ---
    print("\n🧪 Running in DEMO mode with synthetic data")
    
    np.random.seed(42)
    H, W = 100, 100
    n_configs = 4
    
    u_truth = np.ones((H, W)) * 2.0
    v_truth = np.ones((H, W)) * 1.0
    valid_mask = np.ones((H, W), dtype=bool)
    valid_mask[:10, :] = False  # Boundary
    valid_mask[-10:, :] = False
    valid_mask[:, :10] = False
    valid_mask[:, -10:] = False
    
    all_metrics = {}
    all_flows = {}
    config_hashes = [f"config_{i:04d}" for i in range(n_configs)]
    
    for i, h in enumerate(config_hashes):
        noise = 0.3 + i * 0.2
        u = u_truth + np.random.randn(H, W) * noise
        v = v_truth + np.random.randn(H, W) * noise
        all_flows[h] = (u, v)
        
        epe = np.sqrt((u - u_truth)**2 + (v - v_truth)**2)
        all_metrics[h] = {
            "perturbation_rms": epe * (0.8 + np.random.rand(H, W) * 0.4),
            "bidirectional": epe * (0.6 + np.random.rand(H, W) * 0.4),
            "photometric": epe * (1.0 + np.random.rand(H, W) * 0.3),
            "traction": np.random.rand(H, W) * 2,  # Uncorrelated (bad metric)
        }
    
    # --- END DEMO MODE ---
    
    print("\n" + "=" * 70)
    print("RUNNING SELECTIONS")
    print("=" * 70)
    
    # Run EPE-based selection if GT available
    has_gt = u_truth is not None and v_truth is not None
    
    if has_gt:
        epe_results = run_epe_selection(
            all_flows, u_truth, v_truth, valid_mask,
            epe_power=config.evaluation.epe_power
        )
        print(f"\n   📊 EPE Oracle sets the target: {epe_results['oracle_mean_epe_power']:.4f}")
    
    # Run each selection method
    selection_results = {}
    
    for name, params in config.selection.items():
        results = run_selection(
            selection_name=name,
            params=params,
            all_metrics=all_metrics,
            all_flows=all_flows,
            valid_mask=valid_mask,
            u_truth=u_truth,
            v_truth=v_truth,
            epe_power=config.evaluation.epe_power
        )
        selection_results[name] = results
    
    # Optimization if requested
    if args.optimize:
        from optimization.weight_optimizer import optimize_weights, save_optimization_result
        
        print("\n" + "=" * 70)
        print("OPTIMIZING WEIGHTS")
        print("=" * 70)
        
        if not has_gt:
            print("❌ ERROR: Optimization requires ground truth")
            sys.exit(1)
        
        # Determine which selections to optimize
        if args.optimize == "all":
            to_optimize = list(config.selection.keys())
        else:
            if args.optimize not in config.selection:
                print(f"❌ ERROR: Unknown selection '{args.optimize}'")
                print(f"   Available: {list(config.selection.keys())}")
                sys.exit(1)
            to_optimize = [args.optimize]
        
        for name in to_optimize:
            params = config.selection[name]
            
            print(f"\n🔧 Optimizing: {name}")
            
            # Get enabled metrics
            metric_names = params.enabled_metrics
            
            result = optimize_weights(
                all_metrics=all_metrics,
                all_flows=all_flows,
                u_truth=u_truth,
                v_truth=v_truth,
                normalize=params.normalize,
                aggregation=params.aggregation,
                power=params.power,
                valid_mask=valid_mask,
                metric_names=metric_names,
                n_trials=args.n_trials,
                study_name=f"optimize_{name}",
                storage_path=None,  # In-memory for demo
                show_progress=True
            )
            
            # Print comparison
            original_score = selection_results[name]["epe_stats"]["oracle_mean_epe_power"]
            print(f"\n   Original EPE^{params.power}: {original_score:.4f}")
            print(f"   Optimized EPE^{params.power}: {result.best_score:.4f}")
            improvement = 100 * (original_score - result.best_score) / original_score
            print(f"   Improvement: {improvement:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if has_gt:
        print(f"\n{'Method':<20} {'Oracle EPE^n':<15} {'Best Single':<15} {'vs EPE Oracle':<15}")
        print("-" * 65)
        
        epe_oracle = epe_results['oracle_mean_epe_power']
        print(f"{'EPE (target)':<20} {epe_oracle:<15.4f} {epe_results['scores'][epe_results['best_config_hash']]:<15.4f} {'—':<15}")
        
        for name, results in selection_results.items():
            if results["epe_stats"]:
                oracle = results["epe_stats"]["oracle_mean_epe_power"]
                best = results["epe_stats"]["best_mean_epe_power"]
                gap = 100 * (oracle - epe_oracle) / epe_oracle
                print(f"{name:<20} {oracle:<15.4f} {best:<15.4f} {f'+{gap:.1f}%':<15}")
    
    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
