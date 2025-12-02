# Self-Supervised Optical Flow Ensemble Selection

A framework for selecting optimal optical flow algorithm configurations **without ground truth** using perturbation-based uncertainty metrics.

## Key Insight

Different optical flow parameters work best in different image regions. This framework:

1. Runs multiple OF configurations in parallel
2. Evaluates each using self-supervised metrics (no ground truth needed)
3. Selects the best configuration per-pixel using ensemble methods

**Result:** 20-40% improvement over any single configuration, approaching oracle performance.

## Method

### Metrics (computed without ground truth)

| Metric | Description |
|--------|-------------|
| **Perturbation** | How much does flow change when input is shifted? Lower = more stable |
| **Consistency** | Forward-backward flow agreement |
| **Photometric** | Brightness error after warping |
| **Traction** | Flow gradient magnitude (smoothness prior) |

### Selection Strategy

For each pixel, compute a penalty from the metrics and select the configuration with minimum penalty:

```
penalty = max(w_pert * pert², w_cons * cons², w_phot * phot²)
selected_config = argmin(penalty)
```

Using `max` (not `sum`) prevents correlated metrics from double-counting.

### Validated Findings

- **raw_max** outperforms MAD-normalized versions
- Equal weights (1, 1, 1, 0) work as well as optimized weights
- Perturbation is the strongest single predictor (ρ = 0.68 with EPE)
- MAD normalization produces **negative** correlation among top configs — actively harmful

## Installation

```bash
# Clone and install dependencies
pip install numpy opencv-python matplotlib tomli optuna scipy
```

## Quick Start

```bash
# 1. Generate synthetic test sequence and run OF sweep
python scripts/run_experiment.py configs/quadrants_quick.toml

# 2. Evaluate selection methods
python scripts/evaluate_weights.py configs/quadrants_quick.toml

# 3. Analyze metric correlations
python scripts/metric_correlations.py configs/quadrants_quick.toml

# 4. Compute cycle consistency
python scripts/cycle_consistency.py configs/quadrants_quick.toml
```

## Project Structure

```
├── configs/                    # TOML experiment configurations
│   ├── quadrants_quick.toml   # Fast test (12 configs)
│   └── quadrants_farneback.toml  # Full sweep (120 configs)
│
├── scripts/                    # Analysis scripts
│   ├── run_experiment.py      # Main pipeline: generate → compute → analyze
│   ├── evaluate_weights.py    # Compare selection methods
│   ├── metric_correlations.py # Spearman correlations with EPE
│   ├── cycle_consistency.py   # Long-range flow composition test
│   └── regret_histogram.py    # Selection regret analysis
│
├── src/
│   ├── core/                  # Data structures, loading
│   ├── optical_flow/          # OF algorithms, parameter sweeps
│   ├── evaluation/            # Metrics, ground truth comparison
│   ├── ensemble/              # Selection methods
│   └── synthesis/             # Test image generation
│
└── data/                      # Generated data (git-ignored)
    └── {movie_hash}/
        ├── frames/            # Input images
        ├── ground_truth/      # True flow (for synthetic)
        └── analysis/{of_hash}/
            ├── sweep/         # Per-config results
            ├── optimization/  # Weight tuning results
            └── figures/       # Generated plots
```

## Configuration

Experiments are defined in TOML files:

```toml
[image]
size = [288, 288]

[temporal]
num_frames = 5

[sprites.upper_left]
midpoint = [72, 72]
motion = [1, 0, 0]  # (dx, dy, rotation) per frame
texture_type = "checkerboard"

[parameter_sweep]
algorithm = "farneback"
winsize = [9, 15, 21]      # 3 values
iterations = [3, 10]        # 2 values  
poly_n = [5, 7]            # 2 values
# → 12 configurations total

[evaluation]
epe_power = 2  # EPE^2 for evaluation
```

## Key Outputs

### Metric Correlations
![Metric Correlations](figures/metric_correlations.png)

Shows Spearman ρ between each metric and EPE. Top row: all configs. Bottom row: top 3 configs per pixel (the fine-grained selection regime where MAD fails).

### Cycle Consistency
![Cycle Consistency](figures/cycle_consistency.png)

Forward + backward flow composition error. Points should return to origin. raw_max achieves near-zero median error; MAD has errors everywhere.

### EPE per Pair
![EPE per Pair](figures/epe_per_pair.png)

Comparison of selection methods against oracle (per-pixel best) baseline.

## Technical Details

### Perturbation Testing

For each OF configuration, we apply symmetric input shifts and measure response:

```python
# For perturbation δ:
flow_plus  = OF(I1, shift(I2, +δ/2))
flow_minus = OF(I1, shift(I2, -δ/2))
perturbation_sensitivity = ||flow_plus - flow_minus|| / ||δ||
```

Ideal algorithm: sensitivity ≈ 1.0 (flow changes proportionally to input shift).

### Flow Composition

For cycle consistency, we compose flows through a sequence:

```python
# Forward: frame 0 → 1 → 2 → ... → N-1
# Backward: frame N-1 → N-2 → ... → 0
# Cycle error = ||final_position - start_position||
```

Requires iterative inverse warping to compute backward flow from forward flow at correct coordinates.

### Why MAX beats SUM

Perturbation and traction are highly correlated (ρ = 0.85). With sum:
```
penalty = pert² + trac² ≈ 2·pert²  # double-weighted
```

With max:
```
penalty = max(pert², trac²)  # redundancy doesn't stack
```

MAX is robust to unknown metric correlations.

## Limitations

- **Photometric** works well on synthetic data but will fail with real-world lighting changes
- **Perturbation** is a coarse filter — good at rejecting bad configs, not at ranking good ones
- **Ground truth required** for validation (but not for deployment)

## Future Work

- Test on Sintel/KITTI benchmarks with real motion blur and lighting
- Extend to learning-based OF methods (RAFT, FlowNet)
- Adaptive per-region parameter selection

## References

- Farnebäck, G. (2003). Two-frame motion estimation based on polynomial expansion.
- Sun, D., et al. (2010). Secrets of optical flow estimation and their principles.
- Baker, S., et al. (2011). A database and evaluation methodology for optical flow.
