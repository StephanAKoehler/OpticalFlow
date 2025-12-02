# File: src/config/config_parser.py
"""
Config parser for optical flow pipeline.

Handles the new selection-based TOML structure with multiple selection methods.
"""

import tomllib
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Import from selection module (adjust path as needed)
# from src.evaluation.selection import SelectionParams


@dataclass
class SelectionParams:
    """Parameters for config selection."""
    name: str
    normalize: str  # "mad" | "none"
    aggregation: str  # "sum" | "max"
    power: float
    weights: dict[str, float]
    
    def __post_init__(self):
        valid_normalize = {"mad", "none"}
        valid_aggregation = {"sum", "max"}
        
        if self.normalize not in valid_normalize:
            print(f"❌ ERROR: normalize must be one of {valid_normalize}, got '{self.normalize}'")
            sys.exit(1)
        
        if self.aggregation not in valid_aggregation:
            print(f"❌ ERROR: aggregation must be one of {valid_aggregation}, got '{self.aggregation}'")
            sys.exit(1)
    
    @property
    def enabled_metrics(self) -> list[str]:
        """Return list of metrics with non-zero weight."""
        return [m for m, w in self.weights.items() if w > 0]
    
    def get_hash(self) -> str:
        """Generate hash of selection parameters for directory naming."""
        import hashlib
        import json
        
        data = {
            "normalize": self.normalize,
            "aggregation": self.aggregation,
            "power": self.power,
            "weights": self.weights
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:8]


@dataclass
class EvaluationParams:
    """Parameters for evaluation (when GT available)."""
    epe_power: float = 2.0


@dataclass
class PerturbationParams:
    """Parameters for perturbation analysis."""
    directions: int = 4
    magnitude: float = 1.0


@dataclass
class VisualizationParams:
    """Parameters for visualization."""
    output: str = "results/output.png"
    save_weights: bool = False
    stride: int = 12
    arrow_scale: float = 2.0
    dpi: int = 150


@dataclass 
class PipelineConfig:
    """Complete pipeline configuration."""
    # Image settings
    image: dict
    
    # Flow settings  
    flow: dict
    
    # Parameter sweep
    parameter_sweep: dict
    
    # Perturbations
    perturbations: PerturbationParams
    
    # Evaluation
    evaluation: EvaluationParams
    
    # Selection methods (multiple allowed)
    selection: dict[str, SelectionParams]
    
    # Visualization
    visualization: VisualizationParams
    
    # Optional preprocessing
    preprocessing: dict = field(default_factory=dict)
    
    # Source file path
    config_path: Optional[Path] = None


# Metric name mapping from old to new
METRIC_NAME_MAP = {
    "traction_A": "traction",
    "traction_B": "traction", 
    "consistency_A": "bidirectional",
    "consistency_B": "bidirectional",
    "photometric_A": "photometric",
    "photometric_B": "photometric",
    "displacements_sensitivity_A2B": "perturbation_rms",
    "displacements_sensitivity_B2A": "perturbation_rms",
}

# Known metric names
KNOWN_METRICS = {"traction", "perturbation_rms", "bidirectional", "photometric"}


def parse_selection_table(name: str, table: dict) -> SelectionParams:
    """
    Parse a single selection table from config.
    
    Args:
        name: Table name (e.g., "mad_sum")
        table: Dict with selection parameters
        
    Returns:
        SelectionParams instance
    """
    # Required fields
    normalize = table.get("normalize")
    if normalize is None:
        print(f"❌ ERROR: [selection.{name}] missing 'normalize' field")
        sys.exit(1)
    
    aggregation = table.get("aggregation")
    if aggregation is None:
        print(f"❌ ERROR: [selection.{name}] missing 'aggregation' field")
        sys.exit(1)
    
    power = table.get("power", 2.0)
    
    # Extract weights (all other numeric fields are weights)
    weights = {}
    for key, value in table.items():
        if key in {"normalize", "aggregation", "power"}:
            continue
        if not isinstance(value, (int, float)):
            print(f"❌ ERROR: [selection.{name}] weight '{key}' must be numeric, got {type(value)}")
            sys.exit(1)
        
        # Map old metric names to new
        metric_name = METRIC_NAME_MAP.get(key, key)
        
        if metric_name not in KNOWN_METRICS:
            print(f"⚠️  WARNING: Unknown metric '{key}' in [selection.{name}]")
        
        weights[metric_name] = float(value)
    
    # Ensure all known metrics have a weight (default 0)
    for metric in KNOWN_METRICS:
        if metric not in weights:
            weights[metric] = 0.0
    
    return SelectionParams(
        name=name,
        normalize=normalize,
        aggregation=aggregation,
        power=power,
        weights=weights
    )


def load_config(config_path: str | Path) -> PipelineConfig:
    """
    Load and parse pipeline configuration from TOML file.
    
    Args:
        config_path: Path to TOML config file
        
    Returns:
        PipelineConfig instance
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"❌ ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    
    # Parse required sections
    if "image" not in raw:
        print(f"❌ ERROR: Missing [image] section in {config_path}")
        sys.exit(1)
    
    if "flow" not in raw:
        print(f"❌ ERROR: Missing [flow] section in {config_path}")
        sys.exit(1)
    
    if "parameter_sweep" not in raw:
        print(f"❌ ERROR: Missing [parameter_sweep] section in {config_path}")
        sys.exit(1)
    
    # Parse perturbations
    pert_raw = raw.get("perturbations", {})
    perturbations = PerturbationParams(
        directions=pert_raw.get("directions", 4),
        magnitude=pert_raw.get("magnitude", 1.0)
    )
    
    # Parse evaluation
    eval_raw = raw.get("evaluation", {})
    evaluation = EvaluationParams(
        epe_power=eval_raw.get("epe_power", 2.0)
    )
    
    # Parse selection methods
    selection = {}
    selection_raw = raw.get("selection", {})
    
    if not selection_raw:
        print(f"⚠️  WARNING: No [selection.*] tables found, using defaults")
        # Create default selection
        selection["default"] = SelectionParams(
            name="default",
            normalize="mad",
            aggregation="sum", 
            power=2.0,
            weights={
                "traction": 0.0,
                "perturbation_rms": 1.0,
                "bidirectional": 1.0,
                "photometric": 1.0,
            }
        )
    else:
        for name, table in selection_raw.items():
            if not isinstance(table, dict):
                print(f"❌ ERROR: [selection.{name}] must be a table")
                sys.exit(1)
            selection[name] = parse_selection_table(name, table)
    
    # Parse visualization
    viz_raw = raw.get("visualization", {})
    visualization = VisualizationParams(
        output=viz_raw.get("output", "results/output.png"),
        save_weights=viz_raw.get("save_weights", False),
        stride=viz_raw.get("stride", 12),
        arrow_scale=viz_raw.get("arrow_scale", 2.0),
        dpi=viz_raw.get("dpi", 150)
    )
    
    return PipelineConfig(
        image=raw["image"],
        flow=raw["flow"],
        parameter_sweep=raw["parameter_sweep"],
        perturbations=perturbations,
        evaluation=evaluation,
        selection=selection,
        visualization=visualization,
        preprocessing=raw.get("preprocessing", {}),
        config_path=config_path
    )


def validate_config(config: PipelineConfig) -> None:
    """
    Validate configuration for common errors.
    
    Args:
        config: PipelineConfig to validate
    """
    # Check algorithm is specified
    if "algorithm" not in config.parameter_sweep:
        print("❌ ERROR: [parameter_sweep] must specify 'algorithm'")
        sys.exit(1)
    
    # Check at least one selection method exists
    if not config.selection:
        print("❌ ERROR: At least one [selection.*] table required")
        sys.exit(1)
    
    # Check each selection method has at least one enabled metric
    for name, params in config.selection.items():
        if not params.enabled_metrics:
            print(f"❌ ERROR: [selection.{name}] has no enabled metrics (all weights are 0)")
            sys.exit(1)
    
    print(f"✓ Config validated: {len(config.selection)} selection method(s)")


def print_config_summary(config: PipelineConfig) -> None:
    """Print human-readable config summary."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📷 Image: {config.image.get('type', 'unknown')}")
    print(f"   Size: {config.image.get('size', 'unknown')}")
    
    print(f"\n🌊 Flow: {config.flow.get('type', 'unknown')}")
    
    print(f"\n🔧 Parameter Sweep: {config.parameter_sweep.get('algorithm', 'unknown')}")
    sweep_params = {k: v for k, v in config.parameter_sweep.items() if k != 'algorithm'}
    for k, v in sweep_params.items():
        if isinstance(v, list):
            print(f"   {k}: {v}")
    
    print(f"\n📊 Perturbations: {config.perturbations.directions} dirs, mag={config.perturbations.magnitude}")
    
    print(f"\n📈 Evaluation: EPE^{config.evaluation.epe_power}")
    
    print(f"\n🎯 Selection Methods:")
    for name, params in config.selection.items():
        print(f"   [{name}]")
        print(f"      normalize={params.normalize}, aggregation={params.aggregation}, power={params.power}")
        enabled = [f"{m}={params.weights[m]}" for m in params.enabled_metrics]
        print(f"      weights: {', '.join(enabled)}")
    
    print()


if __name__ == "__main__":
    import sys
    
    print("🔧 Config Parser Test")
    print("=" * 50)
    
    # Create a test config file
    test_config = """
# Test config

[image]
type = "checkerboard"
size = [288, 288]
square_size = 25

[flow]
type = "uniform"
motion = [1, 0]

[parameter_sweep]
algorithm = "farneback"
winsize = [5, 15, 25]
poly_n = [5, 7]

[perturbations]
directions = 4
magnitude = 1.0

[evaluation]
epe_power = 2

[selection.mad_sum]
normalize = "mad"
aggregation = "sum"
power = 2
traction = 0.0
perturbation_rms = 1.0
bidirectional = 1.0
photometric = 1.0

[selection.raw_max]
normalize = "none"
aggregation = "max"
power = 2
traction = 0.0
perturbation_rms = 1.0
bidirectional = 0.5
photometric = 0.5

[visualization]
output = "results/test.png"
"""
    
    # Write test config
    test_path = Path("/tmp/test_config.toml")
    test_path.write_text(test_config)
    
    # Load and validate
    config = load_config(test_path)
    validate_config(config)
    print_config_summary(config)
    
    # Test hash generation
    print("Selection hashes:")
    for name, params in config.selection.items():
        print(f"  {name}: {params.get_hash()}")
    
    print("\n✅ Config parser test passed!")
