# File: src/core/data_structures.py
"""
Canonical data structures for optical flow pipeline.

This module defines the SINGLE SOURCE OF TRUTH for how data flows through
the pipeline. All other modules should use these functions to create/access
result dictionaries.

Design Principles:
1. Explicit is better than implicit
2. Parameters never change names
3. Clear separation: metadata / params / flows / metrics
4. Easy to serialize and cache
5. Type-safe access via helper functions
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import hashlib
import json
import itertools


# =============================================================================
# TOML Configuration Processing
# =============================================================================

def create_sweep_config(toml_config: dict) -> dict:
    """
    Extract sweep configuration from TOML.
    
    Separates swept parameters (lists) from fixed parameters (scalars).
    
    Args:
        toml_config: Full TOML config dict with 'parameter_sweep' section
        
    Returns:
        {
            'algorithm': str,
            'swept_params': dict,  # e.g. {'preset': ['FAST', 'MEDIUM']}
            'fixed_params': dict   # e.g. {'patch_stride': 4}
        }
        
    Example:
        >>> toml = {'parameter_sweep': {
        ...     'algorithm': 'dis',
        ...     'preset': ['FAST', 'MEDIUM'],
        ...     'iterations': [12, 16],
        ...     'patch_stride': [4]
        ... }}
        >>> config = create_sweep_config(toml)
        >>> config['swept_params']
        {'preset': ['FAST', 'MEDIUM'], 'iterations': [12, 16], 'patch_stride': [4]}
    """
    sweep_section = toml_config['parameter_sweep']
    
    # Extract algorithm
    algorithm = sweep_section['algorithm']
    if isinstance(algorithm, list):
        if len(algorithm) != 1:
            import sys
            print("❌ ERROR: Multiple algorithms not supported")
            sys.exit(1)
        algorithm = algorithm[0]
    
    # Separate swept from fixed params
    swept_params = {}
    fixed_params = {}
    
    for key, value in sweep_section.items():
        if key == 'algorithm':
            continue
        
        # Normalize to list for consistency
        if isinstance(value, list):
            swept_params[key] = value
        else:
            # Treat scalar as single-element list for uniform handling
            swept_params[key] = [value]
    
    return {
        'algorithm': algorithm,
        'swept_params': swept_params,
        'fixed_params': fixed_params  # Currently unused, but reserved
    }


def expand_sweep_configs(sweep_config: dict) -> List[dict]:
    """
    Generate all parameter combinations from sweep config.
    
    Returns flat parameter dictionaries (one per configuration).
    Keys are EXACTLY as they appear in TOML - no prefixes, no nesting.
    
    Args:
        sweep_config: Output from create_sweep_config()
        
    Returns:
        List of parameter dicts, each containing:
            - 'algorithm': str
            - All swept parameters with concrete values
            
    Example:
        >>> sweep_config = {
        ...     'algorithm': 'dis',
        ...     'swept_params': {
        ...         'preset': ['FAST', 'MEDIUM'],
        ...         'iterations': [12, 16]
        ...     },
        ...     'fixed_params': {}
        ... }
        >>> configs = expand_sweep_configs(sweep_config)
        >>> len(configs)
        4
        >>> configs[0]
        {'algorithm': 'dis', 'preset': 'FAST', 'iterations': 12}
    """
    algorithm = sweep_config['algorithm']
    swept = sweep_config['swept_params']
    
    # Get all combinations of swept params
    param_names = sorted(swept.keys())  # Sort for deterministic ordering
    param_values = [swept[k] for k in param_names]
    
    configs = []
    for combo in itertools.product(*param_values):
        # Create flat dict with algorithm + all params
        params = {'algorithm': algorithm}
        for name, value in zip(param_names, combo):
            params[name] = value
        configs.append(params)
    
    return configs


def get_swept_param_names(sweep_config: dict) -> List[str]:
    """
    Get sorted list of parameter names being swept.
    
    Returns parameter names in alphabetical order for consistent visualization.
    
    Args:
        sweep_config: Output from create_sweep_config()
        
    Returns:
        Sorted list of parameter names (excludes 'algorithm')
        
    Example:
        >>> config = {'swept_params': {'preset': [...], 'iterations': [...]}}
        >>> get_swept_param_names(config)
        ['iterations', 'preset']
    """
    return sorted(sweep_config['swept_params'].keys())


# =============================================================================
# Result Dictionary Creation
# =============================================================================

def create_result_dict(params: dict,
                      flows: dict,
                      metrics: dict,
                      config_name: str,
                      config_id: str = None) -> dict:
    """
    Create canonical result dictionary with strict structure.
    
    This is the ONLY way to create result dicts. All results should use
    this structure for consistency.
    
    Args:
        params: Algorithm parameters (flat dict, EXACTLY as in TOML)
                Must include 'algorithm' key
        flows: Flow field arrays, keys: u_AB, v_AB, u_BA, v_BA, u_sym_A, etc.
        metrics: Metric arrays, keys: traction_A, consistency_A, etc.
        config_name: Human-readable identifier (e.g., "DIS_F_sc0_it16_p8")
        config_id: Hash identifier (auto-generated if None)
        
    Returns:
        Structured dict with four top-level keys:
        {
            'metadata': {
                'config_id': str,      # 12-char hash
                'config_name': str,    # Human-readable
                'algorithm': str       # Algorithm type
            },
            'params': dict,            # Flat param dict (no 'algorithm' key)
            'flows': dict,             # Flow field arrays
            'metrics': dict            # Metric arrays
        }
        
    Example:
        >>> params = {'algorithm': 'dis', 'preset': 'FAST', 'iterations': 16}
        >>> flows = {'u_AB': np.zeros((10, 10)), 'v_AB': np.zeros((10, 10))}
        >>> metrics = {'traction_A': np.zeros((10, 10))}
        >>> result = create_result_dict(params, flows, metrics, 'DIS_F_it16')
        >>> result['metadata']['algorithm']
        'dis'
        >>> result['params']['preset']
        'FAST'
        >>> 'algorithm' in result['params']
        False
    """
    # Generate config_id if not provided
    if config_id is None:
        # Hash params for deterministic ID
        params_str = json.dumps(params, sort_keys=True)
        config_id = hashlib.sha256(params_str.encode()).hexdigest()[:12]
    
    # Extract algorithm and create params dict without it
    algorithm = params['algorithm']
    params_clean = {k: v for k, v in params.items() if k != 'algorithm'}
    
    return {
        'metadata': {
            'config_id': config_id,
            'config_name': config_name,
            'algorithm': algorithm
        },
        'params': params_clean,  # No 'algorithm' key here
        'flows': flows,
        'metrics': metrics
    }


# =============================================================================
# Accessor Functions (Type-Safe Extraction)
# =============================================================================

def extract_metadata(result: dict) -> dict:
    """
    Extract metadata from result dict.
    
    Args:
        result: Canonical result dict
        
    Returns:
        Metadata dict with keys: config_id, config_name, algorithm
    """
    return result['metadata']


def extract_params(result: dict) -> dict:
    """
    Extract parameters from result dict.
    
    Returns flat dict with parameter values (no 'algorithm' key).
    
    Args:
        result: Canonical result dict
        
    Returns:
        Parameters dict (flat, no nesting)
        
    Example:
        >>> result = create_result_dict(...)
        >>> params = extract_params(result)
        >>> params['preset']
        'FAST'
    """
    return result['params']


def extract_flows(result: dict) -> dict:
    """
    Extract flow fields from result dict.
    
    Args:
        result: Canonical result dict
        
    Returns:
        Flows dict with keys: u_AB, v_AB, u_BA, v_BA, u_sym_A, v_sym_A, etc.
    """
    return result['flows']


def extract_metrics(result: dict) -> dict:
    """
    Extract metrics from result dict.
    
    Args:
        result: Canonical result dict
        
    Returns:
        Metrics dict with keys: traction_A, consistency_A, photometric_A, etc.
    """
    return result['metrics']


def get_config_name(result: dict) -> str:
    """Get human-readable config name."""
    return result['metadata']['config_name']


def get_config_id(result: dict) -> str:
    """Get hash-based config identifier."""
    return result['metadata']['config_id']


def get_algorithm(result: dict) -> str:
    """Get algorithm type."""
    return result['metadata']['algorithm']


# =============================================================================
# Backward Compatibility Adapters
# =============================================================================

def flatten_for_selection(result: dict) -> dict:
    """
    Flatten result for ensemble selection (backward compatibility).
    
    The selection module expects flows and metrics at the top level.
    This adapter maintains compatibility while we migrate.
    
    Args:
        result: Canonical result dict
        
    Returns:
        Flattened dict with structure:
        {
            'config_name': str,
            'u_AB': array, 'v_AB': array, ...  # All flows
            'traction_A': array, ...           # All metrics
        }
    """
    return {
        'config_name': result['metadata']['config_name'],
        **result['flows'],
        **result['metrics']
    }


def flatten_for_visualization(result: dict) -> dict:
    """
    Flatten result for legacy visualization code.
    
    Some visualization functions expect params at top level.
    Use this adapter during migration.
    
    Args:
        result: Canonical result dict
        
    Returns:
        Flattened dict with params, flows, and metrics at top level
    """
    return {
        'config_name': result['metadata']['config_name'],
        'algorithm': result['metadata']['algorithm'],
        **result['params'],
        **result['flows'],
        **result['metrics']
    }


# =============================================================================
# Validation
# =============================================================================

def validate_result_dict(result: dict, raise_on_error: bool = True) -> bool:
    """
    Validate result dictionary structure.
    
    Checks for required keys and basic type correctness.
    
    Args:
        result: Dict to validate
        raise_on_error: If True, raise ValueError on validation failure
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If validation fails and raise_on_error=True
    """
    errors = []
    
    # Check top-level keys
    required_keys = {'metadata', 'params', 'flows', 'metrics'}
    missing = required_keys - set(result.keys())
    if missing:
        errors.append(f"Missing top-level keys: {missing}")
    
    # Check metadata structure
    if 'metadata' in result:
        meta_required = {'config_id', 'config_name', 'algorithm'}
        meta_missing = meta_required - set(result['metadata'].keys())
        if meta_missing:
            errors.append(f"Missing metadata keys: {meta_missing}")
    
    # Check params is dict
    if 'params' in result and not isinstance(result['params'], dict):
        errors.append(f"params must be dict, got {type(result['params'])}")
    
    # Check flows is dict
    if 'flows' in result and not isinstance(result['flows'], dict):
        errors.append(f"flows must be dict, got {type(result['flows'])}")
    
    # Check metrics is dict
    if 'metrics' in result and not isinstance(result['metrics'], dict):
        errors.append(f"metrics must be dict, got {type(result['metrics'])}")
    
    if errors:
        if raise_on_error:
            raise ValueError("Result dict validation failed:\n  " + "\n  ".join(errors))
        return False
    
    return True


# =============================================================================
# Serialization Helpers
# =============================================================================

def result_to_json_serializable(result: dict) -> dict:
    """
    Convert result dict to JSON-serializable format.
    
    Converts numpy arrays to lists for JSON serialization.
    
    Args:
        result: Canonical result dict
        
    Returns:
        Dict with arrays converted to lists
    """
    def convert_arrays(obj):
        if isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    return convert_arrays(result)


if __name__ == "__main__":
    print("✅ Data structures module loaded")
    print("\n📋 Testing basic functionality...")
    
    # Test sweep config creation
    test_toml = {
        'parameter_sweep': {
            'algorithm': 'dis',
            'preset': ['FAST', 'MEDIUM'],
            'iterations': [12, 16],
            'patch_stride': [4]
        }
    }
    
    sweep_config = create_sweep_config(test_toml)
    print(f"\n✅ Sweep config created")
    print(f"   Algorithm: {sweep_config['algorithm']}")
    print(f"   Swept params: {list(sweep_config['swept_params'].keys())}")
    
    # Test config expansion
    configs = expand_sweep_configs(sweep_config)
    print(f"\n✅ Expanded to {len(configs)} configurations")
    print(f"   First config: {configs[0]}")
    
    # Test result dict creation
    test_params = {'algorithm': 'dis', 'preset': 'FAST', 'iterations': 16}
    test_flows = {'u_AB': np.zeros((10, 10)), 'v_AB': np.zeros((10, 10))}
    test_metrics = {'traction_A': np.zeros((10, 10))}
    
    result = create_result_dict(test_params, test_flows, test_metrics, 'TEST_CONFIG')
    print(f"\n✅ Result dict created")
    print(f"   Config ID: {result['metadata']['config_id']}")
    print(f"   Params keys: {list(result['params'].keys())}")
    
    # Test validation
    is_valid = validate_result_dict(result)
    print(f"\n✅ Validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Test accessors
    params = extract_params(result)
    flows = extract_flows(result)
    metrics = extract_metrics(result)
    print(f"\n✅ Accessors work correctly")
    
    # Test backward compatibility
    flattened = flatten_for_selection(result)
    print(f"\n✅ Backward compatibility adapter works")
    print(f"   Flattened keys: {list(flattened.keys())[:5]}...")
    
    print("\n✨ All data structure tests passed!")
