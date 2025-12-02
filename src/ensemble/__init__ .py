# File: src/ensemble/__init__.py
"""
Ensemble selection and oracle computation for optical flow.
"""

from .oracle import compute_oracle_selection, build_oracle_flows

__all__ = [
    'compute_oracle_selection',
    'build_oracle_flows',
]
