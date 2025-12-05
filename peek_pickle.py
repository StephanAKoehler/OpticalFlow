#!/usr/bin/env python3
# File: peek_pickle.py
"""Quick peek at results_full.pkl structure."""

import pickle
import sys
from pathlib import Path


def describe_value(v, indent=""):
    """Describe a single value."""
    if hasattr(v, 'shape'):
        return f"{type(v).__name__} {v.shape} {v.dtype}"
    elif isinstance(v, dict):
        return f"dict[{len(v)}] keys={list(v.keys())[:5]}{'...' if len(v) > 5 else ''}"
    elif isinstance(v, list):
        return f"list[{len(v)}]"
    else:
        return f"{type(v).__name__} = {repr(v)[:50]}"


def peek(path: Path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"📦 {path}")
    print(f"   Type: {type(data).__name__}")
    print()
    
    # Handle list (e.g., list of config results)
    if isinstance(data, list):
        print(f"   Length: {len(data)}")
        if not data:
            return
        
        first = data[0]
        print(f"   First item: {type(first).__name__}")
        print()
        
        if isinstance(first, dict):
            print("   First item contents:")
            for k, v in first.items():
                print(f"      {k}: {describe_value(v)}")
    
    # Handle dict
    elif isinstance(data, dict):
        print(f"   Keys: {list(data.keys())}")
        print()
        
        for key, val in data.items():
            print(f"   [{key}]: {describe_value(val)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python peek_pickle.py <path_to_pickle>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"❌ Not found: {path}")
        sys.exit(1)
    
    peek(path)
