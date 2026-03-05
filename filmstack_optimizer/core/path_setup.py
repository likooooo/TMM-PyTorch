"""
Path setup so core can import torch_tmm and dep. Add repo root, artifacts, and libs to sys.path.
"""
import os
import sys

# filmstack_optimizer/core/path_setup.py -> repo root is 2 levels up
_OPTIMIZER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_OPTIMIZER_ROOT)

def setup_paths():
    """Add artifacts, libs, and repo root to sys.path so torch_tmm and dep are importable."""
    for name in ("artifacts", "libs"):
        path = os.path.join(_REPO_ROOT, name)
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)


def get_dep_path():
    """Return path to filmstack_optimizer/dep (for refractiveindex)."""
    return os.path.join(_OPTIMIZER_ROOT, "dep")
