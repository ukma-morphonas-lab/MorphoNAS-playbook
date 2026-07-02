"""MorphoNAS-playbook — hands-on tutorials for morphogenetic neural architecture search.

First workshop: SIFR-2026 "The Genomic Bottleneck". This package bundles a pinned
snapshot of the MorphoNAS engine (see ``engine/PINNED_COMMIT.txt``) plus notebook-ready
helpers, so a Colab runtime can ``pip install`` it and drive the hands-on with no setup.

The vendored engine uses flat module imports (``from grid import Grid``); on import we
put ``engine/`` on ``sys.path`` so those resolve, and set a headless SDL driver so
gymnasium/pygame can render ``rgb_array`` frames without a display.
"""
import os as _os
import sys as _sys

# headless-safe rendering for gym/pygame rgb_array (no X display on a Colab VM)
_os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# the vendored engine expects its own directory on sys.path (flat imports)
_ENGINE = _os.path.join(_os.path.dirname(__file__), "engine")
if _ENGINE not in _sys.path:
    _sys.path.insert(0, _ENGINE)

from .core import (  # noqa: E402  (engine path must be set first, above)
    random_genome, grow, graph_of, n_nodes, n_edges,
    is_connected, is_usable, make_propagator, rollout_once, evaluate,
    task_dims, TASKS,
    GRID, MORPHOGENS, MAX_GROWTH_STEPS, INPUT_DIM, OUTPUT_DIM, NUM_ROLLOUTS,
)
from . import growth, control, evolve, assets  # noqa: E402  (Parts 1, 2, 3&4 + fallbacks)

__all__ = [
    # core primitives
    "random_genome", "grow", "graph_of", "n_nodes", "n_edges",
    "is_connected", "is_usable", "make_propagator", "rollout_once", "evaluate",
    "task_dims", "TASKS",
    # part modules
    "growth", "control", "evolve", "assets",
]

__version__ = "0.1.4"
