"""Pre-computed fallback assets so no live beat can fail on the day.

The GA search is the only slow/seed-sensitive step; we ship the *elite genomes*
it produces (tiny JSON) plus their compression trajectories. Growing and
rendering a cached genome is fast and deterministic, so any beat can fall back to
"here's one we evolved earlier".

  load_genome("part3") -> the bloated ~100-neuron CartPole solver (no size penalty)
  load_genome("part4") -> the compressed 7-neuron/10-edge solver (size penalty)
  load_trajectory("part4") -> per-generation neurons/edges/reward for the plot
"""
import os
import json

from genome import Genome  # vendored engine (path set in package __init__)

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")


def path(filename):
    return os.path.join(_ASSETS, filename)


def load_genome(part):
    """part in {"part3", "part4"} -> a Genome (pre-evolved elite)."""
    with open(path(f"{part}_elite.json")) as f:
        return Genome.from_json(json_str=f.read())


def load_trajectory(part):
    """part in {"part3", "part4"} -> dict of per-generation traces."""
    with open(path(f"{part}_trajectory.json")) as f:
        return json.load(f)
