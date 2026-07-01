"""Shared engine primitives for the hands-on: make a random genome on a 10x10
grid, grow it deterministically, inspect connectivity, attach it to a Gym control
task, and roll it out. Thin wrappers over the vendored MorphoNAS engine.

The package ``__init__`` has already put ``engine/`` on ``sys.path`` and set the
headless SDL driver, so the flat engine imports below resolve.
"""
import numpy as np
import networkx as nx

from grid import Grid
from genome import Genome
from neural_propagation import NeuralPropagator

# ---- canonical hands-on defaults (from the ExpB cartpole config) ----
GRID = 10
MORPHOGENS = 3
MAX_GROWTH_STEPS = 200
INPUT_DIM = 4      # CartPole observations
OUTPUT_DIM = 2     # CartPole actions (argmax)
NUM_ROLLOUTS = 20  # fixed averaging, matches the ExpB config


def make_rng(seed):
    return np.random.default_rng(seed)


def random_genome(seed, size=GRID, morphogens=MORPHOGENS,
                  max_growth_steps=MAX_GROWTH_STEPS):
    """A random genome constrained to a `size`x`size` lattice."""
    rng = make_rng(seed)
    return Genome.random(
        rng, size_x=size, size_y=size,
        num_morphogens=morphogens, max_growth_steps=max_growth_steps,
    )


def grow(genome):
    """Deterministic development: seed one progenitor, run to maturity."""
    grid = Grid(genome)
    grid.add_cell((grid.size_x // 2, grid.size_y // 2), "progenitor")
    for _ in range(grid.max_growth_steps):
        grid.step()
    grid.final_step()
    return grid


def graph_of(grid):
    return grid.get_graph()


def n_nodes(G):
    return G.number_of_nodes()


def n_edges(G):
    return G.number_of_edges()


def is_connected(G):
    """No unreachable nodes: non-empty and weakly connected (ignore edge dir)."""
    return G.number_of_nodes() > 0 and nx.is_weakly_connected(G)


def is_usable(G, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
    """Big enough to wire as a controller and has no floating nodes."""
    return G.number_of_nodes() >= (input_dim + output_dim) and is_connected(G)


def make_propagator(G, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
    return NeuralPropagator(
        G=G, input_dim=input_dim, output_dim=output_dim,
        activation_function=NeuralPropagator.tanh_activation,
        extra_thinking_time=2, additive_update=False, device="cpu",
    )


def rollout_once(G, env_name="CartPole-v1", seed=0,
                 input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
    """One episode; fresh recurrent state. Returns total reward, or None on failure."""
    import gymnasium as gym
    try:
        prop = make_propagator(G, input_dim, output_dim)
        env = gym.make(env_name, render_mode=None)
        obs, _ = env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            prop.propagate(np.asarray(obs, dtype=float).flatten())
            action = int(prop.get_output().argmax().item())
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            total += r
        env.close()
        return total
    except Exception:
        return None


def evaluate(G, env_name="CartPole-v1", episodes=NUM_ROLLOUTS, base_seed=0,
             input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
    """Mean reward over `episodes` seeds. Returns (mean, list). None-mean if unusable."""
    rewards = []
    for s in range(episodes):
        r = rollout_once(G, env_name, base_seed + s, input_dim, output_dim)
        if r is None:
            return None, []
        rewards.append(r)
    return float(np.mean(rewards)), rewards
