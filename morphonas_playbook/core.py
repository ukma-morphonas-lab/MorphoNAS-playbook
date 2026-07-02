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

# Discrete-action Gym tasks the hands-on supports out of the box. input/output
# dims match each env's observation/action spaces; `solved` is the mean-reward bar
# used only for run_ga's telemetry (higher is better for all of these). The engine
# (GymFitnessFunction) infers dims itself, so any Gym env in its PASSING_SCORES
# works too — these presets just save a gym.make() lookup and set a sensible bar.
TASKS = {
    "CartPole-v1":    {"input_dim": 4, "output_dim": 2, "solved": 475.0, "best": 500},
    "Acrobot-v1":     {"input_dim": 6, "output_dim": 3, "solved": -100.0, "best": -60},
    "MountainCar-v0": {"input_dim": 2, "output_dim": 3, "solved": -110.0, "best": -100},
    "LunarLander-v3": {"input_dim": 8, "output_dim": 4, "solved": 200.0, "best": 300},
}


def task_dims(env_name):
    """(input_dim, output_dim) for a task: from TASKS if known, else inferred from
    the Gym env's observation/action spaces (discrete-action envs)."""
    if env_name in TASKS:
        return TASKS[env_name]["input_dim"], TASKS[env_name]["output_dim"]
    import gymnasium as gym
    env = gym.make(env_name)
    obs, act = env.observation_space, env.action_space
    idim = obs.n if hasattr(obs, "n") else int(np.prod(obs.shape))
    odim = act.n if hasattr(act, "n") else int(np.prod(act.shape))
    env.close()
    return int(idim), int(odim)


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
                 input_dim=None, output_dim=None):
    """One episode; fresh recurrent state. Returns total reward, or None on failure.
    Dims default to the task's (inferred from env_name)."""
    import gymnasium as gym
    if input_dim is None or output_dim is None:
        di, do = task_dims(env_name)
        input_dim = di if input_dim is None else input_dim
        output_dim = do if output_dim is None else output_dim
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
             input_dim=None, output_dim=None):
    """Mean reward over `episodes` seeds. Returns (mean, list). Dims default to the task's."""
    if input_dim is None or output_dim is None:
        di, do = task_dims(env_name)
        input_dim = di if input_dim is None else input_dim
        output_dim = do if output_dim is None else output_dim
    rewards = []
    for s in range(episodes):
        r = rollout_once(G, env_name, base_seed + s, input_dim, output_dim)
        if r is None:
            return None, []
        rewards.append(r)
    return float(np.mean(rewards)), rewards
