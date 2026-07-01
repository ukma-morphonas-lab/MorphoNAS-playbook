"""PARTS 3 & 4: evolving CartPole controllers with a GA (the hands-on climax).

  * genomes grow into recurrent controllers  (core.grow)
  * a genetic algorithm optimises the population  (MorphoNAS GeneticAlgorithm)
  * fitness = CartPole performance, optionally times a size penalty
    (MorphoNAS GymFitnessFunction)

PART 3  (penalize_connections=False): no size pressure. Because a few percent of
    random genomes already solve CartPole (the innate-ability result), gen-0 is
    nearly pre-solved -- a 500-step solver appears within a generation or two, but
    it is a *bloated* network (~100 neurons, hundreds of connections).

PART 4  (penalize_connections=True): the same GA, but fitness now multiplies
    performance by a penalty on the number of connections. The genomic bottleneck
    bites: evolution must compress the controller toward a tiny ideal network.

The editable *fitness* is the "students change their own part" seam -- see
build_fitness() and the two scorer classes.

NOTE ON MULTIPROCESSING: the scorer classes below are top-level (picklable) so the
GA's ProcessPoolExecutor can ship them to workers under macOS/Colab 'spawn'. If a
runtime's multiprocessing misbehaves, call run_ga(..., max_workers=1) for a serial
(still fast) run.
"""
import os
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from . import core as H

from genome import Genome                                        # vendored engine
from genetic_algorithm import GeneticAlgorithm
from fitness_functions import GymFitnessFunction
from convergence_strategies import DefaultConvergenceStrategy

CARTPOLE = "CartPole-v1"


# ======================================================================
#  EDITABLE SEAM -- the part students rewrite for PART 4.
#  PART 3 -> build_fitness(penalize_connections=False)
#  PART 4 -> build_fitness(penalize_connections=True)   # add the size penalty
# ======================================================================
def build_fitness(penalize_connections=False,
                  num_rollouts=H.NUM_ROLLOUTS,
                  rollout_seed=0,
                  # penalty knobs (used only when penalize_connections=True)
                  min_connection_fitness=0.8,
                  max_unpenalized_connections=50,
                  connection_half_decay=1000):
    """Create the MorphoNAS GymFitnessFunction (the editable fitness object).

    Fitness of a grown network:
        avg_reward = mean CartPole reward over `num_rollouts` episodes
        base       = sigmoid((avg_reward - 195) / 10)     # in (0, 1), ~1.0 above ~350
        fitness    = base                                 # PART 3
        fitness    = base * connection_penalty(edges)     # PART 4

    The penalty is 1.0 up to `max_unpenalized_connections` edges, then decays
    toward `min_connection_fitness`. Lower the threshold/floor to squeeze harder
    (see the "strong" preset in __main__).
    """
    targets = {"env_name": CARTPOLE, "num_rollouts": num_rollouts, "seed": rollout_seed}
    return GymFitnessFunction(
        targets,
        penalize_connections=penalize_connections,
        min_connection_fitness=min_connection_fitness,
        max_unpenalized_connections=max_unpenalized_connections,
        connection_half_decay=connection_half_decay,
    )


class GrowScorer:
    """Picklable genome -> fitness: grow the genome, then score the network.

    Top-level class (holds only picklable state) so the GA's ProcessPoolExecutor
    can ship it to worker processes under 'spawn'.
    """

    def __init__(self, fitness_function):
        self.ff = fitness_function

    def __call__(self, genome):
        grid = H.grow(genome)
        return float(self.ff.evaluate(grid))


class SizePenaltyScorer:
    """Alternative PART-4 seam: a hand-written, picklable `genome -> float` fitness.

    Same base performance term as the engine, but a transparent *node-count*
    penalty (drives toward the 6-neuron minimum more legibly than the edge penalty):

        fitness = sigmoid((reward - 195)/10) * exp(-lam * max(0, n_neurons - 6))

    Pass run_ga(scorer=SizePenaltyScorer(lam=0.05)). Must be a top-level class.
    """

    def __init__(self, lam=0.03, num_rollouts=H.NUM_ROLLOUTS, rollout_seed=0):
        self.lam = lam
        self.num_rollouts = num_rollouts
        self.rollout_seed = rollout_seed
        self.min_nodes = H.INPUT_DIM + H.OUTPUT_DIM   # 6 = 4 inputs + 2 outputs

    def __call__(self, genome):
        grid = H.grow(genome)
        G = grid.get_graph()
        if G.number_of_nodes() < self.min_nodes:
            return 0.0
        mean, _ = H.evaluate(G, episodes=self.num_rollouts, base_seed=self.rollout_seed)
        if mean is None:
            return 0.0
        base = 1.0 / (1.0 + np.exp(-(mean - 195.0) / 10.0))
        n = int(grid.neuron_count())
        penalty = np.exp(-self.lam * max(0, n - self.min_nodes))
        return float(base * penalty)


# ======================================================================
#  GA driver
# ======================================================================
@dataclass
class GAResult:
    penalize_connections: bool
    population: int
    generations_run: int
    seed: int
    rollout_seed: int
    best_fitness_per_gen: list = field(default_factory=list)
    elite_reward_per_gen: list = field(default_factory=list)
    elite_neurons_per_gen: list = field(default_factory=list)
    elite_edges_per_gen: list = field(default_factory=list)
    wall_per_gen: list = field(default_factory=list)
    first_solver_gen: Optional[int] = None
    solver_reward_threshold: float = 475.0
    final_neurons: Optional[int] = None
    final_edges: Optional[int] = None
    final_reward: Optional[float] = None
    wall_total: float = 0.0
    best_genome_json: Optional[str] = None

    def summary(self):
        tag = "PART 4 (penalize_connections=True)" if self.penalize_connections \
            else "PART 3 (penalize_connections=False)"
        lines = [f"== {tag} | pop={self.population} seed={self.seed} =="]
        lines.append(f"wall total: {self.wall_total:.1f}s over {self.generations_run} generations "
                     f"({self.wall_total/max(1,self.generations_run):.1f}s/gen)")
        if self.first_solver_gen is not None:
            lines.append(f"first 500-step solver (reward>={self.solver_reward_threshold:.0f}): "
                         f"generation {self.first_solver_gen}")
        else:
            lines.append(f"no 500-step solver (reward>={self.solver_reward_threshold:.0f}) reached")
        lines.append(f"final elite: reward={self.final_reward:.0f}  "
                     f"neurons={self.final_neurons}  connections={self.final_edges}")
        lines.append("gen | best_fit  elite_reward  neurons  edges  wall(s)")
        for g in range(len(self.best_fitness_per_gen)):
            lines.append(f"{g:3d} | {self.best_fitness_per_gen[g]:.6f}  "
                         f"{self.elite_reward_per_gen[g]:11.0f}  "
                         f"{self.elite_neurons_per_gen[g]:7d}  "
                         f"{self.elite_edges_per_gen[g]:5d}  "
                         f"{self.wall_per_gen[g]:6.1f}")
        return "\n".join(lines)


def _elite_stats(ga, rollout_seed, num_rollouts):
    """Grow the current best-fitness genome; return (genome, reward, neurons, edges)."""
    scores = np.asarray(ga.current_fitness_scores, dtype=float)
    idx = int(np.argmax(scores))
    genome = ga.population[idx]
    grid = H.grow(genome)
    G = grid.get_graph()
    neurons = int(grid.neuron_count())
    edges = int(G.number_of_edges())
    if G.number_of_nodes() >= (H.INPUT_DIM + H.OUTPUT_DIM):
        mean, _ = H.evaluate(G, episodes=num_rollouts, base_seed=rollout_seed)
        reward = float(mean) if mean is not None else 0.0
    else:
        reward = 0.0
    return genome, reward, neurons, edges


def run_ga(penalize_connections=False,
           population=200,
           max_generations=15,
           num_rollouts=H.NUM_ROLLOUTS,
           grid=H.GRID,
           seed=12345,
           rollout_seed=0,
           max_workers=6,
           scorer=None,
           penalty_kwargs=None,
           stop_on_solver=True,
           stop_neurons=None,
           min_generations=0,
           solver_reward_threshold=475.0,
           verbose=True):
    """Evolve a population of CartPole controllers. Returns a GAResult.

    penalize_connections: PART 3 -> False, PART 4 -> True (ignored if `scorer` given).
    stop_on_solver: PART 3 stops at the first 500-step solver; PART 4 sets it False
                    to keep watching the network shrink under the size penalty.
    max_workers: parallel fitness workers; set 1 if a runtime's multiprocessing balks.
    """
    if scorer is None:
        fitness = build_fitness(penalize_connections=penalize_connections,
                                num_rollouts=num_rollouts, rollout_seed=rollout_seed,
                                **(penalty_kwargs or {}))
        scorer = GrowScorer(fitness)

    ga = GeneticAlgorithm(
        population_size=population,
        max_generations=max_generations,
        fitness_fn=scorer,
        grid_size_x=grid,
        grid_size_y=grid,
        num_morphogens=H.MORPHOGENS,
        max_growth_steps=H.MAX_GROWTH_STEPS,
        mutation_rate=0.3,
        seed=seed,
        max_workers=max_workers,
        use_tournament=True,
        tournament_size=7,
        selection_pressure=0.2,
        use_steady_state=True,
        steady_state_replace_rate=0.4,
        use_elitism=True,
        num_elite=4,
        convergence_strategy=DefaultConvergenceStrategy(convergence_threshold=1.1),
    )

    res = GAResult(penalize_connections=penalize_connections, population=population,
                   generations_run=0, seed=seed, rollout_seed=rollout_seed,
                   solver_reward_threshold=solver_reward_threshold)

    t_start = time.time()

    # gen 0: build + evaluate the initial (random) population
    ga.population = [
        Genome.random(
            ga.rng, size_x=grid, size_y=grid,
            num_morphogens=H.MORPHOGENS, max_growth_steps=H.MAX_GROWTH_STEPS)
        for _ in range(population)
    ]
    ga.fitness_scores = ga._evaluate_solutions(ga.population)

    def _record(gen, t_gen):
        _, reward, neurons, edges = _elite_stats(ga, rollout_seed, num_rollouts)
        best_fit = float(np.max(np.asarray(ga.current_fitness_scores, dtype=float)))
        res.best_fitness_per_gen.append(best_fit)
        res.elite_reward_per_gen.append(reward)
        res.elite_neurons_per_gen.append(neurons)
        res.elite_edges_per_gen.append(edges)
        res.wall_per_gen.append(t_gen)
        if res.first_solver_gen is None and reward >= solver_reward_threshold:
            res.first_solver_gen = gen
        if verbose:
            print(f"  gen {gen:2d}: best_fit={best_fit:.6f}  elite_reward={reward:5.0f}  "
                  f"neurons={neurons:3d}  edges={edges:4d}  ({t_gen:.1f}s)")
        return reward

    reward0 = _record(0, time.time() - t_start)

    solved = (reward0 >= solver_reward_threshold)
    gen = 0
    while gen < max_generations:
        if stop_on_solver and solved and gen >= min_generations:
            break
        t0 = time.time()
        cont = ga.step()
        gen = ga.generation
        reward = _record(gen, time.time() - t0)
        solved = solved or (reward >= solver_reward_threshold)
        # stop as soon as the elite is small AND still solves. The generation at
        # which compression lands shifts with hardware (floating-point drift in the
        # rollouts), so a fixed max_generations cap is brittle; this adapts.
        if (stop_neurons is not None and gen >= min_generations
                and res.elite_reward_per_gen[-1] >= solver_reward_threshold
                and res.elite_neurons_per_gen[-1] <= stop_neurons):
            break
        if not cont:
            break

    res.generations_run = gen
    res.wall_total = time.time() - t_start
    res.final_reward = res.elite_reward_per_gen[-1]
    res.final_neurons = res.elite_neurons_per_gen[-1]
    res.final_edges = res.elite_edges_per_gen[-1]
    try:
        res.best_genome_json = ga.best_solution.to_json()
    except Exception:
        res.best_genome_json = None
    return res


def save_result(res: GAResult, path):
    with open(path, "w") as f:
        json.dump(asdict(res), f, indent=2)


# strong (hands-on) size-penalty preset: clean 100 -> ~6-7 neuron compression by gen ~5
STRONG_PENALTY = dict(max_unpenalized_connections=6,
                      min_connection_fitness=0.2,
                      connection_half_decay=200)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_demo_out")
    os.makedirs(out, exist_ok=True)
    # seed=1: gen-0 has only a *bloated* 100-neuron solver, so PART 4 has to compress it.
    SEED = int(os.environ.get("GA_SEED", "1"))
    WORKERS = int(os.environ.get("GA_WORKERS", "6"))

    print("######## PART 3: no size pressure ########")
    r3 = run_ga(penalize_connections=False, population=200, max_generations=6,
                seed=SEED, max_workers=WORKERS, stop_on_solver=True, min_generations=3)
    print(r3.summary())
    save_result(r3, os.path.join(out, "part3_result.json"))

    print("\n######## PART 4: strong connection penalty ########")
    r4 = run_ga(penalize_connections=True, population=200, max_generations=15,
                seed=SEED, max_workers=WORKERS, stop_on_solver=False,
                penalty_kwargs=STRONG_PENALTY)
    print(r4.summary())
    save_result(r4, os.path.join(out, "part4_strong_result.json"))
