# MorphoNAS-playbook

Hands-on tutorials for **morphogenetic neural architecture search** — compact
genomes that *grow* neural networks through simulated development, instead of
training weights from scratch.

Built on [MorphoNAS](https://github.com/sergemedvid/MorphoNAS) (Glybovets & Medvid,
*Cybernetics and Systems Analysis*, Springer 2026; [arXiv:2507.13785](https://arxiv.org/abs/2507.13785)).
A pinned snapshot of the engine is vendored in `morphonas_playbook/engine/`
(commit in `engine/PINNED_COMMIT.txt`), so the package is self-contained and
version-frozen — nothing to clone, nothing to drift.

## Workshops

### SIFR-2026 — "The Genomic Bottleneck"

3rd Summer School on Research Frontiers in Intelligent Systems (UCU, Lviv, 2026).
Notebook: [`notebooks/sifr-2026-genomic-bottleneck.ipynb`](notebooks/sifr-2026-genomic-bottleneck.ipynb) —
open it in Google Colab (CPU runtime is fine; no GPU needed).

You will:

1. **Grow** a neural network from a compact genome and watch the morphogen field
   drive cell division and axon guidance.
2. **Attach** grown networks to CartPole and discover that some already balance
   the pole **with zero learning** (the innate-ability result).
3. **Evolve** a controller with a genetic algorithm — trivially easy without size
   pressure, because a few percent of random genomes already work.
4. Add a **size penalty** and watch evolution compress a 100-neuron network down
   to a ~6-neuron ideal controller — the genomic bottleneck in action.

## Install

```bash
pip install "git+https://github.com/ukma-morphonas-lab/MorphoNAS-playbook.git"
```

```python
import morphonas_playbook as mp

# grow a random genome into a network
G = mp.graph_of(mp.grow(mp.random_genome(seed=39)))
print(mp.n_nodes(G), "neurons,", mp.n_edges(G), "connections")

# evaluate it on CartPole with no learning
mean, _ = mp.evaluate(G, episodes=20)
print("mean reward:", mean, "/ 500")
```

## API

- `mp.random_genome(seed)`, `mp.grow(genome)`, `mp.graph_of(grid)`, `mp.evaluate(G)`
- `mp.growth` — Part 1: `find_connected_genome`, `animate_growth`, `show_network`
- `mp.control` — Part 2: `attach_and_eval`, `find_innate_solver`, `render_rollout`, `measure`
- `mp.evolve` — Parts 3&4: `run_ga`, `build_fitness`, `GrowScorer`, `SizePenaltyScorer`
- `mp.assets` — pre-evolved fallback genomes: `load_genome("part3"|"part4")`

## Related repositories

- **[MorphoNAS](https://github.com/sergemedvid/MorphoNAS)** — the reference
  implementation and the engine this playbook vendors. Glybovets & Medvid,
  *MorphoNAS: Embryogenic Neural Architecture Search through Morphogen-Guided
  Development*, Cybernetics and Systems Analysis (Springer, 2026);
  [arXiv:2507.13785](https://arxiv.org/abs/2507.13785).
- **[MorphoNAS-PL](https://github.com/ukma-morphonas-lab/MorphoNAS-PL)** — the
  synaptic-plasticity / "grow-then-learn" extension: development builds the
  scaffold and a fast reward-driven loop completes it. Medvid et al., EvoSelf @
  GECCO 2026; [arXiv:2604.03386](https://arxiv.org/abs/2604.03386).

## License / citation

Engine © the MorphoNAS authors (see the
[MorphoNAS](https://github.com/sergemedvid/MorphoNAS) repository). Please cite the
MorphoNAS paper (arXiv:2507.13785) when using this material.
