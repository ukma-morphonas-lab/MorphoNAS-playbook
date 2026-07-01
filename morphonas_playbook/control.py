"""PART 2: innate control (grow -> attach -> roll out, NO evolution).

The scientific point: a MorphoNAS genome grows a recurrent network that can
*already* control CartPole with zero learning. Sample random genomes, grow each,
attach it as an RNN controller, evaluate on CartPole-v1 (max reward 500).

Framing follows the reach-map / lecture definition (verified against
papers/developmental-priors/analysis/reach-map-findings.md):

  * a network is VALID (grown enough to be a controller) if it has >=6 neurons
    and >=5 edges  (NO connectivity requirement -- matches the reach-map).
  * it is COMPETENT if mean reward over 20 rollouts >= 200 -- the "beats a
    do-nothing null policy" bar (CartPole's null policy survives ~9 steps).
    The lecture's "~1 in 20 (4.72%)" is competent-over-valid.
  * a "perfect" solver (mean >= 475, essentially the full 500-step episode) is a
    rarer sub-corner (~1.5-2% of valid) -- that's what we render as the hero.

So the innate rate is reported over BOTH denominators: over valid grown nets
(~1 in 20, the slide number) and over all sampled genomes (~1 in 80, since only
~26% of random genomes grow into a valid controller).
"""
import os
import time

import numpy as np

from . import core as H

# CartPole bars (max reward = 500)
NULL_BAR = 200    # "beats a do-nothing policy" -- the reach-map / lecture competence bar
SOLVE = 475       # "perfect": balances essentially the whole episode (hero render)
SCREEN_EPISODES = 5     # cheap screening pass
CANON_EPISODES = H.NUM_ROLLOUTS  # 20, canonical averaging (matches ExpB config)
RECONFIRM = 150         # screen-mean above this -> re-confirm at CANON_EPISODES

HERO_SOLVER_SEED = 235  # first random genome that perfectly solves CartPole (also 1027)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------
def is_valid(G):
    """Reach-map 'validity': grown enough to wire as a controller.

    >=6 neurons (4 inputs + 2 outputs) and >=5 edges. Deliberately NO
    connectivity requirement, so the denominator matches the lecture's 4.72%.
    """
    return H.n_nodes(G) >= (H.INPUT_DIM + H.OUTPUT_DIM) and H.n_edges(G) >= 5


# ---------------------------------------------------------------------------
# Notebook-ready helpers
# ---------------------------------------------------------------------------
def attach_and_eval(genome, episodes=CANON_EPISODES, base_seed=0):
    """Grow a genome, attach it as a CartPole controller, return mean reward.

    Returns None if the genome grows into a network too small to wire or the
    propagator fails on it.
    """
    G = H.graph_of(H.grow(genome))
    if not is_valid(G):
        return None
    mean, _ = H.evaluate(G, episodes=episodes, base_seed=base_seed)
    return mean


def find_innate_solver(start_seed=0, threshold=SOLVE, episodes=CANON_EPISODES,
                       max_tries=2000, verbose=False):
    """Loop random genomes from `start_seed` until one *perfectly* solves CartPole.

    Cheap screen then confirm. Returns (seed, genome, G, mean) for the first
    genome whose confirmed mean reward >= `threshold`, else None.
    """
    for k in range(max_tries):
        seed = start_seed + k
        genome = H.random_genome(seed)
        G = H.graph_of(H.grow(genome))
        if not is_valid(G):
            continue
        screen, _ = H.evaluate(G, episodes=SCREEN_EPISODES, base_seed=0)
        if screen is None or screen < threshold:
            continue
        mean, _ = H.evaluate(G, episodes=episodes, base_seed=0)
        if mean is not None and mean >= threshold:
            if verbose:
                print(f"first solver at seed {seed} "
                      f"(after {k+1} genomes tried), mean={mean:.1f}")
            return seed, genome, G, mean
    return None


def _load_overlay_font(size=16):
    """Small legible TTF if the OS has one, else PIL's default bitmap font."""
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_rollout(G, out_gif_path, seed=0, max_steps=500, stride=3):
    """Drive CartPole with controller G and write a small, looping balancing GIF.

    rgb_array frames, PIL resize//2, adaptive palette, frame subsample (`stride`)
    to stay well under 1 MB. Each saved frame is labelled (top-left) with its TRUE
    env step index, so the count climbs 0 -> ~500 even though frames are
    subsampled. Loops forever (loop=0). Returns env steps survived.
    """
    import gymnasium as gym
    from PIL import Image, ImageDraw

    prop = H.make_propagator(G)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, done, t = [], False, 0   # frames: (true_step_index, rgb_array)
    while not done and t < max_steps:
        prop.propagate(np.asarray(obs, dtype=float).flatten())
        action = int(prop.get_output().argmax().item())
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
        if t % stride == 0:
            frames.append((t, env.render()))
        t += 1
    env.close()
    if not frames:
        print("GIF: no frames rendered")
        return 0

    INK = (44, 42, 40)   # #2C2A28, dark against CartPole's white background
    font = _load_overlay_font(16)
    imgs = []
    for step_idx, fr in frames:
        im = Image.fromarray(fr)
        im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
        ImageDraw.Draw(im).text((6, 4), f"step {step_idx}", fill=INK, font=font)
        imgs.append(im.convert("P", palette=Image.ADAPTIVE))
    os.makedirs(os.path.dirname(os.path.abspath(out_gif_path)), exist_ok=True)
    imgs[0].save(out_gif_path, save_all=True, append_images=imgs[1:],
                 duration=40, loop=0, optimize=True)
    kb = os.path.getsize(out_gif_path) / 1024
    print(f"wrote {os.path.basename(out_gif_path)}  "
          f"({len(imgs)} frames, {kb:.0f} KB, survived {t} env steps)")
    return t


# ---------------------------------------------------------------------------
# Measurement: innate rate over N random genomes (reach-map framing)
# ---------------------------------------------------------------------------
def _bucket(mean):
    if mean is None:
        return "failed"
    if mean >= SOLVE:
        return ">=475 (perfect)"
    if mean >= NULL_BAR:
        return "200-474 (competent)"
    if mean >= 50:
        return "50-199 (partial)"
    return "<50 (falls fast)"


def measure(N=500, start_seed=0, verbose=True):
    """Screen N random genomes @10x10, re-confirm candidates, report innate rates.

    Denominator = VALID grown nets (>=6 neurons, >=5 edges), matching the
    reach-map. Headline competent rate (mean>=200) over valid should land near
    the lecture's ~4.72% (~1 in 20); also reported over all genomes (~1 in 80).
    """
    t0 = time.time()
    valid_means = []
    solver_seeds = []
    n_valid = n_failed = 0

    for k in range(N):
        seed = start_seed + k
        G = H.graph_of(H.grow(H.random_genome(seed)))
        if not is_valid(G):
            continue
        n_valid += 1
        screen, _ = H.evaluate(G, episodes=SCREEN_EPISODES, base_seed=0)
        if screen is None:
            n_failed += 1
            valid_means.append(None)
            continue
        mean = screen
        if screen >= RECONFIRM:
            conf, _ = H.evaluate(G, episodes=CANON_EPISODES, base_seed=0)
            mean = conf if conf is not None else screen
        valid_means.append(mean)
        if mean is not None and mean >= SOLVE:
            solver_seeds.append((seed, mean))
    dt = time.time() - t0

    valid = [m for m in valid_means if m is not None]
    n_solve = sum(1 for m in valid if m >= SOLVE)
    n_comp = sum(1 for m in valid if m >= NULL_BAR)

    buckets = {}
    for m in valid_means:
        buckets[_bucket(m)] = buckets.get(_bucket(m), 0) + 1

    if verbose:
        print(f"\n== innate rate over N={N} random genomes @10x10 "
              f"(seeds {start_seed}..{start_seed+N-1}) ==")
        print(f"  runtime: {dt:.1f}s  ({1000*dt/N:.0f} ms/genome incl. rollouts)")
        print(f"  valid (>=6 neurons & >=5 edges): {n_valid}/{N} "
              f"({100*n_valid/N:.1f}%)   propagator-failed: {n_failed}")
        print(f"  COMPETENT (mean>=200, 'beats null'): "
              f"{n_comp}/{n_valid} over valid ({100*n_comp/max(n_valid,1):.1f}%)  |  "
              f"{n_comp}/{N} over all ({100*n_comp/N:.2f}%)")
        print(f"  PERFECT   (mean>=475): "
              f"{n_solve}/{n_valid} over valid ({100*n_solve/max(n_valid,1):.1f}%)  |  "
              f"{n_solve}/{N} over all ({100*n_solve/N:.2f}%)")
        print("  mean-reward buckets (valid genomes):")
        for b in [">=475 (perfect)", "200-474 (competent)", "50-199 (partial)",
                  "<50 (falls fast)", "failed"]:
            if b in buckets:
                print(f"      {b:22s}: {buckets[b]}")
        if solver_seeds:
            print("  perfect-solver seeds (seed: mean over 20):")
            for s, m in solver_seeds[:12]:
                print(f"      seed {s}: {m:.1f}")

    return {
        "N": N, "start_seed": start_seed, "runtime_s": dt,
        "n_valid": n_valid, "n_failed": n_failed,
        "n_competent": n_comp, "n_perfect": n_solve,
        "competent_rate_valid": n_comp / max(n_valid, 1),
        "competent_rate_all": n_comp / N,
        "solver_seeds": solver_seeds, "buckets": buckets,
    }


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_demo_out")
    os.makedirs(out, exist_ok=True)
    measure(N=300, start_seed=0)
    print("\n== hero solver render ==")
    genome = H.random_genome(HERO_SOLVER_SEED)
    G = H.graph_of(H.grow(genome))
    mean, _ = H.evaluate(G, episodes=CANON_EPISODES)
    print(f"hero seed {HERO_SOLVER_SEED}: nodes {H.n_nodes(G)} edges {H.n_edges(G)} "
          f"mean {mean:.1f}")
    steps = render_rollout(G, os.path.join(out, "cartpole-innate-solver.gif"))
    print(f"part2 demo done (survived {steps} steps)")
