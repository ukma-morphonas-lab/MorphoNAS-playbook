"""PART 1: live growth.

Notebook-ready helpers for the "watch a network grow" beat:

  find_connected_genome(...)  -> loop random genomes until one is legible,
                                 reporting how many tries it took (the live
                                 "found one at try #N" moment).
  animate_growth(genome, ...) -> a small looping GIF of the morphogen field +
                                 cells + axons developing, step by step.
  show_network(G, ...)        -> a clean networkx render of the grown net.

Reuses the engine's own cell/axon drawing (see _draw.py) so the look matches the
lecture deck visuals.
"""
import os
import time

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid import Grid  # vendored engine (path set in package __init__)

from .core import (
    random_genome, grow, graph_of,
    n_nodes, n_edges, is_connected, is_usable,
    GRID, MAX_GROWTH_STEPS,
)
from ._draw import _draw_cell_borders_and_axons

# ---- deck palette (matches figures/morphonas/*) ----
BG = "#FAF4E9"
INK = "#2C2A28"
MUTE = "#8A857E"
TEAL = "#1B6E6A"
AMBER = "#E7A83E"

HERO_SEED = 39  # curated legible example (8 neurons); students may change it


# ------------------------------------------------------------------ predicates
def legible(G, lo=6, hi=30):
    """A live-demo-friendly net: connected, big enough to be a controller,
    small enough to actually read (not a full-grid ~100-neuron blob)."""
    return is_connected(G) and lo <= n_nodes(G) <= hi


def connected(G):
    """Loosest useful predicate: connected and non-trivial (>=6 neurons)."""
    return is_usable(G)


# --------------------------------------------------------------- search helper
def find_connected_genome(start_seed=0, predicate=legible, max_tries=2000,
                          size=GRID, verbose=True):
    """Loop random genomes from `start_seed` until one grows a net matching
    `predicate`. Returns (seed, genome, grid, G, tries).

    `tries` is how many genomes we grew (the live "found one at try #N" beat).
    """
    for tries in range(1, max_tries + 1):
        seed = start_seed + tries - 1
        genome = random_genome(seed=seed, size=size)
        grid = grow(genome)
        G = graph_of(grid)
        if predicate(G):
            if verbose:
                print(f"found at try #{tries}: seed={seed} "
                      f"nodes={n_nodes(G)} edges={n_edges(G)}")
            return seed, genome, grid, G, tries
    raise RuntimeError(f"no genome matched predicate within {max_tries} tries")


# --------------------------------------------------------------- field helpers
def _rgb_field(grid):
    """Morphogen concentrations -> an RGB image (white = empty)."""
    rgb = np.ones((grid.size_x, grid.size_y, 3))
    for j in range(min(3, grid.num_morphogens)):
        rgb[:, :, j] -= np.clip(grid.get_morphogen_array(j), 0, 1)
    return np.clip(rgb, 0, 1)


# --------------------------------------------------------------- growth GIF
def animate_growth(genome, out_gif_path, size=GRID, max_step=MAX_GROWTH_STEPS,
                   stride=2, hold_last=12, downscale=2, verbose=True):
    """Replay development and write a small looping GIF of field + cells + axons.

    Captures densely while cells appear, tapers through the diffusion tail, draws
    each frame with the engine's cell/axon helper, and packs an optimized GIF via
    PIL (shared palette, downscaled). Returns out_gif_path.
    """
    from PIL import Image
    t0 = time.time()

    grid = Grid(genome)
    grid.add_cell((grid.size_x // 2, grid.size_y // 2), "progenitor")

    capture = (set(range(0, 24, 1))
               | set(range(24, 60, stride))
               | set(range(60, max_step + 1, 2 * stride)))
    capture.add(max_step)

    frames = []
    for i in range(max_step + 1):
        if i in capture:
            if i == max_step:
                grid.final_step()
            frames.append(_render_frame(grid, i))
        if i < max_step:
            grid.step()

    frames.extend([frames[-1]] * hold_last)  # hold the mature net at the end

    rgbs = []
    for arr in frames:
        im = Image.fromarray(arr).convert("RGB")
        if downscale > 1:
            im = im.resize((im.width // downscale, im.height // downscale),
                           Image.LANCZOS)
        rgbs.append(im)

    # one shared palette (from the richest final frame) keeps the title colour
    # stable and compresses consistently; dither=NONE keeps text crisp
    pal = rgbs[-1].convert("P", palette=Image.ADAPTIVE, colors=64,
                           dither=Image.NONE)
    imgs = [im.quantize(palette=pal, dither=Image.NONE) for im in rgbs]

    os.makedirs(os.path.dirname(os.path.abspath(out_gif_path)), exist_ok=True)
    imgs[0].save(out_gif_path, save_all=True, append_images=imgs[1:],
                 duration=80, loop=0, optimize=True)
    dt = time.time() - t0
    kb = os.path.getsize(out_gif_path) / 1024
    if verbose:
        print(f"animate_growth: {len(imgs)} frames, {kb:.0f} KB, "
              f"{dt:.1f}s -> {out_gif_path}")
    return out_gif_path


def _render_frame(grid, step):
    """Draw one growth frame (field + borders + axons) to an RGBA array."""
    rgb = _rgb_field(grid)
    fig, ax = plt.subplots(figsize=(5.0, 5.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("white")
    ax.imshow(rgb, vmin=0, vmax=1, interpolation="nearest")
    _draw_cell_borders_and_axons(ax, grid, rgb, scale=1.2)
    ax.set_xlim(-0.5, grid.size_x - 0.5)
    ax.set_ylim(grid.size_y - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"development — step {step}", fontsize=15, fontweight="bold",
                 color=INK, pad=8)
    ax.text(0.5, -0.03, f"{grid.neuron_count()} neurons",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, color=MUTE)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return buf


# --------------------------------------------------------------- network PNG
def _separate_overlaps(pos, tol=0.06):
    """Spread out nodes that share (near-)identical layout coordinates so every
    neuron is visible (symmetric neurons otherwise stack). Layout shape kept."""
    import math
    from collections import defaultdict
    groups = defaultdict(list)
    for node, (x, y) in pos.items():
        groups[(round(x / tol), round(y / tol))].append(node)
    for members in groups.values():
        if len(members) < 2:
            continue
        cx = np.mean([pos[m][0] for m in members])
        cy = np.mean([pos[m][1] for m in members])
        r = tol * 1.3
        for k, m in enumerate(sorted(members)):
            a = 2 * math.pi * k / len(members)
            pos[m] = np.array([cx + r * math.cos(a), cy + r * math.sin(a)])
    return pos


def show_network(G, out_png_path, title=None, dpi=72, verbose=True):
    """Render a grown network cleanly (kamada-kawai; source nodes highlighted).

    Colours by role in the graph: nodes with no incoming edge (signal sources)
    in amber, the rest teal.
    """
    mapping = {node: idx for idx, node in enumerate(sorted(G.nodes()))}
    Gv = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        Gv.add_edge(mapping[u], mapping[v], **d)
    for node in G.nodes():
        Gv.add_node(mapping[node])

    sources = [i for i in Gv.nodes() if Gv.in_degree(i) == 0]
    rest = [i for i in Gv.nodes() if i not in sources]

    try:
        pos = nx.kamada_kawai_layout(Gv)
    except Exception:
        pos = nx.spring_layout(Gv, seed=3, k=1.4, iterations=200)
    pos = _separate_overlaps(pos)

    nnodes, nedges = Gv.number_of_nodes(), Gv.number_of_edges()
    node_size = max(700, min(2600, int(30000 / max(nnodes, 1))))

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    nx.draw_networkx_edges(
        Gv, pos, ax=ax, edge_color=MUTE, width=1.6, alpha=0.7,
        arrows=True, arrowsize=14, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08", node_size=node_size)
    for nodes, col in [(rest, TEAL), (sources, AMBER)]:
        if nodes:
            nx.draw_networkx_nodes(Gv, pos, nodelist=nodes, ax=ax,
                                   node_color=col, node_size=node_size,
                                   edgecolors=BG, linewidths=2.0)
    if nnodes <= 30:
        nx.draw_networkx_labels(Gv, pos, ax=ax, font_size=11,
                                font_weight="bold", font_color="white")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=AMBER, label="source neuron (no incoming edge)"),
        Patch(facecolor=TEAL, label="downstream neuron"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2,
        frameon=False, fontsize=12, handlelength=1.1)

    if title is None:
        title = f"A grown network  ({nnodes} neurons, {nedges} connections)"
    ax.set_title(title, fontsize=19, fontweight="bold", color=INK, pad=12)
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.07)
    os.makedirs(os.path.dirname(os.path.abspath(out_png_path)), exist_ok=True)
    fig.savefig(out_png_path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    if verbose:
        kb = os.path.getsize(out_png_path) / 1024
        print(f"show_network: {nnodes} nodes {nedges} edges, {kb:.0f} KB "
              f"-> {out_png_path}")
    return out_png_path


# --------------------------------------------------------------- measurement
def measure(N=500, size=GRID, verbose=True):
    """Grow N random genomes (seeds 0..N-1) and summarize the distribution."""
    t0 = time.time()
    rows = []
    for s in range(N):
        G = graph_of(grow(random_genome(seed=s, size=size)))
        rows.append((s, n_nodes(G), n_edges(G), is_connected(G), is_usable(G)))
    dt = time.time() - t0

    nodes = np.array([r[1] for r in rows])
    conn = sum(r[3] for r in rows)
    usable = sum(r[4] for r in rows)
    b_deg = int(np.sum(nodes < 6))
    b_mid = int(np.sum((nodes >= 6) & (nodes <= 30)))
    b_lg = int(np.sum(nodes > 30))

    if verbose:
        print(f"== measure: N={N} @ {size}x{size} ==")
        print(f"grew {N} in {dt:.1f}s  ({1000*dt/N:.1f} ms/genome)")
        print(f"nodes: min={nodes.min()} med={int(np.median(nodes))} "
              f"mean={nodes.mean():.1f} max={nodes.max()}")
        print(f"buckets: degenerate <6 = {b_deg} ({100*b_deg/N:.0f}%) | "
              f"mid 6-30 = {b_mid} ({100*b_mid/N:.0f}%) | "
              f"large >30 = {b_lg} ({100*b_lg/N:.0f}%)")
        print(f"connected: {conn}/{N} ({100*conn/N:.0f}%)   "
              f"usable(>=6 & connected): {usable}/{N} ({100*usable/N:.0f}%)")
        legible_rows = sorted(
            [r for r in rows if r[3] and 6 <= r[1] <= 30],
            key=lambda r: (r[1], r[2]))
        print(f"legible (connected, 6-30 nodes): {len(legible_rows)} "
              f"({100*len(legible_rows)/N:.1f}%)")
        for seed, nn, ee, c, u in legible_rows[:10]:
            print(f"    seed={seed:4d}  nodes={nn:3d}  edges={ee:3d}")
    return dict(N=N, ms_per_genome=1000 * dt / N, nodes=nodes, connected=conn,
                usable=usable, buckets=(b_deg, b_mid, b_lg), rows=rows)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_demo_out")
    os.makedirs(out, exist_ok=True)
    hero = random_genome(seed=HERO_SEED)
    hero_grid = grow(hero)
    hero_G = graph_of(hero_grid)
    print(f"hero seed {HERO_SEED}: nodes={n_nodes(hero_G)} edges={n_edges(hero_G)}")
    animate_growth(hero, os.path.join(out, "growth.gif"))
    show_network(hero_G, os.path.join(out, "network.png"))
    print("part1 demo done")
