#!/usr/bin/env python3
"""Figure: the 31 reference maps laid out by how different they are in brain space.

Each point is one evaluable reference map (selected, non-degenerate), coloured by project with the
shared palette (scripts/project_palette.py). The same 26 references as Figure 4: the "all"
targets that pool their whole project are left out; the two aggregates of sibling targets
(emotion regulation reappraisal, cue reactivity reward) stay in, drawn as diamonds.
Distance is 1 - r between the unthresholded z maps
inside the brain mask, embedded in two dimensions by metric MDS (64 inits). Axes are arbitrary:
only distances carry meaning.

WHY MDS, NOT t-SNE. With 31 maps the global layout -- which topics sit near which -- is the point
of the figure, and t-SNE does not preserve it. Shepard rho between map distance and plotted
distance is 0.83 for MDS against 0.48-0.75 for t-SNE (perplexity 5-12); local neighbourhoods
are kept about equally (trustworthiness at k = 5: 0.88 MDS, 0.87-0.95 t-SNE).

What it shows: maps cluster by project (mean r = 0.50 within a project, 0.20 between), so the
benchmark spans distinct regions of brain space rather than variants of one generic map, and
siblings within a project sit close together -- the separation a method must resolve to tell
them apart.

Writes results/figures/figure_target_map_embedding.{pdf,png} and
results/target_map_embedding.csv (coordinates, for reuse).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_text_to_map as E  # noqa: E402
import project_palette as P  # noqa: E402
from make_difficulty_profile_figure import CATCH_ALL  # noqa: E402  (same set as Figure 4)

AGGREGATE_MARKER = "D"

INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"


def load_maps():
    import nibabel as nib
    ev = {(r["project"], r["target"]): r
          for r in csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))}
    keys = [k for k in ((r["project"], r["target"])
                        for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv")))
            if ev[k]["reference_degenerate"] == "0" and k not in CATCH_ALL]
    mask = nib.load(str(NMB / "data" / "brain_mask_mni152_2mm.nii.gz")).get_fdata() > 0
    X = np.array([nib.load(str(E.reference_dir(*k) / "z.nii.gz")).get_fdata()[mask]
                  for k in keys])
    return keys, X


def embed(D: np.ndarray) -> np.ndarray:
    from sklearn.manifold import MDS
    return MDS(n_components=2, dissimilarity="precomputed", n_init=64, random_state=0,
               normalized_stress="auto").fit_transform(D)


def place_labels(fig, ax, anchors, labels, colours, fontsize=5.6, gap=5.0):
    """Greedy placement: each label takes the first of eight positions around its point that
    overlaps no placed label and no other point; failing that, the least-overlapping one."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    pts = to_px(anchors)
    pt_boxes = [Bbox.from_extents(x - 4.5, y - 4.5, x + 4.5, y + 4.5) for x, y in pts]
    px_per_pt = fig.dpi / 72
    g = gap * px_per_pt
    cands = [(g, 0, "left", "center"), (-g, 0, "right", "center"), (0, g, "center", "bottom"),
             (0, -g, "center", "top"), (g, g, "left", "bottom"), (-g, g, "right", "bottom"),
             (g, -g, "left", "top"), (-g, -g, "right", "top")]
    placed = []
    # crowded points first, so they get the best positions
    dens = [np.sum(np.hypot(*(pts - p).T) < 40 * px_per_pt) for p in pts]
    for i in sorted(range(len(pts)), key=lambda i: -dens[i]):
        best, best_cost = None, np.inf
        for dx, dy, ha, va in cands:
            t = ax.annotate(labels[i], anchors[i], xytext=(dx / px_per_pt, dy / px_per_pt),
                            textcoords="offset points", ha=ha, va=va, fontsize=fontsize,
                            color=colours[i], zorder=4)
            bb = t.get_window_extent(renderer).expanded(1.04, 1.1)
            cost = sum(_overlap(bb, o) for o in placed)
            cost += sum(_overlap(bb, pb) for k, pb in enumerate(pt_boxes) if k != i)
            ax_bb = ax.get_window_extent(renderer)
            if bb.x0 < ax_bb.x0 or bb.x1 > ax_bb.x1 or bb.y0 < ax_bb.y0 or bb.y1 > ax_bb.y1:
                cost += 1e6
            if cost < best_cost:
                if best is not None:
                    best[0].remove()
                best, best_cost = (t, bb), cost
            else:
                t.remove()
            if cost == 0:
                break
        placed.append(best[1])


def _overlap(a, b) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5,
                         "text.color": INK})
    keys, X = load_maps()
    R = np.corrcoef(X)
    Y = embed(1 - R)
    projects = np.array([k[0] for k in keys])
    iu = np.triu_indices(len(keys), 1)
    same = projects[iu[0]] == projects[iu[1]]
    r_within, r_between = R[iu][same].mean(), R[iu][~same].mean()

    with open(REPO / "results" / "target_map_embedding.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "target", "mds_1", "mds_2"])
        for k, (a, b) in zip(keys, Y):
            w.writerow([*k, round(float(a), 5), round(float(b), 5)])

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for k, (a, b) in zip(keys, Y):
        p = k[0]
        agg = k in P.AGGREGATE_LABELS
        ax.scatter([a], [b], s=38 if agg else 42, color=P.colour(p),
                   marker=AGGREGATE_MARKER if agg else P.marker(p), edgecolors="white",
                   linewidths=0.6, zorder=3)
    pad = 0.10 * np.ptp(Y, axis=0)
    ax.set_xlim(Y[:, 0].min() - pad[0], Y[:, 0].max() + pad[0])
    ax.set_ylim(Y[:, 1].min() - pad[1], Y[:, 1].max() + pad[1])
    place_labels(fig, ax, Y, [P.target_label(*k) for k in keys],
                 [P.colour(k[0]) for k in keys])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(RULE)
    ax.set_xlabel("MDS 1", color=MUTED); ax.set_ylabel("MDS 2", color=MUTED)
    handles = [Line2D([], [], ls="", marker=mk, color=c, mec="white", ms=5.5, label=lab)
               for key, (lab, _, c, mk) in P.PROJECTS.items() if key in set(projects)]
    handles.append(Line2D([], [], ls="", marker=AGGREGATE_MARKER, color=MUTED, mec="white",
                          ms=5, label="aggregate of sibling targets"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=6, handletextpad=0.3, labelspacing=0.5)
    out = REPO / "results" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_target_map_embedding.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    print(f"  wrote results/figures/figure_target_map_embedding.{{pdf,png}} "
          f"(r within {r_within:.2f}, between {r_between:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
