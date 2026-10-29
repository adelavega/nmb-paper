#!/usr/bin/env python3
"""Figure: the overlap plane -- study overlap x analysis overlap, with map similarity as colour.

Every within-project pair of targets (n = 71) is one point: how much of their STUDY sets they
share (x) against how much of their ANALYSIS sets they share (y), coloured by how similar their
reference maps are. The plane reads as quadrants:

    upper right   shared studies AND analyses       -> similar maps   (n = 22, mean r = +0.86)
    lower right   shared studies, disjoint analyses -> dissimilar     (n =  2, mean r = -0.04)
    lower left    disjoint studies and analyses     -> intermediate   (n = 45, mean r = +0.40)

The lower-right quadrant is the resource's argument in two points: emotion-regulation targets
built from the same papers whose maps are uncorrelated, a difference no article inclusion list
can record.

DESIGN NOTES, LEARNED FROM THE DATA RATHER THAN ASSUMED

- Analysis overlap is NOT bounded by study overlap: 11 of 71 pairs sit slightly above the y = x
  diagonal, because Jaccard denominators differ when an umbrella target nests another. So the
  diagonal is drawn as a reference, not as the edge of a forbidden region.
- 12 pairs fall at exactly (0, 0) with map r spanning +0.04 to +0.51. Coordinate stacks take a
  small deterministic ring offset (radius 0.022) so every pair stays visible; the offset is
  cosmetic and identical across runs.
- Map similarity is encoded as COLOUR (the validated single-hue ramp shared with the matrix
  figure), not circle size: the claim is a gradient -- dark along the diagonal, pale at lower
  right -- and luminance shows gradients where area does not. Two pairs with slightly negative
  r (min -0.05) are shown at the ramp's zero end; the caption says so.
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RAMP = ["#8ABADB", "#6EA7CE", "#5392BF", "#3A7CAC", "#246495", "#124D79", "#06365A"]
INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"
Q = 0.3          # quadrant guide
JITTER_R = 0.022  # ring radius for exact coordinate stacks


def ring_offsets(n: int):
    if n == 1:
        return [(0.0, 0.0)]
    return [(JITTER_R * math.cos(2 * math.pi * k / n),
             JITTER_R * math.sin(2 * math.pi * k / n)) for k in range(n)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=REPO / "results" / "figures")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 6.5, "axes.linewidth": 0.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })
    cmap = LinearSegmentedColormap.from_list("nmb_seq", RAMP)

    rows = list(csv.DictReader(open(REPO / "results" / "target_structure.csv")))
    sj = np.array([float(r["study_jaccard"]) for r in rows])
    aj = np.array([float(r["analysis_jaccard"]) for r in rows])
    mr = np.array([float(r["map_r"]) for r in rows])

    # spread exact coordinate stacks on a small ring so none of the 71 pairs hides another
    key = [(round(x, 3), round(y, 3)) for x, y in zip(sj, aj)]
    groups = collections.defaultdict(list)
    for i, k in enumerate(key):
        groups[k].append(i)
    xs, ys = sj.astype(float).copy(), aj.astype(float).copy()
    for k, idxs in groups.items():
        for (dx, dy), i in zip(ring_offsets(len(idxs)), idxs):
            xs[i] += dx
            ys[i] += dy

    fig, ax = plt.subplots(figsize=(3.6, 3.35))
    fig.subplots_adjust(left=0.15, right=0.86, top=0.93, bottom=0.14)

    ax.axhline(Q, color=RULE, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.axvline(Q, color=RULE, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.text(0.80, 0.845, "analyses mirror studies", rotation=45, fontsize=5.0,
            color=MUTED, ha="center", va="center",
            rotation_mode="anchor", transform=ax.transData)

    sc = ax.scatter(xs, ys, c=np.clip(mr, 0, 1), cmap=cmap, vmin=0, vmax=1,
                    s=30, edgecolors="white", linewidths=0.5, zorder=3)

    def region(label_xy, ha, cond, name):
        sel = mr[cond]
        if len(sel):
            ax.text(*label_xy, f"{name}\nn = {len(sel)} · $\\bar r$ = {st.mean(sel):+.2f}",
                    ha=ha, va="center", fontsize=5.3, color=INK, linespacing=1.45, zorder=4)

    # Quadrant position carries the semantics, so the labels can be terse -- the long forms
    # collided with the diagonal cluster in the lower-left quadrant.
    region((0.97, 0.42), "right", (sj >= Q) & (aj >= Q), "shared both")
    region((0.97, 0.12), "right", (sj >= Q) & (aj < Q), "same studies,\ndisjoint analyses")
    # The lower-left quadrant's upper band is occupied by the diagonal itself, so this label
    # stacks three short lines hard against the axis where nothing plots.
    ll = mr[(sj < Q) & (aj < Q)]
    ax.text(0.012, 0.255, f"disjoint both\nn = {len(ll)}\n$\\bar r$ = {st.mean(ll):+.2f}",
            ha="left", va="center", fontsize=5.3, color=INK, linespacing=1.45, zorder=4)

    ax.set_xlim(-0.045, 1.045)
    ax.set_ylim(-0.045, 1.045)
    ax.set_xlabel("Study overlap between targets (Jaccard)", fontsize=6.5)
    ax.set_ylabel("Analysis overlap between targets (Jaccard)", fontsize=6.5)
    ax.text(1.0, 1.015, "71 within-project target pairs", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.4, color=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_axisbelow(True)

    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Spatial correlation between reference maps ($r$)", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.2, length=2, color=MUTED)
    cb.outline.set_color(RULE)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / f"figure_overlap_quadrants.{ext}", dpi=400,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {args.output_dir.relative_to(REPO)}/figure_overlap_quadrants.{{pdf,png}}")
    print(f"  {len(rows)} pairs; {sum(1 for v in groups.values() if len(v) > 1)} coordinate "
          f"stacks spread on rings; {int((aj > sj + 1e-9).sum())} pairs above the diagonal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
