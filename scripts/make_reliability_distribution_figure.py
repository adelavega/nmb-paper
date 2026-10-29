#!/usr/bin/env python3
"""Supplementary figure: split-half reliability of every benchmark reference.

One row per benchmark target (35), sorted by mean reliability. The dot is the Spearman-Brown
corrected split-half correlation averaged over ten random study-level splits; the whisker spans
the minimum and maximum across splits. Colour = project (scripts/project_palette.py); diamonds
mark the two aggregates of sibling targets (reappraisal, cue reward); hollow symbols mark the
four degenerate references, which have no voxels surviving FDR correction and are excluded from
evaluation. The dashed line is the 0.6 threshold used to define reliable references in Figure 4.
The right-hand column gives each target's number of contributing studies.

Data: results/reference_reliability.csv (scripts/compute_reference_reliability.py).
Writes results/figures/figure_reliability_distribution.{pdf,png}.
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
import project_palette as P  # noqa: E402

INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"
THRESHOLD = 0.6
AGGREGATE_MARKER = "D"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.7,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    sel = [(r["project"], r["target"])
           for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv"))]
    rows = {(r["project"], r["target"]): r
            for r in csv.DictReader(open(REPO / "results" / "reference_reliability.csv"))}
    keys = sorted(sel, key=lambda k: float(rows[k]["reliability_mean"]))   # lowest at bottom
    n = len(keys)
    ys = np.arange(n)

    fig = plt.figure(figsize=(5.6, 7.6))
    gs = fig.add_gridspec(1, 2, left=0.36, right=0.97, top=0.95, bottom=0.10,
                          width_ratios=(1.0, 0.11), wspace=0.04)
    ax = fig.add_subplot(gs[0])
    ax_n = fig.add_subplot(gs[1], sharey=ax)

    ax.axvline(THRESHOLD, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(THRESHOLD + 0.01, n - 0.3, f"reliable (≥ {THRESHOLD})", fontsize=7, color=MUTED,
            va="bottom", ha="left")
    for y, k in zip(ys, keys):
        r = rows[k]
        p = k[0]
        c = P.colour(p)
        mean, lo, hi = (float(r[f"reliability_{s}"]) for s in ("mean", "min", "max"))
        degenerate = r["reference_degenerate"] == "1"
        ax.plot([lo, hi], [y, y], color=c, lw=1.4, alpha=0.6, zorder=2, solid_capstyle="butt")
        mk = AGGREGATE_MARKER if k in P.AGGREGATE_LABELS else P.marker(p)
        ax.scatter([mean], [y], s=46 if mk == "o" else 40, marker=mk,
                   facecolors="white" if degenerate else c, edgecolors=c,
                   linewidths=1.2 if degenerate else 0.6, zorder=3)
        ax_n.text(0.5, y, r["n_studies"], ha="center", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(ys, [f"{P.PROJECTS[k[0]][1]} · {P.target_label(*k)}" for k in keys])
    for tick, k in zip(ax.get_yticklabels(), keys):
        tick.set_color(P.colour(k[0]))
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.8, n - 0.2 + 0.6)
    lo_all = min(float(rows[k]["reliability_min"]) for k in keys)
    ax.set_xlim(min(-0.05, lo_all - 0.03), 1.0)
    ax.set_xlabel("Split-half reliability\n(Spearman–Brown corrected; whiskers: range over 10 splits)")
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax_n.set_title("studies", fontsize=7, color=MUTED, pad=2)
    ax_n.set_xticks([])
    ax_n.tick_params(axis="y", left=False, labelleft=False)
    for sp in ax_n.spines.values():
        sp.set_visible(False)

    handles = [Line2D([], [], ls="", marker="o", color=MUTED, mec=MUTED, ms=5.5,
                      label="reference"),
               Line2D([], [], ls="", marker=AGGREGATE_MARKER, color=MUTED, mec=MUTED, ms=5,
                      label="aggregate of sibling targets"),
               Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, mew=1.2, ms=5.5,
                      label="degenerate (excluded)")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3,
               frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)

    out = REPO / "results" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_reliability_distribution.{ext}", dpi=400,
                    bbox_inches="tight", facecolor="white")
    v = [float(rows[k]["reliability_mean"]) for k in keys if rows[k]["reference_degenerate"] == "0"]
    print(f"  wrote results/figures/figure_reliability_distribution.{{pdf,png}} "
          f"({n} targets; evaluable mean {np.mean(v):.2f}, median {np.median(v):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
