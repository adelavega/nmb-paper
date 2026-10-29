#!/usr/bin/env python3
"""Figure: how well each method tells a reference apart from its own-project siblings.

The sibling version of panel a of figure_difficulty_profile. One row per evaluable reference that
has at least one evaluable sibling in its project (30; VBM of PTSD has a single target), sorted
by sibling discrimination averaged over the four methods, hardest at the top. Sibling
discrimination is the share of the target's siblings whose reference the method's map
correlates with less than with the target's own reference (0.5 is chance;
results/specificity_scores.csv, scripts/eval_specificity.py). Open marks are the individual
methods, the filled dot their mean, coloured by project (scripts/project_palette.py).

Rows are limited to references with split-half reliability of at least 0.7; scores are
unchanged, each target still being compared with all of its evaluable siblings, which the
"sib." column counts. A target has only one to four siblings, so a single method's score takes few values (with three
siblings: 0, 1/3, 2/3 or 1); the "sib." column gives the count. Siblings are the other evaluable
targets of the project, umbrella targets included.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_palette as P  # noqa: E402
from make_difficulty_profile_figure import (INK, MARKERS, MIN_RELIABILITY, MUTED,  # noqa: E402
                                            RULE, SHORT)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK,
                         "xtick.labelsize": 5.8, "ytick.labelsize": 6.0})
    sib = defaultdict(dict)
    per_project = defaultdict(set)
    for r in csv.DictReader(open(REPO / "results" / "specificity_scores.csv")):
        k = (r["project"], r["target"])
        per_project[r["project"]].add(k)
        if r["sibling_win"] not in ("", "nan"):
            sib[k][r["method"]] = float(r["sibling_win"])
    rel = {(r["project"], r["target"]): float(r["reliability_mean"])
           for r in csv.DictReader(open(REPO / "results" / "reference_reliability.csv"))}
    keys = sorted((k for k in sib if rel[k] >= MIN_RELIABILITY),
                  key=lambda k: np.mean(list(sib[k].values())))   # hardest first
    n = len(keys)
    ys = np.arange(n)[::-1]

    fig = plt.figure(figsize=(4.0, 6.0))
    gs = fig.add_gridspec(1, 2, left=0.34, right=0.98, top=0.93, bottom=0.12,
                          width_ratios=(2.4, 0.3), wspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax_n = fig.add_subplot(gs[1], sharey=ax)

    ax.axvline(0.5, color=MUTED, lw=0.7, ls="--", zorder=1)
    ax.text(0.515, n - 0.35, "chance", fontsize=5.6, color=MUTED, va="bottom")
    for y, k in zip(ys, keys):
        c = P.colour(k[0])
        vals = list(sib[k].values())
        mean = np.mean(vals)
        ax.plot([min(vals), max(vals)], [y, y], color=c, lw=0.6, alpha=0.45, zorder=1)
        for m, v in sib[k].items():
            ax.scatter([v], [y], s=9, marker=MARKERS[m], facecolors="none", edgecolors=c,
                       linewidths=0.5, alpha=0.8, zorder=2)
        ax.scatter([mean], [y], s=26, color=c, marker=P.marker(k[0]), edgecolors="white",
                   linewidths=0.5, zorder=3)
        ax_n.text(0.5, y, str(len(per_project[k[0]]) - 1), ha="center", va="center",
                  fontsize=6, color=MUTED)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Sibling discrimination\n(share of own-project siblings beaten)")
    ax.set_yticks(ys, [f"{P.PROJECTS[k[0]][1]} · {P.target_label(*k)}" for k in keys])
    for tick, k in zip(ax.get_yticklabels(), keys):
        tick.set_color(P.colour(k[0]))
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_title(f"Sibling discrimination (references with reliability ≥ {MIN_RELIABILITY})",
                 loc="left", fontsize=7,
                 fontweight="bold", x=-0.55)
    ax_n.set_xlim(0, 1)
    ax_n.set_xticks([])
    ax_n.set_title("sib.", fontsize=6.0, color=MUTED)
    ax_n.tick_params(axis="y", left=False, labelleft=False)
    for a in (ax, ax_n):
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        a.spines["bottom"].set_color(RULE)
    ax_n.spines["bottom"].set_visible(False)
    for a in (ax,):
        a.grid(axis="x", alpha=0.25, lw=0.4)
        a.set_axisbelow(True)

    handles = [Line2D([], [], ls="", marker=MARKERS[m], mfc="none", mec=MUTED, ms=3.6,
                      label=SHORT[m]) for m in MARKERS]
    handles.append(Line2D([], [], ls="", marker="o", color=MUTED, mec="white", ms=4.5,
                          label="mean"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=5,
               frameon=False, fontsize=5.8, handletextpad=0.2, columnspacing=0.9)

    out = REPO / "results" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_sibling_profile.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    print(f"  wrote results/figures/figure_sibling_profile.{{pdf,png}} ({n} targets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
