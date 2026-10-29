#!/usr/bin/env python3
"""Figure: reliability vs identification, one panel per automated method.

The per-method version of figure_hard_cases. x = the reference's split-half reliability; y = that
method's identification of the reference: the share of the other 30 references its map
correlates with less than with the correct one (0.5 is chance). From
results/specificity_scores.csv (scripts/eval_specificity.py). Panels in the order of the method
comparison figure: NeuroQuery, NeuroVLM, Neurosynth association, Neurosynth uniformity (both
Neurosynth maps held out: benchmark studies removed from the corpus).

Each panel has its own least-squares trend, Spearman rho and a 95% CI from a bootstrap over the
9 projects (targets within a project are not independent). The dotted lines are the median
reliability (shared) and that method's median identification; the shaded corner -- above-median
reliability, below-median identification -- marks reliable references the method misses.
Colours as in figure_hard_cases; emotion regulation targets are labelled in every panel.
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
import make_hard_cases_figure as H  # noqa: E402  (shared colours and names)
import project_palette as P  # noqa: E402

METHODS = ("NeuroQuery", "NeuroVLM", "Neurosynth association", "Neurosynth uniformity")
N_BOOT = 5000
LABELLED = ("emotion_regulation_2022",)


def project_bootstrap_ci(x, y, projects, rng):
    from scipy import stats
    groups = [np.flatnonzero(projects == q) for q in np.unique(projects)]
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, len(g)) for g in
                              (groups[k] for k in rng.integers(0, len(groups), len(groups)))])
        if np.ptp(x[idx]) > 0 and np.ptp(y[idx]) > 0:
            boots.append(stats.spearmanr(x[idx], y[idx])[0])
    return np.percentile(boots, [2.5, 97.5])


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy import stats
    from make_target_map_embedding import place_labels

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5,
                         "text.color": H.INK, "axes.labelcolor": H.INK,
                         "xtick.color": H.MUTED, "ytick.color": H.MUTED,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    rel = {(r["project"], r["target"]): float(r["reliability"])
           for r in csv.DictReader(open(REPO / "results" / "specificity_by_target.csv"))}
    ident = defaultdict(dict)
    for r in csv.DictReader(open(REPO / "results" / "specificity_scores.csv")):
        ident[r["method"]][(r["project"], r["target"])] = float(r["identification"])
    keys = sorted(rel, key=lambda k: k[0] in H.GROUPS)       # highlighted groups drawn last
    x = np.array([rel[k] for k in keys])
    projects = np.array([k[0] for k in keys])
    mx = np.median(x)
    rng = np.random.default_rng(0)

    fig, grid = plt.subplots(2, 2, figsize=(7.0, 6.4), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.08, hspace=0.24, left=0.09,
                                              right=0.99, top=0.93, bottom=0.12))
    for ax, method, letter in zip(grid.ravel(), METHODS, "abcd"):
        y = np.array([ident[method][k] for k in keys])
        my = np.median(y)
        ax.fill_between([mx, 1.0], 0, my, color=H.SHADE, zorder=0, lw=0)
        ax.axvline(mx, color=H.RULE, lw=0.6, ls=":", zorder=1)
        ax.axhline(my, color=H.RULE, lw=0.6, ls=":", zorder=1)
        ax.axhline(0.5, color=H.MUTED, lw=0.6, ls="--", zorder=1)
        b = np.polyfit(x, y, 1)
        xx = np.array([0.12, 0.98])
        ax.plot(xx, np.polyval(b, xx), color=H.INK, lw=1.4, alpha=0.45, zorder=1)
        rho, p = stats.spearmanr(x, y)
        lo, hi = project_bootstrap_ci(x, y, projects, rng)
        p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.06, 1.015, f"trend ρ = {rho:+.2f} ({p_txt}), 95% CI {lo:+.2f} to {hi:+.2f}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=5.6, color=H.MUTED)
        print(f"  {method:<24} rho {rho:+.2f} {p_txt}  CI {lo:+.2f} to {hi:+.2f}")
        lab_xy, lab_txt, lab_col = [], [], []
        for k, xi, yi in zip(keys, x, y):
            c = H.GROUPS.get(k[0], (None, H.OTHER))[1]
            hl = k[0] in H.GROUPS
            ax.scatter([xi], [yi], s=30 if hl else 16, color=c, edgecolors="white",
                       linewidths=0.5, zorder=4 if hl else 3,
                       marker="^" if k[0].startswith("vbm") else "o")
            # every point is passed so labels avoid all of them; only ER gets text
            lab_xy.append((xi, yi))
            lab_txt.append(P.target_label(*k) if k[0] in LABELLED else "")
            lab_col.append(c)
        ax.set_xlim(0.1, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(alpha=0.2, lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(H.RULE)
        ax.text(0.0, 1.075, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontweight="bold", fontsize=8)
        ax.text(0.06, 1.075, method, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.5, fontweight="bold")
        place_labels(fig, ax, np.array(lab_xy), lab_txt, lab_col, fontsize=5.6, gap=4.0)
    for ax in grid[1]:
        ax.set_xlabel("Reference reliability (split-half ceiling)")
    for ax in grid[:, 0]:
        ax.set_ylabel("Identification (share of other references beaten)")

    handles = [Line2D([], [], ls="", marker="o", color=c, mec="white", ms=5, label=lab)
               for lab, c in H.GROUPS.values()]
    handles += [Line2D([], [], ls="", marker="o", color=H.OTHER, mec="white", ms=4.5,
                       label="Other (▲ VBM)"),
                Line2D([], [], color=H.INK, lw=1.4, alpha=0.45, label="Trend"),
                Line2D([], [], color=H.MUTED, lw=0.6, ls="--", label="Chance"),
                Patch(facecolor=H.SHADE, edgecolor="none", label="Reliable but unpredicted")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.0), ncol=7,
               frameon=False, fontsize=6, handletextpad=0.3, columnspacing=1.2)

    out = REPO / "results" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_hard_cases_by_method.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    print("  wrote results/figures/figure_hard_cases_by_method.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
