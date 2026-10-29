#!/usr/bin/env python3
"""Figure: text-to-map tool performance against reference reliability.

One panel per tool on identical axes, 2 x 2: NeuroQuery and NeuroVLM above, Neurosynth's two
held-out maps below (every benchmark study removed from the v7 corpus before fitting) -- association (the reverse-inference map Neurosynth shows users) and uniformity (the
forward-inference map, which asks the same "where do these studies activate?" question as the
references, and uses the same MKDA estimator). x = the reference's split-half ceiling (Spearman-Brown
corrected reliability); y = the tool's r2 against that reference, inside the brain mask. The
dashed line y = x is the ceiling itself: a perfect model of the expert synthesis is expected to
reach about the reference's reliability and no further, so the vertical gap to the line is the
explainable signal the tool misses.

Points are the selected targets with a non-empty reference (31). Colour picks out the three
projects that carry the story -- executive function and problem solving, the only topics the
tools recover, and emotion regulation, whose references are as reliable as theirs but are not
recovered -- against grey for the rest; project alone explains 81-83% of the variance in tool r2.
Triangles are structural (VBM) targets. Palette validated (three-slot categorical, worst CVD
dE 11.0, all >= 3:1 on surface).

What it shows: every tool tracks reliability and every point sits below the ceiling. Reliability
bounds the score, so a positive correlation is expected; the gap to the line is the finding.
Neurosynth uniformity sits far closer to it than the other three (mean r2 0.27 held out vs 0.07-0.15), and
Neurosynth association scores like NeuroQuery.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
HIGHLIGHT = {"executive_function": ("Executive function", "#0072B2"),
             "problem_solving": ("Problem solving", "#009E73"),
             "emotion_regulation_2022": ("Emotion regulation", "#D55E00")}
OTHER = "#B5B5B5"
INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 6.5, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    rd = lambda f: list(csv.DictReader(open(REPO / f)))
    sel = {(r["project"], r["target"]) for r in rd(NMB / "data" / "selected_targets.csv")}
    ev = {(r["project"], r["target"]): r for r in rd("results/text_to_map_baselines.csv")}
    # Neurosynth panels use the held-out maps (benchmark studies removed from the corpus
    # before fitting), so a Neurosynth term map never contains the target's own studies.
    for r in rd("results/text_to_map_baselines_neurosynth_heldout.csv"):
        for col in ("neurosynth_assoc_r2", "neurosynth_unif_r2"):
            ev[(r["project"], r["target"])][col] = r[col]
    rel = {(r["project"], r["target"]): r for r in rd("results/reference_reliability.csv")}
    keys = [k for k in sorted(sel) if ev[k]["reference_degenerate"] == "0"]

    fig, grid = plt.subplots(2, 2, figsize=(7.2, 6.4), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.08, hspace=0.16, left=0.08, right=0.98,
                                              top=0.93, bottom=0.08))
    axes = grid.ravel()
    tools = (("neuroquery_r2", "NeuroQuery"), ("neurovlm_r2", "NeuroVLM"),
             ("neurosynth_assoc_r2", "Neurosynth association (held out)"),
             ("neurosynth_unif_r2", "Neurosynth uniformity (held out)"))
    for ax, (col, name), letter in zip(axes, tools, "abcd"):
        ax.plot([0, 1], [0, 1], color=RULE, lw=0.8, ls="--", zorder=1)
        ax.text(0.90, 0.875, "ceiling", rotation=45, ha="center", va="bottom",
                fontsize=5.6, color=MUTED, rotation_mode="anchor")
        xs, ys = [], []
        # grey first so the highlighted projects draw on top
        for k in sorted(keys, key=lambda k: k[0] in HIGHLIGHT):
            v = ev[k][col]
            if v in ("", None):
                continue
            x = float(rel[k]["reliability_mean"])
            lo, hi = float(rel[k]["reliability_min"]), float(rel[k]["reliability_max"])
            y = float(v)
            xs.append(x)
            ys.append(y)
            c = HIGHLIGHT.get(k[0], (None, OTHER))[1]
            ax.plot([lo, hi], [y, y], color=c, lw=0.6, alpha=0.6, zorder=2,
                    solid_capstyle="butt")
            ax.scatter([x], [y], s=26, marker="^" if k[0].startswith("vbm") else "o",
                       color=c, edgecolors="white", linewidths=0.5, zorder=3)
        rho, p = stats.spearmanr(xs, ys)
        ax.text(0.06, 1.035, name, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.5, fontweight="bold")
        p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.06, 0.985, f"Spearman $\\rho$ = {rho:+.2f}, {p_txt}, n = {len(xs)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color=MUTED)
        ax.set_xlim(0.1, 0.95)
        ax.set_ylim(-0.02, 0.95)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(RULE)
        if letter in "cd":
            ax.set_xlabel("Reference reliability (split-half ceiling)")
        ax.text(0.0, 1.035, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontweight="bold", fontsize=8)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Tool $r^2$ against the reference")
    handles = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="white",
                      markersize=4.8, label=lab) for lab, c in HIGHLIGHT.values()]
    handles += [Line2D([], [], marker="o", ls="", color=OTHER, markeredgecolor="white",
                       markersize=4.8, label="Other projects"),
                Line2D([], [], marker="^", ls="", color=OTHER, markeredgecolor="white",
                       markersize=5.0, label="Structural (VBM)"),
                Line2D([], [], color=MUTED, lw=0.6, label="ceiling range over splits")]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.94), frameon=False, fontsize=5.4,
                   handletextpad=0.3, labelspacing=0.3)

    out = REPO / "results" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_tool_vs_reliability.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    print("  wrote results/figures/figure_tool_vs_reliability.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
