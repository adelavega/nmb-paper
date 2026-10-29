#!/usr/bin/env python3
"""Figures for scripts/eval_specificity.py: which method wins, and which references are predictable.

figure_method_specificity    one panel per score (raw r2, identification, sibling
                             discrimination; residual r2 stays in the tables, not the figure); each method's mean over the 31 targets with its 95%
                             cluster-bootstrap CI over projects. Brackets mark pairs that differ
                             after Holm correction (exact sign-flip test over the 9 projects).
                             The story is the contrast between panel a and panels b-c: raw r2
                             separates the methods (it rewards sharing the generic component),
                             but on picking out the correct reference, or telling apart one
                             project's targets, no pair differs after correction.

figure_reference_predictability   one row per reference, sorted by residual r2 averaged over the
                             four methods. Small open marks are the individual methods, the
                             filled dot their mean; panel b is identification (rank of the
                             correct reference among all 31; 0.5 is chance) in the same row
                             order, panel c the reference's split-half reliability. Colours as in
                             the reliability figure.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
METHODS = ("NeuroQuery", "NeuroVLM", "Neurosynth association", "Neurosynth uniformity")
SHORT = {"NeuroQuery": "NQ", "NeuroVLM": "NVLM", "Neurosynth association": "NS assoc",
         "Neurosynth uniformity": "NS unif"}
MARKERS = {"NeuroQuery": "o", "NeuroVLM": "s", "Neurosynth association": "D",
           "Neurosynth uniformity": "^"}
SCORES = (("raw_r2", "Raw $r^2$\n(against own reference)", None),
          ("identification", "Identification\n(rank among 31 references)", 0.5),
          ("sibling_win", "Sibling discrimination\n(beats own-project siblings)", 0.5))
HIGHLIGHT = {"executive_function": ("Executive function", "#0072B2"),
             "problem_solving": ("Problem solving", "#009E73"),
             "emotion_regulation_2022": ("Emotion regulation", "#D55E00")}
OTHER = "#9A9A9A"
INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"
PROJECT_ABBR = {"executive_function": "EF", "problem_solving": "PS",
                "emotion_regulation_2022": "ER", "social": "Social",
                "decision_making": "DM", "dementia": "Dementia", "cue_reactivity": "Cue",
                "vbm_of_ptsd": "VBM PTSD", "vbm_of_substance_use": "VBM SUD"}


def label(project: str, target: str) -> str:
    import project_palette as P
    return f"{P.PROJECTS[project][1]} · {P.target_label(project, target)}"


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.set_axisbelow(True)


def method_figure(plt):
    summ = defaultdict(dict)
    for r in csv.DictReader(open(REPO / "results" / "specificity_summary.csv")):
        summ[r["score"]][r["method"]] = r
    pairs = defaultdict(list)
    for r in csv.DictReader(open(REPO / "results" / "specificity_pairwise.csv")):
        if float(r["p_holm"]) < 0.05:
            pairs[r["score"]].append(r)

    fig, axes = plt.subplots(1, len(SCORES), figsize=(6.0, 2.5), sharey=True,
                             gridspec_kw=dict(wspace=0.12, left=0.13, right=0.99,
                                              top=0.80, bottom=0.22))
    ys = np.arange(len(METHODS))[::-1]
    for ax, (score, title, chance), letter in zip(axes, SCORES, "abc"):
        hi_x = 0.0
        for y, m in zip(ys, METHODS):
            r = summ[score][m]
            mu, lo, hi = float(r["mean"]), float(r["ci_low"]), float(r["ci_high"])
            ax.plot([lo, hi], [y, y], color=MUTED, lw=1.2, solid_capstyle="butt", zorder=2)
            ax.scatter([mu], [y], s=30, color=INK, edgecolors="white", linewidths=0.6, zorder=3)
            hi_x = max(hi_x, hi)
        if chance is not None:
            ax.axvline(chance, color=RULE, lw=0.8, ls="--", zorder=1)
            ax.text(chance, ys[0] + 0.55, "chance", ha="center", va="bottom", fontsize=5.4,
                    color=MUTED)
        # significance brackets to the right of the intervals
        x0 = hi_x * 1.04 if chance is None else min(1.0, hi_x + 0.03)
        step = (hi_x if chance is None else 0.5) * 0.07
        for k, r in enumerate(sorted(pairs.get(score, []),
                                     key=lambda r: abs(METHODS.index(r["method_a"]) -
                                                       METHODS.index(r["method_b"])))):
            ya, yb = ys[METHODS.index(r["method_a"])], ys[METHODS.index(r["method_b"])]
            xb = x0 + k * step
            ax.plot([xb - step * 0.3, xb, xb, xb - step * 0.3], [ya, ya, yb, yb],
                    color=MUTED, lw=0.5)
        n_br = len(pairs.get(score, []))
        if chance is None:
            ax.set_xlim(0, x0 + (n_br + 0.5) * step)
        else:
            ax.set_xlim(0.35, 1.0)
        ax.set_ylim(-0.6, len(METHODS) - 0.2)
        ax.set_yticks(ys, METHODS)
        ax.grid(axis="x", alpha=0.25, lw=0.4)
        _style(ax)
        ax.set_title(title, fontsize=6.6, loc="left", pad=4)
        ax.text(-0.02, 1.13, letter, transform=ax.transAxes,
                ha="right", va="bottom", fontweight="bold", fontsize=8)
        ax.tick_params(axis="y", length=0)
    return fig


def predictability_figure(plt):
    from matplotlib.lines import Line2D
    from scipy import stats

    per = defaultdict(dict)
    for r in csv.DictReader(open(REPO / "results" / "specificity_scores.csv")):
        per[(r["project"], r["target"])][r["method"]] = r
    rel = {(r["project"], r["target"]): float(r["reliability_mean"])
           for r in csv.DictReader(open(REPO / "results" / "reference_reliability.csv"))}
    mean_of = lambda k, s: np.mean([float(per[k][m][s]) for m in METHODS])
    keys = sorted(per, key=lambda k: mean_of(k, "residual_r2"))
    ys = np.arange(len(keys))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 6.2), sharey=True,
                             gridspec_kw=dict(width_ratios=(1.25, 1.0, 0.55), wspace=0.14,
                                              left=0.24, right=0.99, top=0.93, bottom=0.075))
    panels = (("residual_r2", "Residual $r^2$ (target-specific signal)", None),
              ("identification", "Identification (rank among 31)", 0.5))
    for ax, (score, title, chance), letter in zip(axes, panels, "ab"):
        for y, k in zip(ys, keys):
            vals = [float(per[k][m][score]) for m in METHODS]
            c = HIGHLIGHT.get(k[0], (None, OTHER))[1]
            ax.plot([min(vals), max(vals)], [y, y], color=c, lw=0.6, alpha=0.5, zorder=1)
            for m, v in zip(METHODS, vals):
                ax.scatter([v], [y], s=9, marker=MARKERS[m], facecolors="none",
                           edgecolors=c, linewidths=0.5, alpha=0.8, zorder=2)
            ax.scatter([np.mean(vals)], [y], s=26, color=c, edgecolors="white",
                       linewidths=0.5, zorder=3)
        if chance is not None:
            ax.axvline(chance, color=RULE, lw=0.8, ls="--", zorder=0)
            ax.set_xlim(0, 1.02)
        else:
            ax.axvline(0, color=RULE, lw=0.8, zorder=0)
        ax.set_title(title, fontsize=7, loc="left", fontweight="bold")
        ax.set_xlabel("signed $r^2$ after removing each set's mean map" if chance is None else
                      "share of the other 30 references beaten", fontsize=5.8, color=MUTED)
        ax.text(-0.02, 1.012, letter, transform=ax.transAxes, ha="right", va="bottom",
                fontweight="bold", fontsize=8)
        ax.grid(axis="x", alpha=0.25, lw=0.4)
        _style(ax)
    ax = axes[2]
    for y, k in zip(ys, keys):
        c = HIGHLIGHT.get(k[0], (None, OTHER))[1]
        ax.barh(y, rel[k], height=0.6, color=c, alpha=0.75)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_title("Reliability", fontsize=7, loc="left", fontweight="bold")
    ax.text(-0.04, 1.012, "c", transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold", fontsize=8)
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    _style(ax)
    axes[0].set_yticks(ys, [label(*k) for k in keys], fontsize=6)
    axes[0].tick_params(axis="y", length=0)
    for tick, k in zip(axes[0].get_yticklabels(), keys):
        tick.set_color(HIGHLIGHT.get(k[0], (None, INK))[1])
    axes[0].set_ylim(-0.7, len(keys) - 0.3)

    rho, p = stats.spearmanr([rel[k] for k in keys], [mean_of(k, "residual_r2") for k in keys])
    p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    axes[2].set_xlabel(f"split-half ceiling\n$\\rho$ = {rho:+.2f} with a\n({p_txt})",
                       fontsize=5.8, color=MUTED)
    handles = [Line2D([], [], ls="", marker=MARKERS[m], mfc="none", mec=MUTED, ms=4,
                      label=SHORT[m]) for m in METHODS]
    handles.append(Line2D([], [], ls="", marker="o", color=MUTED, mec="white", ms=5,
                          label="mean of the four"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.24, -0.015), ncol=5,
               frameon=False, fontsize=6, handletextpad=0.2, columnspacing=1.2)
    return fig


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5,
                         "xtick.labelsize": 6, "ytick.labelsize": 6.5, "axes.linewidth": 0.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK})
    out = REPO / "results" / "figures"
    for name, make in (("figure_method_specificity", method_figure),
                       ("figure_reference_predictability", predictability_figure)):
        fig = make(plt)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=400, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote results/figures/{name}.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
