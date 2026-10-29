#!/usr/bin/env python3
"""Figure: each text-to-map tool scored against three estimates of the same reference.

Every selected target is re-estimated three ways from the same expert-selected analyses:
MKDA density (the benchmark reference), and MKDA chi-squared against Neurosynth v7 with all
benchmark studies removed (scripts/run_chi2_meta.py), which yields a uniformity map (forward
inference, the density-like quantity) and an association map (reverse inference: activation
selective for this topic relative to the literature at large). One panel per tool, 2 x 2 in the
same order as the reliability figure. Each thin line is one target's r2 against the three
flavours, inside the brain mask; the black line is the mean; the dashed grey line is the mean of
the project-pooled arm (real coordinates from the target's own project), an optimistic bound on
what coordinate data can reach against each flavour.

Points are the selected targets with a non-empty reference (31). Colours as in the reliability
figure (palette validated, three-slot categorical).

What it shows: changing the reference from density to uniformity helps every tool slightly
(uniformity is density with a correction for corpus base rates). Association collapses
everything, the pooled arm included (0.68 -> 0.18), so it is a noisier, different quantity rather
than a fairer yardstick. The one like-for-like match -- Neurosynth association against the
association reference, both reverse-inference contrasts against Neurosynth v7 -- is the only arm
that scores higher on association than density (0.089 -> 0.113, Wilcoxon p = 0.14), driven by
dementia and structural targets, where removing the generic task-activation pattern helps.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

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
FLAVOURS = (("", "Density\n(reference)"), ("_uniformity", "Chi-sq.\nuniformity"),
            ("_association", "Chi-sq.\nassociation"))
TOOLS = (("neuroquery", "NeuroQuery"), ("neurovlm", "NeuroVLM"),
         ("neurosynth_assoc", "Neurosynth association"),
         ("neurosynth_unif", "Neurosynth uniformity"))


def _val(row, col):
    v = row.get(col, "")
    return float(v) if v not in ("", None) else np.nan


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    sel = {(r["project"], r["target"])
           for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv"))}
    rows = [r for r in csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))
            if (r["project"], r["target"]) in sel and r["reference_degenerate"] == "0"]
    xs = np.arange(len(FLAVOURS))
    pooled = np.nanmean([[_val(r, f"pooled_r2{s}") for s, _ in FLAVOURS] for r in rows], 0)

    fig, grid = plt.subplots(2, 2, figsize=(6.4, 6.0), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.10, hspace=0.20, left=0.09,
                                              right=0.98, top=0.94, bottom=0.08))
    axes = grid.ravel()
    for ax, (tool, name), letter in zip(axes, TOOLS, "abcd"):
        M = []
        for r in sorted(rows, key=lambda r: r["project"] in HIGHLIGHT):
            ys = [_val(r, f"{tool}_r2{s}") for s, _ in FLAVOURS]
            if np.isnan(ys).any():
                continue
            M.append(ys)
            c = HIGHLIGHT.get(r["project"], (None, OTHER))[1]
            hi = r["project"] in HIGHLIGHT
            ax.plot(xs, ys, color=c, lw=0.7 if hi else 0.5, alpha=0.75 if hi else 0.6,
                    zorder=3 if hi else 2, marker="^" if r["project"].startswith("vbm") else "o",
                    ms=2.6, mec="white", mew=0.3)
        mean = np.mean(M, 0)
        ax.plot(xs, pooled, color=MUTED, lw=1.0, ls="--", zorder=4)
        ax.plot(xs, mean, color=INK, lw=1.8, marker="o", ms=4.5, mec="white", mew=0.6, zorder=5)
        for x, m in zip(xs, mean):
            ax.text(x, m - 0.035, f"{m:.2f}", ha="center", va="top", fontsize=6.2,
                    fontweight="bold", color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
        ax.text(0.0, 1.03, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontweight="bold", fontsize=8)
        ax.text(0.06, 1.03, name, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.5, fontweight="bold")
        ax.set_xlim(-0.25, len(FLAVOURS) - 0.55)
        ax.set_ylim(-0.06, 0.92)
        ax.set_xticks(xs, [lab for _, lab in FLAVOURS])
        ax.grid(axis="y", alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(RULE)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Tool $r^2$ against the reference")
    axes[0].text(xs[0] - 0.05, pooled[0] + 0.035, "pooled-map arm (mean)", fontsize=5.8,
                 color=MUTED, ha="left", va="bottom")
    handles = [Line2D([], [], color=c, lw=0.9, marker="o", ms=3, mec="white", label=lab)
               for lab, c in HIGHLIGHT.values()]
    handles += [Line2D([], [], color=OTHER, lw=0.9, marker="o", ms=3, mec="white",
                       label="Other projects"),
                Line2D([], [], color=INK, lw=1.8, marker="o", ms=4, mec="white",
                       label="Mean over targets")]
    axes[1].legend(handles=handles, loc="upper right", frameon=False, fontsize=5.6,
                   handletextpad=0.4, labelspacing=0.3)

    out = REPO / "results" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_reference_flavours.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    print("  wrote results/figures/figure_reference_flavours.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
