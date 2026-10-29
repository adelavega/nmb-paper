#!/usr/bin/env python3
"""Figure: how distinct are the benchmark's targets, and what makes them differ?

PANEL A -- a 38 x 38 spatial similarity matrix between target reference maps, ordered by
project so the block structure is visible. Shows that targets resemble their siblings
(r = +0.55) more than strangers (r = +0.18) without being interchangeable.

PANELS B AND C -- the same 71 within-project pairs on identical axes, against STUDY overlap (b)
and ANALYSIS overlap (c). The argument is the difference between the panels: b has catastrophic
outliers -- the emotion-regulation pairs share ~60% of their studies yet their maps are
uncorrelated -- and in c those same pairs fall exactly where their zero analysis overlap predicts.
Study overlap mispredicts; analysis overlap does not (rho +0.73 vs +0.88). Panel c also carries
the converse phenomenon: executive-function pairs with no shared evidence whose maps still
correlate at r ~ 0.7, convergent anatomy rather than shared curation.

An earlier version drew both measures on one axis as a paired dumbbell. It was replaced: 65 of
71 segments were near zero length (mean |analysis J - study J| = 0.048), the shared axis could
no longer name what it measured, and the statistical claim lived in a text box rather than in
the geometry. Two panels whose only difference is the x variable make the comparison itself the
picture.

DESIGN NOTES

Panel A uses a single-hue sequential ramp, not a diverging one: only 5% of off-diagonal values
are negative and none below -0.06, so the data is magnitude, not polarity, and a diverging scale
would invent a midpoint the data does not have. The ramp's light end clears the panel surface at
2.02:1 so the lowest cells remain visible as cells.

Panel B carries two series, so it carries a legend; the discordant pairs are direct-labelled
rather than every point. Both palettes pass the six-check validator (categorical: worst adjacent
CVD dE 21.9; ordinal: monotone lightness, light end 2.02:1).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))

# Single-hue sequential ramp, validated: monotone L, adjacent dL >= 0.06, light end 2.02:1.
RAMP = ["#8ABADB", "#6EA7CE", "#5392BF", "#3A7CAC", "#246495", "#124D79", "#06365A"]
# Categorical pair, validated: CVD dE 21.9 (protan), normal dE 31.2, both >= 3:1 on surface.
ACCENT = "#D55E00"   # emotion regulation -- the discordant project
BASE = "#0072B2"     # every other project
INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"

SHORT = {
    "cue_reactivity": "Cue reactivity", "decision_making": "Decision making",
    "dementia": "Dementia", "emotion_regulation_2022": "Emotion regulation",
    "executive_function": "Executive function", "problem_solving": "Problem solving",
    "social": "Social cognition", "vbm_of_ptsd": "VBM PTSD",
    "vbm_of_substance_use": "VBM substance use",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=REPO / "results" / "figures")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 6.5, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })
    cmap = LinearSegmentedColormap.from_list("nmb_seq", RAMP)

    # ---- data
    mrows = list(csv.reader(open(REPO / "results" / "target_similarity_matrix.csv")))
    names = mrows[0][1:]
    M = np.array([[float(x) for x in r[1:]] for r in mrows[1:]])
    proj = [n.split("/")[0] for n in names]
    order = sorted(range(len(names)), key=lambda i: (proj[i], names[i]))
    M, names, proj = M[np.ix_(order, order)], [names[i] for i in order], [proj[i] for i in order]

    pairs = list(csv.DictReader(open(REPO / "results" / "target_structure.csv")))

    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.22, 1], hspace=0.16, wspace=0.38,
                          left=0.14, right=0.975, top=0.91, bottom=0.14)

    # ---------- Panel A
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    bounds, start = [], 0
    for i in range(1, len(proj) + 1):
        if i == len(proj) or proj[i] != proj[start]:
            bounds.append((start, i - 1, proj[start]))
            start = i
    for s, e, _ in bounds[:-1]:
        for v in (e + 0.5,):
            ax.axhline(v, color="white", lw=1.1)
            ax.axvline(v, color="white", lw=1.1)
    ax.set_xticks([(s + e) / 2 for s, e, _ in bounds])
    ax.set_xticklabels([SHORT.get(p, p) for _, _, p in bounds], rotation=42,
                       ha="right", fontsize=5.4)
    ax.set_yticks([(s + e) / 2 for s, e, _ in bounds])
    ax.set_yticklabels([SHORT.get(p, p) for _, _, p in bounds], fontsize=5.4)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_color(RULE)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Spatial correlation between reference maps ($r$)", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.4, length=2, color=MUTED)
    cb.outline.set_color(RULE)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=8, pad=6)

    # ---------- Panels B and C: identical axes, one x variable each.
    sj = np.array([float(p["study_jaccard"]) for p in pairs])
    aj = np.array([float(p["analysis_jaccard"]) for p in pairs])
    mr = np.array([float(p["map_r"]) for p in pairs])
    is_er = np.array([p["project"] == "emotion_regulation_2022" for p in pairs])

    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 1], sharex=axb, sharey=axb)

    for ax, x, name in ((axb, sj, "Study overlap"), (axc, aj, "Analysis overlap")):
        ax.axhline(0, color=RULE, lw=0.6, zorder=1)
        # Fill carries the ER distinction as well as hue, so the panels survive greyscale.
        ax.scatter(x[~is_er], mr[~is_er], s=20, facecolors="white", edgecolors=BASE,
                   linewidths=0.9, zorder=3)
        ax.scatter(x[is_er], mr[is_er], s=26, color=ACCENT, edgecolors="white",
                   linewidths=0.5, zorder=4)
        rho = stats.spearmanr(x, mr)[0]
        # Above the axes, not inside them: interior corners are all data in one panel or the
        # other, and the two panels should carry the header in the same place.
        ax.text(1.0, 1.015, f"{name}   $\\rho$ = {rho:+.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.8, color=INK)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.18, 1.1)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(RULE)
        ax.set_ylabel("Map similarity ($r$)", fontsize=6.2)

    plt.setp(axb.get_xticklabels(), visible=False)
    axc.set_xlabel("Overlap between the two targets (Jaccard; 71 within-project pairs)",
                   fontsize=6.5)

    # b: the pairs study overlap mispredicts. One label for the cluster, not one per point.
    disc = is_er & (sj > 0.5) & (mr < 0.3)
    if disc.any():
        cx, cy = float(sj[disc].mean()), float(mr[disc].mean())
        axb.annotate("same studies,\ndisjoint analyses",
                     xy=(cx + 0.015, cy + 0.03), xytext=(0.82, 0.24),
                     ha="center", va="center", fontsize=5.4, color=ACCENT, linespacing=1.4,
                     arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.6,
                                     shrinkA=2, shrinkB=4))
    # The converse case -- executive-function pairs with no shared evidence whose maps still
    # correlate at r ~ 0.7 -- is real but gets no in-panel label: panel c's point cloud leaves no
    # collision-free harbor for two lines of text (verified against renders), and it is an aside,
    # not the claim. It lives in the caption instead.

    axb.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=BASE,
               markeredgewidth=0.9, markersize=4.2, label="Other projects"),
        Line2D([], [], marker="o", ls="", color=ACCENT, markersize=4.4,
               label="Emotion regulation")],
        loc="upper right", bbox_to_anchor=(1.0, 0.70), frameon=False, fontsize=5.4,
        handletextpad=0.35, labelspacing=0.3, borderaxespad=0.1)
    axb.set_title("b", loc="left", fontweight="bold", fontsize=8, pad=4)
    axc.set_title("c", loc="left", fontweight="bold", fontsize=8, pad=4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / f"figure_target_structure.{ext}", dpi=400,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {args.output_dir.relative_to(REPO)}/figure_target_structure.{{pdf,png}}")
    print(f"  panel a: {M.shape[0]}x{M.shape[1]} targets in {len(bounds)} project blocks")
    print(f"  panels b/c: {len(pairs)} pairs; rho study vs analysis = "
          f"{stats.spearmanr(sj, mr)[0]:+.2f} / {stats.spearmanr(aj, mr)[0]:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
