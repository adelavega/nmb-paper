#!/usr/bin/env python3
"""Figure (Results section 2): how reliable each reference is, and how much selection it needs.

x = the reference's split-half reliability (study-level halves, Spearman-Brown corrected);
y = selection dependence, 1 - min(1, pooled r2 / ceiling): the share of the reference's reliable
signal that a map pooling every expert-selected analysis in the project misses. 0 means the
project pool already recovers the target; 1 means pooling recovers none of its reliable signal.
Points are the 31 evaluable selected targets, coloured by project with the shared palette
(scripts/project_palette.py). Umbrella targets (>= 95% of their project's analyses) are at 0 by
construction and drawn hollow; the four degenerate substance-use references (reliability ~0) are
omitted. Targets needing selection (y > 0.1) are labelled; with all six hues in play, labels are
the required secondary encoding.

Data are shared with panel a of figure_difficulty_routes (same loader), which colours the same
points by analysis-only separability instead of by project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_difficulty_figure as D  # noqa: E402
import project_palette as P  # noqa: E402

INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"
LABEL_ABOVE = 0.10
OFFSETS = {("emotion_regulation_2022", "maintain"): (6, 0, "left"),
           ("emotion_regulation_2022", "increase"): (-6, 0, "right"),
           ("emotion_regulation_2022", "decrease"): (6, -2, "left"),
           ("emotion_regulation_2022", "reappraisal"): (6, 4, "left"),
           ("executive_function", "flexibility"): (-6, 0, "right"),
           ("problem_solving", "visuospatialreasoning_mni_final"): (-6, 0, "right")}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    pts, n_deg = D.load()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for p in sorted(pts, key=lambda p: p["dep"]):
        proj = p["k"][0]
        c = P.colour(proj)
        ax.scatter([p["ceil"]], [p["dep"]], s=34, marker=P.marker(proj),
                   facecolors="white" if p["umb"] else c, edgecolors=c,
                   linewidths=0.9 if p["umb"] else 0.5, zorder=3)
        if p["dep"] > LABEL_ABOVE:
            dx, dy, ha = OFFSETS.get(p["k"], (6, 0, "left"))
            ax.annotate(f"{P.PROJECTS[proj][1]} {P.target_label(*p['k'])}",
                        (p["ceil"], p["dep"]), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=5.8, color=c, fontweight="bold")
    n_zero = sum(p["dep"] < 0.05 and not p["umb"] for p in pts)
    n_umb = sum(p["umb"] for p in pts)
    ax.text(1.03, 0.30, f"{n_zero} of {len(pts) - n_umb} non-umbrella targets\nneed almost no "
            f"selection (< 0.05):\nthe project pool recovers them", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.8, color=MUTED)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(-0.04, 0.75)
    ax.set_xlabel("Reference reliability (split-half ceiling)")
    ax.set_ylabel("Selection dependence\n(reliable signal the project-pooled map misses)")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    present = {p["k"][0] for p in pts}
    handles = [Line2D([], [], ls="", marker=mk, color=c, mec=c, ms=5, label=lab)
               for key, (lab, _, c, mk) in P.PROJECTS.items() if key in present]
    handles.append(Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, ms=5,
                          label=f"umbrella target (n = {n_umb}), 0 by construction"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=6, handletextpad=0.3, labelspacing=0.45)
    out = REPO / "results" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_selection_dependence.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    print(f"  wrote results/figures/figure_selection_dependence.{{pdf,png}} "
          f"({len(pts)} targets, {n_deg} degenerate omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
