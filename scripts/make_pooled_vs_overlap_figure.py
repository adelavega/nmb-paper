#!/usr/bin/env python3
"""Figure: a target's share of its project's papers vs how closely the pooled map matches it.

x = the share of the project's studies (PMIDs) the target draws on; y = r2 between the target's
reference map and the map pooling every annotated analysis in its project. One point per
comparison (25: selected targets, umbrella and degenerate excluded), coloured by project
(scripts/project_palette.py). The dashed line is the least-squares trend over every project but
emotion regulation -- what a target's PMID overlap alone would predict. Emotion regulation's
targets fall below it: decrease and reappraisal draw on almost every paper in the project yet
the pooled map misses part of them, because they take only some of the results those papers
report. Data: results/pooled_vs_overlap.csv (scripts/analyse_pooled_vs_overlap.py).
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
ER = "emotion_regulation_2022"
LABEL = {(ER, "decrease"): (-6, -2, "right"), (ER, "reappraisal"): (-6, 5, "right"),
         (ER, "maintain"): (6, -3, "left"), (ER, "increase"): (6, 0, "left"),
         ("decision_making", "rdm_july2019"): (6, -4, "left")}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy import stats

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    rows = [r for r in csv.DictReader(open(REPO / "results" / "pooled_vs_overlap.csv"))
            if r["umbrella"] == "0" and r["degenerate"] == "0"]
    x = np.array([float(r["study_share"]) for r in rows])
    y = np.array([float(r["pooled_r2"]) for r in rows])
    er = np.array([r["project"] == ER for r in rows])
    b = np.polyfit(x[~er], y[~er], 1)
    rho_all = stats.spearmanr(x, y)[0]
    rho_wo = stats.spearmanr(x[~er], y[~er])

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    xx = np.linspace(0, 1, 50)
    yy = np.polyval(b, xx)
    ax.plot(xx[yy <= 1], yy[yy <= 1], color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.text(0.47, np.polyval(b, 0.47) - 0.035, "expected from study overlap\n(trend without ER)",
            ha="left", va="top", fontsize=5.8, color=MUTED)
    for r, xi, yi in zip(rows, x, y):
        p = r["project"]
        ax.scatter([xi], [yi], s=34, color=P.colour(p), marker=P.marker(p), edgecolors="white",
                   linewidths=0.6, zorder=3)
        k = (p, r["target"])
        if k in LABEL:
            dx, dy, ha = LABEL[k]
            ax.annotate(f"{P.PROJECTS[p][1]} {P.target_label(*k)}", (xi, yi), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center", fontsize=5.8,
                        color=P.colour(p), fontweight="bold")
    ax.text(0.98, 0.03, f"Spearman ρ = {rho_all:+.2f}, all 25 targets\n"
            f"{rho_wo[0]:+.2f} without emotion regulation (p < 0.001)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=MUTED)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Share of the project's studies the target draws on")
    ax.set_ylabel("Pooled-map correspondence with the target ($r^2$)")
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    present = {r["project"] for r in rows}
    handles = [Line2D([], [], ls="", marker=mk, color=c, mec="white", ms=5, label=lab)
               for key, (lab, _, c, mk) in P.PROJECTS.items() if key in present]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=6, handletextpad=0.3, labelspacing=0.45)
    out = REPO / "results" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_pooled_vs_overlap.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    print("  wrote results/figures/figure_pooled_vs_overlap.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
