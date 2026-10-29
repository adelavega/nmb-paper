#!/usr/bin/env python3
"""Figure: two routes to needing analysis-level selection.

PANEL A -- difficulty taxonomy. x = how reproducible the reference is (split-half ceiling);
y = how much of its reliable signal the project-pooled map misses (selection dependence,
1 - min(1, pooled r2 / ceiling)). Colour = analysis-only separability from sibling targets
(minus the mean of analysis Jaccard - study Jaccard): dark targets share their siblings' papers
but not their analyses. The upper half is reached by two kinds of target: dark ones (shared
literature) and pale ones (a literature of their own, diluted by the project pool).

PANEL B -- the direct test that separates them. x = selection dependence under pooling (panel A's
y); y = the deficit when the right papers are found but every analysis in them is kept
(1 - r2, article-level arm vs a refit of the target's own analyses). Points on the diagonal lose
as much with the right papers as with the whole project: finding papers does not help. Points on
the floor are rescued by finding the right papers. The dark targets stay up; the pale ones drop.

Umbrella targets (>= 95% of their project pool) sit at zero dependence by construction and are
drawn hollow; degenerate references (ceiling ~0, no reproducible signal) are omitted from both
panels and counted in the caption. Colour ramp and accent validated with the dataviz six-check
validator (sequential: monotone L, light end 2.02:1).
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics as st
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_palette as P  # noqa: E402
RAMP = ["#8ABADB", "#6EA7CE", "#5392BF", "#3A7CAC", "#246495", "#124D79", "#06365A"]
INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"
STRUCTURAL = {"vbm_of_ptsd", "vbm_of_substance_use"}
SHORT = {"emotion_regulation_2022": "ER", "executive_function": "EF", "problem_solving": "PS",
         "social": "Soc", "cue_reactivity": "Cue", "decision_making": "DM",
         "dementia": "Dem", "vbm_of_ptsd": "PTSD", "vbm_of_substance_use": "SUD"}


def load():
    rd = lambda f: list(csv.DictReader(open(REPO / f)))
    sel = [(r["project"], r["target"]) for r in rd(NMB / "data" / "selected_targets.csv")]
    ev = {(r["project"], r["target"]): r for r in rd("results/text_to_map_baselines.csv")}
    rel = {(r["project"], r["target"]): float(r["reliability_mean"])
           for r in rd("results/reference_reliability.csv")}
    art = {(r["project"], r["target"]): r for r in rd("results/article_level_arm.csv")}
    gaps = collections.defaultdict(list)
    for r in rd("results/target_structure.csv"):
        a, b = (r["project"], r["target_a"]), (r["project"], r["target_b"])
        if a in set(sel) and b in set(sel):
            g = float(r["analysis_jaccard"]) - float(r["study_jaccard"])
            gaps[a].append(g)
            gaps[b].append(g)
    pts = []
    for k in sel:
        e = ev[k]
        if e["reference_degenerate"] == "1":
            continue
        umb = e["umbrella_target"] == "1"
        c = rel[k]
        dep = 0.0 if umb else 1 - min(1.0, float(e["pooled_r2"]) / c)
        pts.append(dict(k=k, ceil=c, dep=dep, umb=umb,
                        sep=max(0.0, -st.mean(gaps[k])) if gaps[k] else 0.0,
                        art=None if umb else 1 - float(art[k]["r2_article"]),
                        struct=k[0] in STRUCTURAL))
    n_deg = sum(ev[k]["reference_degenerate"] == "1" for k in sel)
    return pts, n_deg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=REPO / "results" / "figures")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 6.5, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    cmap = LinearSegmentedColormap.from_list("nmb_seq", RAMP)
    pts, n_deg = load()
    vmax = max(p["sep"] for p in pts)
    norm = Normalize(0, vmax)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                                   gridspec_kw=dict(wspace=0.32, left=0.08, right=0.89,
                                                    top=0.88, bottom=0.14))

    def draw(ax, xs, ys, subset):
        for p, x, y in zip(subset, xs, ys):
            col = cmap(norm(p["sep"]))
            ax.scatter([x], [y], s=30, marker="^" if p["struct"] else "o",
                       facecolors="white" if p["umb"] else col,
                       edgecolors=col if p["umb"] else "white",
                       linewidths=1.0 if p["umb"] else 0.5, zorder=3)

    def style(ax):
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(RULE)

    # ---------- A: taxonomy
    axa.axvspan(0.7, 1.0, ymin=0, ymax=1, color="#F2F2F2", zorder=0, lw=0)
    axa.axhline(0.2, color=RULE, lw=0.6, ls="--", zorder=1)
    draw(axa, [p["ceil"] for p in pts], [p["dep"] for p in pts], pts)
    axa.set_xlim(0.1, 0.95)
    axa.set_ylim(-0.05, 0.72)
    axa.set_xlabel("Reference reliability (split-half ceiling)")
    axa.set_ylabel("Selection dependence\n(reliable signal missed by pooling the project)")
    # Lower labels sit just under the dashed boundary, the only band free of points: the floor
    # itself is where most targets are.
    for (x, y, s, ha) in ((0.93, 0.69, "reliable,\nneeds selection", "right"),
                          (0.93, 0.185, "reliable,\nrecoverable from the topic", "right"),
                          (0.12, 0.69, "underpowered,\nneeds selection", "left"),
                          (0.12, 0.185, "underpowered", "left")):
        axa.text(x, y, s, ha=ha, va="top", fontsize=5.4, color=MUTED, linespacing=1.3)
    style(axa)
    axa.set_title("a", loc="left", fontweight="bold", fontsize=8, pad=5)

    # ---------- B: direct test
    sub = [p for p in pts if p["art"] is not None]
    lim = 0.7
    axb.plot([0, lim], [0, lim], color=RULE, lw=0.7, ls="--", zorder=1)
    axb.text(0.60, 0.645, "right papers\ndo not help", rotation=0, ha="center", va="bottom",
             fontsize=5.4, color=MUTED, linespacing=1.3)
    axb.text(0.68, 0.13, "floor: no sibling contamination\n(annotated analyses only)", ha="right",
             va="center", fontsize=5.4, color=MUTED, linespacing=1.3)
    draw(axb, [p["dep"] for p in sub], [p["art"] for p in sub], sub)
    offsets = {("emotion_regulation_2022", "maintain"): ((-6, 3), "right"),
               ("emotion_regulation_2022", "increase"): ((6, -2), "left"),
               ("emotion_regulation_2022", "decrease"): ((6, 2), "left"),
               ("emotion_regulation_2022", "reappraisal"): ((-6, 4), "right"),
               ("executive_function", "flexibility"): ((2, 7), "left"),
               ("problem_solving", "visuospatialreasoning_mni_final"): ((-14, 9), "center"),
               ("problem_solving", "demand_mni_final"): ((6, 0), "left")}
    lab = {k: (f"{P.PROJECTS[k[0]][1]} {P.target_label(*k)}", off, ha)
           for k, (off, ha) in offsets.items()}
    for p in sub:
        if p["k"] in lab:
            text, off, ha = lab[p["k"]]
            axb.annotate(text, (p["dep"], p["art"]), textcoords="offset points", xytext=off,
                         ha=ha, va="center", fontsize=5.4, color=INK)
    axb.set_xlim(-0.03, lim)
    axb.set_ylim(-0.03, lim)
    axb.set_xlabel("Signal lost by pooling the project")
    axb.set_ylabel("Signal lost with the right papers,\nall annotated analyses kept (1 − $r^2$)")
    style(axb)
    axb.set_title("b", loc="left", fontweight="bold", fontsize=8, pad=5)

    # shared legend + colourbar
    cax = fig.add_axes([0.915, 0.30, 0.012, 0.50])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Analysis-only separability\nfrom sibling targets", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.2, length=2, color=MUTED)
    cb.outline.set_color(RULE)
    handles = [Line2D([], [], marker="o", ls="", color="#5392BF", markeredgecolor="white",
                      markersize=4.6, label="functional"),
               Line2D([], [], marker="^", ls="", color="#5392BF", markeredgecolor="white",
                      markersize=4.8, label="structural (VBM)"),
               Line2D([], [], marker="o", ls="", markerfacecolor="white",
                      markeredgecolor="#5392BF", markersize=4.6, label="umbrella target")]
    axa.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.83), frameon=False,
               fontsize=5.4, handletextpad=0.3, labelspacing=0.3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / f"figure_difficulty_routes.{ext}", dpi=400,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote results/figures/figure_difficulty_routes.{{pdf,png}}  "
          f"(panel a {len(pts)} targets, panel b {len(sub)}; {n_deg} degenerate omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
