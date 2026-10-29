#!/usr/bin/env python3
"""Figure (Results section 1): how much sibling targets share their studies and their analyses.

One point per pair of non-umbrella sibling targets within a project. x = study Jaccard (the
papers the two targets draw on); y = analysis Jaccard (the individual results). Colour =
project (scripts/project_palette.py). On the dashed diagonal the two targets share studies and
analyses equally; below it they share papers but take different results from them -- different
contrasts from the same studies, which only analysis-level records can separate. Pairs at the
origin are drawn from separate literatures.

Data: results/target_study_overlap_pairs.csv (scripts/describe_target_overlap.py). Umbrella
targets are excluded (they contain their siblings by construction); VBM of PTSD has a single
target and no pairs. Excluded: the "all" targets and the reappraisal and cue-reward supersets. Colour identifies the project (legend); text labels point out notable
pairs, named by their two targets. What the diagonal means and which projects sit entirely at
the origin are stated in the caption, not on the plot.
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

INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"
NEAR_ZERO = 0.05
# notable pairs: (project, target_a, target_b) -> (label, label xy in data units, ha)
NOTABLE = {
    ("emotion_regulation_2022", "decrease", "maintain"):
        ("decrease vs maintain", (0.555, 0.105), "center"),
    ("emotion_regulation_2022", "increase", "maintain"):
        ("increase vs maintain,\nincrease vs decrease", (0.235, 0.045), "left"),
    ("cue_reactivity", "2_drug_neutral_2020_wbonly", "3_natural_neutral_2020_wbonly"):
        ("drug cues vs\nnatural reward cues", (0.0, 0.28), "left"),
    ("social", "self_merged", "others_merged"):
        ("self vs others", (0.28, 0.15), "left"),
}
# dementia pairs, labelled by default (--no-dementia-labels to omit)
DEMENTIA = {
    ("dementia", "decrease", "functional"):
        ("decreases vs functional,\ndecreases vs structural", (0.515, 0.60), "right"),
    ("dementia", "functional", "structural"):
        ("functional vs structural", (0.30, 0.235), "left"),
}
LEADER_MIN = 0.03   # draw a leader line when the label sits this far from its point


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dementia-labels", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results" / "figures")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    pairs = defaultdict(list)
    where = {}
    for r in csv.DictReader(open(REPO / "results" / "target_study_overlap_pairs.csv")):
        if r["umbrella_pair"] == "0":
            xy = (float(r["study_jaccard"]), float(r["analysis_jaccard"]))
            pairs[r["project"]].append(xy)
            where[(r["project"], r["target_a"], r["target_b"])] = xy

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], color=RULE, lw=0.8, ls="--", zorder=1)
    ax.text(0.40, 0.425, "shared equally", rotation=45, rotation_mode="anchor", ha="center",
            va="bottom", fontsize=5.8, color=MUTED)
    origin = []
    for p in P.PROJECTS:
        if p not in pairs:
            continue
        xy = np.array(pairs[p])
        ax.scatter(xy[:, 0], xy[:, 1], s=30, color=P.colour(p), marker=P.marker(p),
                   edgecolors="white", linewidths=0.6, zorder=3, alpha=0.95)
        if (xy.max(axis=1) < NEAR_ZERO).all():
            origin.append(p)
    labels = {**NOTABLE, **({} if args.no_dementia_labels else DEMENTIA)}
    for key, (text, (lx, ly), ha) in labels.items():
        px, py = where[key]
        lead = np.hypot(lx - px, ly - py) > LEADER_MIN + 0.03
        ax.annotate(text, (px, py), xytext=(lx, ly), textcoords="data", ha=ha, va="center",
                    fontsize=6.2, color=P.colour(key[0]), fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=P.colour(key[0]), lw=0.5,
                                    shrinkA=1, shrinkB=3) if lead else None)
    # zoomed to the data (max 0.61 studies, 0.57 analyses); equal axes keep the diagonal at 45 deg
    ax.set_xlim(-0.02, 0.65)
    ax.set_ylim(-0.02, 0.65)
    ax.set_aspect("equal")
    ax.set_xlabel("Studies shared by the two targets (Jaccard)")
    ax.set_ylabel("Analyses shared by the two targets (Jaccard)")
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    handles = [Line2D([], [], ls="", marker=P.marker(p), color=P.colour(p), mec="white",
                      ms=5, label=P.PROJECTS[p][0]) for p in P.PROJECTS if p in pairs]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=6, handletextpad=0.3, labelspacing=0.45)
    out = args.output_dir
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_target_overlap.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    print(f"  wrote {out}/figure_target_overlap.{{pdf,png}} (at origin: {origin})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
