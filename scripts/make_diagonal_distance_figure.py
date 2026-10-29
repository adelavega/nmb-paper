#!/usr/bin/env python3
"""Figure: how far each target pair falls from the "analyses mirror studies" diagonal.

For every within-project pair of targets (n = 71) the bar is analysis overlap minus study
overlap (Jaccard). Zero means the pair shares exactly as much of its analyses as of its studies
-- an article inclusion list would describe it faithfully. Negative means the pair shares
studies but NOT the analyses within them: the information only an analysis-level record holds.
(This is the vertical distance from y = x; perpendicular distance is the same divided by sqrt 2,
so the ranking is identical.)

Every project but one stays within +/-0.11 of the line. Emotion regulation averages -0.30 and
reaches -0.63: its targets draw on the same papers and select different results within them.

DESIGN NOTES

Bars are GROUPED by project, sorted within each block, blocks in the same order as the matrix
figure -- not sorted globally. A global sort would interleave the projects, putting any colour
next to any other, and eight categorical hues cannot pass the colour-vision checks under that
all-pairs condition (the reference palette caps it at three). Grouped, each colour only ever
touches its slot neighbours, which the eight-slot order is validated for (worst adjacent CVD
dE 9.1, normal-vision dE 19.6). Grouping is also the better reading: which projects sit on the
line, and which one falls off it.

Three light slots sit below 3:1 contrast on the surface, so identity is never colour-alone:
every block carries its project name directly beneath it.

Decision making's three pairs are exactly zero (its targets share neither studies nor analyses),
so zero-height bars are drawn as short stubs CENTRED on the zero line rather than left
invisible -- an upward stub would misread as a small positive value.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))

# Reference categorical palette in its validated slot order (adjacent pairlist).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"
SHORT = {
    "cue_reactivity": "Cue\nreactivity", "decision_making": "Decision\nmaking",
    "dementia": "Dementia", "emotion_regulation_2022": "Emotion\nregulation",
    "executive_function": "Executive\nfunction", "problem_solving": "Problem\nsolving",
    "social": "Social\ncognition", "vbm_of_substance_use": "VBM\nsubstance use",
}
STUB = 0.006   # visible height for an exactly-zero bar


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=REPO / "results" / "figures")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 6.5, "axes.linewidth": 0.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })

    rows = list(csv.DictReader(open(REPO / "results" / "target_structure.csv")))
    by = collections.defaultdict(list)
    for r in rows:
        d = float(r["analysis_jaccard"]) - float(r["study_jaccard"])
        by[r["project"]].append((d, r["target_a"], r["target_b"]))
    projects = sorted(by)                       # same block order as the matrix figure
    if len(projects) > len(SLOTS):
        raise SystemExit(f"{len(projects)} projects > {len(SLOTS)} validated slots")
    colour = {p: SLOTS[i] for i, p in enumerate(projects)}   # slot follows block position

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.92, bottom=0.25)

    x, gap, centres, label_targets = 0.0, 1.6, [], []
    for p in projects:
        vals = sorted(by[p], key=lambda t: t[0])          # most negative first
        start = x
        for d, ta, tb in vals:
            if abs(d) > 1e-9:
                ax.bar(x, d, width=0.78, color=colour[p], linewidth=0, zorder=3)
            else:
                # Centred ON the line, above it in z-order: an upward stub would read as a
                # small positive value, which these pairs do not have.
                ax.bar(x, STUB, bottom=-STUB / 2, width=0.78, color=colour[p],
                       linewidth=0, zorder=4)
            if p == "emotion_regulation_2022" and d < -0.5:
                label_targets.append((x, d, f"{ta} / {tb}"))
            x += 1
        centres.append(((start + x - 1) / 2, p))
        x += gap

    ax.axhline(0, color=MUTED, lw=0.6, zorder=2)
    ax.set_xlim(-1.2, x - gap + 0.2)
    ax.set_ylim(-0.70, 0.14)
    ax.set_xticks([c for c, _ in centres])
    ax.set_xticklabels([SHORT.get(p, p) for _, p in centres], fontsize=5.6, linespacing=1.2)
    ax.tick_params(axis="x", length=0, pad=4)
    for lbl, (_, p) in zip(ax.get_xticklabels(), centres):
        lbl.set_color(INK)
    ax.set_ylabel("Analysis − study overlap\n(signed distance from diagonal)", fontsize=6.3)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(RULE)

    # Direct-label only the two pairs that carry the argument, from the open side.
    for i, (bx, d, txt) in enumerate(sorted(label_targets, key=lambda t: t[1])):
        ax.annotate(txt, xy=(bx + 0.45, d + 0.01), xytext=(bx + 3.2, -0.60 + 0.085 * i),
                    ha="left", va="center", fontsize=5.3, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                    shrinkA=1, shrinkB=1))

    ax.text(0.0, 1.02, "71 within-project target pairs · 0 = analyses shared exactly as much as "
            "studies · below 0 = shared studies, different analyses",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.4, color=MUTED)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / f"figure_diagonal_distance.{ext}", dpi=400,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {args.output_dir.relative_to(REPO)}/figure_diagonal_distance.{{pdf,png}}")
    for p in projects:
        v = [t[0] for t in by[p]]
        print(f"    {p:<26} n={len(v):<3} mean {np.mean(v):+.3f}  min {min(v):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
