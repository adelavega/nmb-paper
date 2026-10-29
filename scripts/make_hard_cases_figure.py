#!/usr/bin/env python3
"""Figure: the references that are reliable but that no automated method predicts.

x = the reference's split-half reliability; y = identification: the share of the other 30
references that a method's map correlates with less than with the correct one (0.5 is chance),
averaged over the four methods (NeuroQuery, NeuroVLM, held-out Neurosynth association and
uniformity), with the range across methods as a whisker. From scripts/eval_specificity.py.
Identification is used rather than r2 because a component shared by every reference -- where
studies tend to activate at all -- moves all 31 correlations together and cannot change the rank.
Dotted lines are the medians over the 31 targets; the shaded corner -- above-median reliability,
below-median identification -- is "reliable but unpredicted": the expert map is stable, so the
miss is the method's, not the reference's noise.

Colour is the mechanism that makes these targets hard, not the project as such:

    emotion regulation (orange)  sibling targets share studies and differ only in which
                                 contrast was taken (analysis-only separability gap -0.14 to
                                 -0.48; every other project's is within +-0.05 of zero). A
                                 method must select the correct contrast within a paper.
    cue reactivity (magenta)      siblings are variants of one paradigm (reward / drug / natural
                                 cue vs neutral) drawn from different studies; methods recover
                                 the paradigm but sibling discrimination is exactly at chance.

    executive function (blue)    the contrast case: equally reliable, and predicted -- working
                                 memory, inhibition and the pooled EF map are the best-recovered
                                 references in the benchmark. Flexibility, the one EF target
                                 drawn from a small separate literature, is neither.

Palette: three highlight slots validated (worst CVD dE 21.8), grey for the rest; EF keeps the
blue it has in the other figures.
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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_palette as P  # noqa: E402
# legend labels are names only; what each group illustrates is in the figure caption
GROUPS = {"emotion_regulation_2022": ("Emotion regulation", "#D55E00"),
          "cue_reactivity": ("Cue reactivity", "#A0479A"),
          "executive_function": ("Executive function", "#0072B2")}
OTHER = "#B5B5B5"
INK, MUTED, RULE, SHADE = "#1A1A1A", "#5A5A5A", "#D8D8D8", "#F3EDE8"
_LABELLED = (("emotion_regulation_2022", "decrease"), ("emotion_regulation_2022", "increase"),
             ("emotion_regulation_2022", "maintain"), ("emotion_regulation_2022", "reappraisal"),
             ("cue_reactivity", "1_reward_neutral_2020_wbonly"),
             ("cue_reactivity", "2_drug_neutral_2020_wbonly"),
             ("cue_reactivity", "3_natural_neutral_2020_wbonly"),
             ("executive_function", "working_memory"), ("executive_function", "all"),
             ("executive_function", "inhibition"), ("executive_function", "flexibility"))
NAMES = {k: f"{P.PROJECTS[k[0]][1]} {P.target_label(*k)}" for k in _LABELLED}
# label offsets in points: (dx, dy, ha)
OFFSETS = {("emotion_regulation_2022", "decrease"): (6, 0, "left"),
           ("emotion_regulation_2022", "increase"): (-6, 0, "right"),
           ("emotion_regulation_2022", "maintain"): (-6, -6, "right"),
           ("emotion_regulation_2022", "reappraisal"): (6, -6, "left"),
           ("cue_reactivity", "1_reward_neutral_2020_wbonly"): (6, -2, "left"),
           ("cue_reactivity", "2_drug_neutral_2020_wbonly"): (-8, -10, "right"),
           ("cue_reactivity", "3_natural_neutral_2020_wbonly"): (0, 10, "center"),
           ("executive_function", "flexibility"): (-6, -10, "right")}
EF_CLUSTER = ("executive_function", ("working_memory", "inhibition", "all"))

def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    rows = list(csv.DictReader(open(REPO / "results" / "specificity_by_target.csv")))
    key = lambda r: (r["project"], r["target"])
    rel = np.array([float(r["reliability"]) for r in rows])

    fig, ax = plt.subplots(figsize=(4.6, 4.3),
                           gridspec_kw=dict(left=0.12, right=0.98, top=0.95, bottom=0.18))
    ylim, chance = (0.0, 1.10), 0.5
    y = np.array([float(r["identification_mean"]) for r in rows])
    lo = np.array([float(r["identification_min"]) for r in rows])
    hi = np.array([float(r["identification_max"]) for r in rows])
    mx, my = np.median(rel), np.median(y)
    ax.fill_between([mx, 1.0], ylim[0], my, color=SHADE, zorder=0, lw=0)
    ax.axvline(mx, color=RULE, lw=0.6, ls=":", zorder=1)
    ax.axhline(my, color=RULE, lw=0.6, ls=":", zorder=1)
    ax.axhline(chance, color=MUTED, lw=0.6, ls="--", zorder=1)
    ax.text(0.105, chance + 0.012, "chance", fontsize=5.6, color=MUTED, va="bottom")
    # least-squares trend over all 31 targets, with Spearman rho and a project-level bootstrap
    # interval (targets within a project are not independent)
    from scipy import stats
    b = np.polyfit(rel, y, 1)
    xx = np.array([0.12, 0.98])
    ax.plot(xx, np.polyval(b, xx), color=INK, lw=1.4, alpha=0.45, zorder=1)
    rho, p_rho = stats.spearmanr(rel, y)
    projects = np.array([r["project"] for r in rows])
    groups = [np.flatnonzero(projects == q) for q in np.unique(projects)]
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(10000):
        idx = np.concatenate([rng.choice(g, len(g)) for g in
                              (groups[k] for k in rng.integers(0, len(groups), len(groups)))])
        if np.ptp(rel[idx]) > 0 and np.ptp(y[idx]) > 0:
            boots.append(stats.spearmanr(rel[idx], y[idx])[0])
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    ax.text(0.105, 0.03, f"trend over all {len(rows)} targets: Spearman ρ = {rho:+.2f} "
            f"(p = {p_rho:.3f})\n95% CI {lo_ci:+.2f} to {hi_ci:+.2f}, bootstrap over the "
            f"{len(groups)} projects", ha="left", va="bottom", fontsize=5.6, color=MUTED)
    order = sorted(range(len(rows)), key=lambda i: rows[i]["project"] in GROUPS)
    for i in order:
        p = rows[i]["project"]
        c = GROUPS.get(p, (None, OTHER))[1]
        hl = p in GROUPS
        ax.plot([rel[i], rel[i]], [lo[i], hi[i]], color=c, lw=0.6,
                alpha=0.55 if hl else 0.35, zorder=2)
        ax.scatter([rel[i]], [y[i]], s=34 if hl else 18, color=c, edgecolors="white",
                   linewidths=0.6, zorder=4 if hl else 3,
                   marker="^" if p.startswith("vbm") else "o")
        name = NAMES.get(key(rows[i]))
        if key(rows[i]) in OFFSETS:
            dx, dy, ha = OFFSETS[key(rows[i])]
            ax.annotate(name, (rel[i], y[i]), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=5.8, color=c, fontweight="bold")
    # the three predicted EF targets sit together at the top: one label for the cluster
    ef = [i for i, r in enumerate(rows)
          if r["project"] == EF_CLUSTER[0] and r["target"] in EF_CLUSTER[1]]
    ax.text(np.mean(rel[ef]), 1.035, "EF working memory, inhibition, all tasks", ha="center",
            va="bottom", fontsize=5.8, color=GROUPS[EF_CLUSTER[0]][1], fontweight="bold")
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Reference reliability (split-half ceiling)")
    ax.set_ylabel("Identification: share of the other 30 references beaten\n"
                  "(mean over the four methods)")
    ax.grid(alpha=0.2, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)

    handles = [Line2D([], [], ls="", marker="o", color=c, mec="white", ms=5.5, label=lab)
               for lab, c in GROUPS.values()]
    handles += [Line2D([], [], ls="", marker="o", color=OTHER, mec="white", ms=4.5,
                       label="Other (▲ VBM)"),
                Line2D([], [], color=MUTED, lw=0.6, label="Range across methods"),
                Line2D([], [], color=INK, lw=1.4, alpha=0.45, label="Trend"),
                Patch(facecolor=SHADE, edgecolor="none", label="Reliable but unpredicted")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.10, 0.0), ncol=4,
               frameon=False, fontsize=6, handletextpad=0.3, columnspacing=1.5)

    out = REPO / "results" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_hard_cases.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    print("  wrote results/figures/figure_hard_cases.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
