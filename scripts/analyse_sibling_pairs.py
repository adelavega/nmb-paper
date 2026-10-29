#!/usr/bin/env python3
"""Which sibling references are genuinely different, yet the automated methods cannot tell apart?

Unit: a pair of sibling targets (A, B) from the same project, both evaluable and neither an
umbrella target (an umbrella contains its siblings by construction). For each pair:

    reference_r        voxelwise correlation between reference A and reference B inside the
                       brain mask: how similar the two expert maps are
    reference_r_corr   the same, disattenuated for the references' split-half reliability
                       (r / sqrt(rel_A * rel_B)); above ~1 means the two differ by no more than
                       their own noise
    margin_<method>    how much better the method's maps match their own reference than the
                       sibling's, averaged over the two directions:
                       ([r(mapA, refA) - r(mapA, refB)] + [r(mapB, refB) - r(mapB, refA)]) / 2
                       > 0 the method separates the pair; ~0 it cannot
    correct_<method>   share of the two directions in which the own reference wins (0, .5, 1)
    margin_mean        margin averaged over the four methods

Writes results/sibling_pair_discrimination.csv and
results/figures/figure_sibling_pairs.{pdf,png}: x = reference_r, y = margin_mean, one point per
pair, coloured by project; hollow where either reference has reliability < LOW_REL, since a
low correlation between noisy maps need not mean the topics differ.
"""

from __future__ import annotations

import csv
import inspect
import itertools
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
import eval_specificity as S  # noqa: E402
import eval_text_to_map as E  # noqa: E402
import project_palette as P  # noqa: E402

INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"
LOW_REL = 0.5
DIFFERENT_BELOW = 0.5        # reference r below this: the two expert maps differ
NOT_SEPARATED_BELOW = 0.02   # mean margin below this: the methods barely separate the pair
UMBRELLA_FRACTION = 0.95


def load():
    import nibabel as nib
    from nilearn.image import resample_to_img
    E._RESAMPLE_PARAMS = set(inspect.signature(resample_to_img).parameters)
    ev = {(r["project"], r["target"]): r
          for r in csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))}
    keys = [k for k in ((r["project"], r["target"])
                        for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv")))
            if ev[k]["reference_degenerate"] == "0"
            and float(ev[k]["fraction_of_project_pool"]) < UMBRELLA_FRACTION]
    grid = nib.load(str(E.reference_dir(*keys[0]) / "z.nii.gz"))
    brain = np.asarray(E.resample(nib.load(str(NMB / "data" / "brain_mask_mni152_2mm.nii.gz")),
                                  grid).dataobj) > 0.5

    def vec(path):
        a = np.asarray(E.resample(nib.load(str(path)), grid).dataobj, dtype=float)[brain]
        a[~np.isfinite(a)] = 0.0
        return a

    R = np.array([vec(E.reference_dir(*k) / "z.nii.gz") for k in keys])
    maps = {m: np.array([vec(S.method_path(m, *k, ev[k]["neurosynth_term"])) for k in keys])
            for m in S.METHODS}
    return keys, R, maps


def main() -> int:
    keys, R, maps = load()
    rel = {(r["project"], r["target"]): float(r["reliability_mean"])
           for r in csv.DictReader(open(REPO / "results" / "reference_reliability.csv"))}
    RR = S.corr_matrix(R, R)
    C = {m: S.corr_matrix(X, R) for m, X in maps.items()}
    idx = {k: i for i, k in enumerate(keys)}
    by_project = defaultdict(list)
    for k in keys:
        by_project[k[0]].append(k)

    rows = []
    for project, ks in sorted(by_project.items()):
        for a, b in itertools.combinations(ks, 2):
            i, j = idx[a], idx[b]
            row = dict(project=project, target_a=a[1], target_b=b[1],
                       reliability_a=round(rel[a], 4), reliability_b=round(rel[b], 4),
                       reference_r=round(float(RR[i, j]), 4),
                       reference_r_corr=round(float(RR[i, j] / np.sqrt(max(rel[a], 1e-6) *
                                                                       max(rel[b], 1e-6))), 4))
            margins = []
            for m in S.METHODS:
                c = C[m]
                d_a, d_b = c[i, i] - c[i, j], c[j, j] - c[j, i]
                margins.append((d_a + d_b) / 2)
                row[f"margin_{m}"] = round(float((d_a + d_b) / 2), 4)
                row[f"correct_{m}"] = (int(d_a > 0) + int(d_b > 0)) / 2
            row["margin_mean"] = round(float(np.mean(margins)), 4)
            rows.append(row)
    with open(REPO / "results" / "sibling_pair_discrimination.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    from scipy import stats
    x = np.array([r["reference_r"] for r in rows])
    y = np.array([r["margin_mean"] for r in rows])
    rho, p = stats.spearmanr(x, y)
    print(f"{len(rows)} sibling pairs; reference r vs mean margin: Spearman {rho:+.2f} "
          f"(p = {p:.3g})")
    for r in sorted(rows, key=lambda r: r["margin_mean"]):
        print(f"  {r['project'][:12]:<13}{r['target_a'][:22]:<23}{r['target_b'][:22]:<23}"
              f"ref r {r['reference_r']:+.2f} (corr {r['reference_r_corr']:+.2f})  "
              f"margin {r['margin_mean']:+.3f}  rel {r['reliability_a']:.2f}/"
              f"{r['reliability_b']:.2f}")
    make_figure(rows, rho, p)
    return 0


def make_figure(rows, rho, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from make_target_map_embedding import place_labels

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.fill_between([-0.12, DIFFERENT_BELOW], -0.1, NOT_SEPARATED_BELOW, color="#F3EDE8",
                    zorder=0, lw=0)
    ax.text(-0.11, -0.095, "different maps,\nnot separated", ha="left", va="bottom",
            fontsize=5.8, color="#8A5A3C", fontstyle="italic")
    ax.axhline(0, color=MUTED, lw=0.7, ls="--", zorder=1)
    ax.text(0.30, -0.004, "no separation", ha="center", va="top", fontsize=5.6, color=MUTED)
    xy, labels, colours = [], [], []
    for r in rows:
        c = P.colour(r["project"])
        noisy = min(r["reliability_a"], r["reliability_b"]) < LOW_REL
        ax.scatter([r["reference_r"]], [r["margin_mean"]], s=34, marker=P.marker(r["project"]),
                   facecolors="white" if noisy else c, edgecolors=c,
                   linewidths=0.9 if noisy else 0.5, zorder=3)
        xy.append((r["reference_r"], r["margin_mean"]))
        # label the pairs in the shaded corner, and the one well-separated ER pair for contrast
        corner = r["reference_r"] < DIFFERENT_BELOW and r["margin_mean"] < NOT_SEPARATED_BELOW
        contrast = (r["project"], r["target_a"], r["target_b"]) == (
            "emotion_regulation_2022", "maintain", "reappraisal")
        if corner or contrast:
            labels.append(f"{P.target_label(r['project'], r['target_a'])} / "
                          f"{P.target_label(r['project'], r['target_b'])}")
        else:
            labels.append("")
        colours.append(c)
    ax.set_xlim(-0.12, 1.0)
    ax.set_ylim(-0.1, 0.22)
    ax.set_xlabel("Similarity of the two expert reference maps (r)")
    ax.set_ylabel("Automated separation of the pair\n(own-reference advantage, mean of 4 methods)")
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.text(0.99, 0.98, f"{len(rows)} sibling pairs; Spearman ρ = {rho:+.2f} (p = {p:.2f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=MUTED)
    place_labels(fig, ax, np.array(xy), labels, colours, fontsize=5.6, gap=4.0)
    present = {r["project"] for r in rows}
    handles = [Line2D([], [], ls="", marker=mk, color=c, mec=c, ms=5, label=lab)
               for key, (lab, _, c, mk) in P.PROJECTS.items() if key in present]
    handles.append(Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, ms=5,
                          label=f"a reference with reliability < {LOW_REL}"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=6, handletextpad=0.3, labelspacing=0.45)
    out = REPO / "results" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_sibling_pairs.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    print("  wrote results/sibling_pair_discrimination.csv, "
          "results/figures/figure_sibling_pairs.{pdf,png}")


if __name__ == "__main__":
    raise SystemExit(main())
