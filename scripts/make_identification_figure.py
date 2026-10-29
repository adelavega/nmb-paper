#!/usr/bin/env python3
"""Figure: which references one automated method can pick out, and what it mistakes the rest for.

One row per evaluable reference (31), sorted by identification: the share of the other 30
references that the method's map correlates with less than with the correct one (0.5 is
chance; 1 means the method's map is closer to its own reference than to any other). Rows are
coloured by project (scripts/project_palette.py). For references the method identifies poorly
(below CONFUSED_BELOW), the right-hand column names the reference its map resembles most, with
both correlations: the map's r with its own reference, and with that closest other reference.

No reliability axis: for the methods with the best identification (NeuroQuery, Neurosynth
association) identification does not track split-half reliability (see
figure_hard_cases_by_method), so the hard cases are shown as they are.

    python scripts/make_identification_figure.py [--method "NeuroQuery"]

Writes results/figures/figure_identification_<method>.{pdf,png}.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
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
CONFUSED_BELOW = 0.75


def load(method: str):
    import nibabel as nib
    from nilearn.image import resample_to_img
    E._RESAMPLE_PARAMS = set(inspect.signature(resample_to_img).parameters)
    ev = {(r["project"], r["target"]): r
          for r in csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))}
    keys = [k for k in ((r["project"], r["target"])
                        for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv")))
            if ev[k]["reference_degenerate"] == "0"]
    grid = nib.load(str(E.reference_dir(*keys[0]) / "z.nii.gz"))
    brain = np.asarray(E.resample(nib.load(str(NMB / "data" / "brain_mask_mni152_2mm.nii.gz")),
                                  grid).dataobj) > 0.5

    def vec(path):
        a = np.asarray(E.resample(nib.load(str(path)), grid).dataobj, dtype=float)[brain]
        a[~np.isfinite(a)] = 0.0
        return a

    R = np.array([vec(E.reference_dir(*k) / "z.nii.gz") for k in keys])
    X = np.array([vec(S.method_path(method, *k, ev[k]["neurosynth_term"])) for k in keys])
    return keys, S.corr_matrix(X, R)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="NeuroQuery", choices=S.METHODS)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK,
                         "xtick.labelsize": 6, "ytick.labelsize": 6.2})
    keys, C = load(args.method)
    n = len(keys)
    ident = np.array([(C[i] < C[i, i]).sum() / (n - 1) for i in range(n)])
    # best first at the top; ties broken by the map's correlation with its own reference
    order = sorted(range(n), key=lambda i: (ident[i], C[i, i]))
    ys = np.arange(n)

    fig, ax = plt.subplots(figsize=(6.4, 5.6),
                           gridspec_kw=dict(left=0.27, right=0.62, top=0.95, bottom=0.08))
    ax.axvline(0.5, color=MUTED, lw=0.7, ls="--", zorder=1)
    ax.text(0.515, n - 0.2, "chance", ha="left", va="bottom", fontsize=5.8, color=MUTED)
    for y, i in zip(ys, order):
        p = keys[i][0]
        c = P.colour(p)
        ax.plot([0, ident[i]], [y, y], color=c, lw=0.6, alpha=0.35, zorder=1)
        ax.scatter([ident[i]], [y], s=30, color=c, marker=P.marker(p), edgecolors="white",
                   linewidths=0.5, zorder=3)
        if ident[i] < CONFUSED_BELOW:
            j = int(np.argmax(np.where(np.arange(n) == i, -np.inf, C[i])))
            q = keys[j][0]
            ax.text(1.06, y, f"→ {P.PROJECTS[q][1]} {P.target_label(*keys[j])}",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6,
                    color=P.colour(q), fontweight="bold")
            ax.text(1.72, y, f"r {C[i, i]:+.2f} own, {C[i, j]:+.2f} closest",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.6,
                    color=MUTED)
    ax.text(1.06, n - 0.2, f"Map most resembles (identification < {CONFUSED_BELOW})",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=6,
            color=MUTED)
    ax.set_yticks(ys, [f"{P.PROJECTS[keys[i][0]][1]} · {P.target_label(*keys[i])}"
                       for i in order])
    for tick, i in zip(ax.get_yticklabels(), order):
        tick.set_color(P.colour(keys[i][0]))
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, n + 0.4)
    ax.set_xlabel(f"{args.method}: identification\n(share of the other {n - 1} references beaten)")
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.set_title(f"{args.method}: mean identification {ident.mean():.2f}; "
                 f"top-1 for {np.mean(ident >= 1 - 1e-9):.0%} of references",
                 fontsize=6.5, loc="left", color=MUTED)

    slug = args.method.lower().replace(" ", "_")
    out = REPO / "results" / "figures"
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_identification_{slug}.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    print(f"  wrote results/figures/figure_identification_{slug}.{{pdf,png}} "
          f"(mean {ident.mean():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
