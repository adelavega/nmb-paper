#!/usr/bin/env python3
"""Figure: which references are hard for automated methods, and that the hard ones are not bad.

"ALL" TARGETS ARE LEFT OUT. Targets that pool their whole project -- executive function all,
social all, problem solving global, dementia all, VBM of substance use all drug classes -- are
excluded both as rows and as competitors: they overlap their own sub-targets by construction.
Aggregate sub-targets stay in and are labelled as such: emotion regulation reappraisal (contains
every decrease and increase analysis) and cue reactivity reward (contains every drug- and
natural-cue study). VBM of PTSD stays as a competitor (a single contrast; an umbrella only
because its project has one target). Identification is recomputed among the remaining 26
references, so values differ from figure_method_specificity (all 31).

PANEL A -- difficulty profile. One row per remaining reference with split-half reliability of at
least MIN_RELIABILITY (lower ones are too noisy to call hard), hardest at the top, sorted by
identification averaged over the four methods: the share of the other 24 references a method's
map correlates with less than with the correct one (0.5 is chance). Open marks are the
individual methods, the filled dot their mean, coloured by project (scripts/project_palette.py).

PANELS B AND C -- emotion regulation decrease, and the worst-identified cue-reactivity target
that is not an aggregate, both with the same method (--method, default NeuroQuery): the reference map; that
method's map for the target; and the other reference (of the 24) that map correlates with most
-- the reference it is mistaken for. Each glass brain shows the map's top 5% of in-mask voxels
(positive values), so maps on different scales are comparable; r values are voxelwise
correlations of the unthresholded maps inside the brain mask.
"""

from __future__ import annotations

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
RAMP = ["#8ABADB", "#6EA7CE", "#5392BF", "#3A7CAC", "#246495", "#124D79", "#06365A"]
MARKERS = {"NeuroQuery": "o", "NeuroVLM": "s", "Neurosynth association": "D",
           "Neurosynth uniformity": "^"}
SHORT = {"NeuroQuery": "NQ", "NeuroVLM": "NVLM", "Neurosynth association": "NS assoc",
         "Neurosynth uniformity": "NS unif"}
# the "all" targets, which pool their whole project; the aggregate sub-targets (reappraisal,
# cue reward) stay in and are labelled as aggregates (project_palette.AGGREGATE_LABELS)
CATCH_ALL = {("executive_function", "all"), ("social", "all_merged"),
             ("problem_solving", "global_calculationandvisuospatialandword_mni_final"),
             ("dementia", "all"), ("vbm_of_substance_use", "all_drug_classes")}
FIRST_EXEMPLAR = ("emotion_regulation_2022", "decrease")
SECOND_EXEMPLAR_PROJECT = "cue_reactivity"      # its worst-identified remaining target
TOP_FRACTION = 0.05
MIN_RELIABILITY = 0.6     # rows shown: references at least this reliable (scores are unchanged)


def profile_data():
    """Identification among the non-catch-all references, from the maps themselves."""
    import nibabel as nib
    from nilearn.image import resample_to_img
    E._RESAMPLE_PARAMS = set(inspect.signature(resample_to_img).parameters)
    ev = {(r["project"], r["target"]): r
          for r in csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))}
    sel = [(r["project"], r["target"])
           for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv"))]
    cands = [k for k in sel if ev[k]["reference_degenerate"] == "0" and k not in CATCH_ALL]
    rel = {(r["project"], r["target"]): float(r["reliability_mean"])
           for r in csv.DictReader(open(REPO / "results" / "reference_reliability.csv"))}
    grid = nib.load(str(E.reference_dir(*cands[0]) / "z.nii.gz"))
    brain = np.asarray(E.resample(nib.load(str(NMB / "data" / "brain_mask_mni152_2mm.nii.gz")),
                                  grid).dataobj) > 0.5

    def vec(path):
        a = np.asarray(E.resample(nib.load(str(path)), grid).dataobj, dtype=float)[brain]
        a[~np.isfinite(a)] = 0.0
        return a

    R = np.array([vec(E.reference_dir(*k) / "z.nii.gz") for k in cands])
    X = {m: np.array([vec(S.method_path(m, *k, ev[k]["neurosynth_term"])) for k in cands])
         for m in S.METHODS}
    C = {m: S.corr_matrix(X[m], R) for m in S.METHODS}
    n = len(cands)
    ident = {k: {m: float((C[m][i] < C[m][i, i]).sum() / (n - 1)) for m in S.METHODS}
             for i, k in enumerate(cands)}
    keys = [k for k in cands if rel[k] >= MIN_RELIABILITY]
    mean = {k: float(np.mean(list(ident[k].values()))) for k in keys}
    keys.sort(key=lambda k: mean[k])                       # hardest first
    cache = dict(cands=cands, R=R, X=X, C=C, brain=brain, grid=grid, ev=ev)
    return keys, ident, mean, rel, cache


def exemplar_maps(target, ident, method, cache):
    """Reference, the method's map, and the other reference that map resembles most."""
    import nibabel as nib
    cands, C, brain, grid, ev = (cache[k] for k in ("cands", "C", "brain", "grid", "ev"))
    n_studies = {(r["project"], r["target"]): int(r["n_studies"])
                 for r in csv.DictReader(open(REPO / "results" / "pooled_vs_overlap.csv"))}
    i = cands.index(target)
    row = np.where(np.arange(len(cands)) == i, -np.inf, C[method][i])
    j = int(np.argmax(row))
    arr = {"ref": cache["R"][i], "tool": cache["X"][method][i], "other": cache["R"][j]}
    shown = {}
    for k, v in arr.items():
        vol = np.zeros(brain.shape, dtype=np.float32)
        vol[brain] = np.clip(v, 0, None)
        thr = np.quantile(v, 1 - TOP_FRACTION)
        shown[k] = (nib.Nifti1Image(vol, grid.affine), thr,
                    np.quantile(vol[brain & (vol > thr)], 0.99))
    term = ev[target]["neurosynth_term"]
    query = (f"term \u201c{term}\u201d" if method.startswith("Neurosynth")
             else f"\u201c{ev[target]['query']}\u201d")
    return dict(shown=shown, method=method, query=query, closest=cands[j],
                ident=ident[target][method], n_studies=n_studies[target],
                tool_ref=float(C[method][i, i]), tool_other=float(C[method][i, j]))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="NeuroQuery", choices=S.METHODS)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from nilearn import plotting

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": 0.8,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 8.0})
    keys, ident, mean, rel, cache = profile_data()
    cue = [k for k in keys if k[0] == SECOND_EXEMPLAR_PROJECT and k not in P.AGGREGATE_LABELS]
    EXEMPLARS = (FIRST_EXEMPLAR, min(cue, key=lambda k: mean[k]))
    n = len(keys)
    ys = np.arange(n)[::-1]                                # hardest at the top

    fig = plt.figure(figsize=(9.6, 4.6))
    left = fig.add_gridspec(1, 1, left=0.15, right=0.42, top=0.92, bottom=0.16)
    ax_id = fig.add_subplot(left[0])

    # panel a: identification by four methods
    ax_id.axvline(0.5, color=MUTED, lw=1.1, ls="--", zorder=1)
    ax_id.text(0.515, n - 0.45, "chance", fontsize=7.5, color=MUTED, va="bottom")
    for y, k in zip(ys, keys):
        c = P.colour(k[0])
        vals = list(ident[k].values())
        ax_id.plot([min(vals), max(vals)], [y, y], color=c, lw=1.4, alpha=0.5, zorder=1)
        for m, v in ident[k].items():
            ax_id.scatter([v], [y], s=24, marker=MARKERS[m], facecolors="none", edgecolors=c,
                          linewidths=1.0, alpha=0.85, zorder=2)
        ax_id.scatter([mean[k]], [y], s=60, color=c, marker=P.marker(k[0]),
                      edgecolors="white", linewidths=0.5, zorder=3)
    ax_id.set_xlim(0, 1.03)
    ax_id.set_xlabel(f"Identification (share of {len(cache['cands']) - 1} other references beaten)\n"
                     f"references with split-half reliability ≥ {MIN_RELIABILITY}")
    ax_id.set_yticks(ys, [f"{P.PROJECTS[k[0]][1]} · {P.target_label(*k)}" for k in keys])
    for tick, k in zip(ax_id.get_yticklabels(), keys):
        tick.set_color(P.colour(k[0]))
        if k in EXEMPLARS:
            tick.set_fontweight("bold")
    ax_id.tick_params(axis="y", length=0)
    ax_id.set_ylim(-0.7, n - 0.3)
    ax_id.set_title("a  Identification by four methods", loc="left", fontsize=9.5,
                    fontweight="bold", x=-0.45)
    for ax in (ax_id,):
        ax.grid(axis="x", alpha=0.3, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(RULE)

    handles = [Line2D([], [], ls="", marker=MARKERS[m], mfc="none", mec=MUTED, ms=5.5, mew=1.0,
                      label=SHORT[m]) for m in MARKERS]
    handles.append(Line2D([], [], ls="", marker="o", color=MUTED, mec="white", ms=7,
                          label="mean"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, -0.045), ncol=5,
               frameon=False, fontsize=7.5, handletextpad=0.2, columnspacing=1.0)

    # panels b and c: two reliable hard cases
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
    summary = {}
    for col, (target, letter) in enumerate(zip(EXEMPLARS, "bc")):
        ex = exemplar_maps(target, ident, args.method, cache)
        summary[target] = {k: ex[k] for k in ("method", "closest", "ident", "tool_ref",
                                              "tool_other")}
        name = f"{P.PROJECTS[target[0]][1]} \u00b7 {P.target_label(*target)}"
        other = f"{P.PROJECTS[ex['closest'][0]][1]} \u00b7 {P.target_label(*ex['closest'])}"
        rows = (("ref", f"{name} reference\n{ex['n_studies']} studies, reliability "
                        f"{rel[target]:.2f}", P.colour(target[0])),
                ("tool", f"{ex['method']}, {ex['query']}\nr = {ex['tool_ref']:+.2f} with the "
                         f"reference", INK),
                ("other", f"{other} reference\nr = {ex['tool_other']:+.2f} with the "
                          f"{ex['method']} map", P.colour(ex["closest"][0])))
        x0 = 0.455 + col * 0.275
        grid = fig.add_gridspec(3, 1, left=x0, right=x0 + 0.255, top=0.86, bottom=0.03,
                                hspace=0.55)
        for i, (key, title, colour) in enumerate(rows):
            ax = fig.add_subplot(grid[i])
            ax.axis("off")
            img, thr, vmax = ex["shown"][key]
            plotting.plot_glass_brain(img, display_mode="ortho", threshold=thr, vmax=vmax,
                                      cmap=cmap, plot_abs=False, colorbar=False, axes=ax,
                                      annotate=False, black_bg=False, alpha=0.35)
            ax.set_title(title, fontsize=7.5, loc="left", pad=2, color=colour)
        fig.text(x0, 0.955, f"{letter}  {name}", fontsize=9.5, fontweight="bold", ha="left",
                 va="bottom")

    out = REPO / "results" / "figures"
    suffix = "" if args.method == "NeuroQuery" else "_" + args.method.lower().replace(" ", "_")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_difficulty_profile{suffix}.{ext}", dpi=400,
                    bbox_inches="tight", facecolor="white")
    print(f"  wrote results/figures/figure_difficulty_profile{suffix}.{{pdf,png}}")
    for t, v in summary.items():
        print(f"    {t}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
