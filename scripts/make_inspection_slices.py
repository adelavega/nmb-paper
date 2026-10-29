#!/usr/bin/env python3
"""Inspection figure: thresholded maps for every target, reference vs NeuroQuery vs NeuroVLM.

For eyeballing, not publication. One row per target, three blocks per row (NeuroMetaBench
reference, NeuroQuery, NeuroVLM), each shown on the same fixed axial slices so rows are directly
comparable. Each row is labelled with the tools' r2 against the reference.

THRESHOLDS -- the three maps are not on the same scale, so "p < 0.01" has to be stated per map:

    reference    MKDA z, uncorrected, one-sided p < 0.01  ->  z > 2.326
    NeuroQuery   its output is a z map (brain_map == z_map in this version)  ->  z > 2.326
    Neurosynth   association and uniformity maps are both z maps  ->  z > 2.326 each
    NeuroVLM     NOT a z map (values ~0-0.4; nothing ever exceeds 2.33), so p < 0.01 is undefined.
                 Shown instead: the top-k voxels, k = the reference's own p < 0.01 extent. This is
                 the same rule the top-k Dice metric uses, and it is the fairest visual match.

ORDER AND SCOPE -- by default only the 35 selected targets (data/selected_targets.csv). Projects
are ordered by their mean NeuroQuery r2 over those targets, best first (from
results/r2_by_project_selected.csv); within a project, targets are ordered by the reference's
split-half ceiling, most reliable first (results/reference_reliability.csv). Reading down a page
therefore separates "the tool fails on this project" from "this reference is too noisy to
recover". --all-targets restores all 38.

Writes results/figures/inspection/inspection_slices_p{1..n}.png and a combined PDF.
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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_palette as P  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_mask import brain_mask  # noqa: E402
import eval_text_to_map as E  # noqa: E402

Z_P01 = 2.326
CUTS = [-18, -2, 14, 30, 46]
ROWS_PER_PAGE = 10
SHORT = {"cue_reactivity": "cue", "decision_making": "decision", "dementia": "dementia",
         "emotion_regulation_2022": "emo-reg", "executive_function": "exec-fn",
         "problem_solving": "prob-solv", "social": "social", "vbm_of_ptsd": "vbm-ptsd",
         "vbm_of_substance_use": "vbm-sud"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path,
                    default=REPO / "results" / "figures" / "inspection")
    ap.add_argument("--all-targets", action="store_true",
                    help="include all 38 reference maps, not just the 35 selected targets")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import nibabel as nib
    from nilearn import plotting
    from nilearn.image import resample_to_img
    E._RESAMPLE_PARAMS = set(inspect.signature(resample_to_img).parameters)

    inside = brain_mask()
    ev = {(r["project"], r["target"]): r
          for r in csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))}
    rel_path = REPO / "results" / "reference_reliability.csv"
    rel = ({(r["project"], r["target"]): float(r["reliability_mean"])
            for r in csv.DictReader(open(rel_path))} if rel_path.exists() else {})

    selected = {(r["project"], r["target"])
                for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv"))}
    proj_nq = {r["project"]: float(r["neuroquery_mean_r2"])
               for r in csv.DictReader(open(REPO / "results" / "r2_by_project_selected.csv"))
               if r["project"] != "ALL" and r["neuroquery_mean_r2"] not in ("", None)}
    ns_term = {(r["project"], r["target"]): r["neurosynth_term"]
               for r in csv.DictReader(open(REPO / "data" / "neurosynth_terms.csv"))}
    targets = [x for x in E.targets_with_umbrella_flag()
               if args.all_targets or (x[0], x[1]) in selected]
    targets.sort(key=lambda x: (-proj_nq.get(x[0], -1.0), x[0],
                                -rel.get((x[0], x[1]), -9.0), x[1]))

    rows = []
    for proj, targ, umb, _ in targets:
        rd = E.reference_dir(proj, targ)
        if rd is None:
            continue
        ref = nib.load(str(rd / "z.nii.gz"))
        g = ref.get_fdata()
        k = int(((g > Z_P01) & inside).sum())

        def thresholded(arr, rule):
            out = np.zeros_like(arr)
            if rule == "z":
                keep = (arr > Z_P01) & inside
            else:  # top-k, extent matched to the reference
                if k == 0:
                    return out, 0
                vals = np.where(inside & np.isfinite(arr), arr, -np.inf)
                idx = np.argpartition(vals.ravel(), -k)[-k:]
                keep = np.zeros(arr.size, bool)
                keep[idx] = True
                keep = keep.reshape(arr.shape)
            out[keep] = arr[keep]
            return out, int(keep.sum())

        cells = []
        gr, nr = thresholded(g, "z")
        cells.append(("NeuroMetaBench  z>2.33", nib.Nifti1Image(gr, ref.affine), nr))
        nq = REPO / "results" / "neuroquery_maps" / f"{proj}__{targ}.nii.gz"
        if nq.exists():
            a = np.asarray(nib.load(str(nq)).dataobj, dtype=float)
            ar, na = thresholded(a, "z")
            cells.append(("NeuroQuery  z>2.33", nib.Nifti1Image(ar, ref.affine), na))
        else:
            cells.append(("NeuroQuery", None, 0))
        ns_dir = REPO / "results" / "neurosynth_maps" / ns_term.get((proj, targ), "?").replace(" ", "_")
        for fname, lab in (("z_desc-association.nii.gz", "Neurosynth assoc"),
                           ("z_desc-uniformity.nii.gz", "Neurosynth unif")):
            f = ns_dir / fname
            if f.exists():
                a = np.asarray(nib.load(str(f)).dataobj, dtype=float)
                ar, na = thresholded(a, "z")
                cells.append((f"{lab}  z>2.33", nib.Nifti1Image(ar, ref.affine), na))
            else:
                cells.append((lab, None, 0))
        nv = REPO / "data" / "neurovlm_maps" / f"{proj}__{targ}.nii.gz"
        if nv.exists():
            a = np.asarray(E.resample(nib.load(str(nv)), ref).dataobj, dtype=float)
            ar, na = thresholded(a, "topk")
            cells.append((f"NeuroVLM  top-{k:,} voxels", nib.Nifti1Image(ar, ref.affine), na))
        else:
            cells.append(("NeuroVLM", None, 0))

        e = ev.get((proj, targ), {})
        fmt = lambda v: f"{float(v):.2f}" if v not in ("", None) else "--"
        flags = ("  [umbrella]" if umb else "") + \
                ("  [degenerate]" if e.get("reference_degenerate") == "1" else "")
        label = (f"{SHORT.get(proj, proj)} (project NQ mean {proj_nq.get(proj, float('nan')):.3f})\n"
                 f"  {P.target_label(proj, targ)}{flags}\n"
                 f"r²  NQ {fmt(e.get('neuroquery_r2'))}  NVLM {fmt(e.get('neurovlm_r2'))}  "
                 f"NS a/u {fmt(e.get('neurosynth_assoc_r2'))}/{fmt(e.get('neurosynth_unif_r2'))}"
                 + (f"   ceiling {rel[(proj, targ)]:.2f}" if (proj, targ) in rel else "")
                 + f"\nquery: \"{e.get('query', '')[:38]}\"\nNS term: \"{ns_term.get((proj, targ), '')}\"")
        rows.append((label, cells))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = [rows[i:i + ROWS_PER_PAGE] for i in range(0, len(rows), ROWS_PER_PAGE)]
    pdf_path = args.output_dir / "inspection_slices.pdf"
    with PdfPages(pdf_path) as pdf:
        for pi, page in enumerate(pages, 1):
            fig = plt.figure(figsize=(30, 1.6 * len(page) + 0.7))
            gs = fig.add_gridspec(len(page), 6, width_ratios=[0.78, 1, 1, 1, 1, 1],
                                  wspace=0.03, hspace=0.12,
                                  left=0.005, right=0.995, top=1 - 0.95 / fig.get_figheight(),
                                  bottom=0.01)
            for ri, (label, cells) in enumerate(page):
                lax = fig.add_subplot(gs[ri, 0])
                lax.axis("off")
                lax.text(0.0, 0.5, label, va="center", ha="left", fontsize=8.5,
                         family="monospace", linespacing=1.5)
                for ci, (title, img, n) in enumerate(cells):
                    ax = fig.add_subplot(gs[ri, ci + 1])
                    if img is None:
                        ax.axis("off")
                        ax.text(0.5, 0.5, f"{title}: no map", ha="center", va="center",
                                fontsize=9, color="#888888")
                        continue
                    data = np.asarray(img.dataobj)
                    vmax = float(np.nanmax(data)) if np.any(data > 0) else 1.0
                    plotting.plot_stat_map(img, display_mode="z", cut_coords=CUTS,
                                           axes=ax, colorbar=False, annotate=False,
                                           draw_cross=False, black_bg=False,
                                           threshold=1e-9, vmax=vmax, cmap="hot")
                    if ri == 0:
                        ax.set_title(title.split("  ")[0], fontsize=11, pad=4)
                    ax.text(0.995, 0.02, f"{n:,} vox", transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=7.5, color="#333333")
            fig.text(0.005, 1 - 0.08 / fig.get_figheight(),
                     f"page {pi}/{len(pages)}   reference, NeuroQuery & Neurosynth: z > 2.33 (p < 0.01, "
                     f"uncorrected, one-sided)   NeuroVLM: top-k voxels matched to the "
                     f"reference's p < 0.01 extent (its output is not a z map)   "
                     f"axial z = {CUTS}\nsorted: projects by mean NeuroQuery r² (best first), "
                     f"targets within a project by reference ceiling (most reliable first)",
                     fontsize=9, color="#444444", va="top")
            png = args.output_dir / f"inspection_slices_p{pi}.png"
            fig.savefig(png, dpi=110, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            print(f"  wrote {png.relative_to(REPO)}  ({len(page)} targets)")
    print(f"  wrote {pdf_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
