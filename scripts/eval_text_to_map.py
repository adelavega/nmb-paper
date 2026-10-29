#!/usr/bin/env python3
"""Benchmark published text-to-map tools against the NeuroMetaBench reference maps.

THE QUESTION

Automated neuroimaging meta-analysis tools take a text query and return a brain map. None of
them exposes which studies or analyses it used, so the only evaluation available is end-to-end:
how close is the returned map to what an expert meta-analysis of the same question produced?

ARMS

    neuroquery   Dockes et al. 2020, a regularised text-to-map regression over ~13k studies.
                 Generated here from the published model.
    neurovlm     embedding-based text-to-map. Maps are pre-generated (the model needs its own
                 environment) and read from --neurovlm-maps.
    neurosynth   Yarkoni et al. 2011, term-based MKDA chi-squared over the 14,371-study v7
                 corpus (scripts/run_neurosynth_arm.py). One vocabulary term per target, from
                 data/neurosynth_terms.csv. Scored on BOTH of its outputs: `neurosynth_assoc`
                 (association test, the map Neurosynth shows users) and `neurosynth_unif`
                 (uniformity test, the closer analogue of a density reference).
    pooled       every analysis in the project's studyset, meta-analysed together. No text, no
                 tool: this is the benchmark answering "what if you pooled the right literature
                 but did not pick the right analyses within it?" It is the internal reference
                 point that makes a text-to-map score interpretable.

                 An OPTIMISTIC bound, not a ceiling. The benchmark records only analyses the
                 expert selected for some target (all 2,219 belong to >= 1 target; no orphans),
                 so this pool has already been filtered for relevance. A method that extracted
                 every analysis an included paper reports would pool more, including irrelevant
                 ones, and score lower. See run_pooled_baseline.py.

                 Note also that the pooled arm is HANDED the expert's study list, while the
                 text-to-map arms must find the literature from a query. The arms are not
                 matched on what they are given, so the gap between them is not a decomposition
                 of anything.

METRICS

    r2        squared Pearson correlation between unthresholded maps, inside the brain mask.
              The primary measure. Text-to-map tools emit no thresholded map, so this is the
              only metric all arms can share.
    topk_dice Dice between the k highest-valued voxels of each map and the k suprathreshold
              voxels of the reference, where k is taken from the reference's own FDR-corrected
              map. Form-insensitive: it asks whether the maps point at the same places without
              requiring the tool to produce a calibrated statistic.

Writes results/text_to_map_baselines.csv.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_mask import common_mask  # noqa: E402

UNCORRECTED = "z.nii.gz"
# Chi-squared reference flavours, produced by scripts/run_chi2_meta.py. `association` is the
# reverse-inference map: activation selective for this topic rather than for the literature at
# large. Scored alongside density so the paper can answer "are these tools being penalised for
# doing reverse inference?" rather than assume either way.
CHI2_FLAVOURS = {"uniformity": "z_desc-uniformity.nii.gz",
                 "association": "z_desc-association.nii.gz"}
CORRECTED = "z_corr-FDR_method-indep.nii.gz"
DICE_Z = 1.96
MIN_TOPK = 50

# A target selecting at least this fraction of its project's analyses is treated as an
# umbrella: the project-pooled arm reproduces it, so scoring the two against each other is
# circular. Exact equality is too strict -- executive_function/all takes 259 of 271 analyses
# (95.6%) and problem_solving/global_... takes 281 of 282 (99.6%), and both score r2 >= 0.994
# against the pooled map. They are circular in substance while passing an == test.
UMBRELLA_FRACTION = 0.95

# One query per target contrast. The wording follows the construct the source meta-analysis
# names, not the benchmark's internal column key. The whole-brain-only variants of the cue
# reactivity targets are the same construct as their parent, so they share a query.
QUERIES: dict[tuple[str, str], str] = {
    ("cue_reactivity", "1_reward_neutral_2020"): "cue reactivity",
    ("cue_reactivity", "1_reward_neutral_2020_wbonly"): "cue reactivity",
    ("cue_reactivity", "2_drug_neutral_2020"): "drug cue reactivity",
    ("cue_reactivity", "2_drug_neutral_2020_wbonly"): "drug cue reactivity",
    ("cue_reactivity", "3_natural_neutral_2020"): "natural reward cue reactivity",
    ("cue_reactivity", "3_natural_neutral_2020_wbonly"): "natural reward cue reactivity",
    ("decision_making", "adm_july2019"): "decision making under ambiguity",
    ("decision_making", "perceptual_dm_july2019"): "perceptual decision making",
    ("decision_making", "rdm_july2019"): "risky decision making",
    ("dementia", "all"): "dementia",
    ("dementia", "decrease"): "decreased brain activity in dementia",
    ("dementia", "functional"): "functional activation in dementia",
    ("dementia", "structural"): "gray matter atrophy in dementia",
    ("emotion_regulation_2022", "decrease"): "decreasing negative emotion",
    ("emotion_regulation_2022", "increase"): "increasing emotional response",
    ("emotion_regulation_2022", "maintain"): "maintaining emotional response",
    ("emotion_regulation_2022", "reappraisal"): "cognitive reappraisal",
    ("executive_function", "all"): "executive function",
    ("executive_function", "flexibility"): "cognitive flexibility task switching",
    ("executive_function", "inhibition"): "response inhibition",
    ("executive_function", "working_memory"): "working memory",
    ("problem_solving", "calculation_mni_final"): "mental arithmetic calculation",
    ("problem_solving", "demand_mni_final"): "logical reasoning",
    ("problem_solving",
     "global_calculationandvisuospatialandword_mni_final"): "problem solving reasoning",
    ("problem_solving", "visuospatialreasoning_mni_final"): "visuospatial reasoning",
    ("problem_solving", "wordproblems_mni_final"): "verbal problem solving",
    ("social", "affiliation_merged"): "attachment and affiliation",
    ("social", "all_merged"): "social cognition",
    ("social", "others_merged"): "person perception of others",
    ("social", "self_merged"): "self referential processing",
    ("social", "soccomm_merged"): "social communication",
    ("vbm_of_ptsd", "nonptsdgtptsd_merged"): "posttraumatic stress disorder gray matter",
    ("vbm_of_substance_use", "alcohol"): "alcohol dependence gray matter",
    ("vbm_of_substance_use", "all_drug_classes"): "substance use disorder gray matter",
    ("vbm_of_substance_use", "cannabis"): "cannabis use gray matter",
    ("vbm_of_substance_use", "nicotine"): "nicotine smoking gray matter",
    ("vbm_of_substance_use", "opioids"): "opioid dependence gray matter",
    ("vbm_of_substance_use", "stimulants"): "stimulant use gray matter",
}

_RESAMPLE_PARAMS: set[str] = set()
# Brain masking is ON. --no-brain-mask exists only to reproduce the sensitivity check
# reported in the Methods; it is never the number to publish. See r2().
_BRAIN_MASK = True


def targets() -> list[tuple[str, str]]:
    """Every (project, target) the benchmark ships a reference map for."""
    return [(p, c) for p, c, _, _ in targets_with_umbrella_flag()]


def targets_with_umbrella_flag() -> list[tuple[str, str, bool, float]]:
    """(project, target, is_umbrella, fraction_of_project_pool).

    A target is `umbrella` when it selects at least UMBRELLA_FRACTION of every analysis in its
    project's studyset, so the project-pooled arm effectively reproduces it. Seven of the 38
    qualify at 0.95. Those are excluded from the pooled-vs-expert comparison, which would
    otherwise be circular; they remain valid targets for the text-to-map arms, which never see
    the analyses.
    """
    out = []
    for ann_path in sorted(NMB.glob("data/nimads/*/merged/nimads_annotation.json")):
        project = ann_path.parent.parent.name
        ann = json.loads(ann_path.read_text())
        ann = ann[0] if isinstance(ann, list) else ann
        notes = ann.get("notes", []) or []
        total = len({n.get("analysis") for n in notes})
        for col in sorted(k for k, v in (ann.get("note_keys") or {}).items() if v == "boolean"):
            n_col = sum(1 for n in notes if (n.get("note") or {}).get(col))
            frac = n_col / total if total else 0.0
            out.append((project, col, frac >= UMBRELLA_FRACTION, round(frac, 4)))
    return out


def reference_dir(project: str, target: str) -> Path | None:
    """analysis/<project>/<target>, tolerating the case differences in social/."""
    base = NMB / "analysis" / project
    exact = base / target
    if (exact / UNCORRECTED).exists():
        return exact
    for cand in base.glob("*"):
        if cand.name.lower().replace("-", "_") == target.lower() and \
                (cand / UNCORRECTED).exists():
            return cand
    return None


def resample(img, ref):
    from nilearn.image import resample_to_img
    kw = dict(interpolation="continuous", force_resample=True)
    if "copy_header" in _RESAMPLE_PARAMS:
        kw["copy_header"] = True
    return resample_to_img(img, ref, **kw)


def r2(a: np.ndarray, g: np.ndarray) -> float | str:
    """Squared Pearson correlation over voxels finite in both maps, inside the brain mask.

    The mask is not cosmetic. MKDA maps are exactly 0 outside the brain, while the text-to-map
    tools are not: after resampling, 64% of NeuroVLM's out-of-brain voxels are exactly zero
    (agreeing with the reference's zeros, which inflates r-squared) whereas 42% of NeuroQuery's
    are non-zero (disagreeing, which deflates it). Unmasked, those 674,146 out-of-brain voxels
    are 75% of the correlation's n and move the arms in OPPOSITE directions -- NeuroVLM +0.096,
    NeuroQuery -0.016 -- so an unmasked comparison partly measures how each tool treats space
    outside the brain. Masked, NeuroVLM leads NeuroQuery 1.6x; unmasked it appears to lead 3.2x.
    """
    m = common_mask(a, g, brain=_BRAIN_MASK)
    if m.sum() < 2 or a[m].std() == 0 or g[m].std() == 0:
        return ""
    return round(float(np.corrcoef(a[m].ravel(), g[m].ravel())[0, 1] ** 2), 6)


def top_k_mask(a: np.ndarray, k: int, inside: np.ndarray) -> np.ndarray:
    """The k highest-valued voxels within the brain mask."""
    vals = np.where(inside & np.isfinite(a), a, -np.inf)
    idx = np.argpartition(vals.ravel(), -k)[-k:]
    m = np.zeros(a.size, dtype=bool)
    m[idx] = True
    return m.reshape(a.shape)


def top_k_dice(a: np.ndarray, g_supra: np.ndarray, k: int, inside: np.ndarray) -> float:
    ma = top_k_mask(a, k, inside)
    inter = int((ma & g_supra).sum())
    total = int(ma.sum()) + int(g_supra.sum())
    return 0.0 if total == 0 else round(2.0 * inter / total, 6)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=REPO / "results" / "text_to_map_baselines.csv")
    ap.add_argument("--neurovlm-maps", type=Path, default=REPO / "data" / "neurovlm_maps",
                    help="pre-generated NeuroVLM maps, named <project>__<target>.nii.gz")
    ap.add_argument("--neuroquery-cache", type=Path, default=REPO / "results" / "neuroquery_maps",
                    help="where generated NeuroQuery maps are written")
    ap.add_argument("--neurosynth-maps", type=Path, default=REPO / "results" / "neurosynth_maps")
    ap.add_argument("--neurosynth-terms", type=Path,
                    default=REPO / "data" / "neurosynth_terms.csv")
    ap.add_argument("--pooled-dir", type=Path, default=REPO / "results" / "pooled_all_analyses",
                    help="output of scripts/run_pooled_baseline.py, one dir per project")
    ap.add_argument("--chi2-dir", type=Path,
                    default=REPO / "results" / "chi2_neurosynth_v7",
                    help="MKDA chi-squared references; those columns are skipped when absent")
    ap.add_argument("--skip-neuroquery", action="store_true")
    ap.add_argument("--no-brain-mask", action="store_true",
                    help="Sensitivity check only: correlate over the whole volume. The "
                         "published numbers are brain-masked -- see r2().")
    args = ap.parse_args(argv)

    import nibabel as nib
    from nilearn.image import resample_to_img
    global _RESAMPLE_PARAMS
    _RESAMPLE_PARAMS = set(inspect.signature(resample_to_img).parameters)

    global _BRAIN_MASK
    _BRAIN_MASK = not args.no_brain_mask
    from map_mask import brain_mask
    inside = brain_mask()
    if not _BRAIN_MASK:
        print("!! --no-brain-mask: r-squared over the whole volume. Sensitivity check "
              "only, not publishable.")

    model = None
    if not args.skip_neuroquery:
        from neuroquery import fetch_neuroquery_model, NeuroQueryModel
        print("loading the NeuroQuery model")
        model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())
    args.neuroquery_cache.mkdir(parents=True, exist_ok=True)

    ns_term = ({(r["project"], r["target"]): r["neurosynth_term"]
                for r in csv.DictReader(open(args.neurosynth_terms))}
               if args.neurosynth_terms.exists() else {})

    rows, notes = [], []
    for project, target, is_umbrella, pool_frac in targets_with_umbrella_flag():
        rd = reference_dir(project, target)
        if rd is None:
            notes.append(f"{project}/{target}: no reference map")
            continue
        ref_img = nib.load(str(rd / UNCORRECTED))
        g = ref_img.get_fdata()

        # k and the suprathreshold set come from the reference's own corrected map.
        corr = rd / CORRECTED
        g_supra = (nib.load(str(corr)).get_fdata() > DICE_Z) & inside if corr.exists() \
            else np.zeros_like(g, dtype=bool)
        k = int(g_supra.sum())
        degenerate = k == 0

        row = {"project": project, "target": target,
               "query": QUERIES.get((project, target), ""),
               "reference_suprathreshold_voxels": k,
               "reference_degenerate": int(degenerate),
               "umbrella_target": int(is_umbrella),
               "fraction_of_project_pool": pool_frac}

        arm_arrays: dict[str, np.ndarray] = {}

        # --- NeuroQuery
        query = QUERIES.get((project, target))
        if model is not None and query:
            pred = resample(model(query)["brain_map"], ref_img)
            nib.save(pred, args.neuroquery_cache / f"{project}__{target}.nii.gz")
            a = np.asarray(pred.dataobj)
            arm_arrays["neuroquery"] = a
            row["neuroquery_r2"] = r2(a, g)
            row["neuroquery_topk_dice"] = (top_k_dice(a, g_supra, k, inside)
                                           if k >= MIN_TOPK else "")
        elif not query:
            notes.append(f"{project}/{target}: no query defined")

        # --- NeuroVLM (pre-generated)
        nv = args.neurovlm_maps / f"{project}__{target}.nii.gz"
        if nv.exists():
            nvr = resample(nib.load(str(nv)), ref_img)
            a = np.asarray(nvr.dataobj)
            arm_arrays["neurovlm"] = a
            row["neurovlm_r2"] = r2(a, g)
            row["neurovlm_topk_dice"] = top_k_dice(a, g_supra, k, inside) if k >= MIN_TOPK else ""
        else:
            notes.append(f"{project}/{target}: no NeuroVLM map")

        # --- Neurosynth: one term per target, both outputs scored
        term = ns_term.get((project, target))
        if term:
            row["neurosynth_term"] = term
            for arm, fname in (("neurosynth_assoc", "z_desc-association.nii.gz"),
                               ("neurosynth_unif", "z_desc-uniformity.nii.gz")):
                fpath = args.neurosynth_maps / term.replace(" ", "_") / fname
                if not fpath.exists():
                    notes.append(f"{project}/{target}: no Neurosynth map for '{term}'")
                    continue
                img = nib.load(str(fpath))
                a = (img.get_fdata() if img.shape == g.shape
                     else np.asarray(resample(img, ref_img).dataobj))
                arm_arrays[arm] = a
                row[f"{arm}_r2"] = r2(a, g)
                row[f"{arm}_topk_dice"] = (top_k_dice(a, g_supra, k, inside)
                                           if k >= MIN_TOPK else "")

        # --- pooled-all-analyses (internal, tool-free)
        pooled = args.pooled_dir / project / "pooled" / UNCORRECTED
        if pooled.exists():
            a = nib.load(str(pooled)).get_fdata()
            if a.shape == g.shape:
                arm_arrays["pooled"] = a
                row["pooled_r2"] = r2(a, g)
                row["pooled_topk_dice"] = (top_k_dice(a, g_supra, k, inside)
                                           if k >= MIN_TOPK else "")
                # Each tool against the PROJECT-POOLED map: does the tool recover the topic,
                # even if it misses the target? Separates two failure modes. If a tool lands
                # near the pooled map but not the target, it has the topic and lacks the
                # contrast-level distinction. If it misses the pooled map too, it fails at the
                # topic level -- which is what happens for emotion regulation (NeuroQuery 0.02,
                # NeuroVLM 0.07, against 0.3 for executive function and problem solving), so
                # its low scores cannot be attributed to the targets' analysis-level separability.
                for arm in ("neuroquery", "neurovlm"):
                    if arm in arm_arrays:
                        row[f"{arm}_r2_vs_pooled"] = r2(arm_arrays[arm], a)
        # --- the same arms, scored against the chi-squared reference flavours
        for flavour, fname in CHI2_FLAVOURS.items():
            fpath = args.chi2_dir / project / target / fname
            if not fpath.exists():
                continue
            gf = nib.load(str(fpath)).get_fdata()
            for arm, a in arm_arrays.items():
                if a.shape == gf.shape:
                    row[f"{arm}_r2_{flavour}"] = r2(a, gf)

        rows.append(row)

    if not rows:
        print("nothing evaluated", file=sys.stderr)
        return 1

    fields = ["project", "target", "query", "reference_suprathreshold_voxels",
              "reference_degenerate", "umbrella_target", "fraction_of_project_pool",
              "neuroquery_r2", "neuroquery_topk_dice",
              "neurovlm_r2", "neurovlm_topk_dice", "pooled_r2", "pooled_topk_dice",
              "neuroquery_r2_uniformity", "neuroquery_r2_association",
              "neurovlm_r2_uniformity", "neurovlm_r2_association",
              "pooled_r2_uniformity", "pooled_r2_association",
              "neuroquery_r2_vs_pooled", "neurovlm_r2_vs_pooled",
              "neurosynth_term", "neurosynth_assoc_r2", "neurosynth_assoc_topk_dice",
              "neurosynth_unif_r2", "neurosynth_unif_topk_dice",
              "neurosynth_assoc_r2_uniformity", "neurosynth_assoc_r2_association",
              "neurosynth_unif_r2_uniformity", "neurosynth_unif_r2_association"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k2: r.get(k2, "") for k2 in fields})

    scored = [r for r in rows if not r["reference_degenerate"]]
    n_umb = sum(r["umbrella_target"] for r in rows)
    print(f"\n{len(rows)} target contrasts; {len(scored)} with a non-empty reference map; "
          f"{n_umb} umbrella targets (>= {UMBRELLA_FRACTION:.0%} of their project's pool)")
    for arm in ("neuroquery", "neurovlm", "neurosynth_assoc", "neurosynth_unif", "pooled"):
        # The pooled arm reproduces an umbrella target exactly, so scoring it there is circular.
        elig = [r for r in scored if not (arm == "pooled" and r["umbrella_target"])]
        v = [float(r[f"{arm}_r2"]) for r in elig if r.get(f"{arm}_r2") not in ("", None)]
        d = [float(r[f"{arm}_topk_dice"]) for r in elig
             if r.get(f"{arm}_topk_dice") not in ("", None)]
        if v:
            tag = "  (umbrella targets excluded)" if arm == "pooled" else ""
            print(f"  {arm:<17} mean r2 {st.mean(v):.3f}  median {st.median(v):.3f}  n={len(v)}"
                  + (f"   |  top-k dice mean {st.mean(d):.3f} (n={len(d)})" if d else "") + tag)
    umb = [f"{r['project']}/{r['target']} ({r['fraction_of_project_pool']:.0%})"
           for r in rows if r["umbrella_target"]]
    if umb:
        print(f"\n  umbrella, excluded from the pooled arm as circular: {', '.join(umb)}")
    if any(r.get("neuroquery_r2_association") not in ("", None) for r in rows):
        print(f"\n  r2 by reference flavour (degenerate excluded; pooled also excludes umbrella)")
        print(f"  {'arm':<18}" + "".join(f"{f:>16}" for f in
                                         ("density", "uniformity", "association")))
        for arm in ("neuroquery", "neurovlm", "neurosynth_assoc", "neurosynth_unif", "pooled"):
            elig = [r for r in scored if not (arm == "pooled" and r["umbrella_target"])]
            line = f"  {arm:<18}"
            for f, col in (("density", f"{arm}_r2"),
                           ("uniformity", f"{arm}_r2_uniformity"),
                           ("association", f"{arm}_r2_association")):
                v = [float(r[col]) for r in elig if r.get(col) not in ("", None)]
                line += f"{st.mean(v):>10.3f} (n={len(v):>2})" if v else f"{'--':>16}"
            print(line)

    deg = [f"{r['project']}/{r['target']}" for r in rows if r["reference_degenerate"]]
    if deg:
        print(f"\n  reference map has no suprathreshold voxel (cannot be scored on overlap): "
              f"{', '.join(deg)}")
    for n in notes:
        print(f"  note: {n}")
    print(f"\n  wrote {args.output.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
