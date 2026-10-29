#!/usr/bin/env python3
"""How reliable is each reference map? A split-half noise ceiling for every target.

WHY THIS EXISTS

Tools are scored by how well their predicted map correlates with a target's reference map
(results/text_to_map_baselines.csv). But a reference map is itself an estimate: an MKDA fit to a
finite sample of studies, and a different sample of equally eligible studies would have produced a
somewhat different map. No predictor -- not even the true population map -- can correlate with the
reference beyond the reference's own reliability. Without that ceiling a score of r2 = 0.12 is
uninterpretable: it could be a poor tool on a crisp target, or a near-perfect tool on a target
whose reference is mostly sampling noise. This script measures the ceiling per target, so tool
scores can be read against an honest upper bound and under-powered targets can be flagged.

THE PROCEDURE, for one target contrast:

    select the analyses the annotation marks true    (as the reference fit does)
    drop analyses with no finite coordinate          (as the reference fit does)
    group the surviving analyses by study
    for split = 0 .. n_splits - 1:
        shuffle the studies with numpy default_rng(seed=split), cut into two halves
        fit MKDA density on each half                (unthresholded z only)
        r_half = Pearson r of the two z maps inside the brain mask
        r_xx   = 2 r_half / (1 + r_half)             (Spearman-Brown)
    report mean / min / max over splits

WHY SPLIT BY STUDY, NOT BY ANALYSIS

Analyses from the same study share participants, scanner, preprocessing and often a baseline
condition, so their foci are correlated. Splitting at the analysis level would put sibling
analyses on both sides and the two halves would agree partly because they are partly the same
data, inflating the ceiling. The study is the independent sampling unit of a meta-analysis
(NiMARE's MKDA itself models within-study dependence by taking the max over a study's kernels
per experiment), so the halves here never share a study. Study ids in these studysets are PMIDs
and are unique within a project.

WHY SPEARMAN-BROWN, AND HOW TO READ r_xx

Each half has roughly half the evidence of the full reference, so the raw split correlation
understates the reliability of a map built from all of it. Spearman-Brown projects a half-length
reliability to full length: r_xx = 2 r / (1 + r). Under the classical model reference = signal +
independent noise, the correlation of the reference with its own noise-free signal is sqrt(r_xx),
so r_xx is approximately the largest r-squared ANY model can attain against this reference --
including a perfect one. Dividing a tool's r2 by r_xx therefore gives the fraction of the
EXPLAINABLE variance the tool recovers, which is the number comparable across targets of very
different reliability.

Caveats worth carrying into the paper. (1) Spearman-Brown assumes the halves are parallel
measurements; MKDA z is not linear in sample size (z grows with n), but the correlation is
scale-free, so the projection concerns pattern reliability only. (2) For small targets (vbm_of_ptsd
22 studies, vbm_of_substance_use/opioids 11) each half is five to eleven studies and the estimate is
itself noisy -- the min-max range across splits says how noisy. A low or unstable ceiling there is
the finding, not a failure. (3) Normalised scores can exceed 1 when a tool's r2 beats the estimated
ceiling, which happens when the ceiling is low and imprecise; read those as "at ceiling".
(4) The ceiling bounds predictors that are INDEPENDENT of the reference's sampling noise -- the
text-to-map tools, which never see these studies. It does not bound the pooled arm, which is
fit to a superset of the reference's own studies and so reproduces the reference's noise as well
as its signal; pooled r2 / r_xx sits at ~1 and above, and that is the expected consequence, not
a tool beating a perfect model. (5) Where r_xx <= 0 (opioids) no ratio is defined and the cell is
left empty; where r_xx is near zero the ratio explodes, so cross-target summaries should use the
median or the reliable subset (r_xx >= LOW_RELIABILITY), both of which are printed.

SPEED AND SIDE EFFECTS

The half-fits call MKDADensity().fit(dataset) directly -- no CBMAWorkflow, no corrector, no
diagnostics (an earlier job here ran 20x slower because a default Jackknife diagnostic slipped in),
and nothing is written to disk except the final CSV. Single process: memory on the machines this
runs on is tight.

Writes results/reference_reliability.csv: per target, the split counts, every split's raw r, the
reliability mean/min/max, the umbrella/degenerate flags carried over from
results/text_to_map_baselines.csv, and each tool's r2 next to r2 / r_xx. The pooled arm's
normalised score is left empty for umbrella targets, where pooled reproduces the reference by
construction and its r2 is excluded throughout the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_mask import common_mask  # noqa: E402
from nimads_meta import (  # noqa: E402
    _analysis_ids_for_column,
    _has_valid_coordinates,
    boolean_columns,
    load_studyset_and_annotation,
    name_by_id,
)
import eval_text_to_map as E  # noqa: E402

ARMS = ("neuroquery", "neurovlm", "pooled")
# Below this the reference is noisy enough that a tool's raw r2 materially understates it.
LOW_RELIABILITY = 0.7


def spearman_brown(r: float) -> float:
    """Full-length reliability from a split-half correlation."""
    if not math.isfinite(r) or r <= -1:
        return float("nan")
    return 2.0 * r / (1.0 + r)


def target_studies(studyset, annotation: dict, column: str) -> dict[str, list[str]]:
    """{study_id: [analysis_id, ...]} for the analyses the reference fit for `column` uses."""
    ids = _analysis_ids_for_column(annotation, column)
    if not ids:
        return {}
    sliced = studyset.slice(analyses=ids)
    # NiMARE >= 0.21 returns a view whose .studies still lists the parent's analyses; keep only
    # the selected ones (a no-op under 0.2.1, which returns a copy)
    selected = set(ids)
    out: dict[str, list[str]] = {}
    for s in sliced.studies:
        valid = [a.id for a in s.analyses if a.id in selected and _has_valid_coordinates(a)]
        if valid:
            out[s.id] = valid
    return out


def split_studies(study_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Random disjoint halves of the study list. Odd n puts the extra study in the second half."""
    ordered = sorted(study_ids)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    cut = len(ordered) // 2
    return [ordered[i] for i in perm[:cut]], [ordered[i] for i in perm[cut:]]


def fit_z(studyset, analysis_ids: list[str]) -> np.ndarray | None:
    """Unthresholded MKDA z on the 2mm grid, or None if nothing fittable."""
    from nimare.meta.cbma import MKDADensity

    if not analysis_ids:
        return None
    dataset = studyset.slice(analyses=analysis_ids).to_dataset()
    if len(dataset.ids) == 0 or len(dataset.coordinates) == 0:
        return None
    res = MKDADensity().fit(dataset)
    return np.asarray(res.get_map("z").get_fdata())


def masked_r(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    m = common_mask(a, b)
    if m.sum() < 2 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m].ravel(), b[m].ravel())[0, 1])


def split_half(studyset, studies: dict[str, list[str]], n_splits: int) -> list[float]:
    """Raw split-half r for seeds 0 .. n_splits-1 (NaN where a half could not be fit)."""
    rs = []
    for seed in range(n_splits):
        h1, h2 = split_studies(list(studies), seed)
        z1 = fit_z(studyset, [a for s in h1 for a in studies[s]])
        z2 = fit_z(studyset, [a for s in h2 for a in studies[s]])
        rs.append(masked_r(z1, z2))
    return rs


def _fmt(x: float | None, nd: int = 6):
    return "" if x is None or not math.isfinite(x) else round(x, nd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=REPO / "results" / "reference_reliability.csv")
    ap.add_argument("--n-splits", type=int, default=10, help="seeds 0 .. n-1 (default 10)")
    ap.add_argument("--projects", nargs="*", default=None,
                    help="restrict to these projects (for timing / spot checks)")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="restrict to these target columns")
    ap.add_argument("--baselines", type=Path,
                    default=REPO / "results" / "text_to_map_baselines.csv")
    args = ap.parse_args(argv)

    from nimare.nimads import Studyset

    base = {(r["project"], r["target"]): r for r in csv.DictReader(open(args.baselines))}
    wanted = [(p, t, umb, frac) for p, t, umb, frac in E.targets_with_umbrella_flag()
              if (args.projects is None or p in args.projects)
              and (args.targets is None or t in args.targets)]

    rows, t0 = [], time.time()
    loaded: dict[str, tuple] = {}
    for project, target, is_umbrella, pool_frac in wanted:
        if project not in loaded:
            loaded.clear()  # one project's Studyset in memory at a time
            d = NMB / "data" / "nimads" / project / "merged"
            ss, ann = load_studyset_and_annotation(d / "nimads_studyset.json",
                                                   d / "nimads_annotation.json")
            name_by_id(ss)
            loaded[project] = (Studyset(ss), ann)
        studyset, ann = loaded[project]
        if target not in boolean_columns(ann):
            continue

        tt = time.time()
        studies = target_studies(studyset, ann, target)
        rs = split_half(studyset, studies, args.n_splits)
        rel = [spearman_brown(r) for r in rs]
        ok_r = [r for r in rs if math.isfinite(r)]
        ok_rel = [x for x in rel if math.isfinite(x)]
        b = base.get((project, target), {})
        row = {
            "project": project, "target": target,
            "n_studies": len(studies),
            "n_analyses": sum(len(v) for v in studies.values()),
            "n_studies_half": len(studies) // 2,
            "umbrella_target": int(b.get("umbrella_target", int(is_umbrella))),
            "reference_degenerate": b.get("reference_degenerate", ""),
            "fraction_of_project_pool": pool_frac,
            "n_splits": args.n_splits, "n_splits_ok": len(ok_r),
            "mean_raw_r": _fmt(st.mean(ok_r)) if ok_r else "",
            "reliability_mean": _fmt(st.mean(ok_rel)) if ok_rel else "",
            "reliability_min": _fmt(min(ok_rel)) if ok_rel else "",
            "reliability_max": _fmt(max(ok_rel)) if ok_rel else "",
        }
        for i, r in enumerate(rs):
            row[f"raw_r_split{i}"] = _fmt(r)
        rxx = st.mean(ok_rel) if ok_rel else float("nan")
        for arm in ARMS:
            v = b.get(f"{arm}_r2", "")
            row[f"{arm}_r2"] = v
            excluded = arm == "pooled" and row["umbrella_target"]
            row[f"{arm}_r2_over_rxx"] = (
                _fmt(float(v) / rxx, 4)
                if v not in ("", None) and not excluded and math.isfinite(rxx) and rxx > 0
                else "")
        rows.append(row)
        print(f"  {project}/{target}: {row['n_studies']} studies, {row['n_analyses']} analyses, "
              f"r_half {row['mean_raw_r']}  r_xx {row['reliability_mean']} "
              f"[{row['reliability_min']}, {row['reliability_max']}]  "
              f"({time.time() - tt:.0f}s)", flush=True)

    if not rows:
        print("nothing evaluated", file=sys.stderr)
        return 1

    fields = (["project", "target", "n_studies", "n_analyses", "n_studies_half",
               "umbrella_target", "reference_degenerate", "fraction_of_project_pool",
               "n_splits", "n_splits_ok", "mean_raw_r",
               "reliability_mean", "reliability_min", "reliability_max"]
              + [f"raw_r_split{i}" for i in range(args.n_splits)]
              + [c for arm in ARMS for c in (f"{arm}_r2", f"{arm}_r2_over_rxx")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    rel = [float(r["reliability_mean"]) for r in rows if r["reliability_mean"] != ""]
    print(f"\n{len(rows)} targets, {args.n_splits} study-level splits each, "
          f"{time.time() - t0:.0f}s")
    if rel:
        print(f"  split-half reliability r_xx: mean {st.mean(rel):.3f}  "
              f"median {st.median(rel):.3f}  range {min(rel):.3f} .. {max(rel):.3f}")
    low = [r for r in rows if r["reliability_mean"] == ""
           or float(r["reliability_mean"]) < LOW_RELIABILITY]
    if low:
        print(f"\n  reliability below {LOW_RELIABILITY} (reference is substantially noise):")
        for r in sorted(low, key=lambda r: float(r["reliability_mean"] or "nan")):
            print(f"    {r['project'] + '/' + r['target']:<62} r_xx {r['reliability_mean']}  "
                  f"[{r['reliability_min']}, {r['reliability_max']}]  n_studies {r['n_studies']}")

    print("\n  tool r2, raw vs as a fraction of the ceiling (r2 / r_xx; degenerate references "
          "excluded; pooled also excludes umbrella)")
    for arm in ARMS:
        el = [r for r in rows if str(r["reference_degenerate"]) != "1"
              and r[f"{arm}_r2"] not in ("", None) and r[f"{arm}_r2_over_rxx"] != ""]
        if el:
            raw = [float(r[f"{arm}_r2"]) for r in el]
            nrm = [float(r[f"{arm}_r2_over_rxx"]) for r in el]
            print(f"    {arm:<12} n={len(el):<3} r2 mean {st.mean(raw):.3f}  "
                  f"r2/r_xx mean {st.mean(nrm):.3f}  median {st.median(nrm):.3f}")
            rel_el = [r for r in el if float(r["reliability_mean"]) >= LOW_RELIABILITY]
            if rel_el:
                print(f"    {'':<12} n={len(rel_el):<3} r_xx >= {LOW_RELIABILITY}: r2 mean "
                      f"{st.mean(float(r[f'{arm}_r2']) for r in rel_el):.3f}  r2/r_xx mean "
                      f"{st.mean(float(r[f'{arm}_r2_over_rxx']) for r in rel_el):.3f}")
    print("    (pooled shares the reference's own studies, so the ceiling does not bound it)")
    print(f"\n  wrote {args.output.relative_to(REPO) if args.output.is_relative_to(REPO) else args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
