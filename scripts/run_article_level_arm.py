#!/usr/bin/env python3
"""The article-level arm: the right papers, without choosing analyses within them.

THE QUESTION

A target can need analysis-level selection for two different reasons, and only one of them is
solved by finding the right papers:

    SHARED LITERATURE  its studies are shared with sibling targets, which use different analyses
                       from the same papers (emotion regulation: maintain / decrease / reappraisal
                       draw on ~60% of the same studies and none of the same analyses).
    OWN LITERATURE     its studies are largely its own, but they are a small slice of the project
                       pool, so pooling the project dilutes it (executive_function/flexibility,
                       problem_solving/visuospatial).

The project-pooled arm cannot tell these apart: it fails both. This arm can. For each target it
takes the target's own studies and meta-analyses EVERY annotated analysis they contain -- article
selection done perfectly, analysis selection not done at all. If the hypothesis is right, it
recovers the own-literature targets almost exactly and fails the shared-literature ones.

WHY THE NO-ORPHANS PROPERTY HELPS HERE

The benchmark records only analyses an expert selected for some target; the analyses rejected for
every target were never written down. So the analyses this arm adds beyond the target's own are
exactly the analyses its sibling targets took from the same papers. That is the contamination the
hypothesis is about, isolated.

THE CAVEAT CUTS IN OPPOSITE DIRECTIONS FOR THE TWO HALVES OF THE RESULT

The benchmark only tracks analyses included in some target, so it is biased toward analyses
relevant to at least one question. A real article-level system would also pull in everything else
those papers report -- control contrasts, ROI analyses, unrelated tasks -- none of it recorded here.
Every deficit this arm measures is therefore a LOWER BOUND.

  - A FAILURE is conservative. The emotion-regulation targets lose 0.31-0.50 of their signal from
    sibling analyses alone; unrecorded analyses could only add to that.
  - A RESCUE is not established. A near-zero deficit (EF flexibility 0.04, PS visuospatial 0.01)
    shows only that no recorded sibling analysis shares those papers. Whether the papers' other,
    unrecorded analyses would dilute the map is unknown. Report these as "not contaminated by
    sibling targets", never as "finding the right papers is sufficient".

PROCEDURE, per target: collect the PMIDs of its selected analyses; take every analysis in the
project studyset belonging to those PMIDs (dropping any without coordinates, as the reference fit
does); fit MKDADensity with NiMARE defaults; correlate the unthresholded z with the reference z
inside the brain mask. Fits go straight to MKDADensity -- no CBMAWorkflow, no corrector, no
diagnostics -- because only the unthresholded z is needed. The same path applied to the target's
own analyses reproduces the committed reference map (checked by --verify).

Writes results/article_level_arm.csv.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
import nimads_meta  # noqa: E402
from map_mask import common_mask  # noqa: E402
import eval_text_to_map as E  # noqa: E402


def fit_z(studyset, analysis_ids):
    from nimare.meta.cbma import MKDADensity
    ss = studyset.slice(analyses=list(analysis_ids))
    valid = [a.id for s in ss.studies for a in s.analyses if nimads_meta._has_valid_coordinates(a)]
    ss = ss.slice(analyses=valid)
    res = MKDADensity().fit(ss.to_dataset())
    return np.asarray(res.get_map("z").dataobj, dtype=float), len(valid)


def r2(a, g):
    m = common_mask(a, g)
    return float(np.corrcoef(a[m], g[m])[0, 1] ** 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=REPO / "results" / "article_level_arm.csv")
    ap.add_argument("--only", nargs="*", default=None, help="project/target keys to run")
    ap.add_argument("--verify", action="store_true",
                    help="also refit each target from its own analyses and check it reproduces "
                         "the committed reference map")
    args = ap.parse_args(argv)

    import nibabel as nib
    from nimare.nimads import Studyset

    selected = [(r["project"], r["target"])
                for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv"))]
    if args.only:
        selected = [k for k in selected if f"{k[0]}/{k[1]}" in args.only]
    ev = {(r["project"], r["target"]): r
          for r in csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))}
    rel = {(r["project"], r["target"]): float(r["reliability_mean"])
           for r in csv.DictReader(open(REPO / "results" / "reference_reliability.csv"))}

    cache, rows = {}, []
    for proj, targ in selected:
        if proj not in cache:
            base = NMB / "data" / "nimads" / proj / "merged"
            ssd, ann = nimads_meta.load_studyset_and_annotation(
                base / "nimads_studyset.json", base / "nimads_annotation.json")
            nimads_meta.name_by_id(ssd)
            a2s = {a["id"]: s["id"] for s in ssd["studies"] for a in (s.get("analyses") or [])}
            cache[proj] = (Studyset(copy.deepcopy(ssd)), ann, a2s)
        studyset, ann, a2s = cache[proj]
        own = set(nimads_meta._analysis_ids_for_column(ann, targ))
        studies = {a2s[a] for a in own if a in a2s}
        article = {a for a, s in a2s.items() if s in studies}
        ref = nib.load(str(E.reference_dir(proj, targ) / "z.nii.gz")).get_fdata()

        t0 = time.time()
        z_art, n_art = fit_z(studyset, article)
        # Compare against a refit of the target's own analyses from the SAME current records,
        # not the committed map: 5 committed maps (social others/soccomm/all, SUD nicotine/
        # opioids) were built from an older version of the NiMADS records and differ slightly.
        # Against the refit, the difference measures only the sibling analyses this arm adds.
        z_own, n_own = fit_z(studyset, own)
        row = {"project": proj, "target": targ, "n_studies": len(studies),
               "n_target_analyses": n_own, "n_article_analyses": n_art,
               "extra_analyses": n_art - n_own,
               "r2_article": round(r2(z_art, z_own), 6),
               "r2_article_vs_committed_map": round(r2(z_art, ref), 6),
               "committed_map_matches_records": int(float(np.nanmax(np.abs(z_own - ref))) < 1e-4),
               "ceiling": round(rel[(proj, targ)], 6),
               "reference_degenerate": ev[(proj, targ)]["reference_degenerate"],
               "umbrella_target": ev[(proj, targ)]["umbrella_target"],
               "pooled_r2": ev[(proj, targ)]["pooled_r2"]}
        if args.verify:
            row["verify_max_abs_diff"] = float(np.nanmax(np.abs(z_own - ref)))
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"  {proj}/{targ}: {len(studies)} studies, {len(own)} -> {n_art} analyses "
              f"(+{row['extra_analyses']}), r2_article {row['r2_article']:.3f}, "
              f"ceiling {row['ceiling']:.2f}  [{row['seconds']}s]"
              + (f"  verify maxdiff {row['verify_max_abs_diff']:.2e}" if args.verify else ""),
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {args.output} ({len(rows)} targets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
