#!/usr/bin/env python3
"""Does a target's share of its project's studies predict how well the pooled map stands in for it?

The project-pooled map fits every annotated analysis in a project at once
(scripts/run_pooled_baseline.py). A target built from most of the project's papers should look
like it; a target built from a small slice of the papers should not. That is the expectation from
study (PMID) overlap alone. The interesting targets are those that break it: built from nearly all
of the project's papers yet poorly matched by the pooled map, because they take only some of the
results those papers report.

Per selected target (data/selected_targets.csv), against its project's pooled map, inside the
brain mask:

    pooled_r, pooled_r2   signed Pearson r and its square with the target's reference map
    study_share           target studies / project studies   (the PMID-level expectation)
    analysis_share        target analyses / project analyses
    reliability           the reference's split-half ceiling (results/reference_reliability.csv)

Umbrella targets (analysis_share >= 0.95) match the pooled map by construction and degenerate
references (reliability ~0) have no reproducible signal; both are reported but kept out of the
comparison and the correlations.

Writes results/pooled_vs_overlap.csv and results/pooled_vs_overlap_projects.csv.
"""

from __future__ import annotations

import csv
import inspect
import json
import statistics as st
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
import eval_text_to_map as E  # noqa: E402

UMBRELLA_FRACTION = 0.95


def project_sets(project: str):
    base = NMB / "data" / "nimads" / project / "merged"
    ss = json.loads((base / "nimads_studyset.json").read_text())
    ann = json.loads((base / "nimads_annotation.json").read_text())
    ann = ann[0] if isinstance(ann, list) else ann
    study_of = {a["id"]: s["id"] for s in ss["studies"] for a in s["analyses"]}
    cols = [k for k, v in (ann.get("note_keys") or {}).items() if v == "boolean"]
    per = {c: set() for c in cols}
    for n in ann.get("notes", []):
        a, note = n.get("analysis"), n.get("note") or {}
        if a in study_of:
            for c in cols:
                if note.get(c):
                    per[c].add(a)
    every = set().union(*per.values())
    return per, study_of, every


def main() -> int:
    import nibabel as nib
    from nilearn.image import resample_to_img
    from scipy import stats
    E._RESAMPLE_PARAMS = set(inspect.signature(resample_to_img).parameters)

    selected = defaultdict(list)
    for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv")):
        selected[r["project"]].append(r["target"])
    ev = {(r["project"], r["target"]): r
          for r in csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))}
    rel = {(r["project"], r["target"]): float(r["reliability_mean"])
           for r in csv.DictReader(open(REPO / "results" / "reference_reliability.csv"))}
    brain = nib.load(str(NMB / "data" / "brain_mask_mni152_2mm.nii.gz"))

    rows = []
    for project, targets in sorted(selected.items()):
        per, study_of, every = project_sets(project)
        lower = {c.lower().replace("-", "_"): c for c in per}
        proj_studies = {study_of[a] for a in every}
        pooled_img = nib.load(str(REPO / "results" / "pooled_all_analyses" / project / "pooled"
                                  / "z.nii.gz"))
        mask = np.asarray(E.resample(brain, pooled_img).dataobj) > 0.5
        pooled = pooled_img.get_fdata()[mask]
        for t in targets:
            an = per[lower[t.lower()]]
            ref = nib.load(str(E.reference_dir(project, t) / "z.nii.gz")).get_fdata()[mask]
            r = float(np.corrcoef(ref, pooled)[0, 1]) if ref.std() > 0 else float("nan")
            a_share = len(an) / len(every)
            rows.append(dict(
                project=project, target=t,
                umbrella=int(a_share >= UMBRELLA_FRACTION),
                degenerate=int(ev[(project, t)]["reference_degenerate"] == "1"),
                n_studies=len({study_of[a] for a in an}), n_analyses=len(an),
                study_share=round(len({study_of[a] for a in an}) / len(proj_studies), 4),
                analysis_share=round(a_share, 4),
                reliability=round(rel[(project, t)], 4),
                pooled_r=round(r, 4), pooled_r2=round(r * r, 4)))

    out = REPO / "results"
    with open(out / "pooled_vs_overlap.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    comp = [r for r in rows if not r["umbrella"] and not r["degenerate"]]
    by = defaultdict(list)
    for r in comp:
        by[r["project"]].append(r)
    with open(out / "pooled_vs_overlap_projects.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "n_comparisons", "mean_r2", "min_r2", "max_r2",
                    "mean_study_share", "mean_analysis_share"])
        for p, rs in sorted(by.items(), key=lambda kv: -st.mean(r["pooled_r2"] for r in kv[1])):
            v = [r["pooled_r2"] for r in rs]
            w.writerow([p, len(rs), round(st.mean(v), 3), min(v), max(v),
                        round(st.mean(r["study_share"] for r in rs), 3),
                        round(st.mean(r["analysis_share"] for r in rs), 3)])

    print(f"selected {len(rows)}; umbrella {sum(r['umbrella'] for r in rows)}; degenerate "
          f"{sum(r['degenerate'] for r in rows)}; comparisons {len(comp)} "
          f"from {len(by)} projects")
    v = [r["pooled_r2"] for r in comp]
    pm = [st.mean(r["pooled_r2"] for r in rs) for rs in by.values()]
    print(f"pooled r2: target-weighted mean {st.mean(v):.3f} (range {min(v):.3f}-{max(v):.3f}); "
          f"project-weighted {st.mean(pm):.3f}; signed r min {min(r['pooled_r'] for r in comp):+.3f}")
    for x in ("study_share", "analysis_share"):
        rho, p = stats.spearmanr([r[x] for r in comp], v)
        print(f"pooled r2 vs {x}: Spearman rho {rho:+.2f} (p = {p:.3g})")
    xs = np.array([r["study_share"] for r in comp])
    b = np.polyfit(xs, v, 1)
    for r in comp:
        r["resid"] = r["pooled_r2"] - np.polyval(b, r["study_share"])
    print("largest shortfalls from the study-share expectation:")
    for r in sorted(comp, key=lambda r: r["resid"])[:6]:
        print(f"  {r['project']}/{r['target']}: study share {r['study_share']:.2f}, analysis "
              f"share {r['analysis_share']:.2f}, pooled r2 {r['pooled_r2']:.3f} "
              f"(expected {r['pooled_r2'] - r['resid']:.3f})")
    print("  wrote results/pooled_vs_overlap.csv, results/pooled_vs_overlap_projects.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
