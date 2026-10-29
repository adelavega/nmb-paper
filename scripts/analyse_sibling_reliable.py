#!/usr/bin/env python3
"""Sibling discrimination restricted to reliable references.

Sibling discrimination (scripts/eval_specificity.py): for a target, the share of its own-project
siblings whose reference a method's map correlates with less than with the target's own
reference (0.5 = chance). Here both the targets and the siblings they are compared against are
limited to references with split-half reliability >= MIN_RELIABILITY, so that a sibling cannot
be "beaten" merely because its reference is noisy. Two reference sets:

    figure3   the 31 evaluable references (as in Figure 3), reliable ones only
    figure4   the same without the five "all" targets that pool their project (as in Figure 4)

Targets without a reliable sibling are dropped. Intervals: two-stage cluster bootstrap over the
projects that remain (10,000 draws); with this few projects the intervals are wide by design.

Writes results/sibling_discrimination_reliable.csv.
"""

from __future__ import annotations

import csv
import inspect
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
from make_difficulty_profile_figure import CATCH_ALL  # noqa: E402

MIN_RELIABILITY = 0.6
N_BOOT = 10_000


def main() -> int:
    import nibabel as nib
    from nilearn.image import resample_to_img
    E._RESAMPLE_PARAMS = set(inspect.signature(resample_to_img).parameters)
    ev = {(r["project"], r["target"]): r
          for r in csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))}
    rel = {(r["project"], r["target"]): float(r["reliability_mean"])
           for r in csv.DictReader(open(REPO / "results" / "reference_reliability.csv"))}
    keys = [k for k in ((r["project"], r["target"])
                        for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv")))
            if ev[k]["reference_degenerate"] == "0" and rel[k] >= MIN_RELIABILITY]
    grid = nib.load(str(E.reference_dir(*keys[0]) / "z.nii.gz"))
    brain = np.asarray(E.resample(nib.load(str(NMB / "data" / "brain_mask_mni152_2mm.nii.gz")),
                                  grid).dataobj) > 0.5

    def vec(path):
        a = np.asarray(E.resample(nib.load(str(path)), grid).dataobj, dtype=float)[brain]
        a[~np.isfinite(a)] = 0.0
        return a

    R = np.array([vec(E.reference_dir(*k) / "z.nii.gz") for k in keys])
    C = {m: S.corr_matrix(np.array([vec(S.method_path(m, *k, ev[k]["neurosynth_term"]))
                                    for k in keys]), R) for m in S.METHODS}
    rng = np.random.default_rng(0)
    out_rows = []
    for name, use in (("figure3", set(keys)), ("figure4", set(keys) - CATCH_ALL)):
        idx = [i for i, k in enumerate(keys) if k in use]
        by_proj = defaultdict(list)
        for i in idx:
            by_proj[keys[i][0]].append(i)
        rows = []          # (project, target index, {method: score})
        for p, members in by_proj.items():
            if len(members) < 2:
                continue
            for i in members:
                sib = [j for j in members if j != i]
                rows.append((p, i, {m: float(np.mean([C[m][i, i] > C[m][i, j] for j in sib]))
                                    for m in S.METHODS}))
        projects = np.array([p for p, _, _ in rows])
        print(f"\n{name}: {len(rows)} reliable targets with reliable siblings, "
              f"{len(set(projects))} projects ({', '.join(sorted(set(projects)))})")
        for m in S.METHODS:
            v = np.array([sc[m] for _, _, sc in rows])
            lo, hi = np.percentile(S.cluster_bootstrap(v, projects, rng), [2.5, 97.5])
            print(f"  {m:<24} mean {v.mean():.2f}  95% CI {lo:.2f}-{hi:.2f}")
            out_rows.append(dict(reference_set=name, method=m, n_targets=len(rows),
                                 n_projects=len(set(projects)), mean=round(float(v.mean()), 4),
                                 ci_low=round(float(lo), 4), ci_high=round(float(hi), 4)))
        print("  per target (mean over methods):")
        for p, i, sc in sorted(rows, key=lambda r: np.mean(list(r[2].values()))):
            print(f"    {np.mean(list(sc.values())):.2f}  {p}/{keys[i][1]}")
    with open(REPO / "results" / "sibling_discrimination_reliable.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print("\n  wrote results/sibling_discrimination_reliable.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
