#!/usr/bin/env python3
"""Mean r2 per project, restricted to the selected target set.

WHICH TARGETS

data/selected_targets.csv lists the 35 targets paired with an evaluated synthesis question (the
`annotation_mappings` keys of each project's nmb_mappings.json in autonima-results). The benchmark
ships 38 reference maps; the 3 left out are cue_reactivity's non-whole-brain variants
(1_reward_neutral_2020, 2_drug_neutral_2020, 3_natural_neutral_2020). Their `_wbonly`
counterparts are the ones scored, which is correct for coordinate-based meta-analysis: the
non-wbonly sets include region-of-interest analyses, which a whole-brain CBMA should not pool.

EXCLUSIONS, applied per row and reported, not hidden

    degenerate reference (no suprathreshold voxel; split-half ceiling ~0)  -> excluded, all arms
    umbrella target (>= 95% of project pool)                               -> excluded, pooled arm

All r2 values come from results/text_to_map_baselines.csv, which computes them inside the
228,483-voxel MNI152 2 mm brain mask (scripts/map_mask.py).

Writes results/r2_by_project_selected.csv.
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
ARMS = ("neuroquery", "neurovlm", "neurosynth_assoc", "neurosynth_unif", "pooled")


def main() -> int:
    sel = {(r["project"], r["target"])
           for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv"))}
    ev = [r for r in csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))
          if (r["project"], r["target"]) in sel]
    relp = REPO / "results" / "reference_reliability.csv"
    rel = ({(r["project"], r["target"]): float(r["reliability_mean"])
            for r in csv.DictReader(open(relp))} if relp.exists() else {})

    def vals(rows, arm):
        return [float(r[f"{arm}_r2"]) for r in rows
                if r["reference_degenerate"] == "0" and r.get(f"{arm}_r2") not in ("", None)
                and not (arm == "pooled" and r["umbrella_target"] == "1")]

    out = []
    projects = sorted({r["project"] for r in ev})
    print(f"{len(ev)} selected targets in {len(projects)} projects\n")
    print(f"{'project':<25}{'sel':>4}{'deg':>4}" + "".join(f"{a[:16]:>17}" for a in ARMS)
          + f"{'ceiling':>9}")
    for proj in projects + ["ALL"]:
        rows = ev if proj == "ALL" else [r for r in ev if r["project"] == proj]
        rec = {"project": proj, "n_selected": len(rows),
               "n_degenerate": sum(r["reference_degenerate"] == "1" for r in rows)}
        line = f"{proj:<25}{len(rows):>4}{rec['n_degenerate']:>4}"
        for arm in ARMS:
            v = vals(rows, arm)
            rec[f"{arm}_n"] = len(v)
            rec[f"{arm}_mean_r2"] = round(st.mean(v), 4) if v else ""
            line += f"{st.mean(v):>11.3f} (n={len(v):>2})" if v else f"{'--':>17}"
        c = [rel[(r["project"], r["target"])] for r in rows
             if r["reference_degenerate"] == "0" and (r["project"], r["target"]) in rel]
        rec["mean_ceiling"] = round(st.mean(c), 4) if c else ""
        line += f"{st.mean(c):>9.2f}" if c else f"{'--':>9}"
        if proj == "ALL":
            print("-" * len(line))
        print(line)
        out.append(rec)

    path = REPO / "results" / "r2_by_project_selected.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\n  wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
