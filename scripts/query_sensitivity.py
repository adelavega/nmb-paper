#!/usr/bin/env python3
"""How much does a text-to-map baseline's score depend on how the query is worded?

WHY THIS EXISTS

Text-to-map tools are only as good as the query they are given, so a reviewer will reasonably
ask whether the queries were chosen to make the tools look bad. This measures the spread across
plausible phrasings of the same target, so the paper can report it instead of asserting fairness.

WHAT IT FOUND (Neurosynth-free, NeuroQuery only; run 2026-09-24)

1. The shipped queries are already at or near the best of the sampled variants -- best in 4 of
   the 5 targets probed. No evidence the tools are being disadvantaged by wording.

2. Wording matters a great deal. executive_function/inhibition ranges 0.035 ("inhibition") to
   0.234 ("response inhibition"), a 7x swing on one target. Any single-query result carries that
   uncertainty and the paper must say so.

3. Longer, paper-derived queries are WORSE, not better. Feeding the source paper's title and
   keywords -- an obvious-looking "principled" rule -- scored 0.018 on reappraisal against 0.071
   for "cognitive reappraisal", and 0.024 vs 0.093 on social. NeuroQuery does better with a
   short construct term than with an abstract, despite being designed to accept abstracts.

WHAT NOT TO DO

Do not pick the best-scoring query per target. That is tuning on the benchmark, and it would
make the comparison meaningless in exactly the way the benchmark exists to prevent. The queries
must be fixed by a rule stated in advance; this script measures the consequence of that choice,
it does not optimise it.

Writes results/query_sensitivity.csv.
"""

from __future__ import annotations

import argparse
import csv
import inspect
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
import eval_text_to_map as E  # noqa: E402

# Plausible alternative phrasings per target. Written to span the space a competent user might
# try -- terse construct, the shipped query, and a longer descriptive form -- NOT to find a
# maximum.
VARIANTS: dict[tuple[str, str], list[str]] = {
    ("emotion_regulation_2022", "reappraisal"): [
        "reappraisal",
        "emotion regulation reappraisal of negative affect",
        "Reappraisal and empathic perspective-taking. Reappraisal; Cognitive perspective taking",
    ],
    ("emotion_regulation_2022", "decrease"): [
        "downregulation of negative emotion", "emotion regulation decrease negative affect"],
    ("social", "all_merged"): [
        "social processes",
        "Neural systems underlying RDoC social constructs. Social processes; Social task"],
    ("social", "self_merged"): ["self referential", "self referential judgment about oneself"],
    ("executive_function", "inhibition"): [
        "inhibition", "inhibitory control go no-go stop signal"],
    ("executive_function", "working_memory"): ["n-back working memory maintenance"],
    ("problem_solving", "calculation_mni_final"): [
        "calculation", "numerical cognition arithmetic problem solving"],
    ("decision_making", "rdm_july2019"): [
        "risk taking", "decision making under risk gambling"],
    ("cue_reactivity", "2_drug_neutral_2020"): [
        "drug craving", "drug cue reactivity craving addiction"],
    ("vbm_of_ptsd", "nonptsdgtptsd_merged"): [
        "ptsd gray matter", "post-traumatic stress disorder voxel based morphometry"],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=REPO / "results" / "query_sensitivity.csv")
    args = ap.parse_args(argv)

    import nibabel as nib
    from nilearn.image import resample_to_img
    from neuroquery import fetch_neuroquery_model, NeuroQueryModel
    E._RESAMPLE_PARAMS = set(inspect.signature(resample_to_img).parameters)
    model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())

    rows = []
    for (project, target), alts in VARIANTS.items():
        rd = E.reference_dir(project, target)
        if rd is None:
            continue
        ref = nib.load(str(rd / "z.nii.gz"))
        g = ref.get_fdata()
        shipped = E.QUERIES.get((project, target))
        for query in ([shipped] if shipped else []) + alts:
            a = np.asarray(E.resample(model(query)["brain_map"], ref).dataobj)
            m = common_mask(a, g)
            if m.sum() < 2 or a[m].std() == 0 or g[m].std() == 0:
                r2 = float("nan")
            else:
                r2 = float(np.corrcoef(a[m].ravel(), g[m].ravel())[0, 1] ** 2)
            rows.append({"project": project, "target": target, "query": query,
                         "is_shipped_query": int(query == shipped), "r2": round(r2, 6)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'target':<46}{'shipped':>9}{'min':>8}{'max':>8}{'spread':>9}  shipped is")
    shipped_is_best = 0
    n = 0
    for (p, t) in VARIANTS:
        sel = [r for r in rows if r["project"] == p and r["target"] == t
               and not np.isnan(r["r2"])]
        if not sel:
            continue
        n += 1
        ship = next((r["r2"] for r in sel if r["is_shipped_query"]), float("nan"))
        vals = [r["r2"] for r in sel]
        best = max(vals)
        rank = sorted(vals, reverse=True).index(ship) + 1 if not np.isnan(ship) else -1
        shipped_is_best += int(rank == 1)
        print(f"{p + '/' + t:<46}{ship:>9.3f}{min(vals):>8.3f}{best:>8.3f}"
              f"{best - min(vals):>9.3f}  rank {rank}/{len(vals)}")
    print(f"\n  the shipped query is the best of its variants in {shipped_is_best}/{n} targets")
    print(f"  mean spread across variants: "
          f"{st.mean(max(r['r2'] for r in rows if r['project'] == p and r['target'] == t) - min(r['r2'] for r in rows if r['project'] == p and r['target'] == t) for p, t in VARIANTS):.3f}")
    print(f"\n  wrote {args.output.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
