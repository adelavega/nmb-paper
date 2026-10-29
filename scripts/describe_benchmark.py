#!/usr/bin/env python3
"""Descriptive statistics for the NeuroMetaBench paper.

Every number the paper quotes about the dataset's composition is derived here, from the data
files, so that none of them has to be transcribed by hand.

THE THREE LAYERS

Availability is uneven across the resource, and the paper presents that as its structure rather
than as a caveat:

    A  protocol    search string, criteria, record counts        data/meta_datasets.csv
    B  inclusion   the final included-study list                 data/included_studies.csv
    C  analysis    analyses, coordinates, target assignments     data/nimads/*/merged/

Writes results/benchmark_descriptives.csv (per project) and prints the paper-facing summary.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))


# The benchmark's targets (data/selected_targets.csv). Composition counts cover every recorded
# analysis by default; 54 cue-reactivity analyses (from 42 studies) report only region-of-interest
# coordinates and contribute to none of the 35 targets. --benchmark-targets-only counts just the analyses that
# contribute to at least one target.
SELECTED = {(r["project"], r["target"].lower())
            for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv"))}


def load_layer_c(benchmark_only: bool = False) -> list[dict]:
    rows = []
    for ss_path in sorted(NMB.glob("data/nimads/*/merged/nimads_studyset.json")):
        project = ss_path.parent.parent.name
        ss = json.loads(ss_path.read_text())
        ann_path = ss_path.parent / "nimads_annotation.json"
        ann = json.loads(ann_path.read_text()) if ann_path.exists() else {}
        ann = ann[0] if isinstance(ann, list) else ann
        cols = sorted(k for k, v in (ann.get("note_keys") or {}).items() if v == "boolean"
                      and (project, k.lower().replace("-", "_")) in SELECTED)
        # an analysis belongs to the collection if it contributes to at least one benchmark target
        keep = {n.get("analysis") for n in ann.get("notes", []) or []
                if any((n.get("note") or {}).get(c) for c in cols)}

        per_study = collections.Counter()
        n_an = n_pt = 0
        for study in ss.get("studies", []):
            pmid = str(study.get("pmid") or study.get("id"))
            for an in study.get("analyses", []) or []:
                if benchmark_only and an.get("id") not in keep:
                    continue
                n_an += 1
                per_study[pmid] += 1
                n_pt += len(an.get("points") or [])

        selected = collections.Counter()
        targets_per_analysis = collections.Counter()
        for note in ann.get("notes", []) or []:
            hits = [c for c in cols if (note.get("note") or {}).get(c)]
            for c in hits:
                selected[c] += 1
            if hits:
                targets_per_analysis[note.get("analysis")] = len(hits)

        rows.append({
            "project": project,
            "studies": len(per_study),
            "analyses": n_an,
            "coordinates": n_pt,
            "targets": len(cols),
            "analyses_per_study_mean": round(n_an / len(per_study), 3) if per_study else 0,
            "analyses_per_study_max": max(per_study.values()) if per_study else 0,
            "studies_multi_analysis": sum(1 for v in per_study.values() if v > 1),
            "_per_study": list(per_study.values()),
            "_selected": dict(selected),
            "_multi_target": sum(1 for v in targets_per_analysis.values() if v > 1),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path,
                    default=REPO / "results" / "benchmark_descriptives.csv")
    ap.add_argument("--benchmark-targets-only", action="store_true",
                    help="count only analyses that contribute to at least one benchmark target")
    args = ap.parse_args(argv)

    rows = load_layer_c(args.benchmark_targets_only)
    if not rows:
        print("no NiMADS studysets found", file=sys.stderr)
        return 1

    # ---- layers A and B
    reg = list(csv.DictReader(open(NMB / "data" / "meta_datasets.csv")))
    done = [x for x in reg if x["inclusion_status"].strip() == "Done"]
    inc = list(csv.DictReader(open(NMB / "data" / "included_studies.csv")))
    alls = list(csv.DictReader(open(NMB / "data" / "all_studies.csv")))
    years = [int(x["year"]) for x in reg if x["year"].strip().isdigit()]

    print("=" * 78)
    print("LAYER A -- protocol metadata (data/meta_datasets.csv)")
    print("=" * 78)
    print(f"  meta-analyses registered      {len(reg)}   ({len(done)} marked complete)")
    print(f"  publication years             {min(years)}-{max(years)}")
    for f, label in (("search", "search string"), ("inclusion", "inclusion criteria"),
                     ("exclusion", "exclusion criteria"),
                     ("search_results_n", "records identified"),
                     ("selected_n", "records selected")):
        n = sum(1 for x in reg if x[f].strip())
        print(f"  with {label:<22}{n:>3}/{len(reg)}")
    meth = collections.Counter(x["method"].strip() for x in reg if x["method"].strip())
    print(f"  original methods              "
          f"{', '.join(f'{k} ({v})' for k, v in meth.most_common(4))}")

    print("\n" + "=" * 78)
    print("LAYER B -- inclusion lists")
    print("=" * 78)
    print(f"  included-study links          {len(inc)} over "
          f"{len({x['meta_pmid'] for x in inc})} meta-analyses")
    st_counts = collections.Counter(x["final_status"] for x in alls)
    print(f"  full screening decisions      {len(alls)} over "
          f"{len({x['meta_pmid'] for x in alls})} meta-analyses "
          f"({dict(st_counts)})")

    print("\n" + "=" * 78)
    print("LAYER C -- analysis-level data (data/nimads/*/merged/)")
    print("=" * 78)
    print(f"  {'project':<26}{'studies':>8}{'analyses':>10}{'coords':>9}{'targets':>9}"
          f"{'an/study':>10}{'max':>5}")
    for r in rows:
        print(f"  {r['project']:<26}{r['studies']:>8}{r['analyses']:>10}{r['coordinates']:>9}"
              f"{r['targets']:>9}{r['analyses_per_study_mean']:>10.2f}"
              f"{r['analyses_per_study_max']:>5}")
    T = {k: sum(r[k] for r in rows)
         for k in ("studies", "analyses", "coordinates", "targets", "studies_multi_analysis")}
    print("  " + "-" * 76)
    print(f"  {'TOTAL':<26}{T['studies']:>8}{T['analyses']:>10}{T['coordinates']:>9}"
          f"{T['targets']:>9}")

    allc = [v for r in rows for v in r["_per_study"]]
    print(f"\n  THE UNIT ARGUMENT")
    print(f"    analyses per study          mean {st.mean(allc):.2f}  median "
          f"{st.median(allc):.0f}  max {max(allc)}")
    print(f"    studies with >1 analysis    {T['studies_multi_analysis']}/{T['studies']} "
          f"({100 * T['studies_multi_analysis'] / T['studies']:.0f}%)")
    multi_t = sum(r["_multi_target"] for r in rows)
    print(f"    analyses serving >1 target  {multi_t}/{T['analyses']} "
          f"({100 * multi_t / T['analyses']:.0f}%)")
    frac = [n / r["analyses"] for r in rows for n in r["_selected"].values() if r["analyses"]]
    print(f"    a target uses a median      {100 * st.median(frac):.0f}% of its project's pool "
          f"(range {100 * min(frac):.0f}%-{100 * max(frac):.0f}%, n={len(frac)} targets)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in rows[0] if not k.startswith("_")]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {args.output.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
