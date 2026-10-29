#!/usr/bin/env python3
"""Meta-analyse every analysis in a project, ignoring the target-contrast annotation.

WHY

This is the benchmark's internal reference point. It answers: if you had exactly the right
literature -- every study the expert included -- but did not distinguish which analyses within
those papers addressed which question, how well would you recover a given target's map?

It uses no external tool, no search and no text. The gap between this arm and a target's own
reference map is a property of the dataset, and it is what establishes that the analysis-level
annotation carries information an article-level resource could not encode.

Within a project this produces ONE map, compared against every target in that project -- which is
the point: a method that cannot select analyses returns the same answer to every sub-question in
a domain.

WHAT THIS ARM IS NOT: IT IS AN OPTIMISTIC BOUND, NOT A CEILING

The benchmark records only analyses the expert SELECTED for at least one target. Verified: all
2,219 analyses across the nine projects belong to >= 1 target arm, and not one is an orphan. The
analyses an expert read and rejected as irrelevant to every target were never written down.

So "pool everything in the project" pools the union of expert-selected analyses -- a set that has
already been filtered for relevance at the analysis level. A real article-level method, which
extracts every analysis a included paper reports and selects none of them, would pool strictly
more, including the irrelevant ones, and would score LOWER than this arm.

This arm therefore overstates what article-level-only synthesis can achieve. Report it as an
optimistic bound on that class of method, never as a ceiling, and never as an estimate of what
a real pool-everything pipeline would produce. Closing the gap would require annotating the
rejected analyses too, which is a benchmark extension and out of scope here.

Writes results/pooled_all_analyses/<project>/pooled/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nimads_meta  # noqa: E402


def pooled_annotation(studyset_data: dict) -> dict:
    """An annotation marking every analysis true for a single column, 'pooled'."""
    notes = []
    for study in studyset_data.get("studies", []):
        for analysis in study.get("analyses", []) or []:
            notes.append({"analysis": analysis.get("id"),
                          "study": study.get("id"),
                          "note": {"pooled": True}})
    return {"note_keys": {"pooled": "boolean"}, "notes": notes}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=NMB / "data")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO / "results" / "pooled_all_analyses")
    ap.add_argument("--projects", nargs="*", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)

    from nimare.nimads import Studyset

    projects = sorted(p.parent.parent.name for p in
                      (args.data_dir / "nimads").glob("*/merged/nimads_studyset.json"))
    if args.projects:
        projects = [p for p in projects if p in args.projects]
    if not projects:
        print("no projects found", file=sys.stderr)
        return 1

    ok = 0
    for project in projects:
        ss_path = args.data_dir / "nimads" / project / "merged" / "nimads_studyset.json"
        out = args.output_dir / project
        target = out / "pooled"
        if args.skip_existing and target.exists() and any(target.iterdir()):
            print(f"{project}: already present, skipping")
            continue

        studyset_data = json.loads(ss_path.read_text())
        nimads_meta.name_by_id(studyset_data)
        ann = pooled_annotation(studyset_data)
        n_an = len(ann["notes"])
        n_st = len(studyset_data.get("studies", []))
        print(f"\n{project}: pooling {n_an} analyses from {n_st} studies")

        target.mkdir(parents=True, exist_ok=True)
        result = nimads_meta.run_meta_analysis_for_column(
            studyset=Studyset(studyset_data),
            annotation_data=ann,
            column="pooled",
            output_dir=target,
        )
        if result is not None:
            ok += 1

    print(f"\n{ok}/{len(projects)} pooled baselines written under "
          f"{args.output_dir.relative_to(REPO)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
