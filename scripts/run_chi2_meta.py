#!/usr/bin/env python3
"""Re-estimate benchmark targets as MKDA chi-squared maps against a reference corpus.

THE HYPOTHESIS THIS TESTS

The benchmark's reference maps are MKDA *density* maps: forward inference, "where do studies on
this topic report activation?" Text-to-map tools may be closer to *specificity*: "where is
activation more associated with this topic than with the literature at large?" Comparing a
specificity-like prediction against a density reference would understate those tools for a reason
that has nothing to do with their quality.

MKDA chi-squared makes the comparison on the specificity footing. For each target it contrasts
the target's selected analyses (dataset1) against a large reference corpus (dataset2), producing

    z_desc-uniformity    forward inference (Neurosynth's "uniformity test"); the
                         density-like map: where do studies on this topic report activation?
    z_desc-association   reverse inference (Neurosynth's "association test"); activation here
                         is selectively associated with this topic rather than with the
                         literature at large

`z_desc-association` is the map to compare against the text-to-map tools: it is the "unique
above and beyond a reference corpus" quantity those tools are closer to producing.

THE REFERENCE CORPUS MATTERS AND IS NOT NEUTRAL

Chi-squared asks "compared to what?", so the answer depends on the corpus. Neurosynth v7 is the
default here because it is one API call and is what the tools were largely trained on. Benchmark studies
are removed from the reference set before fitting -- without that they sit on both sides of the
contrast and the test is attenuated, and this is not a rounding detail: 435 of the benchmark's
1,033 distinct PMIDs (42%) are in Neurosynth v7. By default every benchmark study is removed
once, giving a single fixed reference corpus shared by all targets; that is both 38x less work
and the cleaner contrast, since no target is scored against a differently-constituted corpus.

Writes results/chi2_<corpus>/<project>/<target>/.
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


def selected_dataset(studyset_data: dict, annotation_data: dict, column: str):
    """A NiMARE Dataset of just the analyses this target selects."""
    from nimare.nimads import Studyset

    ids = nimads_meta._analysis_ids_for_column(annotation_data, column)
    if not ids:
        return None, set()
    ss = Studyset(studyset_data).slice(analyses=ids)
    valid = [a.id for s in ss.studies for a in s.analyses
             if nimads_meta._has_valid_coordinates(a)]
    if not valid:
        return None, set()
    ss = ss.slice(analyses=valid)
    pmids = {str(s.id) for s in ss.studies}
    ds = ss.to_dataset()
    return (ds if len(ds.ids) else None), pmids


def reference_minus(reference, exclude_pmids: set[str]):
    """The reference corpus with the target's own studies removed.

    Leaving the target's studies in both arms of the contrast attenuates the chi-squared, and it
    is not a rounding detail here: 435 of the benchmark's 1,033 distinct PMIDs (42%) are in
    Neurosynth v7.

    `study_id` is taken from the coordinates table rather than parsed out of the experiment id,
    so this does not depend on the '<pmid>-<contrast>' id convention holding.
    """
    sid = (reference.coordinates[["id", "study_id"]]
           .drop_duplicates("id").astype(str).set_index("id")["study_id"])
    keep = [i for i in reference.ids if sid.get(str(i)) not in exclude_pmids]
    return reference.slice(keep), len(reference.ids) - len(keep)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path,
                    default=REPO / "data" / "reference_corpora"
                    / "neurosynth_v7_dataset.pkl.gz")
    ap.add_argument("--corpus-name", default="neurosynth_v7")
    ap.add_argument("--data-dir", type=Path, default=NMB / "data")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--projects", nargs="*", default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--overlap", choices=("global", "per-target", "none"), default="global",
                    help="which studies to remove from the reference arm. 'global' (default) "
                         "removes every benchmark study once, giving one fixed reference corpus "
                         "shared by all targets -- faster and cleaner. 'per-target' removes only "
                         "that target's studies, rebuilding the reference each time. 'none' "
                         "keeps the overlap, which attenuates the contrast.")
    args = ap.parse_args(argv)

    out_root = args.output_dir or (REPO / "results" / f"chi2_{args.corpus_name}")

    from nimare.dataset import Dataset
    from nimare.correct import FDRCorrector
    from nimare.meta.cbma import MKDAChi2
    from nimare.workflows import PairwiseCBMAWorkflow

    if not args.reference.exists():
        print(f"reference corpus not found: {args.reference}\n"
              f"fetch it first (nimare.extract.fetch_neurosynth)", file=sys.stderr)
        return 1
    print(f"loading reference corpus {args.reference.name}")
    reference = Dataset.load(str(args.reference))
    print(f"  {len(reference.ids)} experiments, {len(reference.coordinates)} coordinates")

    projects = sorted(p.parent.parent.name for p in
                      (args.data_dir / "nimads").glob("*/merged/nimads_studyset.json"))
    if args.projects:
        projects = [p for p in projects if p in args.projects]

    # One fixed reference corpus for every target. Besides being 38x less work, this is the
    # cleaner contrast: every target is scored against the same corpus, so cross-target
    # comparisons are not confounded by which studies happened to be removed.
    fixed_reference = None
    if args.overlap == "global":
        all_bench = set()
        for f in (args.data_dir / "nimads").glob("*/merged/nimads_studyset.json"):
            for s in json.loads(f.read_text()).get("studies", []):
                all_bench.add(str(s.get("pmid") or s.get("id")))
        fixed_reference, dropped_global = reference_minus(reference, all_bench)
        print(f"  fixed reference: {len(fixed_reference.ids)} experiments "
              f"({dropped_global} of the benchmark's {len(all_bench)} studies removed once)")

    done = failed = 0
    for project in projects:
        base = args.data_dir / "nimads" / project / "merged"
        studyset_data, annotation_data = nimads_meta.load_studyset_and_annotation(
            base / "nimads_studyset.json", base / "nimads_annotation.json")
        nimads_meta.name_by_id(studyset_data)
        columns = nimads_meta.boolean_columns(annotation_data, args.targets)

        for column in columns:
            out = out_root / project / column
            if args.skip_existing and out.exists() and any(out.iterdir()):
                print(f"{project}/{column}: present, skipping")
                continue

            ds1, pmids = selected_dataset(studyset_data, annotation_data, column)
            if ds1 is None:
                print(f"{project}/{column}: no usable analyses, skipping")
                continue

            if args.overlap == "global":
                ds2, dropped = fixed_reference, 0
            elif args.overlap == "none":
                ds2, dropped = reference, 0
            else:
                ds2, dropped = reference_minus(reference, pmids)

            print(f"\n{project}/{column}: {len(ds1.ids)} target experiments vs "
                  f"{len(ds2.ids)} reference"
                  + (f" ({dropped} overlapping removed)" if dropped else ""))
            out.mkdir(parents=True, exist_ok=True)
            try:
                wf = PairwiseCBMAWorkflow(
                    estimator=MKDAChi2(),
                    corrector=FDRCorrector(),
                    # [] disables. `None` does NOT: PairwiseCBMAWorkflow resolves it to
                    # a default Jackknife, which re-fits the chi-squared once per
                    # experiment in dataset1 -- 25 refits against 14k reference studies
                    # for the smallest target, 856 for social. That was the whole cost.
                    diagnostics=[],
                    output_dir=str(out),
                )
                wf.fit(ds1, ds2)
                done += 1
            except Exception as e:
                failed += 1
                print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"\n{done} fitted, {failed} failed -> {out_root}")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
