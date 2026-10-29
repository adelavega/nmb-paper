#!/usr/bin/env python3
"""The Neurosynth arm: classic term-based meta-analysis over the 14,371-study v7 corpus.

WHAT NEUROSYNTH COMPUTES

For a vocabulary term, studies whose abstract tf-idf weight for the term exceeds 0.001 are
contrasted with every other study in the corpus by MKDA chi-squared (prior 0.5), giving

    z_desc-association   Neurosynth's headline map ("association test", formerly reverse
                         inference): is activation here selectively associated with the term?
    z_desc-uniformity    the forward map ("uniformity test"): do studies with the term activate
                         here more than chance?

Both are scored. The association map is what Neurosynth shows a user; the uniformity map is the
closer analogue of the benchmark's MKDA density references, so reporting only one would let the
choice of output flatter or penalise the tool.

THE CORPUS IS USED AS PUBLISHED

Unlike scripts/run_chi2_meta.py, which builds REFERENCE maps and so removes benchmark studies
from the contrast set, this is a tool arm: Neurosynth as a user meets it. No studies are removed.
42% of benchmark PMIDs are in Neurosynth v7, so -- as for NeuroQuery and NeuroVLM -- scores are
retrospective recovery, not generalisation to unseen studies.

TERMS

data/neurosynth_terms.csv fixes one vocabulary term per target, derived from the same query the
other tools received and chosen before any Neurosynth map was computed. Several sibling targets
share a term (Neurosynth cannot express their distinction); each distinct term is fitted once.

HELD-OUT VARIANT (--exclude-benchmark)

The uniformity map is an MKDA estimate built from real coordinates, so benchmark studies inside a
term set are partly the reference recovering itself: on average 18% of a target's reference
studies fall inside its term set (up to 65%). --exclude-benchmark removes every benchmark study
from the corpus before fitting, the same global exclusion run_chi2_meta.py applies to reference
maps. Comparing the two variants bounds how much of the Neurosynth score is leakage.

Writes results/neurosynth_maps/<term>/z_desc-{association,uniformity}.nii.gz
(or results/neurosynth_maps_heldout/ with --exclude-benchmark).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
PREFIX = "terms_abstract_tfidf__"
FREQ_THRESHOLD = 0.001   # Neurosynth's own default


def write_capped(img, path: Path) -> int:
    """Write a z map with +/-inf capped at its largest finite magnitude. Returns voxels capped.

    For large term sets (working memory, reward, emotion) the chi-squared p-value underflows to
    0 at the strongest voxels and z becomes +inf. Left in, those voxels are not just a plotting
    nuisance: every r2 in this repo is computed over voxels finite in both maps, so the map's
    PEAK would be silently dropped from its own correlation. Capping keeps them as the peak.
    """
    import nibabel as nib
    import numpy as np
    a = np.asarray(img.dataobj, dtype=np.float32)
    bad = ~np.isfinite(a)
    if bad.any():
        fin = a[np.isfinite(a)]
        a = np.where(np.isposinf(a), fin.max(), np.where(np.isneginf(a), fin.min(), a))
    nib.Nifti1Image(a, img.affine).to_filename(str(path))
    return int(bad.sum())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=REPO / "data" / "reference_corpora"
                    / "neurosynth_v7_terms_dataset.pkl.gz")
    ap.add_argument("--terms-table", type=Path, default=REPO / "data" / "neurosynth_terms.csv")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results" / "neurosynth_maps")
    ap.add_argument("--terms", nargs="*", default=None, help="fit only these terms")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--exclude-benchmark", action="store_true",
                    help="remove every benchmark study from the corpus first (held-out variant)")
    args = ap.parse_args(argv)

    from nimare.dataset import Dataset
    from nimare.meta.cbma import MKDAChi2

    terms = sorted({r["neurosynth_term"] for r in csv.DictReader(open(args.terms_table))})
    if args.terms:
        terms = [t for t in terms if t in args.terms]
    print(f"loading {args.dataset.name}")
    ds = Dataset.load(str(args.dataset))
    if args.exclude_benchmark:
        import json
        bench = set()
        for f in (NMB / "data" / "nimads").glob("*/merged/nimads_studyset.json"):
            bench |= {str(s.get("pmid") or s.get("id")) for s in json.loads(f.read_text())["studies"]}
        sid = (ds.coordinates[["id", "study_id"]].drop_duplicates("id").astype(str)
               .set_index("id")["study_id"])
        keep = [i for i in ds.ids if sid.get(str(i)) not in bench]
        print(f"  held out: removing {len(ds.ids) - len(keep)} benchmark studies")
        ds = ds.slice(keep)
        if args.output_dir == REPO / "results" / "neurosynth_maps":
            args.output_dir = REPO / "results" / "neurosynth_maps_heldout"
    print(f"  {len(ds.ids)} experiments; fitting {len(terms)} term(s)")

    for term in terms:
        out = args.output_dir / term.replace(" ", "_")
        if args.skip_existing and (out / "z_desc-association.nii.gz").exists():
            print(f"{term}: present, skipping")
            continue
        t0 = time.time()
        with_term = ds.get_studies_by_label(PREFIX + term, label_threshold=FREQ_THRESHOLD)
        without = sorted(set(ds.ids) - set(with_term))
        res = MKDAChi2().fit(ds.slice(with_term), ds.slice(without))
        out.mkdir(parents=True, exist_ok=True)
        for name in ("z_desc-association", "z_desc-uniformity"):
            write_capped(res.get_map(name), out / f"{name}.nii.gz")
        print(f"{term:<22} {len(with_term):>5} vs {len(without):>5} studies  "
              f"{time.time() - t0:6.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
