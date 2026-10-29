#!/usr/bin/env python3
"""Score every arm against each flavour of reference map.

THE QUESTION

The benchmark's reference maps are MKDA *density*: forward inference, "where do studies on this
topic report activation?" Text-to-map tools may be closer to *association*: "where is activation
selectively related to this topic rather than to the literature at large?" Scoring an
association-like prediction against a density reference would understate those tools for a reason
that has nothing to do with their quality.

This scores each arm against all three references so the question is answerable rather than
assumed:

    density       analysis/<project>/<target>/z.nii.gz          the benchmark's own reference
    uniformity    chi-squared forward inference                  should track density closely
    association   chi-squared reverse inference                  the "unique above and beyond"
                                                                 quantity

If the hypothesis holds, the tools' r2 rises substantially against `association` while staying
flat against `uniformity`. If it does not, their low scores are not an inference-direction
artifact.

Writes results/reference_flavour_comparison.csv.
"""

from __future__ import annotations

import argparse
import csv
import collections
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

FLAVOURS = {
    "density": lambda rd, cd: rd / "z.nii.gz",
    "uniformity": lambda rd, cd: cd / "z_desc-uniformity.nii.gz",
    "association": lambda rd, cd: cd / "z_desc-association.nii.gz",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chi2-dir", type=Path,
                    default=REPO / "results" / "chi2_neurosynth_v7")
    ap.add_argument("--output", type=Path,
                    default=REPO / "results" / "reference_flavour_comparison.csv")
    args = ap.parse_args(argv)

    import nibabel as nib
    from nilearn.image import resample_to_img
    E._RESAMPLE_PARAMS = set(inspect.signature(resample_to_img).parameters)

    arms = {
        "neuroquery": lambda p, t: REPO / "results" / "neuroquery_maps" / f"{p}__{t}.nii.gz",
        "neurovlm": lambda p, t: REPO / "data" / "neurovlm_maps" / f"{p}__{t}.nii.gz",
        "pooled": lambda p, t: (REPO / "results" / "pooled_all_analyses" / p / "pooled"
                                / "z.nii.gz"),
    }

    rows = []
    for project, target, umbrella, frac in E.targets_with_umbrella_flag():
        rd = E.reference_dir(project, target)
        cd = args.chi2_dir / project / target
        if rd is None:
            continue
        for flavour, pathfn in FLAVOURS.items():
            gpath = pathfn(rd, cd)
            if not gpath.exists():
                continue
            gimg = nib.load(str(gpath))
            g = gimg.get_fdata()
            for arm, afn in arms.items():
                if arm == "pooled" and umbrella:
                    continue          # circular: the pooled set IS this target
                apath = afn(project, target)
                if not apath.exists():
                    continue
                aimg = nib.load(str(apath))
                a = (np.asarray(E.resample(aimg, gimg).dataobj)
                     if aimg.shape != g.shape else aimg.get_fdata())
                m = common_mask(a, g)
                if m.sum() < 2 or a[m].std() == 0 or g[m].std() == 0:
                    continue
                rows.append({
                    "project": project, "target": target, "umbrella_target": int(umbrella),
                    "reference_flavour": flavour, "arm": arm,
                    "r2": round(float(np.corrcoef(a[m].ravel(), g[m].ravel())[0, 1] ** 2), 6),
                })

    if not rows:
        print("nothing scored -- has the chi-squared run finished?", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    grid = collections.defaultdict(list)
    for r in rows:
        grid[(r["arm"], r["reference_flavour"])].append(r["r2"])
    print(f"\nmean r2 by arm x reference flavour  "
          f"({len({(r['project'], r['target']) for r in rows})} targets)\n")
    print(f"{'arm':<14}" + "".join(f"{f:>16}" for f in FLAVOURS))
    for arm in arms:
        line = f"{arm:<14}"
        for f in FLAVOURS:
            v = grid[(arm, f)]
            line += f"{st.mean(v):>10.3f} (n={len(v):>2})" if v else f"{'--':>16}"
        print(line)
    print(f"\n  wrote {args.output.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
