#!/usr/bin/env python3
"""How distinct are the targets, and what makes two targets produce different maps?

THREE QUESTIONS, ONE SET OF MEASUREMENTS

1. SPATIAL STRUCTURE. A 38x38 correlation matrix between target reference maps, in brain space.
   Establishes that the benchmark's targets are distinguishable rather than 38 views of nine
   topics -- the precondition for any of its comparisons meaning anything.

2. WHAT DRIVES THE DIFFERENCE. For every within-project pair of targets, the overlap of their
   STUDY sets and of their ANALYSIS sets, each against the similarity of their maps. If article
   overlap explained map similarity, article-level curation records would be sufficient and this
   resource would not need to exist.

3. DOES TOOL PERFORMANCE TRACK ANY OF IT? Whether a text-to-map tool does better on targets that
   are typical of their project and worse on distinctive ones.

WHAT THEY FOUND (run 2026-10-06, 38 reference maps, 71 within-project pairs)

Targets are distinct: within-project pairs correlate r = +0.55 on average against r = +0.18
between projects, so targets resemble their siblings more than strangers, but are far from
interchangeable.

Analysis overlap predicts map similarity better than study overlap -- Spearman rho = +0.875 vs
+0.727. That gap is the quantitative case for annotating at the analysis level.

The decisive cases are in emotion regulation, where targets share most of their literature and
none of their analyses:

    maintain / reappraisal   study overlap 0.63   analysis overlap 0.00   map r = -0.03
    decrease / maintain      study overlap 0.61   analysis overlap 0.00   map r = -0.05

Two targets built from the same papers, selecting disjoint results within them, produce
uncorrelated maps. No article inclusion list can express that difference.

Projects vary widely in how much their targets share: decision making 0.00 and substance use
0.07 study overlap (targets drawn from essentially separate literatures) against dementia 0.56
and cue reactivity 0.43. Emotion regulation is the instructive one, with high study overlap
(0.47) and low analysis overlap (0.17).

CORRECTION (2026-10-07): DISTINCTIVENESS DOES NOT SURVIVE CONTROLLING FOR REFERENCE RELIABILITY.

Section 3 reports that tool r2 falls as a target's map becomes more distinctive from its siblings
(NeuroVLM rho = -0.45, p = 0.013), which was read as "tools recover the topic but miss the target".
That reading is not supported. Distinctive targets are the noisy ones -- distinctiveness correlates
rho = -0.59 with the split-half reliability in results/reference_reliability.csv -- and once that
ceiling is partialled out the association vanishes: NeuroVLM -0.29 (p = 0.12), NeuroQuery -0.11
(p = 0.53). A target that looks unlike its siblings is, mostly, a target with a less reliable
reference, and no tool can track an unreliable map. Report the partial, not the raw coefficient.

Writes results/target_structure.csv (pairs) and results/target_similarity_matrix.csv.
"""

from __future__ import annotations

import argparse
import csv
import collections
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_mask import brain_mask  # noqa: E402
import eval_text_to_map as E  # noqa: E402


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def load_sets() -> dict:
    """{(project, target): (analysis_ids, study_pmids)}"""
    out = {}
    for ap in sorted(NMB.glob("data/nimads/*/merged/nimads_annotation.json")):
        proj = ap.parent.parent.name
        ss = json.loads((ap.parent / "nimads_studyset.json").read_text())
        ann = json.loads(ap.read_text())
        ann = ann[0] if isinstance(ann, list) else ann
        a2s = {a.get("id"): str(s.get("pmid") or s.get("id"))
               for s in ss.get("studies", []) for a in (s.get("analyses") or [])}
        for col in sorted(k for k, v in (ann.get("note_keys") or {}).items() if v == "boolean"):
            A = {n["analysis"] for n in (ann.get("notes") or [])
                 if (n.get("note") or {}).get(col) and n.get("analysis")}
            out[(proj, col)] = (A, {a2s[a] for a in A if a in a2s})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-out", type=Path, default=REPO / "results" / "target_structure.csv")
    ap.add_argument("--matrix-out", type=Path,
                    default=REPO / "results" / "target_similarity_matrix.csv")
    args = ap.parse_args(argv)

    import nibabel as nib
    inside = brain_mask()
    sets = load_sets()

    keys, vecs = [], []
    for key in sets:
        rd = E.reference_dir(*key)
        if rd is None:
            continue
        a = nib.load(str(rd / "z.nii.gz")).get_fdata()
        if a[inside].std() > 0:
            keys.append(key)
            vecs.append(a[inside].ravel())
    M = np.corrcoef(np.vstack(vecs))

    args.matrix_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.matrix_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target"] + [f"{p}/{t}" for p, t in keys])
        for i, (p, t) in enumerate(keys):
            w.writerow([f"{p}/{t}"] + [round(float(x), 6) for x in M[i]])

    idx = {k: i for i, k in enumerate(keys)}
    within = [M[idx[a], idx[b]] for a, b in itertools.combinations(keys, 2) if a[0] == b[0]]
    between = [M[idx[a], idx[b]] for a, b in itertools.combinations(keys, 2) if a[0] != b[0]]
    print(f"1. SPATIAL STRUCTURE  ({len(keys)} targets)")
    print(f"   within-project pairs  n={len(within):<4} mean r={st.mean(within):+.3f}")
    print(f"   between-project pairs n={len(between):<4} mean r={st.mean(between):+.3f}")

    rows = []
    for a, b in itertools.combinations(keys, 2):
        if a[0] != b[0]:
            continue
        A1, S1 = sets[a]
        A2, S2 = sets[b]
        rows.append({"project": a[0], "target_a": a[1], "target_b": b[1],
                     "study_jaccard": round(jaccard(S1, S2), 4),
                     "analysis_jaccard": round(jaccard(A1, A2), 4),
                     "map_r": round(float(M[idx[a], idx[b]]), 4)})
    with open(args.pairs_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n2. WHAT DRIVES MAP SIMILARITY  ({len(rows)} within-project pairs)")
    sj = [r["study_jaccard"] for r in rows]
    aj = [r["analysis_jaccard"] for r in rows]
    mr = [r["map_r"] for r in rows]
    for lbl, v in (("study overlap", sj), ("analysis overlap", aj)):
        rho, p = stats.spearmanr(v, mr)
        print(f"   {lbl:<18} vs map similarity  rho={rho:+.3f}  p={p:.2g}")

    print(f"\n   by project ({'study J':>9}{'analysis J':>12}{'map r':>8})")
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r["project"]].append(r)
    for proj, v in sorted(byp.items(), key=lambda kv: st.mean(x["study_jaccard"] for x in kv[1])):
        print(f"     {proj:<26}{st.mean(x['study_jaccard'] for x in v):>9.3f}"
              f"{st.mean(x['analysis_jaccard'] for x in v):>12.3f}"
              f"{st.mean(x['map_r'] for x in v):>8.3f}")

    print("\n   same studies, disjoint analyses, uncorrelated maps:")
    for r in sorted(rows, key=lambda r: r["study_jaccard"] - r["map_r"], reverse=True)[:4]:
        print(f"     {r['project']}: {r['target_a']} / {r['target_b']}  "
              f"studyJ={r['study_jaccard']:.2f} analysisJ={r['analysis_jaccard']:.2f} "
              f"r={r['map_r']:+.2f}")

    print("\n3. DOES TOOL PERFORMANCE TRACK TARGET DISTINCTIVENESS?")
    distinct = {}
    for k in keys:
        sib = [M[idx[k], idx[o]] for o in keys if o[0] == k[0] and o != k]
        if sib:
            distinct[k] = 1.0 - float(np.mean(sib))
    ev = {(r["project"], r["target"]): r for r in
          csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))
          if not int(r["reference_degenerate"])}
    rel_path = REPO / "results" / "reference_reliability.csv"
    rel = ({(r["project"], r["target"]): float(r["reliability_mean"])
            for r in csv.DictReader(open(rel_path))} if rel_path.exists() else {})
    for arm in ("neuroquery", "neurovlm", "pooled"):
        pts = [(distinct[k], float(ev[k][f"{arm}_r2"])) for k in keys
               if k in distinct and k in ev and ev[k].get(f"{arm}_r2") not in ("", None)
               and not (arm == "pooled" and int(ev[k]["umbrella_target"]))]
        if len(pts) < 6:
            continue
        x, y = zip(*pts)
        rho, p = stats.spearmanr(x, y)
        line = f"   {arm:<12} n={len(pts):<3} distinctiveness vs r2: rho={rho:+.3f} p={p:.3f}"
        if rel:
            ks = [k for k in keys if k in distinct and k in ev and k in rel
                  and ev[k].get(f"{arm}_r2") not in ("", None)
                  and not (arm == "pooled" and int(ev[k]["umbrella_target"]))]
            rx, ry, rz = (stats.rankdata(v) for v in (
                [distinct[k] for k in ks], [float(ev[k][f"{arm}_r2"]) for k in ks],
                [rel[k] for k in ks]))
            resid = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
            pr, pp = stats.pearsonr(resid(rx), resid(ry))
            line += f"   | partial, ceiling held fixed: {pr:+.3f} p={pp:.3f}"
        print(line)

    print(f"\n   wrote {args.pairs_out.relative_to(REPO)} and "
          f"{args.matrix_out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
