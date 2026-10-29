#!/usr/bin/env python3
"""Does analysis-only separability between sibling targets predict selection dependence?

PREDICTOR. For each selected target, the mean over its within-project siblings of
(analysis Jaccard - study Jaccard). Negative = the target shares its siblings' studies but not
their analyses, i.e. it is separable from them only at the analysis level.

OUTCOMES. Selection dependence, the y-axis of the difficulty taxonomy:
1 - min(1, pooled r2 / split-half ceiling), i.e. how much of the target's reliable signal the
project-pooled map fails to capture. Also each tool's r2 / ceiling.

WHAT IT FOUND (2026-10-07; n = 25 selected, non-umbrella, non-degenerate targets)

Overall: yes. rho = -0.48 (p = 0.014), unchanged with ceiling and pool fraction held fixed
(-0.49, p = 0.013); using the most extreme sibling pair, -0.55 (p = 0.004).

BUT IT IS ONE PROJECT. Drop emotion regulation and it vanishes (rho = -0.10, p = 0.65); drop any
other project and it survives (rho -0.45 to -0.65, all p < 0.04). Emotion regulation is the only
project whose targets are separated at the analysis level but not the study level -- every other
target's gap lies within -0.05 of zero -- so the benchmark has almost no variance on this
predictor outside it. The result is consistent with the hypothesis but cannot establish it as
a general property. Do not report the pooled coefficient without the leave-one-project-out result.

Selection dependence also has a second route that this predictor does not capture: targets
drawn from a largely separate literature that is a small slice of the project pool
(executive_function/flexibility 0.31, problem_solving/visuospatial 0.24, both with gap ~0).
An article-level method that found the right studies would recover those; it would not recover
the emotion-regulation targets, whose studies are shared with their siblings.

No relationship with tool recovery once emotion regulation is set aside (NeuroQuery and NeuroVLM
r2/ceiling, rho -0.05 to -0.30, all p > 0.18).
"""
import csv, collections, statistics as st
import numpy as np
from scipy import stats
from pathlib import Path
import os
REPO = Path(__file__).resolve().parent.parent          # nmb-paper
NMB = Path(os.environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
sel ={(r["project"], r["target"]) for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv"))}
ev = {(r["project"], r["target"]): r for r in csv.DictReader(open(REPO / "results/text_to_map_baselines.csv"))}
rel = {(r["project"], r["target"]): float(r["reliability_mean"]) for r in csv.DictReader(open(REPO / "results/reference_reliability.csv"))}

gaps = collections.defaultdict(list)
for r in csv.DictReader(open(REPO / "results/target_structure.csv")):
    a, b = (r["project"], r["target_a"]), (r["project"], r["target_b"])
    if a in sel and b in sel:
        g = float(r["analysis_jaccard"]) - float(r["study_jaccard"])
        gaps[a].append(g); gaps[b].append(g)

rows = []
for k in sorted(sel):
    e = ev[k]
    if e["umbrella_target"] == "1" or e["reference_degenerate"] == "1" or k not in gaps: continue
    c = rel[k]; po = float(e["pooled_r2"])
    rows.append(dict(k=k, gap=st.mean(gaps[k]), gmin=min(gaps[k]), ceil=c,
                     dep=1 - min(1.0, po / c), frac=float(e["fraction_of_project_pool"]),
                     nq=float(e["neuroquery_r2"]) / c, nv=float(e["neurovlm_r2"]) / c))
print(f"n = {len(rows)} targets (selected, non-umbrella, non-degenerate, with >=1 sibling)\n")
for r in sorted(rows, key=lambda r: r["gap"]):
    print(f"  {r['k'][0][:12]:<13}{r['k'][1][:24]:<25} gap {r['gap']:+.3f}  min {r['gmin']:+.3f}  "
          f"sel.dep {r['dep']:.2f}  ceil {r['ceil']:.2f}  NQ/c {r['nq']:.2f}  NV/c {r['nv']:.2f}")

def partial(x, y, zs):
    R = lambda v: stats.rankdata(v)
    Z = np.column_stack([np.ones(len(x))] + [R(z) for z in zs])
    res = lambda v: R(v) - Z @ np.linalg.lstsq(Z, R(v), rcond=None)[0]
    return stats.pearsonr(res(x), res(y))

V = lambda key, rs=rows: np.array([r[key] for r in rs])
print("\nDoes analysis-only separability (gap; more negative = more analysis-only) predict ...")
for out, lab in (("dep", "selection dependence (plot A y-axis)"), ("nq", "NeuroQuery r2/ceiling"), ("nv", "NeuroVLM r2/ceiling")):
    for pred in ("gap", "gmin"):
        rho, p = stats.spearmanr(V(pred), V(out))
        pr, pp = partial(V(pred), V(out), [V("ceil"), V("frac")])
        noer = [r for r in rows if r["k"][0] != "emotion_regulation_2022"]
        r2, p2 = stats.spearmanr(V(pred, noer), V(out, noer))
        print(f"  {lab:<38} by {pred:<4}: rho={rho:+.2f} (p={p:.3f})  | ceiling+pool-fraction held: "
              f"{pr:+.2f} (p={pp:.3f})  | without emotion reg (n={len(noer)}): {r2:+.2f} (p={p2:.3f})")

print("\nleave-one-project-out, gap -> selection dependence:")
for proj in sorted({r["k"][0] for r in rows}):
    sub = [r for r in rows if r["k"][0] != proj]
    rho, p = stats.spearmanr(V("gap", sub), V("dep", sub))
    print(f"  drop {proj:<25} n={len(sub):<3} rho={rho:+.2f}  p={p:.3f}")
