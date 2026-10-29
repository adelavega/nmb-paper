#!/usr/bin/env python3
"""Is difficulty a property of the target or of the tool?

WHY THESE ANALYSES

Prompted by the LLM systematic-review benchmark literature, which reports two things this
resource can test directly. SciLitBench (arXiv:2609.05505) finds that automated performance
degrades as the target becomes more interpretive, and runs ensemble analyses to ask whether
combining systems beats the best single one. NeuroMetaBench can ask both questions of
text-to-map tools, because it records how specific each target is -- the fraction of its
project's analysis pool the target uses -- and scores several tools on the same references.

1. SPECIFICITY GRADIENT. Do the tools do worse on narrow targets than broad ones?
2. SHARED FAILURE. Do the tools fail on the same targets, or on different ones?
3. ENSEMBLE HEADROOM. Would a best-of-both oracle beat the better single tool?

WHAT THEY FOUND (run 2026-09-25, 31-34 non-degenerate targets)

1. No gradient for either tool: NeuroQuery rho = -0.08 (p = 0.67), NeuroVLM rho = -0.10
   (p = 0.58). The internal project-pooled arm, by contrast, shows a strong positive gradient
   (rho = +0.69, p = 0.0001) -- exactly as it must, since a single broad map resembles a broad
   target more than a narrow one. That the internal arm behaves as theory predicts while the
   tools show nothing is the interesting part: the tools are not failing at specificity, they
   are failing at the topic level, which is a more fundamental limitation.

2. The tools fail on the same targets: Spearman rho = +0.85 between their per-target scores.
   Difficulty is a property of the target, not an idiosyncrasy of either system.

3. Consequently the ensemble headroom is negligible: a per-target oracle picking the better of
   the two gains +0.004 over NeuroVLM alone. The tools are redundant rather than complementary.

NeuroVLM does outrank NeuroQuery, and the margin survives project-level clustering: mean
+0.053, ahead in 25 of 31 targets, cluster-bootstrap 95% CI [+0.027, +0.080].

Writes results/tool_failure_modes.csv.
"""

from __future__ import annotations

import argparse
import csv
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
ARMS = ("neuroquery", "neurovlm", "pooled")


def load(path: Path) -> list[dict]:
    return [r for r in csv.DictReader(open(path)) if not int(r["reference_degenerate"])]


def cluster_bootstrap(pairs, resamples=20000, seed=0):
    """95% CI for a paired mean difference, resampling PROJECTS not contrasts.

    Contrasts within a project share studies and a reference record, so resampling them as if
    independent would understate the interval.
    """
    projects = sorted({p for p, _, _ in pairs})
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(resamples):
        pick = rng.choice(len(projects), len(projects), replace=True)
        vals = [d for i in pick for pr, _, d in pairs if pr == projects[i]]
        if vals:
            means.append(float(np.mean(vals)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
                    default=REPO / "results" / "text_to_map_baselines.csv")
    ap.add_argument("--output", type=Path,
                    default=REPO / "results" / "tool_failure_modes.csv")
    args = ap.parse_args(argv)

    rows = load(args.input)
    out = []

    print("1. SPECIFICITY GRADIENT -- does r2 track how broad the target is?\n")
    for arm in ARMS:
        sel = [(float(r["fraction_of_project_pool"]), float(r[f"{arm}_r2"])) for r in rows
               if r.get(f"{arm}_r2") not in ("", None)
               and not (arm == "pooled" and int(r["umbrella_target"]))]
        if len(sel) < 6:
            continue
        x, y = zip(*sel)
        rho, p = stats.spearmanr(x, y)
        med = st.median(x)
        lo = [v for f, v in sel if f <= med]
        hi = [v for f, v in sel if f > med]
        print(f"  {arm:<12} n={len(sel):<3} rho={rho:+.3f} p={p:.4f}   "
              f"narrow {st.mean(lo):.3f} vs broad {st.mean(hi):.3f}")
        out.append({"analysis": "specificity_gradient", "arm": arm, "n": len(sel),
                    "statistic": round(rho, 4), "p_value": round(p, 5),
                    "detail": f"narrow={st.mean(lo):.3f};broad={st.mean(hi):.3f}"})

    print("\n2. SHARED FAILURE -- do the tools fail on the same targets?\n")
    pair = [(r["project"], r["target"], float(r["neuroquery_r2"]), float(r["neurovlm_r2"]))
            for r in rows if r.get("neuroquery_r2") not in ("", None)
            and r.get("neurovlm_r2") not in ("", None)]
    nq = [a for _, _, a, _ in pair]
    nv = [b for _, _, _, b in pair]
    rho, p = stats.spearmanr(nq, nv)
    print(f"  per-target agreement: rho={rho:+.3f} (p={p:.4g}), n={len(pair)}")
    out.append({"analysis": "shared_failure", "arm": "neuroquery_vs_neurovlm",
                "n": len(pair), "statistic": round(rho, 4), "p_value": round(p, 8),
                "detail": "spearman of per-target r2"})

    d = [b - a for _, _, a, b in pair]
    lo, hi = cluster_bootstrap([(p_, t, b - a) for p_, t, a, b in pair])
    w = stats.wilcoxon(nq, nv).pvalue
    print(f"  neurovlm - neuroquery: mean {st.mean(d):+.3f}, ahead in "
          f"{sum(1 for x in d if x > 0)}/{len(d)}, Wilcoxon p={w:.4g}")
    print(f"    cluster-bootstrap 95% CI over projects: [{lo:+.3f}, {hi:+.3f}]")
    out.append({"analysis": "tool_margin", "arm": "neurovlm_minus_neuroquery",
                "n": len(d), "statistic": round(st.mean(d), 4), "p_value": round(w, 8),
                "detail": f"CI=[{lo:+.4f},{hi:+.4f}];ahead={sum(1 for x in d if x > 0)}"})

    print("\n3. ENSEMBLE HEADROOM -- would picking the better tool per target help?\n")
    oracle = st.mean(max(a, b) for _, _, a, b in pair)
    best_single = max(st.mean(nq), st.mean(nv))
    print(f"  per-target oracle {oracle:.3f} vs better single tool {best_single:.3f}"
          f"  -> headroom {oracle - best_single:+.3f}")
    out.append({"analysis": "ensemble_headroom", "arm": "oracle_max", "n": len(pair),
                "statistic": round(oracle - best_single, 4), "p_value": "",
                "detail": f"oracle={oracle:.4f};best_single={best_single:.4f}"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(out[0]))
        w_.writeheader()
        w_.writerows(out)
    print(f"\n  wrote {args.output.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
