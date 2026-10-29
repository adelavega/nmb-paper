#!/usr/bin/env python3
"""Score text-to-map methods on target-specific signal, with project-level uncertainty.

WHY RAW r2 IS NOT ENOUGH

Every forward-inference map contains the same generic component -- where fMRI studies tend to
report activation at all -- and raw r2 rewards a method for sharing it with the reference.
Neurosynth's uniformity map is MKDA over keyword-selected studies, the same estimator as the
references, so it shares that component almost by construction; NeuroQuery leans towards
reverse inference and shares less of it. Raw r2 therefore partly measures inference direction
rather than whether a method found the right topic. Swapping in a chi-squared association
reference does not fix this: it needs a reference corpus, and any corpus favours the tool that
was built on it.

Three scores that a shared component cannot inflate, plus raw r2 for comparison:

    raw_r2          squared Pearson r with the target's own reference, inside the brain mask
                    (reproduces results/text_to_map_baselines.csv).
    residual_r2     the same after removing, from every map, its projection onto the mean map of
                    its own set: each reference is residualised on the mean of the 31
                    references, each method's map on the mean of that method's 31 maps. Signed
                    (sign(r) * r^2) so an anti-correlated map scores below zero. Residualising,
                    not subtracting: subtracting the mean turns a weak map into roughly minus the
                    mean, and since every method's mean resembles the references' mean, the two
                    minus-means correlate for no topical reason. That inflated exactly the weak,
                    atypical targets -- vbm_of_ptsd rose from raw r2 0.04 to 0.21 under
                    subtraction, and is 0.03 residualised.
    identification  rank percentile of the correct reference among all 31: the share of the
                    other references the method's map correlates with less. 0.5 is chance. A
                    component shared by every reference moves all 31 correlations together, so
                    it cannot change the rank.
    sibling_win     the same, restricted to the target's siblings in its own project -- can the
                    method tell apart the meta-analyses that one expert team distinguished?
                    Undefined for single-target projects.

UNCERTAINTY

Targets are not independent: siblings share studies, query vocabulary and a project-specific
literature, and project alone explains 81-83% of the variance in tool r2. Intervals therefore
come from a cluster bootstrap over the 9 projects (projects resampled with replacement, then
targets within each drawn project), 10,000 draws, percentile 95% CIs. Pairwise method
comparisons use an exact sign-flip permutation test on per-project mean differences (2^9 = 512
assignments), Holm-corrected across the six pairs within each score. Nine clusters is few, so
intervals are wide by design; that is the honest width.

Writes results/specificity_scores.csv (per target x method), results/specificity_summary.csv
(per method: mean and CI), results/specificity_pairwise.csv (per pair: difference, CI, p),
results/specificity_by_target.csv (per reference: mean and range over the four methods -- which
references the automated methods can predict at all).
"""

from __future__ import annotations

import csv
import inspect
import itertools
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_text_to_map as E  # noqa: E402

METHODS = ("NeuroQuery", "NeuroVLM", "Neurosynth association", "Neurosynth uniformity")
SCORES = ("raw_r2", "residual_r2", "identification", "sibling_win")
N_BOOT = 10_000
SEED = 0


def method_path(method: str, project: str, target: str, term: str) -> Path:
    term = term.replace(" ", "_")
    return {"NeuroQuery": REPO / "results" / "neuroquery_maps" / f"{project}__{target}.nii.gz",
            "NeuroVLM": REPO / "data" / "neurovlm_maps" / f"{project}__{target}.nii.gz",
            "Neurosynth association":
                REPO / "results" / "neurosynth_maps_heldout" / term / "z_desc-association.nii.gz",
            "Neurosynth uniformity":
                REPO / "results" / "neurosynth_maps_heldout" / term / "z_desc-uniformity.nii.gz",
            }[method]


def _z(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)


def corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between every row of A and every row of B."""
    return _z(A) @ _z(B).T / A.shape[1]


def residualise(X: np.ndarray) -> np.ndarray:
    """Standardise each map, then remove its projection onto the set's (standardised) mean map."""
    m = X.mean(0)
    m = (m - m.mean()) / m.std()
    Z = _z(X)
    return Z - np.outer(Z @ m / len(m), m)


def per_target_scores(X: np.ndarray, R: np.ndarray, projects: np.ndarray) -> dict:
    n = len(R)
    C = corr_matrix(X, R)
    d = np.diag(corr_matrix(residualise(X), residualise(R)))
    out = {"raw_r2": np.diag(C) ** 2, "residual_r2": np.sign(d) * d ** 2,
           "identification": np.array([(C[i] < C[i, i]).sum() / (n - 1) for i in range(n)])}
    sib = []
    for i in range(n):
        js = [j for j in range(n) if projects[j] == projects[i] and j != i]
        sib.append(np.mean([C[i, i] > C[i, j] for j in js]) if js else np.nan)
    out["sibling_win"] = np.array(sib)
    return out


def cluster_bootstrap(values: np.ndarray, projects: np.ndarray, rng) -> np.ndarray:
    """Bootstrap distribution of the mean over targets, resampling projects then targets."""
    groups = [np.flatnonzero(projects == p) for p in np.unique(projects)]
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, len(g)) for g in
                              (groups[k] for k in rng.integers(0, len(groups), len(groups)))])
        draws[b] = np.nanmean(values[idx])
    return draws


def signflip_p(diff: np.ndarray, projects: np.ndarray) -> float:
    """Exact two-sided sign-flip test on per-project mean differences."""
    per = np.array([np.nanmean(d) if np.isfinite(d := diff[projects == p]).any() else np.nan
                    for p in np.unique(projects)])
    per = per[np.isfinite(per)]
    obs = abs(per.mean())
    flips = np.array(list(itertools.product((1, -1), repeat=len(per))))
    return float(np.mean(np.abs((flips * per).mean(1)) >= obs - 1e-12))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def main() -> int:
    import nibabel as nib
    from nilearn.image import resample_to_img
    E._RESAMPLE_PARAMS = set(inspect.signature(resample_to_img).parameters)

    ev = {(r["project"], r["target"]): r
          for r in csv.DictReader(open(REPO / "results" / "text_to_map_baselines.csv"))}
    sel = [(r["project"], r["target"])
           for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv"))]
    keys = [k for k in sel if ev[k]["reference_degenerate"] == "0"]
    projects = np.array([p for p, _ in keys])

    grid = nib.load(str(E.reference_dir(*keys[0]) / "z.nii.gz"))
    brain = np.asarray(E.resample(nib.load(str(NMB / "data" / "brain_mask_mni152_2mm.nii.gz")),
                                  grid).dataobj) > 0.5

    def vec(path) -> np.ndarray:
        a = np.asarray(E.resample(nib.load(str(path)), grid).dataobj, dtype=float)[brain]
        a[~np.isfinite(a)] = 0.0
        return a

    R = np.array([vec(E.reference_dir(p, t) / "z.nii.gz") for p, t in keys])
    scores = {}
    for m in METHODS:
        X = np.array([vec(method_path(m, p, t, ev[(p, t)]["neurosynth_term"])) for p, t in keys])
        scores[m] = per_target_scores(X, R, projects)
        print(f"  {m:<24} raw r2 {scores[m]['raw_r2'].mean():.3f}")

    out = REPO / "results"
    with open(out / "specificity_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "target", "method", *SCORES])
        for m in METHODS:
            for i, (p, t) in enumerate(keys):
                w.writerow([p, t, m, *(round(float(scores[m][s][i]), 6) for s in SCORES)])

    rng = np.random.default_rng(SEED)
    with open(out / "specificity_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "method", "mean", "ci_low", "ci_high", "n_targets"])
        for s in SCORES:
            for m in METHODS:
                v = scores[m][s]
                lo, hi = np.percentile(cluster_bootstrap(v, projects, rng), [2.5, 97.5])
                w.writerow([s, m, round(float(np.nanmean(v)), 4), round(lo, 4), round(hi, 4),
                            int(np.isfinite(v).sum())])

    with open(out / "specificity_pairwise.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "method_a", "method_b", "mean_diff", "ci_low", "ci_high",
                    "p_signflip", "p_holm"])
        for s in SCORES:
            rows = []
            for a, b in itertools.combinations(METHODS, 2):
                diff = scores[a][s] - scores[b][s]
                lo, hi = np.percentile(cluster_bootstrap(diff, projects, rng), [2.5, 97.5])
                rows.append([s, a, b, round(float(np.nanmean(diff)), 4), round(lo, 4),
                             round(hi, 4), round(signflip_p(diff, projects), 4)])
            for r, ph in zip(rows, holm([r[-1] for r in rows])):
                w.writerow([*r, round(ph, 4)])
    # Which references are predictable: per target, the mean over the four methods, with the
    # range across methods as its spread (four methods is too few for an interval).
    rel = {(r["project"], r["target"]): r
           for r in csv.DictReader(open(out / "reference_reliability.csv"))}
    with open(out / "specificity_by_target.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "target", "reliability",
                    *(f"{s}_{k}" for s in SCORES[:3] for k in ("mean", "min", "max")),
                    "methods_top1", "methods_top3"])
        for i, (p, t) in enumerate(keys):
            cells = []
            for sc in SCORES[:3]:
                v = np.array([scores[m][sc][i] for m in METHODS])
                cells += [round(float(v.mean()), 4), round(float(v.min()), 4),
                          round(float(v.max()), 4)]
            ident = np.array([scores[m]["identification"][i] for m in METHODS])
            w.writerow([p, t, rel[(p, t)]["reliability_mean"], *cells,
                        int((ident >= 1 - 1e-9).sum()), int((ident >= 1 - 2 / 30 - 1e-9).sum())])
    print("  wrote results/specificity_{scores,summary,pairwise,by_target}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
