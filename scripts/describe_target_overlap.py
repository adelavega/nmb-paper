#!/usr/bin/env python3
"""How much do sibling targets within a project share their studies? (Results section 1)

Some projects split one literature into sub-questions answered by the same papers (emotion
regulation: decrease / increase / maintain contrasts taken from the same studies); others assemble
each sub-target from a largely separate literature (VBM of substance use: one drug class per
target). This script measures that from the curation records, for the 35 selected targets.

For every pair of targets in a project, from the studies (PMIDs) each target draws on:

    study_jaccard      |A n B| / |A u B|
    study_overlap      |A n B| / min(|A|, |B|)   -- 1 when the smaller target is nested in the
                       larger, which Jaccard hides whenever the two differ in size
    analysis_jaccard   the same Jaccard on analyses, for comparison: a pair can share studies
                       but not analyses (different contrasts from the same papers)

The "all" targets and the two supersets of siblings (see EXCLUDED) contain other targets, so pairs
involving one are flagged (umbrella_pair = 1) and kept out of the project summary. Per target, `shared_share` is the fraction of its studies that also feed at least one
non-umbrella sibling. Per project, `multi_target_share` is the fraction of the project's studies
(over non-umbrella targets) that feed more than one target.

Writes results/target_study_overlap_pairs.csv, results/target_study_overlap_targets.csv and
results/target_study_overlap_projects.csv.
"""

from __future__ import annotations

import collections
import csv
import itertools
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
# Excluded from the pairwise comparison: the "all" targets, which pool their project, and the two
# supersets of siblings -- emotion regulation reappraisal (contains every decrease and increase
# analysis) and cue reactivity reward (contains every drug- and natural-cue study). Pairs with
# them measure nesting, not how the experts split a literature.
EXCLUDED = {("executive_function", "all"), ("social", "all_merged"),
            ("problem_solving", "global_calculationandvisuospatialandword_mni_final"),
            ("dementia", "all"), ("vbm_of_substance_use", "all_drug_classes"),
            ("vbm_of_ptsd", "nonptsdgtptsd_merged"),
            ("emotion_regulation_2022", "reappraisal"),
            ("cue_reactivity", "1_reward_neutral_2020_wbonly")}


def load_project(project: str):
    base = NMB / "data" / "nimads" / project / "merged"
    ss = json.loads((base / "nimads_studyset.json").read_text())
    ann = json.loads((base / "nimads_annotation.json").read_text())
    ann = ann[0] if isinstance(ann, list) else ann
    study_of = {a["id"]: s["id"] for s in ss["studies"] for a in s["analyses"]}
    cols = [k for k, v in (ann.get("note_keys") or {}).items() if v == "boolean"]
    analyses = {c: set() for c in cols}
    annotated = set()
    for n in ann.get("notes", []):
        note = n.get("note") or {}
        a = n.get("analysis")
        if a not in study_of:
            continue
        hit = False
        for c in cols:
            if note.get(c):
                analyses[c].add(a)
                hit = True
        if hit:
            annotated.add(a)
    return analyses, study_of, len(annotated)


def jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else float("nan")


def ovl(a: set, b: set) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else float("nan")


def main() -> int:
    selected = collections.defaultdict(list)
    for r in csv.DictReader(open(NMB / "data" / "selected_targets.csv")):
        selected[r["project"]].append(r["target"])

    pair_rows, target_rows, project_rows = [], [], []
    for project, targets in sorted(selected.items()):
        analyses, study_of, n_annotated = load_project(project)
        lower = {c.lower().replace("-", "_"): c for c in analyses}
        col = {t: lower[t.lower()] for t in targets}
        an = {t: analyses[col[t]] for t in targets}
        stu = {t: {study_of[a] for a in an[t]} for t in targets}
        umb = {t: (project, t) in EXCLUDED for t in targets}
        core = [t for t in targets if not umb[t]]

        for a, b in itertools.combinations(targets, 2):
            pair_rows.append(dict(
                project=project, target_a=a, target_b=b,
                umbrella_pair=int(umb[a] or umb[b]),
                n_studies_a=len(stu[a]), n_studies_b=len(stu[b]),
                shared_studies=len(stu[a] & stu[b]),
                study_jaccard=round(jac(stu[a], stu[b]), 4),
                study_overlap=round(ovl(stu[a], stu[b]), 4),
                analysis_jaccard=round(jac(an[a], an[b]), 4)))

        for t in targets:
            others = set().union(*(stu[o] for o in core if o != t)) if len(core) > 1 else set()
            target_rows.append(dict(
                project=project, target=t, umbrella=int(umb[t]),
                n_studies=len(stu[t]), n_analyses=len(an[t]),
                shared_share=round(len(stu[t] & others) / len(stu[t]), 4)
                if stu[t] and others else ""))

        core_pairs = [p for p in pair_rows if p["project"] == project and not p["umbrella_pair"]]
        counts = collections.Counter(s for t in core for s in stu[t])
        pool = len(counts)
        sj = [p["study_jaccard"] for p in core_pairs]
        so = [p["study_overlap"] for p in core_pairs]
        aj = [p["analysis_jaccard"] for p in core_pairs]
        project_rows.append(dict(
            project=project, n_targets=len(targets), n_umbrella=sum(umb.values()),
            n_core_pairs=len(core_pairs), core_study_pool=pool,
            multi_target_share=round(sum(v > 1 for v in counts.values()) / pool, 4) if pool else "",
            study_jaccard_mean=round(st.mean(sj), 4) if sj else "",
            study_jaccard_min=min(sj) if sj else "", study_jaccard_max=max(sj) if sj else "",
            study_overlap_mean=round(st.mean(so), 4) if so else "",
            analysis_jaccard_mean=round(st.mean(aj), 4) if aj else "",
            study_minus_analysis=round(st.mean(sj) - st.mean(aj), 4) if sj else ""))

    out = REPO / "results"
    for name, rows in (("pairs", pair_rows), ("targets", target_rows),
                       ("projects", project_rows)):
        with open(out / f"target_study_overlap_{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"{'project':<24}{'tgts':>5}{'umb':>4}{'pairs':>6}{'pool':>6}{'multi':>7}"
          f"{'studyJ':>8}{'(range)':>14}{'overlap':>8}{'analysisJ':>10}")
    for r in sorted(project_rows, key=lambda r: (r["study_jaccard_mean"] == "",
                                                 -(r["study_jaccard_mean"] or 0))):
        rng = (f"{r['study_jaccard_min']:.2f}-{r['study_jaccard_max']:.2f}"
               if r["n_core_pairs"] else "")
        f = lambda v: f"{v:.2f}" if v != "" else "–"
        print(f"{r['project']:<24}{r['n_targets']:>5}{r['n_umbrella']:>4}{r['n_core_pairs']:>6}"
              f"{r['core_study_pool']:>6}{f(r['multi_target_share']):>7}"
              f"{f(r['study_jaccard_mean']):>8}{rng:>14}{f(r['study_overlap_mean']):>8}"
              f"{f(r['analysis_jaccard_mean']):>10}")
    print("  wrote results/target_study_overlap_{pairs,targets,projects}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
