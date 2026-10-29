# nmb-paper

Analyses, figures and manuscript drafts for the NeuroMetaBench data paper. 

The dataset itself
(curated NiMADS studysets, target annotations, and the scripts that build the MKDA reference
maps) lives in [neurometabench](https://github.com/adelavega/neurometabench).
## Setup

```bash
pixi install
```

The scripts expect a neurometabench checkout next to this one (`../neurometabench`). If yours
is somewhere else, set `NMB_ROOT=/path/to/neurometabench`. They read these parts of it:

- `data/`: NiMADS studysets and annotations, `selected_targets.csv`, the brain mask
- `analysis/<project>/<target>/`: the MKDA density reference maps. These are not tracked in
  neurometabench; build them there with `pixi run -e nimare021 python scripts/run_meta_analyses.py`
- `scripts/nimads_meta.py`, `scripts/map_mask.py`: imported directly

## Layout

```
scripts/      every analysis, figure and descriptive script for the paper
results/      CSV outputs and working figures (results/figures/)
manuscript/   the numbered figures (the manuscript text is drafted elsewhere)
data/         paper-only inputs: NeuroVLM maps, Neurosynth term list
```

Untracked and regenerable (see `.gitignore`):

- `results/neuroquery_maps/`, `results/neurosynth_maps*/`: cached method maps
- `results/chi2_neurosynth_v7/`, `results/pooled_all_analyses/`: refits
- `data/reference_corpora/`

## Reproducing the paper

Run every command from this directory with `pixi run python scripts/<name>.py`.

1. Method maps and raw scores: `eval_text_to_map.py`, `run_neurosynth_arm.py` (held-out
   Neurosynth), `run_pooled_baseline.py`
2. Reference reliability: `compute_reference_reliability.py` (split-half, 10 seeds), then
   `run_article_level_arm.py`, which reads it
3. Target-specific scores: `eval_specificity.py` writes `results/specificity_*.csv`
4. Descriptives: `describe_benchmark.py`, `describe_target_overlap.py`, `analyse_*.py`
5. Figures:

   | Figure | Script |
   |---|---|
   | 1 | `make_target_overlap_figure.py` |
   | 2 | `make_target_map_embedding.py` |
   | 3 | `make_specificity_figures.py` |
   | 4 | `make_difficulty_profile_figure.py` |
   | S1 | `make_reliability_distribution_figure.py` |

   After running them, `export_paper_figures.py` copies the figures into `manuscript/figures/`
   under their numbers.

`project_palette.py` holds the shared project colours and target display names. The
remaining `make_*` scripts produce exploratory figures that are not in the manuscript.
