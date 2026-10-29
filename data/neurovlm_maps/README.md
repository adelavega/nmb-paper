# NeuroVLM predicted maps

One map per benchmark target contrast, named `<project>__<target>.nii.gz`, produced by querying
NeuroVLM with the query string recorded for that target in
`scripts/eval_text_to_map.py::QUERIES`.

**These are committed rather than generated** because NeuroVLM needs a dependency set that
conflicts with the pinned stack this repo uses to reproduce the reference maps (NiMARE 0.2.1 /
nilearn 0.10.1 / numpy < 2). They were generated in an isolated environment and copied in.

Coverage: 32 of the benchmark's 38 target contrasts. Missing are the three non-whole-brain-only
cue-reactivity variants and the cannabis, opioids and stimulants contrasts of
`vbm_of_substance_use` — the last three have no suprathreshold voxel in their reference map
(the source paper reports no significant result), so they cannot be scored on overlap anyway.

Maps are in whatever grid NeuroVLM emits; `eval_text_to_map.py` resamples each one to the
reference map's grid before comparison, leaving the reference untouched.

To regenerate or extend, record the NeuroVLM version and the environment used here.
