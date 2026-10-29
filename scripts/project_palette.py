"""One colour per project, shared by every figure that colours by topic.

The four projects the Results lean on keep the colours they already had in earlier figures:
executive function blue, problem solving green, emotion regulation orange, cue reactivity
magenta. Social cognition and decision making take the two remaining task-fMRI hues. The three
clinical / structural projects are neutral greys told apart by marker, since nine hues cannot all
be distinguished at once.

Validation (dataviz validate_palette.js, --pairs all): the six hues pass every check; the worst
pair, cue reactivity vs executive function, is dE 6.4 for deuteranopes -- inside the 6-8 band
that is legal only with secondary encoding, so any figure using all six must label its points
or groups directly. Subsets used elsewhere (EF / ER / cue: worst dE 21.8) pass outright.
"""

from __future__ import annotations

# project key -> (display label, short label, colour, marker)
PROJECTS: dict[str, tuple[str, str, str, str]] = {
    "executive_function": ("Executive function", "EF", "#0072B2", "o"),
    "problem_solving": ("Problem solving", "PS", "#009E73", "o"),
    "emotion_regulation_2022": ("Emotion regulation", "ER", "#D55E00", "o"),
    "cue_reactivity": ("Cue reactivity", "Cue", "#A0479A", "o"),
    "social": ("Social cognition", "Social", "#56B4E9", "o"),
    "decision_making": ("Decision making", "DM", "#E6A800", "o"),
    "dementia": ("Dementia", "Dementia", "#6E6E6E", "s"),
    "vbm_of_ptsd": ("VBM of PTSD", "VBM PTSD", "#8C8C8C", "^"),
    "vbm_of_substance_use": ("VBM of substance use", "VBM SUD", "#A8A8A8", "v"),
}


def colour(project: str) -> str:
    return PROJECTS[project][2]


def marker(project: str) -> str:
    return PROJECTS[project][3]


# Display names for the 35 benchmark targets, used by every figure. Aggregates of sibling targets
# are named as such: reappraisal contains all of the decrease and increase analyses (152 of its
# 154); the reward-cue target contains every drug- and natural-cue study.
NICE_NAMES = {
    ("cue_reactivity", "1_reward_neutral_2020_wbonly"): "reward cues (all)",
    ("cue_reactivity", "2_drug_neutral_2020_wbonly"): "drug cues",
    ("cue_reactivity", "3_natural_neutral_2020_wbonly"): "natural reward cues",
    ("decision_making", "adm_july2019"): "ambiguous decisions",
    ("decision_making", "rdm_july2019"): "risky decisions",
    ("decision_making", "perceptual_dm_july2019"): "perceptual decisions",
    ("dementia", "all"): "all findings",
    ("dementia", "decrease"): "decreases",
    ("dementia", "functional"): "functional",
    ("dementia", "structural"): "structural",
    ("emotion_regulation_2022", "decrease"): "decrease",
    ("emotion_regulation_2022", "increase"): "increase",
    ("emotion_regulation_2022", "maintain"): "maintain",
    ("emotion_regulation_2022", "reappraisal"): "reappraisal (dec + inc)",
    ("executive_function", "all"): "all tasks",
    ("executive_function", "working_memory"): "working memory",
    ("executive_function", "inhibition"): "inhibition",
    ("executive_function", "flexibility"): "flexibility",
    ("problem_solving", "global_calculationandvisuospatialandword_mni_final"): "all tasks",
    ("problem_solving", "calculation_mni_final"): "calculation",
    ("problem_solving", "wordproblems_mni_final"): "word problems",
    ("problem_solving", "visuospatialreasoning_mni_final"): "visuospatial reasoning",
    ("problem_solving", "demand_mni_final"): "demand",
    ("social", "all_merged"): "all domains",
    ("social", "self_merged"): "self",
    ("social", "others_merged"): "others",
    ("social", "affiliation_merged"): "affiliation",
    ("social", "soccomm_merged"): "social communication",
    ("vbm_of_ptsd", "nonptsdgtptsd_merged"): "decrease GM",
    ("vbm_of_substance_use", "all_drug_classes"): "all drug classes",
    ("vbm_of_substance_use", "alcohol"): "alcohol",
    ("vbm_of_substance_use", "cannabis"): "cannabis",
    ("vbm_of_substance_use", "nicotine"): "nicotine",
    ("vbm_of_substance_use", "opioids"): "opioids",
    ("vbm_of_substance_use", "stimulants"): "stimulants",
}
AGGREGATE_LABELS = {k: NICE_NAMES[k] for k in (("emotion_regulation_2022", "reappraisal"),
                                                ("cue_reactivity", "1_reward_neutral_2020_wbonly"))}


def target_label(project: str, target: str) -> str:
    """Display name for a target, without the project prefix."""
    if (project, target) in NICE_NAMES:
        return NICE_NAMES[(project, target)]
    return target.replace("_", " ")
