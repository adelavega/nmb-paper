#!/usr/bin/env python3
"""Copy the main-text figures into manuscript/figures/ under their figure numbers.

The figure scripts write to results/figures/; this copies the ones the manuscript uses, renamed
by number, so the paper folder always holds the current versions. Re-run after re-rendering.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
FIGURES = {
    1: "figure_target_overlap",          # study x analysis overlap of sibling targets
    2: "figure_target_map_embedding",    # MDS of the reference maps
    3: "figure_method_specificity",      # raw r2, identification, sibling discrimination
    4: "figure_difficulty_profile",      # difficulty profile + exemplars (NeuroQuery)
    "S1": "figure_reliability_distribution",  # split-half reliability of every reference
}


def main() -> int:
    src, dst = REPO / "results" / "figures", REPO / "manuscript" / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    for n, stem in FIGURES.items():
        for ext in ("pdf", "png"):
            out = dst / f"Figure{n}.{ext}"
            shutil.copy2(src / f"{stem}.{ext}", out)
            print(f"  {stem}.{ext} -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
