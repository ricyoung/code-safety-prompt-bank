# Licensing Notes

This release consolidates prompts from four upstream benchmarks under different terms. This document records each source's licensing status, the practical interpretation adopted here, and the takedown commitment that compensates where no explicit license exists upstream.

## Upstream sources and their licenses

| Source | Prompts in release | License verified at upstream repo | URL |
|---|---|---|---|
| **AdvBench / harmful_behaviors** (Zou et al. 2023) | 129 code + 388 knowledge | **MIT** | https://github.com/llm-attacks/llm-attacks |
| **CySecBench** (Wahréus et al. 2025) | 632 | **MIT** | https://github.com/cysecbench/dataset |
| **RMCBench** (Chen et al. 2024) | 473 | **NO EXPLICIT LICENSE** | https://github.com/qing-yuan233/RMCBench |
| **MalwareBench** (Li et al. 2025) | 320 | **NO EXPLICIT LICENSE** | https://github.com/MAIL-Tele-AI/MalwareBench |

## Interpretation adopted here

- Authors' own contributions — pipeline code, Fleiss κ implementation, consensus labels, figure scripts, dataset cards — are released under the **MIT License** (see `LICENSE`).
- Prompts from MIT-licensed sources (AdvBench, CySecBench) are redistributed under MIT with attribution.
- Prompts from unlicensed sources (RMCBench, MalwareBench) are redistributed under a **research-use fair-use interpretation**, with a standing takedown commitment: if any upstream author requests removal, their prompts will be replaced with UID-only entries (no prompt text) within 30 days.

## Takedown procedure

To request removal of your prompts from this release:

1. Email `ryoung@unlv.edu` identifying the source and prompts (by UID prefix or upstream identifier).
2. The maintainers will acknowledge within 5 business days.
3. Prompt text will be redacted within 30 days of acknowledgment; UIDs and consensus labels will be retained for scientific reproducibility of the companion papers.
4. A user-facing loader script will be added to the repo so downstream researchers can re-fetch the redacted prompts from the upstream source by UID if they have access.

## Courtesy notifications

Before public release, the maintainers will notify the authors of RMCBench and MalwareBench by email, linking to the private staging dataset and offering a 14-day review window. The public release will not proceed until either:

- The upstream authors have responded with no objection, or
- The 14-day window has elapsed with no response.

## Why this matters

Research consolidations of safety datasets live in a gray area of copyright law. By documenting upstream status explicitly, committing to takedown, and pre-notifying authors, this release follows the community norm for responsible dataset aggregation established by prior artifacts including HarmBench and JailbreakBench.

If you use this dataset in your own work, **please cite all four upstream sources** alongside the consolidation paper. The upstream sources did the work of creating the prompts; this release adds only the consensus labeling axis.
