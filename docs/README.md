# docs/ index

**Start here:** `vos_benchmark/nfo_vos_pilot_findings.md` (current results), `vos_benchmark/nfo_failure_log.md`
(one line per dead end), `briefing/index.html` (collaborator briefing).

Layout: one folder per topic; dated specs and plans (`YYYY-MM-DD-*.md`) sit in their topic folder; closed or
superseded work goes to the topic's `archive/`, kept for its results and reasoning.

- `vos_benchmark/` — the NFO video-segmentation benchmark (current work, Oct 2026)
  - `nfo_vos_pilot_findings.md`, `nfo_failure_log.md` — results and dead ends
  - `2026-10-02-nfo-vos-benchmark-pilot-design.md`, `2026-10-05-nfo-vos-benchmark-pilot.md` — protocol (rev. 5) and
    its implementation plan
  - `plan_downstream_benchmark.md` — box AUC + recognition on masked stacks (`benchmark/nfo_vos/eval/`)
  - `hand_annotation_plan.md`, `annotator_cvat_setup.md`, `label_generators_design.md` — how hand labels were made;
    the second annotator's guide; automatic label generators
  - `nfo_pseudo_segmentation_approach.md` — the SAM2-ceiling (former pseudo-GT) generator
  - `gtfree_clean_integral_image.md` — construction of the GT-free integral image C, with equations
  - `gt_conditioned_blend_plan.md` — GT-conditioned exemplar blending
  - `archive/` — full chronological findings log, the A/4 port
- `layer_sam2/` — the learned adapter on frozen SAM2 (code: `benchmark/nfo_vos/adapter/`); see `layer_sam2/README.md`
- `unet/` — the original spatio-temporal U-Net: training-failure root cause (resolved)
- `poster/poster_plan.md` — conference poster: structure, figures, vocabulary, open decisions; `poster/poster_latex/`
  — the poster's LaTeX sources and figures (TUM template and logos local only)
- `briefing/` — collaborator briefing (HTML + figures); published as the claude.ai artifact "Seeing Through Foliage",
  https://claude.ai/artifact/3Fq1La6yPnNqLjQ8SgiP9d (shared with the organisation). The repo copy is the source:
  edit `briefing/index.html`, then republish it to that URL
- blob tracker (Aug–Sep 2026): code (`tracking/`) and design notes (`archive/blob_tracker/`) are at git tag
  `archive/pre-cleanup-2026-10-21`
- `external/` — papers and poster text by others (SatSAM2, ViperSAM, STORE poster); local only, not committed
