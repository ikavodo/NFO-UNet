# docs/ index (by topic)

- `vos_benchmark/` — NFO video-segmentation benchmark (current work, Oct 2026)
  - `nfo_vos_pilot_findings.md` — constructive results log, chronological; the last sections are the current state
  - `nfo_failure_log.md` — one line per dead end
  - `hand_annotation_plan.md`, `label_generators_design.md` — how the hand labels / automatic labels were made
  - `gtfree_clean_composite.md` — construction of the GT-free composite C, with equations
  - `nfo_pseudo_segmentation_approach.md` — the SAM2-ceiling (former pseudo-GT) generator
- `layer_sam2/` — learned in-SAM2 architecture (next step); `plan_feature_layers_training.md` = current training plan
  - `layered_sam2_architecture.md` (read section v2), `plan_gpjatk_layer_sam2.md` (implementation plan),
    `related_work_layered_sam2.md` (novelty pass), `Layered Memory for SAM2 Test Plan.md` (user's test plan)
- `blob_tracker/` — classical blob tracker, scale generalisation, live deployment (Aug–Sep 2026)
- `unet/` — the original spatio-temporal U-Net: training-failure root cause
- `external/` — papers and poster text by others / for reference (SatSAM2, ViperSAM, STORE poster); not committed
- `superpowers/` — dated specs and implementation plans
