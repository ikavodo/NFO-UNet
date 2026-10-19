# docs/ index (by topic)

- `vos_benchmark/` — NFO video-segmentation benchmark (current work, Oct 2026)
  - `nfo_vos_pilot_findings.md` — compressed current findings (task, metrics, results, evidence status); full chronological log in `archive/nfo_vos_pilot_findings_full.md`
  - `nfo_failure_log.md` — one line per dead end
  - `hand_annotation_plan.md`, `label_generators_design.md` — how the hand labels / automatic labels were made
  - `gtfree_clean_composite.md` — construction of the GT-free composite C, with equations
  - `nfo_pseudo_segmentation_approach.md` — the SAM2-ceiling (former pseudo-GT) generator
- `layer_sam2/` — learned in-SAM2 components; see `layer_sam2/README.md` (live: `training/plan_fusion_prompt_adapter.md`)
  - `training/` plans, `data/` synthetic GPJATK testbed, `design/` architecture drafts, user's test plan, novelty pass
- `blob_tracker/` — classical blob tracker, scale generalisation, live deployment (Aug–Sep 2026)
- `unet/` — the original spatio-temporal U-Net: training-failure root cause
- `external/` — papers and poster text by others / for reference (SatSAM2, ViperSAM, STORE poster); not committed
- `superpowers/` — dated specs and implementation plans
