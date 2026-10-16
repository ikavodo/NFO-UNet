# layer_sam2/ — learned in-SAM2 components (index, 2026-10-16)

Status: **live** = current direction · **done** = executed, results inside · **closed** = killed or superseded, kept
for its results and reasoning.

## training/ — what to train
- `plan_fusion_prompt_adapter.md` — **live.** Composite/background fusion adapter → gated dense prompt for a frozen,
  memory-free SAM2 with the tracker point. Gate O1 (GT-prompt oracle) first.
- `plan_feature_layers_training.md` — **done / closed.** Tracker-guided deformable static/dynamic feature layers.
  Steps 1–3 were run (nuisance model, dataset, G1 baselines); §10 is the oracle logit bias inside SAMURAI:
  ΔJ +0.017 < +0.02, so the training-free route was killed. Its §9 math (energy detector vs matched filter) is
  reused by the live plan.

## data/ — the synthetic GPJATK + foliage testbed
- `synthetic_nfo_design.md` — **done.** Dataset design, pre-registered bands, layered-experiment results. Chosen
  setting: perlin clump 24, density ≈ 0.45, occluder ≡ foliage-photo background.
- `plan_gpjatk_layer_sam2.md` — **done.** Implementation plan for the generator (`benchmark/nfo_vos/synth/`) and the
  SAM2 recall sanity check.

## design/ — architecture drafts and prior work
- `layered_sam2_architecture.md` — **closed (v1–v3 drafts).** v3 holds the oracle visibility result: stride 16 fails
  on sub-cell fragments (AUC 0.612), strides 8 / 4 pass. It also holds the true SAM2 cell sizes.
- `layered_memory_test_plan.md` — the user's test plan of 2026-10-12. Option C (dense mask prompt) is the ancestor
  of the live plan; options A / B (memory tokens, differential attention) are out of scope, since the live plan uses
  no memory (user, 2026-10-16).
- `related_work_layered_sam2.md` — novelty pass of 2026-10-14 for the v2 architecture.
