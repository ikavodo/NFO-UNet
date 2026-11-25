# Next session: light, tied, teacher-free feature integral (written 2026-11-25)

Paste the block below as the first message of the new session.

---

We continue the one-pass **feature-integral** line of the NFO adapter (branch `exp/layer-sam-feat`, nothing committed yet: commit
the current work first if I ask). Read first: CLAUDE.md (its "closed directions" line on the one-pass adapter is outdated: the
line is open and is THE line), findings `docs/vos_benchmark/nfo_vos_pilot_findings.md` §10.x items 8, 9, 20–25, failure log
tail, and `docs/layer_sam2/feature_layer_sam.html` (the method page).

**Goal.** One per-frame model (frozen SAM2.1-b+ image encoder, one encoder pass, small learned modules in feature space) that
beats SAM2 video (B0) and SAMURAI (T1) on BOTH box AUC and recognition confidence, near real-time. One output mask for both
metrics (no per-metric heads: selection bias). Elegance and generality matter: the integral should be a clean, teacher-free
feature-space module.

**Where we are (one seed each).**
- Through-the-frozen-pixel-reader student (item 23): held-out seq2–4, 21 untouched crossings, box AUC 0.712 vs pixel adapter 0.720;
  capped by construction (it imitates the pixel route).
- Item 24 (the breakthrough): a training-only amodal head (`AmHead` on the Fusion trunk, `AMODAL=1` in `adapter/fusion_adapter.py`),
  loss on the amodal mask S restricted to Ω (hidden at t, visible in aligned history; plus background), makes the reader depend on
  Ĉ on real data for the first time. seq1, 8 crossings: box reliance +0.101 vs +0.010; recognition 0.611 vs 0.588 (+0.023
  [+0.000, +0.043]); box AUC 0.539 vs 0.551 (n.s.). Integral used there: frozen BigS16 (distilled to the pixel teacher).
- Item 25 speed: GPU tracker port 0.49 vs 4.6 ms/frame with identical tracks (`kernel_track_gpu` in `tools/speed_check.py`,
  not yet moved into `adapter/tracker.py`); `torch.compile` encoder 19.7 vs 25.2 ms; encoder crop is invalid (s16 features
  change, 1 − cos 0.19). Projected ≈ 29 ms/frame, not measured end to end.

**Next experiment (decided).** Train everything from scratch on synthetic only, no pixel teacher:
- Integral: the light `FeatIntegral` (learned α softmax over 6 shifted keys + null slot, convex mix at s16/8/4), no `Refined`,
  no `BigS16`.
- **Tie the evidence mass**: a = 1 − α_∅, removing the separate mass head (`feat_integral.py:132`, `:176-179`) and `mass_loss`.
- Losses: (1) visible loss v3.4, (2) amodal-Ω aux, (3) optional clean-feature loss `feat_loss` on Ω. In `train_step`, `lam`
  currently switches `feat_loss` AND `mass_loss` together: split them.
- Compare against item 24 (`adapter-featbigam`) on seq1's 8 crossings, same laptop environment (@lap), box AUC + recognition,
  each with its -noC control; synthetic readout AMGAP (printed at validation) as the early kill check.

**Known risk to address before running.** Tying was tried before and undone: commit fea385f ("5b gate showed the shared
softmax trades Ĉ (s16 0.493 → 0.412) for mass (AUC 0.799 → 0.885)"). Mechanism to check: in background cells every key ≈ B,
so α_∅ vs keys is not identified and 1 − α_∅ need not mean "person evidence"; under mass_loss + feat_loss the two targets
fought over one softmax. With mass_loss gone and the amodal loss shaping α, it may behave differently, but say how (math first),
and pre-register what would show the tie failed (e.g. a on the person vs background, Ĉ cos to clean T on Ω, AMGAP).

**Rules.** seq2–4 is the held-out test: never use it for selection or screening (the old "s0 screen" leaked 3 test crossings;
screen on seq1 only). Ask before long runs; justify each experiment (question, action per outcome, validity); one variable at
a time where possible; log every result to findings §10.x and dead ends to the failure log the same turn. One pre-authorised
expert design review before implementing. Seeds only on the final configuration; same-environment B0/T1 rescoring only before a
claim. Laptop: `../samurai/.venv/bin/python`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, kill only by PID.
Gotchas: `box_metrics score` overwrites `box_gt.csv` with only the listed arms (backup: `box_gt_heldout_v2.bak.csv`); in run
scripts use `grep --line-buffered` before `tee` or the live log stays empty; `train_step` now returns (loss, aux loss).
Be brief and decisive with me; plain language, few options, no detail dumps.
