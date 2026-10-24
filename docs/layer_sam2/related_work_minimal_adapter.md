# Related work: fewer inputs and encoder passes for the per-frame adapter (2026-10-20)

- **Question searched:** can the per-frame layer adapter on frozen SAM2.1-b+ drop its hand-made inputs (explicit
  feature differences F_X − F_C, F_X − F_B; evidence mass a) and the extra encoder pass for the pixel integral image C (code: composite),
  without losing performance?
- **Run by:** research-companion:brainstormer subagent, about 9 web lookups, ~2 min.
- **Checked against:** `../briefing/index.html` §3 (architecture) and `benchmark/nfo_vos/layers/fusion_adapter.py`
  (`Fusion.z`, `Fusion.forward`).

## Verdict

- **Most threatening prior work:** Airborne Optical Sectioning, especially *Inverse AOS* (static camera, occluded
  walkers; arXiv 2207.13344). It integrates pixels before detection to see people through foliage, which is the
  principle behind our integral image. Cite it as the ancestor of the integral image.
- **Defensible contribution:** open-loop pixel integration (AOS-style) injected into a frozen promptable segmenter,
  against SAM2's closed-loop feature memory.
- **Open gap:** no paper found shows whether **feature-space** aggregation matches **pixel-space** aggregation when
  every single frame is fragmentary. Testing that separates *where* we integrate (pixels vs features) from *how*
  (open-loop vs closed-loop), and either outcome is informative.

## Closest prior work per component

| our component | closest prior work | what it does | how close |
|---|---|---|---|
| pixel integral image C (aligned, change-gated) | Nathan, Kurmi, Schedl, Bimber, *Through-Foliage Tracking with Airborne Optical Sectioning*, J. Remote Sensing 2022, arXiv 2111.06959; *Inverse Airborne Optical Sectioning*, arXiv 2207.13344 | integrate many views/frames before detection; precision ~42% → ~97% for people in forest | ancestor (same principle, no segmenter) |
| multi-frame obstruction removal | Liu et al., *Learning to See Through Obstructions*, CVPR 2020, arXiv 2004.01180 | pixel-level flow + layer decomposition for fences and reflections | partial |
| feature-space alternative to C | Zhu et al., *Deep Feature Flow*, CVPR 2017, arXiv 1611.07715; *FGFA*, ICCV 2017, arXiv 1703.10025; Gadde et al., *NetWarp*, ICCV 2017, arXiv 1708.03088; *MEGA*, CVPR 2020, arXiv 2003.00063 | warp and weight past CNN features; each frame there holds a degraded but **whole** object | partial |
| alignment level | Tian et al., *TDAN*, CVPR 2020, arXiv 1812.02898 | argues feature-level alignment beats image-level warping (flow errors propagate) | partial |
| zero-init residual adapter | Zhang et al., *ControlNet*, ICCV 2023, arXiv 2302.05543; Mou et al., *T2I-Adapter*, AAAI 2024, arXiv 2302.08453 | condition encoder added as zero-initialised / lightweight residuals | same mechanism |
| SAM adapters with hand-made side inputs | Chen et al., *SAM-Adapter*, ICCVW 2023, arXiv 2304.09148; Liu et al., *Explicit Visual Prompting*, CVPR 2023, arXiv 2303.10883; *SAM2-Adapter*, arXiv 2408.04579 | explicit hand-crafted components (e.g. FFT high frequencies) fed through MLPs at multiple scales; ablations show they help | partial |
| prompt-token correction | Ke et al., *HQ-SAM*, NeurIPS 2023, arXiv 2306.01567 | learned output token + fused early/late encoder features | partial |
| explicit difference features | Daudt et al., *Fully Convolutional Siamese Networks for Change Detection*, ICIP 2018, arXiv 1810.08462 (difference vs concatenation: dataset-dependent); *ChangeFormer*, arXiv 2201.01293; *BIT*, arXiv 2103.00208; *Segment Any Change*, NeurIPS 2024, arXiv 2402.01188 (zero-shot change from SAM feature matching); *SAM-CD*, TGRS 2024, arXiv 2309.01429 | difference of bitemporal features as input | partial |
| evidence mass a → learned gate | Bhat et al., *Deep Burst Super-Resolution*, CVPR 2021, arXiv 2101.10997 | per-element learned fusion weights over warped past features; feeding feature **residuals** to the weight predictor beat raw features | closest analogue |
| empty-scene reference | Tezcan et al., *BSUV-Net*, WACV 2020, arXiv 1907.11371 | current frame + two background references concatenated in pixel space; no hand-made mass | partial |
| learned gating, generic | Li et al., *Gated Fully Fusion*, AAAI 2020, arXiv 1904.01803; Oh et al., *STM*, ICCV 2019, arXiv 1904.00607 (cross-attention memory read) | learned per-pixel gates / fully learned memory read | ancestor |
| closed-loop memory baselines | *SAMURAI*, arXiv 2411.11922; *SAM2Long*, arXiv 2410.16268; *DAM4SAM*, CVPR 2025, arXiv 2411.17576 | change memory selection, still write own predictions (closed loop) | baselines |

## Not novel (standard practice)

- Zero-initialised residual conditioning of a frozen backbone (ControlNet, T2I-Adapter).
- Hand-made auxiliary inputs to a SAM adapter (Explicit Visual Prompting, SAM-Adapter).
- Explicit feature differences against a reference image (change detection: FC-Siam-diff, ChangeFormer, BIT).
- Background-reference conditioning (BSUV-Net).
- Pixel integration to see through foliage (AOS).

## Possibly novel

- **Open-loop pixel integration as conditioning for a frozen promptable segmenter, against closed-loop memory.** To
  claim it: the existing ablations (integral image +0.018 box AUC, +0.022 recognition conf; memory arms ≤ +0.012 J bound)
  plus the variant (c) test below.
- **Pixel vs feature aggregation under fragmented occlusion.** To claim it: variant (c) against the pixel integral image,
  ≥ 2 seeds, same synthetic data.

## Analysis of the three reduced-input variants

The argument for keeping the pixel integral image: the encoder f is nonlinear, so f(Σ wᵢ gᵢ xᵢ) ≠ Σ wᵢ gᵢ f(xᵢ) in general.
Under fragmented occlusion each xᵢ holds only slivers, so the encoder never sees a contiguous person in any single frame.

| variant | change | information lost? | risk | verdict |
|---|---|---|---|---|
| (a) | drop the explicit differences, concatenate [F_X, F_C, F_B] | none: after the shared 1×1 projection, the differences are linear in the concatenation, so the first conv can represent them exactly | slower or worse convergence from zero-init on synthetic data | conceptual tidiness only; not worth a run on its own |
| (b) | drop a, learn a gate gᵢ = σ(conv(\|Fᵢ − F_B\|)) | a gate on F_X, F_B alone cannot recover a, because a summarises **past** frames | slivers < 16 px vanish at stride 16; gate at stride 4 | only meaningful inside (c) |
| (c) | feature-space integral image: cache past F at strides 16/8/4, shift by tracker motion, learned gates from (b), recency weights, normalised sum. One encoder pass per frame | the encoder never sees integrated pixels | box AUC and recognition, the metrics where the integral image matters | the decisive experiment |

## Baselines a reviewer will demand

- SAMURAI and SAM2.1 video on the same windows. Already in the benchmark: box AUC (LaSOT OPE), hand J, recognition AP50.
- An AOS-style unmasked integration baseline for recognition. Already present as "no mask" (aligned plain mean).
- DAM4SAM or SAM2Long as a second memory-selection baseline. Not yet run.

## Cheapest kill test

- **Run:** retrain with variant (c)+(b), same synthetic data, ≥ 2 seeds.
- **Compare against:** the full adapter and the no-integral-image arm (`-noC`), on the synthetic val clips and on the seq2–4 downstream metrics.
- **Adopt if:** (c) recovers ≥ half of the integral image's gap (box AUC 0.018, recognition conf 0.022), outside the seed spread.
- **Resolution check:** the gap is resolvable. The 3-seed sd of box AUC is 0.004 and of recognition AP50 is 0.006 (measured 2026-10-20).
- **Don't:** swap feature-space integral images into the current adapter without retraining. The input distribution changes, which confounds the test.

## Citation status

- **Verified** (title and id resolved during the search): 2111.06959, 2207.13344, 2101.10997, 2402.01188, 2408.04579,
  1812.02898, 1907.11371, 1810.08462, 1708.03088.
- **Unverified** (from memory, not re-checked; spot-check before citing): 1611.07715, 1703.10025, 2003.00063,
  2004.01180, 2304.09148, 2303.10883, 2302.05543, 2302.08453, 2306.01567, 2304.12620, 2411.11922, 2410.16268,
  2411.17576, 2201.01293, 2103.00208, 2309.01429, 1904.01803, 1904.00607.
- No corrections to earlier project claims.

## Sources

- [arXiv 2111.06959](https://arxiv.org/abs/2111.06959v2)
- [arXiv 2207.13344](https://export.arxiv.org/pdf/2207.13344)
- [arXiv 2101.10997](https://arxiv.org/pdf/2101.10997)
- [arXiv 2402.01188](https://arxiv.org/html/2402.01188v4)
- [arXiv 2408.04579](https://arxiv.org/pdf/2408.04579)
- [arXiv 1812.02898](https://arxiv.org/pdf/1812.02898)
- [arXiv 1907.11371](https://arxiv.com/abs/1907.11371)
- [arXiv 1810.08462](https://ar5iv.arxiv.org/html/1810.08462)
- [arXiv 1708.03088](https://ar5iv.arxiv.org/html/1708.03088)
