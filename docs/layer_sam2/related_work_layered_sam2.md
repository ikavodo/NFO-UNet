# Related work: learned layered refinement for SAM2 under fragmented occlusion

2026-10-14. Novelty pass for `docs/layer_sam2/layered_sam2_architecture.md`, run by the research-companion brainstormer
(about 20 searches). Question: which components of the proposed method (frozen SAM2 + learned empty-scene comparison
+ person-aligned feature composite + reliability gate + zero-init stride-4/8 residuals, trained on layered synthetic
data) already exist, and what is defensible as new.

## Verdict

- **Most threatening prior work:** BSUV-Net 2.0 (background reference + generalisation to unseen videos + jitter and
  illumination augmentation), with SAM-CD as the structural template (frozen SAM encoder + bi-temporal adapter). A
  reviewer can describe the architecture as "SAM-CD with an empty-scene reference plus EDVR, plugged into SAM2".
- **Defensible contribution:** (a) the measured, stride-explained failure of SAM2 on visible fragments smaller than
  one stride-16 cell (H2: recall 0.00 below half a cell; no paper found that quantifies this), and (b) showing that a
  background layer recovers exactly that size bin. Lead with fragment-size-binned recall, not the module list.

## Closest prior work per component

| Component | Prior work | What it does | Closeness |
|---|---|---|---|
| Person composite: alignment + temporal fusion | EDVR, Wang et al., CVPRW 2019, arXiv 1905.02716 | Pyramid/cascading deformable alignment (PCD) + temporal-spatial attention fusion (TSA) | Same mechanism (module A) |
| | BasicVSR | Standard follow-up to EDVR | Ancestor (unverified in this pass) |
| Learned frame vs empty-scene comparison | BSUV-Net, Tezcan, Ishwar & Konrad, WACV 2020, arXiv 1907.11371 | CNN on current frame + empty and recent backgrounds (+ semantics), illumination augmentation, unseen videos | Same idea (module B) |
| | BSUV-Net 2.0, IEEE Access 2021, arXiv 2101.09585 | Adds spatio-temporal augmentation incl. camera jitter / PTZ | Same, incl. our jitter randomisation |
| | SAM-CD, Ding et al., arXiv 2309.01429 | Frozen FastSAM encoder, Siamese bi-temporal comparison, small trainable adapter | Structural template for module B |
| | Segment Any Change, Zheng et al., NeurIPS 2024 | Training-free bi-temporal matching of SAM latents | Prior for the cosine baseline |
| | ZBS, CVPR 2023 | Zero-shot background subtraction | Unverified |
| Per-pixel reliability gate | (none direct) | Attention-weighted fusion is generic (EDVR's TSA) | Novel only as a layer-explanation framing |
| Zero-init residuals, high-res injection | ControlNet, Zhang & Agrawala, arXiv 2302.05543 | Zero-initialised convolutions for conditioning a frozen model | Standard |
| | HQ-SAM, Ke et al., NeurIPS 2023, arXiv 2306.01567 | Frozen SAM + learned HQ output token + early high-res feature fusion | Very close to the output head of module D |
| | SAM-Adapter, arXiv 2304.09148; SAM2-Adapter, arXiv 2408.04579 | Multi-resolution adapters (camouflage, shadow) | Standard |
| | HQTrack, arXiv 2307.13974 | Tracker + HQ-SAM refiner in video | Video precedent |
| Memory changes / re-prompting | SAMURAI, arXiv 2411.11922 | Motion-aware memory selection | Our T1 |
| | DAM4SAM, CVPR 2025, arXiv 2411.17576 | Distractor-aware memory | Memory variant |
| | SAM2Long, arXiv 2410.16268 | Tree-structured memory | Second-hand only |
| | Segment Any Motion in Videos, CVPR 2025, arXiv 2503.22268 | Long-range trajectories + DINO, iterative SAM2 prompting | Closest "motion cue → SAM2 dense prompt" |
| | FlowP-SAM, arXiv 2404.12389 | Optical flow as SAM prompt | Prompt-level injection |
| Amodal extent | TABE, arXiv 2411.19210 | Zero-shot amodal VOS via video-diffusion outpainting | Amodal ancestor |
| | Diffusion-VAS, Chen, Ramanan & Khurana, CVPR 2025, arXiv 2412.04623 | Amodal masks from modal masks + depth (tested here: FAIL) | Amodal baseline |
| | SAMEO, CVPR 2025, arXiv 2503.06261 | SAM as amodal decoder, synthetic data | Amodal |
| | EoRaS, ICCV 2023, arXiv 2309.13248 | Supervised amodal video segmentation (synthetic MOVi) | Amodal |
| | Amodal SAM, arXiv 2604.20748 (Apr 2026) | SAM + adapter + synthetic occlusion pipeline, video | Closest recent amodal threat (SAM2 use unchecked) |
| Layers / seeing through occluders | Wang & Adelson, IEEE TIP 3(5), 1994 | Layered motion representation | Conceptual ancestor |
| | Vaish, Szeliski, Zitnick, Kang & Levoy, CVPR 2006 | Synthetic aperture, robust median/entropy costs under occlusion | Ancestor of the composite |
| | Airborne optical sectioning: Kurmi, Schedl & Bimber, Sci. Rep. 2022; Nathan et al., J. Remote Sensing 2022 | Integrating views to detect/track people through foliage | Nearest application |
| | Generative Omnimatte, CVPR 2025, arXiv 2411.16683 | Layer decomposition of video | Layers |
| | Simple Copy-Paste, Ghiasi et al., CVPR 2021, arXiv 2012.07177 | Copy-paste augmentation | Our synthetic data |
| The problem | Pegoraro & Pflugfelder, arXiv 2004.13076 | Fragmented occlusion in object detection | Must cite |
| | PETS2021 through-foliage challenge | Detection/tracking through foliage | Must cite |
| | Auer, TU Wien diploma thesis 2022 | Spatio-temporal aggregation, synthetic data, fragmented occlusion | Check overlap with planned claims |

## Not novel (standard practice)
Zero-init residuals; adapters; HQ-style high-res injection; EDVR-style alignment and fusion; BSUV-style comparison
against a background reference with illumination/jitter augmentation; synthetic layered compositing / copy-paste;
memory re-weighting and re-prompting.

## Possibly novel (what would have to be shown)
1. Empty-scene comparison inside a frozen video foundation model's stride-4/8 decoder path: no SAM2 + BSUV hybrid
   found. Must beat BSUV-Net 2.0 ∪ SAM2.
2. Exact visible / amodal / occluder supervision on fragmented occlusion with a sub-cell fragment loss.
3. The phenomenon: a stride-explained recall cliff of SAM2 on sub-cell visible fragments (a fragment of area fraction
   p of a cell contributes about p to the cell's token). Must be shown with size-binned recall on human labels.

## Baselines a reviewer will demand
1. SAM2 fine-tuned on the same synthetic composites (SAM2-Adapter or HQ-SAM-style decoder tuning, same trainable
   parameter count): separates "layer structure helps" from "more data helps".
2. BSUV-Net 2.0 (pretrained, given the same empty scene) alone and unioned with SAM2.
3. Zero-shot trackers prompted identically: SAM2.1, SAMURAI, DAM4SAM; plus the training-free V̂ rule and a
   Segment-Any-Change-style cosine comparison.
Metrics: J&F on NFO (seq2–4 held out); visible-fragment recall binned by size relative to the stride-16 cell
(headline); removed-pixel and leak area; amodal mIoU vs TABE / Diffusion-VAS if extent is claimed. External validity:
PETS2021 through-foliage; CDnet-2014 unseen-video F-measure for the background module. 3 seeds, one variable per
ablation.

## Cheapest kill test
BSUV-Net 2.0 ∪ SAM2 vs V̂ on seq1–4, size-binned. If it closes the sub-cell gap, modules B and D reduce to an
off-the-shelf union, and the paper becomes the failure analysis plus that fix.

## Citation status and corrections
- Verified by search (title and id): all entries with an arXiv id above, Wang & Adelson 1994, Vaish et al. 2006, the
  AOS papers, PETS2021, Auer thesis.
- Unverified: BasicVSR (not checked), ZBS (not found), SAM2Long (second-hand), ControlNet zero-conv detail (id
  resolved, detail not in snippet), Amodal SAM's use of SAM2.
- Corrections to earlier claims in this project: Vaish et al. has five authors (incl. Levoy); airborne optical
  sectioning is not a Nature Machine Intelligence paper as stated from memory on 2026-10-14 — cite Sci. Rep. 2022 /
  J. Remote Sensing 2022.

## Sources
[EDVR](https://arxiv.org/abs/1905.02716v1) · [BSUV-Net](https://arxiv.org/pdf/1907.11371) ·
[BSUV-Net 2.0](https://arxiv.org/pdf/2101.09585) · [SAM-CD](https://arxiv.org/abs/2309.01429v3) ·
[HQ-SAM](https://arxiv.org/pdf/2306.01567) · [HQTrack](https://arxiv.org/pdf/2307.13974) ·
[SAM2-Adapter](https://arxiv.org/pdf/2408.04579) · [SAM-Adapter](https://www.alphaxiv.org/resources/2304.09148) ·
[SAMURAI](https://arxiv.org/pdf/2411.11922) · [DAM4SAM](https://arxiv.org/pdf/2411.17576v2) ·
[Segment Any Motion](https://arxiv.org/pdf/2503.22268) · [FlowP-SAM](https://arxiv.org/html/2404.12389v2) ·
[TABE](https://arxiv.org/pdf/2411.19210) · [Diffusion-VAS](https://arxiv.org/pdf/2412.04623) ·
[SAMEO](https://arxiv.org/pdf/2503.06261) · [EoRaS](https://arxiv.org/pdf/2309.13248) ·
[Amodal SAM](https://www.alphaxiv.org/abs/2604.20748) · [Generative Omnimatte](https://arxiv.org/pdf/2411.16683) ·
[Wang & Adelson](https://persci.mit.edu/pub_abstracts/wang_tr279_abs.html) ·
[Vaish et al. 2006](https://mlanthology.org/cvpr/2006/vaish2006cvpr-reconstructing) ·
[AOS classification](https://pure.fh-ooe.at/en/publications/combined-person-classification-with-airborne-optical-sectioning/) ·
[AOS tracking](https://research.jku.at/en/publications/through-foliage-tracking-with-airborne-optical-sectioning/) ·
[Copy-Paste](https://arxiv.org/abs/2012.07177v2) · [Fragmented occlusion](https://ar5iv.labs.arxiv.org/html/2004.13076) ·
[PETS2021](https://reading-clone.eprints-hosting.org/104305) · [Auer thesis](https://repositum.tuwien.at/handle/20.500.12708/19637)
