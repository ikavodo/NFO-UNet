# Literature/novelty pass, 2026-11-11: SAM2 occlusion adapter + mask-based ape behaviour recognition

Source: research-companion:brainstormer, about 25 minutes of searching. "Unverified" means taken from a search snippet or
from memory, not from the paper itself. Check every item before citing it.

## Changes to the plan
- **The behaviour baseline is no longer the IJCV 2024 table.** Mueller, Lüddecke, Vogg & Ecker, "Domain-Adaptive Pretraining
  Improves Primate Behavior Recognition", CVPR 2025 CV4Animals workshop (arXiv 2509.12193). V-JEPA with domain-adaptive
  pretraining; +6.1 pp accuracy over the previous state of the art on PanAf and +6.3 pp mAP on ChimpACT.
  **Unverified:** whether "PanAf" there means PanAf500, the absolute numbers, and the cropping (reportedly Grounding DINO) and
  head (reportedly an attentive classifier on a frozen backbone).
- **PanAf-FGBG** (Brookes et al., CVPR 2025) already runs SAM2.1-L on PanAf with one point prompt per ape, to erase apes and
  make background-only clips. It shows that background is a confound for behaviour models. Its split into overlapping vs.
  disjoint camera locations is the evaluation reviewers will respect.
- The control for "masks help recognition" is **a box crop with the same backbone and the same head** (one variable). Full
  frame is not that control.

## Closest prior work
| area | work | relation |
|---|---|---|
| SAM2 for occlusion, training-free | SAMURAI (arXiv 2411.11922), SAM2Long (arXiv 2410.16268, venue unverified), DAM4SAM (CVPR 2025), HiM2SAM (arXiv 2507.07603) | baselines; all change memory, not features |
| SAM2, trained | SAMITE (arXiv 2507.21732): prototype memory + position prompts, aimed at occlusion | closest competitor (prompt side) |
| | SAM2-Adapter (arXiv 2408.08870, venue unverified): encoder adapters for camouflage/shadow | closest competitor (feature-residual side) |
| | MOSE decoder fine-tuning (arXiv 2504.08306), SAM2-OV (AAAI 2026) | context |
| SAM-based behaviour | SAMURAI masks → DINOv2 → MLP/LSTM for pigs and cows (citations unverified) | masks used only as preprocessing |
| | Region-Based Representations Revisited (CVPR 2024, arXiv 2402.02352): DINOv2/CLIP features pooled inside SAM masks | image-level mask pooling |
| | arXiv 2412.01136: SAM2 object pointers concatenated over time (referring segmentation) | precedent for the token (ii) feature option |
| object-centric action recognition | ORViT (CVPR 2022), STLT/Something-Else (CVPR 2020), arXiv 2306.11726 | the standard recipe; gains of a few points, largest on fine-grained tasks |
| wildlife SAM2 | Meier & Lüddecke, zero-shot multi-animal tracking (arXiv 2511.02591, ChimpACT); SA-FARI camera-trap masklets (arXiv 2511.15622); MammAlps (arXiv 2503.18223) | SA-FARI = external tracking test with masks |
| object score | no calibration study found. SAM2Long uses the score as a heuristic. RUAC (arXiv 2605.10603) studies mask uncertainty. SAM 3 (arXiv 2511.16719) has an explicit presence head | open; if AP is ranked on obj_score, add a reliability diagram/ECE |

## Novelty, as of this pass
- Not found: learned residuals on **both** SAM2's backbone features and the per-frame prompt, aimed at occlusion.
- Not found: SAM2 masks or features as the input to great-ape behaviour recognition on PanAf.
- **Likely reviewer objection:** "a standard adapter plus ORViT-style mask pooling, with the gain coming from domain
  adaptation rather than occlusion handling." It is answered by:
  - results split into occluded vs. clear frames (needs the hand-labelled subset);
  - DAM4SAM and SAM 3 baselines;
  - **no PanAf training of the adapter**, so domain adaptation cannot explain a gain. The NFO-only adapter is the right
    choice here.
