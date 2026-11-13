"""Layer-SAM on apes, Stage 1 (design: docs/layer_sam2/2026-11-13-ape-stage1-data-design.md). Data paths by env."""
import os
from pathlib import Path

PANAF = Path(os.environ.get('PANAF_REPO', '../PanAf20K')).resolve()          # repo: labels/, results/panaf/
PANAF_ROOT = Path(os.environ.get('PANAF_ROOT', PANAF / 'PanAf500')).resolve()  # PanAf500 videos + annotations
SBR_ROOT = Path(os.environ.get('SBR_ROOT', '../PanAf-SBR/extracted/PanAf-SBR-Masks-24FPS')).resolve()
FGBG_ROOT = Path(os.environ.get('FGBG_ROOT', '../PanAf-FGBG/panaf_fgbg_cr')).resolve()
OUT = Path(os.environ.get('APE_OUT', 'results/ape')).resolve()
