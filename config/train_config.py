import dill
import torch.nn

from config.config import AbstractConfig, available_cpus
from dataset.abstract_dataset import HeatMap
from dataset.kth_dataset import KthDataSet
from logistic_loss import LogisticLoss
from utils.fs_utils import ensure_dir
from utils.transform_utils import rand_h_flip, rand_v_flip, rand_rot_90, reduce_colors, rand_color_swap, \
    rand_zoom_out

base_config = AbstractConfig({
    # dataset
    # dataset type to use
    'dataset_type': None,
    # path to training data root
    'train_data': None,
    # path to evaluation data root
    'eval_data': None,
    # exclude certain folders from the dataset inside of the train and eval data folder
    'exclude_dirs': [],
    # search string specifying which heatmap type to use (either use _gauss or _circle)
    'hm_filter': None,
    # criterion to be used in training
    'criterion': None,
    # batch size for training and evaluation
    'batch_size': None,
    # number of frames in one sequence
    'seq_size': None,
    # every nth frame will be included in the sequences
    'nth_frame': 1,
    # shuffle training dataset
    'shuffle': True,
    # number of workers for loading data samples
    'num_workers': 0,
    # pin memory of the data loader
    'pin_memory': True,
    # transforms to use for training (order matters)
    'train_transforms': [],
    # transforms to use for evaluation (order matters)
    'eval_transforms': [],

    # training
    # number of epochs for training
    'num_epochs': 100,
    # learning rate for training
    'lr': None,
    # every nth minibatch will be printed to the console (running mean loss or evaluation)
    'print_ma': 100,
    # if true, early stopping will be enabled
    'enable_early_stopping': True,
    # how many times the network can stay without improvement before early stopping will trigger
    'early_stopping_patience': 15,
    # True for visualizing, False otherwise
    'visualize': False
})
config = base_config.copy()

# train on KTH (standard person split), eval_data is the held-out validation persons.
# Hyperparameters match the publication (Auer, "Robust object localization under
# fragmented occlusion", 2022): the paper's own head-to-head comparison (Sec 4.3.3) shows
# logistic loss beats MSE on final NFO F1 for the same n5,2/n7,2 networks (0.906 vs 0.739
# for n5,2) AND converges faster (early-stops ~40 epochs vs running the full 100 for MSE)
# - so LogisticLoss, not MSE, despite the earlier color-discretization ablation (Sec 4.3.2)
# using MSE as its baseline. Logistic loss pairs with the CIRCLE heatmap (a classification-
# style {-1,1} target), not GAUSS (a smooth regression target) - hm_filter follows suit.
# lr=1e-3, batch_size=16, up to 100 epochs (early stopping patience=15 already matches the
# paper's cmax=15), color discretization (cbest=2, i.e. 4 colors) + swapping plus
# geometric flip/rotation augmentation.
# seq_size=7: Table 3 of the IEEE paper (Pflugfelder & Auer, AVSS 2021) shows precision
# saturates around 0.96 at N>=7, clearly ahead of N=5's ~0.92 ceiling.
# nth_frame=2 (the paper's frame rate f=2): at either N, f=2 beats f=1 (nth_frame=1, the
# previous default - the worst setting tested) - not a fidelity nuance, a previously-
# missed parameter.
kth_train = {
    'dataset_type': KthDataSet,
    'train_data': 'data/kth_train',
    'eval_data': 'data/kth_val',
    'hm_filter': HeatMap.CIRCLE,
    'criterion': LogisticLoss(),
    'batch_size': 16,
    'seq_size': 7,
    'nth_frame': 2,
    'lr': 1e-3,
    'num_epochs': 100,
    # rand_zoom_out: KTH persons are 71-144px tall at 224px (median 109), NFO's are 45-64px
    # (median 54) - disjoint distributions. Feeding the trained net a 2x-upscaled NFO frame
    # lifts precision 0.52 -> 0.98, so this scale gap *is* the KTH->NFO gap (see
    # docs/training_failure_hypotheses.md). Zooming out during training covers NFO's range.
    'train_transforms': [rand_h_flip(), rand_v_flip(), rand_rot_90(), rand_zoom_out(0.4, 1.0),
                         reduce_colors(4), rand_color_swap()],
    # validate on the same distribution we train on, so the two losses are comparable and
    # early stopping is driven by a like-for-like signal (was []: raw 256-level frames)
    'eval_transforms': [rand_zoom_out(0.4, 1.0), reduce_colors(4)],
    # scale to whichever machine actually runs training (e.g. a remote GPU box)
    'num_workers': min(8, available_cpus()),
}

# Anisotropic-heatmap experiment (see /home/akovi/.claude/plans/sparkling-munching-valiant.md):
# identical to kth_train in every other respect, so the only difference between the two
# training runs is the target image + loss - an isolated comparison of whether the U-Net can
# learn a real-mask-moment-derived covariance target (gen_data/gen_kth_data/
# gen_anisotropic_heatmap.py), and whether doing so also affects plain localization accuracy.
# MSELoss, not LogisticLoss: the anisotropic target is a smooth regression target (a real
# covariance-shaped Gaussian, continuous values), not GAUSS/CIRCLE's classification-style
# {-1,1} target that LogisticLoss is paired with per the kth_train comment above - using
# LogisticLoss here would reproduce the exact GAUSS/LogisticLoss mismatch already documented
# as unsuitable.
kth_train_anisotropic = kth_train.copy()
kth_train_anisotropic.update({
    'hm_filter': HeatMap.ANISO,
    'criterion': torch.nn.MSELoss(),
})


def set_cfg(config_name: str):
    config.replace(config.copy(eval(config_name)))


def persist_cfg(file_path: str):
    ensure_dir(file_path)
    with open(file_path, 'wb') as output_file:
        dill.dump(config, output_file, dill.HIGHEST_PROTOCOL)


def load_cfg(file_path: str):
    with open(file_path, 'rb') as input_file:
        config.replace(dill.load(input_file))
