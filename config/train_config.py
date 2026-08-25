import os

import dill

from config.config import AbstractConfig
from dataset.abstract_dataset import HeatMap
from dataset.kth_dataset import KthDataSet
from logistic_loss import LogisticLoss
from utils.fs_utils import ensure_dir
from utils.transform_utils import rand_h_flip, rand_v_flip, rand_rot_90, reduce_colors, rand_color_swap

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
# nth_frame=2 (the paper's frame rate f=2): Table 3 of the IEEE paper (Pflugfelder & Auer,
# AVSS 2021) shows precision for N=5 (our seq_size) jumps from 0.85 at f=1 (nth_frame=1,
# the previous default - the worst setting tested) to 0.92 at f=2 - not a fidelity nuance,
# a real, previously-missed parameter.
kth_train = {
    'dataset_type': KthDataSet,
    'train_data': 'data/kth_train',
    'eval_data': 'data/kth_val',
    'hm_filter': HeatMap.CIRCLE,
    'criterion': LogisticLoss(),
    'batch_size': 16,
    'seq_size': 5,
    'nth_frame': 2,
    'lr': 1e-3,
    'num_epochs': 100,
    'train_transforms': [rand_h_flip(), rand_v_flip(), rand_rot_90(), reduce_colors(4), rand_color_swap()],
    # scale to whichever machine actually runs training (e.g. a remote GPU box)
    'num_workers': min(8, os.cpu_count() or 1),
}


def set_cfg(config_name: str):
    config.replace(config.copy(eval(config_name)))


def persist_cfg(file_path: str):
    ensure_dir(file_path)
    with open(file_path, 'wb') as output_file:
        dill.dump(config, output_file, dill.HIGHEST_PROTOCOL)


def load_cfg(file_path: str):
    with open(file_path, 'rb') as input_file:
        config.replace(dill.load(input_file))
