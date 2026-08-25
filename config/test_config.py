import dill

from config.config import AbstractConfig, available_cpus
from dataset.abstract_dataset import HeatMap
from dataset.kth_dataset import KthDataSet
from dataset.testing_dataset import TestingDataSet
from eval.threshold_eval import ThresholdEval

base_config = AbstractConfig({
    # dataset (KthDataSet | MnistDataSet | TestingDataSet)
    'dataset_type': None,
    # path to test data root
    'test_data': None,
    # exclude certain folders from the dataset inside of the train and eval data folder
    'exclude_dirs': [],
    # search string specifying which heatmap type to use (either use _gauss or _circle)
    'hm_filter': None,
    # batch size for training and evaluation
    'batch_size': None,
    # number of frames in one sequence
    'seq_size': None,
    # every nth frame will be included in the sequences
    'nth_frame': 1,
    # number of workers for loading data samples
    'num_workers': 0,
    # shuffle the dataset
    'shuffle': True,
    # pin memory of the data loader
    'pin_memory': True,
    # transforms to use for testing (order matters)
    'test_transforms': [],
    # evaluation to use
    'eval_method': None
})
config = base_config.copy()

# evaluate on the real NFO footage (seq1-4). eval_method matches the publication exactly:
# ThresholdEval (Otsu-threshold + contour based, not a plain argmax) with a distance error
# tolerance of 0.1 (10% of the heatmap side length).
nfo_test = {
    'dataset_type': TestingDataSet,
    'test_data': 'data/nfo_processed',
    'hm_filter': HeatMap.CIRCLE,
    'batch_size': 16,
    'seq_size': 7,
    'nth_frame': 2,  # match training's frame rate f=2 (see config/train_config.py)
    'eval_method': ThresholdEval(max_dist_error=0.1),
    'num_workers': min(8, available_cpus()),
}

# diagnostic config: box/F1 evaluation on KTH's own validation set (same domain/resolution
# as training) - isolates "does the eval/matching code work at all" from "did the model
# generalize to NFO", since KthDataSet already carries real ground-truth boxes too.
kth_val_test = {
    'dataset_type': KthDataSet,
    'test_data': 'data/kth_val',
    'hm_filter': HeatMap.CIRCLE,
    'batch_size': 16,
    'seq_size': 7,
    'nth_frame': 2,  # match training's frame rate f=2 (see config/train_config.py)
    'eval_method': ThresholdEval(max_dist_error=0.1),
    'num_workers': min(8, available_cpus()),
}


def set_cfg(config_name: str):
    config.replace(config.copy(eval(config_name)))


def load_cfg(file_path: str):
    with open(file_path, 'rb') as input_file:
        config.replace(dill.load(input_file))
