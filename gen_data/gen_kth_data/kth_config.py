from config.config import AbstractConfig

base_config = AbstractConfig({
    # directory containing the sequence folders, each of them containing the seq images and labels
    'in_dir': 'data/kth_staged',
    # image size of sequences to be generated
    "img_size": 224,
    # if None, bounding boxes will not be averaged, else these are the weights for the mean calculation
    "bb_mean_weights": [1.5, 3, 5, 3, 1.5],
    # radius of circles in heatmap
    'hm_circle_radius': 0.07,
    # root directory which will hold all generated sequences
    "out_dir": 'data/kth_processed',
    # number of generated occlusion files
    'num_occ_samples': 1000,
    # number of iterations specifying how many times the occlusion should be growing and shrinking
    'occ_iterations': 5,
    # probability of a pixel being considered as occlusion (before growing and shrinking)
    'occ_init_prob': 0.08,
    # probability of a noisy pixel to grow within one iteration
    'occ_growing_prob': 0.3,
    # probability of a noisy pixel to shrink within one iteration
    'occ_shrink_prob': 0.3,
})

config = base_config.copy()


def set_cfg(config_name: str):
    global config
    config.replace(eval(config_name))
