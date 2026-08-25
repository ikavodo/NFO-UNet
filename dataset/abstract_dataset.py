from abc import ABC, abstractmethod
from collections import namedtuple
from enum import IntEnum
from typing import List, Dict, Tuple


import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

from utils.bb_utils import parse_bbs
from utils.fs_utils import search_folders, search_files, flatten_list
from utils.transform_utils import resolve_transform, Transform

DatasetEntry = namedtuple('DatasetEntry', 'orig_file gauss_file circle_file bbs')


class HeatMap(IntEnum):
    GAUSS = 0
    CIRCLE = 1


class AbstractDataSet(ABC, Dataset):

    def __init__(self, root_dir: str, seq_length: int, nth_frame: int, exclude_roots: List[str],
                 transforms: List[Transform]):
        assert seq_length % 2 != 0
        self.margin = seq_length // 2
        self.nth_frame = nth_frame

        bbs_per_root, hms_per_root, circle_per_root, imgs_per_root = self.__search_files(root_dir, exclude_roots)
        self.entries, self.idx_mapping = self._construct_ds_entries(bbs_per_root, hms_per_root, circle_per_root,
                                                                    imgs_per_root)
        self.transform = resolve_transform(transforms)

    @staticmethod
    def __search_files(root_dir: str, exclude_roots: List[str]):
        roots = search_folders(root_dir, "_gt")
        roots = [b for b in roots if all(a not in b for a in exclude_roots)]
        # fetch files from file system
        imgs_per_root = [sorted(search_files(root, "_or", True), key=lambda x: x[1]) for root in roots]
        gauss_per_root = [sorted(search_files(root, "_gauss", True), key=lambda x: x[1]) for root in roots]
        circle_per_root = [sorted(search_files(root, "_circle", True), key=lambda x: x[1]) for root in roots]
        bb_files = flatten_list([search_files(root, "groundtruth.txt", True) for root in roots])
        bbs_per_root = [parse_bbs(gt_file.path) for gt_file in bb_files]

        return bbs_per_root, gauss_per_root, circle_per_root, imgs_per_root

    def __len__(self) -> int:
        return len(self.idx_mapping)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        idx = self.idx_mapping.get(idx)
        img = self.entries[idx].orig_file
        gauss = self.entries[idx].gauss_file
        circle = self.entries[idx].circle_file
        frames = [self.entries[idx + i * self.nth_frame].orig_file for i in range(-self.margin, self.margin + 1)]

        img = cv2.imread(img.path, 0)
        frames = np.stack([cv2.imread(frame.path, 0) for frame in frames], axis=2)
        hm = (np.stack([cv2.imread(gauss.path, 0), cv2.imread(circle.path, 0)], axis=2)
              if gauss is not None and circle is not None
              else np.zeros((*img.shape, 2), dtype=np.uint8))
        bbs = self.entries[idx].bbs

        frames, hm, bbs = self.transform(frames, hm, bbs)
        return {'img': img, 'frames': frames, 'hm': hm, 'bbs': bbs}

    @staticmethod
    def collate_fn(batch):
        img = list()
        frames = list()
        hm = list()
        bbs = list()

        for b in batch:
            img.append(b['img'])
            frames.append(b['frames'])
            hm.append(b['hm'])
            bbs.append(b['bbs'])

        frames = torch.stack(frames, dim=0)
        hm = torch.stack(hm, dim=0)
        return img, frames, hm, bbs

    @abstractmethod
    def _construct_ds_entries(self, bbs_per_root, hm_files_per_root, circle_per_root, orig_files_per_root) -> Tuple[
        List[DatasetEntry], Dict[int, int]]:
        pass
