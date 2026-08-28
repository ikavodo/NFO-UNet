from typing import Tuple, Dict, List

from dataset.abstract_dataset import AbstractDataSet, DatasetEntry


class TestingDataSet(AbstractDataSet):

    def __init__(self, root_dir: str, seq_length: int, nth_frame: int, exclude_roots: List[str], transforms: List):
        super().__init__(root_dir=root_dir,
                         seq_length=seq_length,
                         nth_frame=nth_frame,
                         exclude_roots=exclude_roots,
                         transforms=transforms)

    def _construct_ds_entries(self, bbs_per_root, gauss_files_per_root, circle_files_per_root, aniso_files_per_root,
                              orig_files_per_root) -> Tuple[List[DatasetEntry], Dict[int, int]]:
        ds_entries, idx_file_mapping, global_idx = [], {}, 0
        for orig_files, bbs in zip(orig_files_per_root, bbs_per_root):
            for root_idx, frame_of in enumerate(orig_files):
                ds_entries.append(DatasetEntry(orig_file=frame_of,
                                               gauss_file=None,
                                               circle_file=None,
                                               aniso_file=None,
                                               bbs=bbs.get(root_idx, [])))

                # paper sec. 4.2: the measures "assume presence of a person in each test image";
                # rejecting a localisation hypothesis on empty frames is explicitly out of scope.
                # MaxEval always emits one point, so an unannotated frame is a guaranteed fp -
                # skip those windows, mirroring KthDataSet's `and gauss_file is not None`.
                if self.margin * self.nth_frame <= root_idx < len(orig_files) - self.margin * self.nth_frame \
                        and bbs.get(root_idx):
                    idx_file_mapping[len(idx_file_mapping)] = global_idx
                global_idx += 1
        return ds_entries, idx_file_mapping
