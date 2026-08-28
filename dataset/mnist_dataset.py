from typing import Dict, Tuple, List

from dataset.abstract_dataset import AbstractDataSet, DatasetEntry


class MnistDataSet(AbstractDataSet):

    def __init__(self, root_dir: str, seq_length: int, nth_frame: int,
                 exclude_roots: List[str], transforms: List):
        super().__init__(root_dir=root_dir,
                         seq_length=seq_length,
                         nth_frame=nth_frame,
                         exclude_roots=exclude_roots,
                         transforms=transforms)

    def _construct_ds_entries(self, bbs_per_root, gauss_files_per_root, circle_files_per_root, aniso_files_per_root,
                              orig_files_per_root) -> Tuple[List[DatasetEntry], Dict[int, int]]:
        ds_entries, idx_file_mapping, global_idx = [], {}, 0

        for orig_files, gauss_files, circle_files, bbs in zip(orig_files_per_root, gauss_files_per_root, circle_files_per_root, bbs_per_root):
            for root_idx, (frame_of, frame_gauss, frame_circle, frame_bbs) in enumerate(zip(orig_files, gauss_files, circle_files, bbs.values())):
                # MNIST has no anisotropic heatmap generation - not part of this experiment
                ds_entries.append(DatasetEntry(orig_file=frame_of, gauss_file=frame_gauss, circle_file=frame_circle,
                                              aniso_file=None, bbs=frame_bbs))
                if self.is_valid(len(orig_files), root_idx):
                    idx_file_mapping[len(idx_file_mapping)] = global_idx
                global_idx += 1

        return ds_entries, idx_file_mapping

    def is_valid(self, seq_len, root_idx):
        return self.margin * self.nth_frame <= root_idx < seq_len - self.margin * self.nth_frame and root_idx == seq_len // 2
