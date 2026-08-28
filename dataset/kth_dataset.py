from typing import Tuple, Dict, List

from dataset.abstract_dataset import AbstractDataSet, DatasetEntry


class KthDataSet(AbstractDataSet):

    def __init__(self, root_dir: str, seq_length: int, nth_frame: int, exclude_roots: List[str], transforms: List):
        super().__init__(root_dir=root_dir,
                         seq_length=seq_length,
                         nth_frame=nth_frame,
                         exclude_roots=exclude_roots,
                         transforms=transforms)

    @staticmethod
    def get_file_index(file_name: str) -> int:
        return int(''.join(c for c in file_name if c.isdigit()))

    def _construct_ds_entries(self, bbs_per_root, gauss_files_per_root, circle_files_per_root, aniso_files_per_root,
                              orig_files_per_root) -> Tuple[List[DatasetEntry], Dict[int, int]]:
        ds_entries, idx_file_mapping, global_idx = [], {}, 0
        for orig_files, gauss_files, circle_files, aniso_files, bbs in zip(
                orig_files_per_root, gauss_files_per_root, circle_files_per_root, aniso_files_per_root, bbs_per_root):
            hm_dict = {}
            for gauss_file, circle_file in zip(gauss_files, circle_files):
                assert self.get_file_index(gauss_file.name) == self.get_file_index(circle_file.name)
                hm_dict[self.get_file_index(gauss_file.name)] = (gauss_file, circle_file)
            # aniso is a strict subset of gauss/circle (only frames with a valid mask/moments
            # get one - gen_anisotropic_heatmap.py skips the rest), so it's looked up
            # independently by file index rather than zipped 1:1 like gauss/circle above
            aniso_dict = {self.get_file_index(f.name): f for f in aniso_files}

            for root_idx, (frame_of, frame_bbs) in enumerate(zip(orig_files, bbs.values())):
                key = self.get_file_index(frame_of.name)
                gauss_file, circle_file = hm_dict.get(key) if key in hm_dict else (None, None)
                aniso_file = aniso_dict.get(key)
                ds_entries.append(DatasetEntry(orig_file=frame_of, gauss_file=gauss_file, circle_file=circle_file,
                                              aniso_file=aniso_file, bbs=frame_bbs))

                # NOTE: indexability still only requires gauss/circle (unchanged from before
                # aniso existed) so the existing circle-based kth_train config's training set
                # is unaffected - aniso is a strict subset, and a config that trains on ANISO
                # will see some indexable frames fall back to an all-zero aniso target
                # (__getitem__'s per-channel zero-fallback) where no mask/moments exist. Not
                # corrected here; a training/eval script for the anisotropic config should
                # mask those frames out of its loss/metrics rather than treat zero as a real
                # target.
                if self.margin * self.nth_frame <= root_idx < len(orig_files) - self.margin * self.nth_frame and gauss_file is not None:
                    idx_file_mapping[len(idx_file_mapping)] = global_idx
                global_idx += 1
        return ds_entries, idx_file_mapping

