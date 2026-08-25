from abc import abstractmethod, ABC

import numpy as np
import cv2


class AbstractEval(ABC):

    def __init__(self, max_dist_error: float, init_thresh: float):
        self.max_dist_error = max_dist_error
        self.init_thresh = init_thresh

    def retrieve_centers(self, batched_hms: np.ndarray):
        s = batched_hms.shape
        batched_centers, batched_out_hms = [], np.zeros((s[0], s[2], s[3], 3), dtype=np.uint8)
        # normalizing
        for i in range(s[0]):
            hm = self._preprocess(batched_hms, i)

            centers, centers_norm = self.extract_centers(hm)
            batched_centers.append(centers_norm)

            # draw centers found
            hm = AbstractEval._draw_centers_on_hm(centers, hm)
            batched_out_hms[i, ...] = hm

        return batched_centers, batched_out_hms

    def _preprocess(self, batched_hms, i):
        hm = batched_hms[i, 0, ...] * 255
        if self.init_thresh:
            hm[hm < self.init_thresh * 255] = 0
        return hm

    @abstractmethod
    def extract_centers(self, hm):
        pass

    @staticmethod
    def _draw_centers_on_hm(centers, hm):
        hm = np.clip(hm, 0, 255).astype(np.uint8)
        hm = cv2.merge([hm, hm, hm])
        l = 20
        for center in centers:
            c = (int(center[0]), int(center[1]))
            cv2.line(hm, (c[0] - l, c[1] - l), (c[0] + l, c[1] + l), (0, 0, 255), 5)
            cv2.line(hm, (c[0] - l, c[1] + l), (c[0] + l, c[1] - l), (0, 0, 255), 5)
        return hm.astype(np.uint8)

    def calculate_eval_stats(self, batched_gt_bbs, batched_pr_centers):
        tp, fp, fn = 0, 0, 0
        for gt_bbs, pr_centers in zip(batched_gt_bbs, batched_pr_centers):
            # check tp and fn
            for gt_bb in gt_bbs:
                fn += 1
                for pr_center in pr_centers:
                    dist_sq = gt_bb.dist_squared(pr_center)
                    if dist_sq < self.max_dist_error ** 2:
                        tp += 1
                        fn -= 1
                        break

            # check fp
            for pr_center in pr_centers:
                fp += 1
                for gt_bb in gt_bbs:
                    dist_sq = gt_bb.dist_squared(pr_center)
                    if dist_sq < self.max_dist_error ** 2:
                        fp -= 1
                        break

        return tp, fp, fn
