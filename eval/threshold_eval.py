import numpy as np
import cv2

from eval.abstract_eval import AbstractEval
from utils.bb_utils import BoundingBox


class ThresholdEval(AbstractEval):

    def __init__(self, max_dist_error: float, init_thresh: float = None):
        super().__init__(max_dist_error, init_thresh)

    def extract_centers(self, hm):
        norm = self.normalize(hm)
        norm = cv2.medianBlur(norm, 5)
        bbs = []

        if np.amax(norm) > 0:
            _, thresh_hm = cv2.threshold(norm, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh_hm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                hm_copy = np.copy(norm)
                cv2.fillPoly(hm_copy, [c], 0)
                diff = np.where(norm != hm_copy, norm, 0)
                max_index = np.argmax(diff)
                bbs.append(BoundingBox(max_index % 224, max_index // 224, 0, 0))

        # calculate center using bounding box
        centers = [bb.center() for bb in bbs]
        centers_norm = [bb.scale((1 / 224, 1 / 224)).center() for bb in bbs]
        return centers, centers_norm

    def normalize(self, hm):
        norm = np.copy(hm)
        norm -= np.amin(hm)
        if np.amax(norm) <= 0:
            return norm.astype(np.uint8)
        norm /= np.amax(norm)
        norm *= 255
        return norm.astype(np.uint8)
