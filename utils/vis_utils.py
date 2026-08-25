import numpy as np
import cv2


def visualize_train_output(orig_batch, frames_batch, out_batch, hm_batch):
    # read values
    out_batch = out_batch.cpu().detach().numpy()
    hm_batch = hm_batch.cpu().detach().numpy()
    frames_batch = frames_batch.numpy()
    assert len(orig_batch) == out_batch.shape[0] == hm_batch.shape[0]

    for i in range(len(orig_batch)):
        # normalize
        orig, frames, out, hm = orig_batch[i].astype(np.float32) / 255, \
                                frames_batch[i][frames_batch.shape[1] // 2], \
                                out_batch[i], \
                                hm_batch[i]

        out = normalize(out)
        out, hm = cwh_to_hwc(out), cwh_to_hwc(hm)
        padded = pad_imgs([orig, frames, out, hm])
        cv2.imshow('Training', cv2.hconcat(padded))
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_eval_output(orig_batch, frames_batch, out_batch_unprocessed, out_batch_processed, hm_batch):
    # read values
    hm_batch = hm_batch.cpu().detach().numpy()
    frames_batch = frames_batch.numpy()

    for i in range(len(orig_batch)):
        merged = merge_eval_imgs(frames_batch, hm_batch, i, orig_batch, out_batch_processed, out_batch_unprocessed)
        cv2.imshow('input - output - groundtruth', merged)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def visualize_and_compare_eval_output(orig_batch, frames_batch, out_batch_unprocessed1, out_batch_processed1,
                                      gt_1_batch, out_batch_unprocessed2, out_batch_processed2, gt_2_batch):
    # read values
    gt_1_batch = gt_1_batch.cpu().detach().numpy()
    gt_2_batch = gt_2_batch.cpu().detach().numpy()
    frames_batch = frames_batch.numpy()

    for i in range(len(orig_batch)):
        merged1 = merge_eval_imgs(frames_batch, gt_1_batch, i, orig_batch, out_batch_processed1, out_batch_unprocessed1)
        merged2 = merge_eval_imgs(frames_batch, gt_2_batch, i, orig_batch, out_batch_processed2, out_batch_unprocessed2)
        cv2.imshow('Evaluation', cv2.vconcat([merged1, merged2]))
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def merge_eval_imgs(frames_batch, hm_batch, i, orig_batch, out_batch_processed, out_batch_unprocessed):
    orig, frames, out_unprocessed, out_processed, hm = orig_batch[i].astype(np.float32) / 255, \
                                                       frames_batch[i][frames_batch.shape[1] // 2], \
                                                       out_batch_unprocessed[i].astype(np.float32), \
                                                       out_batch_processed[i].astype(np.float32), \
                                                       hm_batch[i]
    out_unprocessed, out_processed = normalize(out_unprocessed), normalize(out_processed)
    out_unprocessed, hm = cwh_to_hwc(out_unprocessed), cwh_to_hwc(hm)
    padded = pad_imgs([orig, frames, out_unprocessed, out_processed, hm])
    return cv2.hconcat(padded)


def normalize(img):
    if img.min() != img.max():
        img -= img.min()
        img /= (img.max() - img.min())
    return img


def cwh_to_hwc(img):
    return np.transpose(img, (1, 2, 0))


def pad_imgs(imgs):
    imgs = [cv2.copyMakeBorder(img, 10, 10, 5, 5, cv2.BORDER_CONSTANT, value=[1, 1, 1]) for img in imgs]
    return [img if len(img.shape) == 3 else cv2.merge([img, img, img]) for img in imgs]
