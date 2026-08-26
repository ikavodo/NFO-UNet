import argparse
import logging
import os

# must be set before the CUDA allocator initializes (i.e. before any torch.cuda call) -
# reduces GPU memory fragmentation, letting larger batch sizes fit without OOM
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
from torch.utils.data.dataloader import DataLoader

from config import test_config as conf
from network.unet import UNet
from utils.log_utils import init_logging
from utils.vis_utils import visualize_eval_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, required=False, help='The config to use for the program execution')
    parser.add_argument('--load-dir', required=True, help='Directory where input will be loaded')
    known, unknown = parser.parse_known_args()
    return known


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    # logging.info("Device is {}".format(device))
    return device


def create_data_loader(files_dir: str):
    # logging.info('Prepare dataloader')
    dataset = c.dataset_type(files_dir,
                             c.seq_size,
                             nth_frame=c.nth_frame,
                             exclude_roots=c.exclude_dirs,
                             transforms=c.test_transforms)
    return DataLoader(dataset,
                      batch_size=c.batch_size,
                      shuffle=c.shuffle,
                      num_workers=c.num_workers,
                      pin_memory=c.pin_memory,
                      collate_fn=c.dataset_type.collate_fn)


def main():
    # logging.info('Starting application')
    eval_data_loader = create_data_loader(c.test_data)
    device = get_device()
    net = UNet(n_channels=c.seq_size, n_classes=1, bilinear=False)

    # logging.info("Loading model from %s", args.load_dir)
    net.load_checkpoint(args.load_dir)
    net.to(device)

    evaluate(device, net, eval_data_loader, c.criterion)


def evaluate(device, net, dl, criterion):
    with torch.no_grad():
        net.eval()
        tp, fp, fn = 0, 0, 0
        ma = []
        for i, batch in enumerate(dl):
            img, frames, hm, bbs_gt = batch[0], batch[1].to(device), batch[2][:, int(c.hm_filter), :, :].unsqueeze(
                1).to(device), batch[3]
            # fp16 autocast only for the forward pass (inference-only path, no grad scaler
            # needed since there's no backward pass here); no-op on CPU
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                out = net(frames)

            # calculate stats for mean loss
            loss = criterion(out, hm).item()
            if loss < float("inf"):
                ma.append(loss)

            # calculate stats for evaluation (f1, fp, ...)
            out = out.cpu().detach().numpy()
            center_pr, out_processed = c.eval_method.retrieve_centers(out)
            _tp, _fp, _fn = c.eval_method.calculate_eval_stats(bbs_gt, center_pr)
            tp, fp, fn = tp + _tp, fp + _fp, fn + _fn
            if c.visualize:
                visualize_eval_output(img, batch[1], out, out_processed, hm)

    f1 = tp / (tp + 0.5 * (fp + fn))
    # the paper reports Prec. = tp/(tp+fp) and the raw tp count (sec. 4.2, tables 2/3), not F1
    prec = tp / (tp + fp) if tp + fp else 0.0
    logging.info(f'Prec. was {round(prec, 6)} (paper metric), F1 score was {round(f1, 6)}')
    logging.info(f'tp: {tp}, fp: {fp}, fn: {fn}')
    # logging.info(f'Mean loss was {round(sum(ma) / len(ma), 8)}')


if __name__ == '__main__':
    args = parse_args()
    init_logging(os.path.join(args.load_dir, 'test.log'))
    conf.load_cfg(os.path.join(args.load_dir, 'train_cfg.pkl'))
    if args.config is not None:
        conf.set_cfg(args.config)
    c = conf.config
    logging.info(c)
    main()

