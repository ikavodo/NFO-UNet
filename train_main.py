import argparse
import logging
import os

# must be set before the CUDA allocator initializes (i.e. before any torch.cuda call) -
# reduces GPU memory fragmentation, letting larger batch sizes fit without OOM
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from config import train_config as conf
from early_stopping import EarlyStopping
from network.unet import UNet
from utils.log_utils import init_logging
from utils.vis_utils import visualize_train_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, required=False, help='The config to use for the program execution')
    parser.add_argument('--save-dir', required=True, help='Directory where output will be saved')
    known, unknown = parser.parse_known_args()
    return known


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # inputs are fixed-size (224x224, fixed batch_size) -> let cudnn pick the fastest conv algorithm
        torch.backends.cudnn.benchmark = True
    logging.info("Device is {}".format(device))
    return device


def create_data_loader(is_train: bool, files_dir: str):
    logging.info(f'Prepare dataloader for {"training" if is_train else "evaluation"}')
    dataset = c.dataset_type(files_dir,
                             c.seq_size,
                             nth_frame=c.nth_frame,
                             exclude_roots=c.exclude_dirs,
                             transforms=c.train_transforms if is_train else c.eval_transforms)
    return DataLoader(dataset,
                      batch_size=c.batch_size,
                      shuffle=is_train and c.shuffle,
                      num_workers=c.num_workers,
                      pin_memory=c.pin_memory,
                      collate_fn=c.dataset_type.collate_fn)


def main():
    logging.info('Starting application')
    early_stopping = EarlyStopping(c.early_stopping_patience)
    train_data_loader = create_data_loader(True, c.train_data)
    eval_data_loader = create_data_loader(False, c.eval_data)
    device = get_device()
    net = UNet(n_channels=c.seq_size, n_classes=1, bilinear=False)
    net.to(device)

    criterion = c.criterion.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=c.lr)
    # GradScaler is required for fp16 training (unlike inference, which only needs autocast) -
    # it rescales gradients to avoid them underflowing to zero in fp16. No-op when disabled.
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(c.num_epochs):
        logging.info(f'Starting epoch {epoch + 1}')
        train(criterion, device, epoch, net, optimizer, train_data_loader, scaler)
        validation_loss = validate(criterion, device, epoch, net, eval_data_loader, early_stopping.best_validation_loss)

        if early_stopping(validation_loss) and c.enable_early_stopping:
            logging.info(f'Early stopping, no improvement for last {c.early_stopping_patience} epochs')
            break

        if early_stopping.save_net():
            logging.info(f'Saving network to {args.save_dir}')
            net.save_checkpoint(args.save_dir)

        logging.info(f'Finished epoch {epoch + 1}')
    logging.info(f"Application finished. Best validation loss was {early_stopping.best_validation_loss}")


def train(criterion, device, epoch, net, opt, dl, scaler):
    net.train()
    xma, ma, = [], []

    for i, batch in enumerate(dl):
        img, frames, hm = batch[0], batch[1].to(device), batch[2][:, int(c.hm_filter), :, :].unsqueeze(1).to(device)
        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            out = net(frames)
            loss = criterion(out, hm)
        scaler.scale(loss).backward()
        # unscale before clipping (standard AMP pattern) - without this, a criterion whose
        # gradient scale differs a lot from what the tuned lr/precision expects (e.g. MSELoss's
        # squared, unbounded error vs. LogisticLoss's bounded log(1+exp(...)) the existing
        # kth_train hyperparameters were actually tuned against) can produce a large early
        # update, overflow fp16's narrow dynamic range, and NaN out the run with no recovery -
        # confirmed happening on kth_train_anisotropic's MSELoss run, never seen on kth_train's
        # LogisticLoss run
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        xma.append(loss.item())
        ma.append(loss.item())
        if c.visualize:
            visualize_train_output(img, batch[1], out, hm)
        if (i + 1) % c.print_ma == 0:
            logging.info(f'[{epoch + 1}/{c.num_epochs} - {i + 1}/{len(dl)}] Last {c.print_ma} mean loss: {round(sum(xma) / len(xma), 8)}')
            xma = []
    logging.info(f"[{epoch + 1}/{c.num_epochs}] Training mean loss was {round(sum(ma)/len(ma), 8)}")


def validate(criterion, device, epoch, net, dl, best_loss):
    with torch.no_grad():
        net.eval()
        ma = []

        for i, batch in enumerate(dl):
            img, frames, hm, bbs_gt = batch[0], batch[1].to(device), batch[2][:, int(c.hm_filter), :, :].unsqueeze(1).to(device), batch[3]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                out = net(frames)
                loss = criterion(out, hm)
            ma.append(loss.item())
            if c.visualize:
                visualize_train_output(img, batch[1], out, hm)

    logging.info(f'[{epoch + 1}/{c.num_epochs}] Validation mean loss was {round(sum(ma)/len(ma), 8)}. Best mean loss so far was {round(best_loss, 8)}')
    return sum(ma)/len(ma)


if __name__ == '__main__':
    args = parse_args()
    init_logging(os.path.join(args.save_dir, 'train.log'))
    if args.config is not None:
        conf.set_cfg(args.config)
    conf.persist_cfg(os.path.join(args.save_dir, 'train_cfg.pkl'))
    c = conf.config
    logging.info(c)
    main()
