import os
import zipfile

import cv2
from tqdm import tqdm

KTH_DIR = 'data/kth'
LABELS_ZIP = 'kth_labels.zip'
OUT_DIR = 'data/kth_staged'
ACTIONS = ('walking', 'jogging', 'running')


def extract_frames(avi_path: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(avi_path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f'{i:05d}.jpg'), frame)
        i += 1
    return i


def main():
    with zipfile.ZipFile(LABELS_ZIP) as z:
        for action in ACTIONS:
            action_dir = os.path.join(KTH_DIR, action)
            avi_files = sorted(f for f in os.listdir(action_dir) if f.endswith('.avi'))
            for fname in tqdm(avi_files, desc=action):
                seq_name = fname[:-len('.avi')]
                label_bytes = z.read(f'kth/{seq_name}/groundtruth.txt')
                n_labels = len(label_bytes.decode().splitlines())

                seq_out = os.path.join(OUT_DIR, seq_name)
                n_frames = extract_frames(os.path.join(action_dir, fname), seq_out)
                assert n_frames == n_labels, f'{seq_name}: {n_frames} frames vs {n_labels} labels'

                with open(os.path.join(seq_out, 'groundtruth.txt'), 'wb') as out_f:
                    out_f.write(label_bytes)


if __name__ == '__main__':
    main()
