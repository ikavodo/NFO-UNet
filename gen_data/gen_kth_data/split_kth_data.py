import os
import re

# standard KTH benchmark person split (Schuldt et al.); test persons are unused here
# since this repo tests on the NFO dataset instead
TRAIN_PERSONS = {11, 12, 13, 14, 15, 16, 17, 18}
VAL_PERSONS = {19, 20, 21, 23, 24, 25, 1, 4}

IN_DIR = 'data/kth_processed'
TRAIN_OUT = 'data/kth_train'
VAL_OUT = 'data/kth_val'


def person_id(seq_gt_name: str) -> int:
    return int(re.match(r'person(\d+)_', seq_gt_name).group(1))


def main():
    os.makedirs(TRAIN_OUT, exist_ok=True)
    os.makedirs(VAL_OUT, exist_ok=True)
    for seq_gt in sorted(os.listdir(IN_DIR)):
        if not seq_gt.endswith('_gt'):
            continue
        pid = person_id(seq_gt)
        if pid in TRAIN_PERSONS:
            out_dir = TRAIN_OUT
        elif pid in VAL_PERSONS:
            out_dir = VAL_OUT
        else:
            continue
        link = os.path.join(out_dir, seq_gt)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(IN_DIR, seq_gt)), link)


if __name__ == '__main__':
    main()
