"""Build a version of a clip where the video FREEZES for longer than the tracker's window, to
demonstrate that a window of duplicate frames yields no box at all.

    python -m tracking.eval.make_frozen_clip --video data/ido_walk.mkv --at 110 330

Why there should be no box, and it is not a special case in the code: MOG2 absorbs a static
frame into its background model at its learning rate, so duplicate frames stop producing
foreground. Measured on ido_walk with bg_frames=30, detections per frame across the start of a
freeze go 3, 3, 1, 0, 0, 0, ... - absorbed in THREE frames. With no detections anywhere in the
13-frame ring buffer, track_blobs finds no track of min_track_length=3, score_and_fit returns
None, and the result carries x=None and box=None. Nothing thresholds or refuses; the estimate
simply does not exist, because every parameter of this tracker is defined on motion.

The boxes that do survive inside a freeze are exactly the emissions whose window still reaches
back into real motion - measured, emissions for frames 110, 111 and 112 of a freeze starting at
110, and no others, at any freeze length. That is a 3-frame tail, i.e. SPAN/2 rounded, and it is
the honest edge of the demonstration rather than a flaw.
"""
import argparse
import os

import cv2
import numpy as np

from tracking.stream.stream import BUFFER, frames_from_source


def freeze(frames, at, length):
    """Insert `length` copies of frames[at] immediately after it. Returns the new list and the
    (start, end) index range of the frozen block in the OUTPUT sequence."""
    return frames[:at + 1] + [frames[at]] * (length - 1) + frames[at + 1:], (at, at + length - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', default='data/ido_walk.mkv')
    p.add_argument('--out', default=None, help='default: <video stem>_frozen.mkv beside it')
    p.add_argument('--at', type=int, nargs='+', default=[110, 330],
                   help='frame index to freeze on, one per walk')
    p.add_argument('--length', type=int, default=None,
                   help=f'frozen block length; default 2*BUFFER = {2 * BUFFER}, so at least one '
                        f'emitted window is entirely duplicates')
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--fps', type=float, default=24.0)
    a = p.parse_args()
    length = a.length or 2 * BUFFER

    frames = list(frames_from_source(a.video, a.scale))
    ranges = []
    for at in sorted(a.at, reverse=True):        # descending, so earlier indices stay valid
        assert 0 <= at < len(frames), f"--at {at} outside 0..{len(frames) - 1}"
        frames, r = freeze(frames, at, length)
        ranges.append(r)
    # re-derive the ranges in the final sequence (each earlier freeze shifts the later ones)
    shift, final = 0, []
    for at in sorted(a.at):
        final.append((at + shift, at + shift + length - 1))
        shift += length - 1

    out = a.out or os.path.splitext(a.video)[0] + '_frozen.mkv'
    h, w = frames[0].shape
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (w, h))
    assert writer.isOpened(), f"cannot open {out} for writing"
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    writer.release()
    print(f"wrote {out}: {len(frames)} frames {w}x{h} @ {a.fps:g}fps "
          f"({len(frames) - shift} original + {shift} duplicated)")
    print(f"frozen blocks in the OUTPUT sequence (pass to --present): "
          + " ".join(f"{lo}:{hi}" for lo, hi in final))
    print(f"each block is {length} frames, against a {BUFFER}-frame ring buffer, so at least "
          f"{length - BUFFER + 1} emitted windows consist only of duplicates")


if __name__ == '__main__':
    main()
