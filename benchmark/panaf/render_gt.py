#!/usr/bin/env python3
"""Render PanAf500 ground truth (boxes, ape_id, species, behaviour) onto a video.

Stdlib only; drawing is done by ffmpeg's drawbox/drawtext filters, one pair per
detection, each enabled on exactly its frame. Deterministic: same input -> same output.

Usage:
    python3 tools/render_gt.py 1W3EJbBJOf            # id, file name or path all work
    python3 tools/render_gt.py PanAf500/videos/1W3EJbBJOf.mp4 -o renders/x.mp4

Overlay convention:
    box colour  = ape_id (fixed palette, consistent across frames of one video)
    box label   = "id<ape_id> <species> <behaviour>"
    top-left    = 1-indexed frame_id (matches the JSON) and number of GT boxes
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__import__('os').environ.get('PANAF_ROOT', '../PanAf20K/PanAf500')).resolve()   # run from the NFO-UNet root
# Tableau-10, distinct and colour-blind-friendly enough for <=10 apes per video.
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
FS = 16  # label font size (px)


def probe_size(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True).stdout
    w, h = map(int, out.strip().split(",")[:2])
    return w, h


def build_filters(ann, w, h):
    f = []
    for fr in ann["annotations"]:
        n = fr["frame_id"] - 1  # JSON is 1-indexed, ffmpeg's n is 0-indexed
        en = f"enable='eq(n,{n})'"
        dets = fr["detections"]
        f.append(f"drawtext=text='GT boxes {len(dets)}':x=8:y=30:fontsize={FS}:"
                 f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4:{en}")
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            # clip: GT boxes can extend slightly outside the frame
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
            bw, bh = max(1, round(x2 - x1)), max(1, round(y2 - y1))
            c = PALETTE[d["ape_id"] % len(PALETTE)]
            f.append(f"drawbox=x={round(x1)}:y={round(y1)}:w={bw}:h={bh}:"
                     f"color={c}:t=3:{en}")
            label = f"id{d['ape_id']} {d['species']} {d['behaviour']}"
            ly = y1 - FS - 8 if y1 >= FS + 8 else y1 + 4  # above box, else inside
            f.append(f"drawtext=text='{label}':x={round(x1)}:y={round(ly)}:"
                     f"fontsize={FS}:fontcolor=white:box=1:boxcolor={c}@0.8:"
                     f"boxborderw=3:{en}")
    # frame counter on every frame, numbered like the JSON (1-indexed)
    f.append(f"drawtext=text='frame_id %{{frame_num}}':start_number=1:x=8:y=8:"
             f"fontsize={FS}:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4")
    return ",\n".join(f)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="PanAf500 video id, file name, or path")
    p.add_argument("-o", "--out", help="output .mp4 (default: renders/<id>_gt.mp4)")
    a = p.parse_args()

    vid = Path(a.video).stem
    video = Path(a.video) if Path(a.video).is_file() else ROOT / "videos" / f"{vid}.mp4"
    jsons = list((ROOT / "annotations").glob(f"*/{vid}.json"))
    if not video.is_file() or len(jsons) != 1:
        raise SystemExit(f"video {video} exists={video.is_file()}, annotation files found: {jsons}")
    ann = json.loads(jsons[0].read_text())
    w, h = probe_size(video)
    out = Path(a.out) if a.out else Path("results/panaf") / f"{vid}_gt.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fs:
        fs.write(build_filters(ann, w, h))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
                    "-filter_script:v", fs.name, "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-an", str(out)], check=True)
    Path(fs.name).unlink()
    print(f"{out}  ({jsons[0].parent.name} split, {w}x{h}, "
          f"{len(ann['annotations'])} annotated frames)")


if __name__ == "__main__":
    main()
