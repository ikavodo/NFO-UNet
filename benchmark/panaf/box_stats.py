#!/usr/bin/env python3
"""Per-video GT box statistics for PanAf500 -> renders/box_stats.csv (stdlib only).

Size classes follow COCO (area in native pixels): S < 32^2, M 32^2..96^2, L > 96^2.
Occlusion cues computed from GT alone:
  gaps     : frames where an ape_id is missing *inside* its own lifetime (first..last frame),
             i.e. the annotator dropped the box mid-track -> candidate full-occlusion events.
  border   : fraction of boxes touching the frame edge (truncation, NOT occlusion; confounder).
  area_cv  : median over tracks of std/mean of box area -> a box that shrinks/grows while the
             ape stays put is a candidate partial occlusion (only if boxes are modal, see notes).
"""
import csv, json, statistics as st, subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__import__('os').environ.get('PANAF_ROOT', '../PanAf20K/PanAf500')).resolve()   # run from the NFO-UNet root
OUT = Path("results/panaf") / "box_stats.csv"
S, L = 32 ** 2, 96 ** 2


def size(video):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True, check=True).stdout
    return tuple(map(int, o.strip().split(",")[:2]))


rows = []
for jf in sorted((ROOT / "annotations").glob("*/*.json")):
    d = json.loads(jf.read_text())
    w, h = size(ROOT / "videos" / f"{d['video']}.mp4")
    areas, border, tracks = [], 0, defaultdict(dict)
    for fr in d["annotations"]:
        for det in fr["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            a = max(0, x2 - x1) * max(0, y2 - y1)
            areas.append(a)
            border += x1 <= 2 or y1 <= 2 or x2 >= w - 3 or y2 >= h - 3
            tracks[det["ape_id"]][fr["frame_id"]] = a
    if not areas:
        continue
    n = len(areas)
    gaps = sum((max(t) - min(t) + 1) - len(t) for t in tracks.values())
    cvs = [st.pstdev(t.values()) / st.mean(t.values()) for t in tracks.values() if len(t) > 1]
    med = st.median(areas)
    rows.append(dict(
        video=d["video"], split=jf.parent.name, w=w, h=h, n_apes=len(tracks), n_boxes=n,
        frac_S=round(sum(a < S for a in areas) / n, 3),
        frac_M=round(sum(S <= a <= L for a in areas) / n, 3),
        frac_L=round(sum(a > L for a in areas) / n, 3),
        median_area=round(med), median_rel_height=round(med ** .5 / h, 3),
        gap_frames=gaps, frac_border=round(border / n, 3),
        area_cv=round(st.median(cvs), 3) if cvs else ""))

rows.sort(key=lambda r: -(r["frac_S"] + r["frac_M"]))
OUT.parent.mkdir(exist_ok=True)
with OUT.open("w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0]); wr.writeheader(); wr.writerows(rows)
print(OUT, len(rows), "videos")
