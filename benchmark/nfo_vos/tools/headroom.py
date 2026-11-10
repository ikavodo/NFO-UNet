"""Headroom checks before porting the method to a new dataset (2026-11-10).

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tools.headroom cdnet b0
    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tools.headroom mvot b0|t1
    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tools.headroom mvot-report

cdnet: SAM2.1 video (B0) on object passes (entry to exit, GT components linked frame to frame; NFO-VOS protocol),
  prompted at entry, 1/3 and 2/3 of each pass, run to the pass end; videos in VIDEOS. t0 prompt = tight box of the
  object's GT motion pixels (box only). Scored per clip, pooled over frames 1..end: precision, recall, J vs CDnet GT (255 = motion; 170 unknown and 85
  outside-ROI ignored; 50 shadow = background). Question: is recall < precision as on NFO (headroom), or ≥ 0.95?
mvot: B0 or T1 (SAMURAI) on 30 random 'similar object' videos (regular subset, seed 0) and 30 matched videos without
  the label (same lighting / tilt / view). t0 prompt = GT box of the first annotated frame. Success AUC (LaSOT: 21
  IoU thresholds 0..1, IoU of the mask's tight box vs GT; empty mask = 0). Failure anatomy (design doc H0b): runs of
  IoU < 0.1 for >= 5 frames, classified by the wrong mask's velocity relative to the target's.
Deviation from the NFO protocol: box-only t0 prompt (no point), as in SAMURAI / SatSAM2. On the laptop both B0 and
T1 come from the SAMURAI repo's sam2 fork (B0 = its sam2.1 config, samurai_mode off).
Outputs: results/benchmark/{cdnet,mvot}/headroom/*.csv, montages in images/benchmark/{cdnet,mvot}/headroom_*.jpg.
"""
import csv, glob, json, os, random, shutil, sys, tempfile, time
import cv2
import numpy as np

from benchmark.nfo_vos.baselines.run_sam2_video import build_predictor
import torch

CDNET = os.environ.get('CDNET_ROOT', os.path.expanduser('~/PycharmProjects/CDnet_2014/dataset'))   # cluster: /storage/user/akid/datasets/CDnet_2014/dataset
MVOT = os.environ.get('MVOT_ROOT', os.path.expanduser('~/PycharmProjects/MVOT'))                      # cluster: /storage/user/akid/datasets/MVOT
WIN = 40
K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))


def run_clip(pred, frame_paths, box_xyxy):
    """SAM2 video on a list of jpgs (symlinked as 0.jpg ..), box prompt at frame 0; returns bool masks [T, H, W]."""
    stage = tempfile.mkdtemp(prefix='headroom_')
    try:
        for i, p in enumerate(frame_paths):
            os.symlink(os.path.abspath(p), os.path.join(stage, f'{i}.jpg'))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            st = pred.init_state(stage, offload_video_to_cpu=True, offload_state_to_cpu=True)
            pred.add_new_points_or_box(st, frame_idx=0, obj_id=0, box=np.array(box_xyxy, np.float32))
            m = np.zeros((len(frame_paths), st['video_height'], st['video_width']), bool)
            for fi, _, lg in pred.propagate_in_video(st):
                m[fi] = (lg[0, 0] > 0).cpu().numpy()
        return m
    finally:
        shutil.rmtree(stage)


def tight(m):
    ys, xs = np.nonzero(m)
    return None if len(xs) == 0 else (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def iou(a, b):
    if a is None or b is None:
        return 0.0
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0.0


# ------------------------------------------------------------------ CDnet
VIDEOS = ('dynamicBackground/fall', 'badWeather/snowFall', 'dynamicBackground/fountain01', 'dynamicBackground/fountain02',
          'dynamicBackground/overpass', 'badWeather/wetSnow')
MIN_PASS = 30


def cdnet_passes(v):
    """Passes = one GT object from entry to exit (NFO-VOS protocol): motion components (merged by a 15 px dilation,
    >= 30 px) linked frame to frame by maximal overlap with the previous frame's region. Returns [(frames, regions)]."""
    a, b = map(int, open(f'{CDNET}/{v}/temporalROI.txt').read().split())
    active, done = [], []
    for k in range(a, b + 1):
        gt = cv2.imread(f'{CDNET}/{v}/groundtruth/gt{k:06d}.png', 0); mot = (gt == 255).astype(np.uint8)
        n, lab, _, _ = cv2.connectedComponentsWithStats(cv2.dilate(mot, K))
        regs = [lab == i for i in range(1, n) if (mot * (lab == i)).sum() >= 30]
        taken, keep = set(), []
        for p in active:
            ov = [(p[1][-1] & r).sum() if i not in taken else 0 for i, r in enumerate(regs)]
            if ov and max(ov) > 0:
                i = int(np.argmax(ov)); taken.add(i); p[0].append(k); p[1].append(regs[i]); keep.append(p)
            elif len(p[0]) >= MIN_PASS:
                done.append(p)
        active = keep + [([k], [r]) for i, r in enumerate(regs) if i not in taken]
    done += [p for p in active if len(p[0]) >= MIN_PASS]
    return done


def cdnet_clips(videos):
    """Per pass, start points at entry, 1/3 and 2/3 of the pass (if >= MIN_PASS frames remain); each clip runs to the
    end of the pass, like the NFO-VOS full-pass protocol."""
    out = []
    for v in videos:
        for frames, regs in cdnet_passes(v):
            L = len(frames)
            for si, s in enumerate((0, L // 3, 2 * L // 3)):
                if L - s >= MIN_PASS:
                    out.append((v, si, frames[s:], regs[s:]))
    return out


def main_cdnet(method, videos=None, tag=''):
    pred = build_predictor(method); rows = []
    os.makedirs('results/benchmark/cdnet/headroom', exist_ok=True)
    clips = cdnet_clips(videos or VIDEOS); print(len(clips), 'clips', flush=True)
    for j, (v, si, ks, regs) in enumerate(clips):
        gts = [cv2.imread(f'{CDNET}/{v}/groundtruth/gt{k:06d}.png', 0) for k in ks]
        m = run_clip(pred, [f'{CDNET}/{v}/input/in{k:06d}.jpg' for k in ks], tight((gts[0] == 255) & regs[0]))
        tp = fp = fn = 0
        for t in range(1, len(ks)):      # target = motion pixels of this pass's object; other objects count as background
            pos, valid = (gts[t] == 255) & regs[t], (gts[t] != 170) & (gts[t] != 85)
            tp += (m[t] & pos & valid).sum(); fp += (m[t] & ~pos & valid).sum(); fn += (~m[t] & pos & valid).sum()
        r = dict(video=v, start=('entry', 'third', 'two_thirds')[si], first_frame=ks[0], n_frames=len(ks),
                 P=tp / max(tp + fp, 1), R=tp / max(tp + fn, 1), J=tp / max(tp + fp + fn, 1))
        rows.append(r); print(j, r, flush=True)
        if j % 3 == 0:   # visual trace: input | GT (this object) | prediction, at the clip's middle frame
            t = len(ks) // 2; im = cv2.imread(f'{CDNET}/{v}/input/in{ks[t]:06d}.jpg')
            ov = im.copy(); ov[m[t]] = (0.5 * ov[m[t]] + [0, 0, 127]).astype(np.uint8)
            g = ((gts[t] == 255) & regs[t]).astype(np.uint8) * 255; g[gts[t] == 170] = 128
            vis = np.hstack([im, cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), ov]); sc = 1200 / vis.shape[1]
            cv2.imwrite(f'images/benchmark/cdnet/headroom_pass_{method}_{v.split("/")[1]}_{ks[0]}.jpg', cv2.resize(vis, None, fx=sc, fy=sc))
    with open(f'results/benchmark/cdnet/headroom/{method}_passes{tag}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for v in sorted({r['video'] for r in rows}):
        rr = [r for r in rows if r['video'] == v]
        print(f'{v}: clips={len(rr)} frames={sum(r["n_frames"] for r in rr)}  P {np.mean([r["P"] for r in rr]):.3f}  '
              f'R {np.mean([r["R"] for r in rr]):.3f}  J {np.mean([r["J"] for r in rr]):.3f}  | J by start: '
              + ' '.join(f'{s} {np.mean([r["J"] for r in rr if r["start"] == s]):.3f}' for s in ('entry', 'third', 'two_thirds') if any(r['start'] == s for r in rr)))


# ------------------------------------------------------------------ MVOT
def mvot_videos():
    rows = [json.loads(l) for l in open(f'{MVOT}/data/metadata.jsonl')]
    reg = sorted([r for r in rows if r['subset'] == 'regular'], key=lambda r: r['sample_id'])
    sim = [r for r in reg if 'yes' in (r.get('similar_values') or [])]
    non = [r for r in reg if 'yes' not in (r.get('similar_values') or [])]
    rng = random.Random(0); pick = rng.sample(sim, 30); used, match = set(), []
    for s in pick:
        key = lambda r, lv: tuple(r[k] for k in ('lighting', 'camera_tilt', 'view')[:lv])
        for lv in (3, 2, 1):
            c = [r for r in non if key(r, lv) == key(s, lv) and r['sample_id'] not in used]
            if c:
                m = rng.choice(c); used.add(m['sample_id']); match.append(m); break
    return [('similar', r) for r in pick] + [('matched', r) for r in match]


def mvot_frames(r):
    d = f'results/benchmark/mvot/frames/{r["sample_id"]}'
    if not os.path.isdir(d) or len(os.listdir(d)) < r['frame_count']:
        os.makedirs(d, exist_ok=True); cap = cv2.VideoCapture(f'{MVOT}/data/{r["file_name"]}'); k = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            cv2.imwrite(f'{d}/{k}.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 95]); k += 1
    return [f'{d}/{k}.jpg' for k in range(len(os.listdir(d)))]


def gt_boxes(r):
    return {a['frame_id'] - 1: (a['bbox_xywh'][0], a['bbox_xywh'][1], a['bbox_xywh'][0] + a['bbox_xywh'][2],
                                a['bbox_xywh'][1] + a['bbox_xywh'][3])
            for a in r['annotations'] if a['bbox_xywh'] and not a.get('source_marked_invalid')}


def main_mvot(method):
    pred = build_predictor(method); out = f'results/benchmark/mvot/headroom/{method}'; os.makedirs(out, exist_ok=True)
    for j, (grp, r) in enumerate(mvot_videos()):
        f = f'{out}/{r["sample_id"]}.npz'
        if os.path.exists(f):
            continue
        gt = gt_boxes(r); t0 = min(gt); paths = mvot_frames(r)[t0:]
        t_s = time.time(); m = run_clip(pred, paths, gt[t0])
        boxes = np.array([tight(x) or (-1, -1, -1, -1) for x in m], np.float32)
        cents = np.array([(np.nonzero(x)[1].mean(), np.nonzero(x)[0].mean()) if x.any() else (np.nan, np.nan) for x in m])
        np.savez_compressed(f, boxes=boxes, cents=cents, t0=t0, group=grp)
        print(j, grp, r['sample_id'], f'{(time.time() - t_s) / len(paths) * 1000:.0f} ms/frame', flush=True)


def report_mvot():
    th = np.linspace(0, 1, 21); vids = mvot_videos(); rows = []
    for method in ('b0', 't1'):
        for grp, r in vids:
            f = f'results/benchmark/mvot/headroom/{method}/{r["sample_id"]}.npz'
            if not os.path.exists(f):
                continue
            z = np.load(f); gt = gt_boxes(r); t0 = int(z['t0'])
            ts = [t for t in range(1, len(z['boxes'])) if t0 + t in gt]
            pb = lambda t: None if z['boxes'][t][0] < 0 else tuple(z['boxes'][t])
            ious = np.array([iou(pb(t), gt[t0 + t]) for t in ts])
            auc = float(np.mean([(ious > x).mean() for x in th]))
            # failure anatomy: runs of IoU < 0.1 for >= 5 frames
            cls, i = [], 0
            while i < len(ts):
                if ious[i] < 0.1:
                    e = i
                    while e < len(ts) and ious[e] < 0.1:
                        e += 1
                    if e - i >= 5:
                        seg = ts[i:e]; c = z['cents'][seg]; g = np.array([((gt[t0 + t][0] + gt[t0 + t][2]) / 2, (gt[t0 + t][1] + gt[t0 + t][3]) / 2) for t in seg])
                        L = np.median([max(gt[t0 + t][2] - gt[t0 + t][0], gt[t0 + t][3] - gt[t0 + t][1]) for t in seg])
                        ok = ~np.isnan(c[:, 0])
                        if ok.sum() < len(seg) / 2:
                            cls.append('empty')
                        else:
                            cc, gg = c[ok], g[ok]; n = max(len(cc) - 1, 1)
                            vp, vg = (cc[-1] - cc[0]) / n, (gg[-1] - gg[0]) / n
                            dv = np.linalg.norm(vp - vg)
                            cls.append('static' if np.linalg.norm(vp) < 0.2 and np.linalg.norm(vg) >= 0.5
                                       else 'co-moving' if dv < L / 30 else 'different')
                    i = e
                else:
                    i += 1
            rows.append(dict(method=method, group=grp, video=r['sample_id'], auc=auc, fail_events=len(cls),
                             **{k: cls.count(k) for k in ('co-moving', 'static', 'different', 'empty')}))
    os.makedirs('results/benchmark/mvot/headroom', exist_ok=True)
    with open('results/benchmark/mvot/headroom/report.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rng = np.random.default_rng(0)
    for method in ('b0', 't1'):
        a = {g: np.array([r['auc'] for r in rows if r['method'] == method and r['group'] == g]) for g in ('similar', 'matched')}
        if not len(a['similar']) or not len(a['matched']):
            continue
        d = [rng.choice(a['similar'], len(a['similar'])).mean() - rng.choice(a['matched'], len(a['matched'])).mean() for _ in range(2000)]
        print(f'{method}: AUC similar {a["similar"].mean():.3f} (n={len(a["similar"])})  matched {a["matched"].mean():.3f} (n={len(a["matched"])})'
              f'  diff {a["similar"].mean() - a["matched"].mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]')
        for g in ('similar', 'matched'):
            rr = [r for r in rows if r['method'] == method and r['group'] == g]
            print(f'   {g}: failure events {sum(r["fail_events"] for r in rr)} =', {k: sum(r[k] for r in rr) for k in ('co-moving', 'static', 'different', 'empty')},
                  f'videos with >=1 failure: {sum(r["fail_events"] > 0 for r in rr)}')


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'cdnet':   # optional: video list after the method, e.g. dynamicBackground/overpass badWeather/wetSnow
        main_cdnet(sys.argv[2], sys.argv[3:] or None, '_' + '_'.join(v.split('/')[1] for v in sys.argv[3:]) if sys.argv[3:] else '')
    elif what == 'mvot':
        main_mvot(sys.argv[2])
    elif what == 'mvot-report':
        report_mvot()
