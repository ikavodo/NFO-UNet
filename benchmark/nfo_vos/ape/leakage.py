"""Camera-leakage clusters for PanAf600 (PanAf500 + the 100 PanAf-SBR videos), which carry no camera ids.

Per video: the per-pixel median of 16 evenly spaced gray frames (removes a moving ape; the scene stays), embedded with
DINOv2-small (Oquab et al., arXiv 2304.07193; HF facebook/dinov2-small, frozen): [CLS ; mean patch token], L2-normed.
Same-camera threshold calibrated on PanAf-FGBG foreground clips, which have camera ids (disjoint.csv): 150 cameras with
>= 3 clips, 3 clips each; the threshold is set so that 0.1% of different-camera pairs exceed it, and the share of
same-camera pairs it catches (recall) is reported. (A 95%-recall threshold linked 68% of different-camera pairs, so
single linkage chained all 600 videos into one cluster: day and IR clips of one camera embed far apart.)
Leaked = a non-eval video whose similarity to ANY eval-subset video exceeds the threshold (direct, no chaining).
Clusters (connected components at the same threshold) are written for inspection only.
Writes results/ape/leakage.csv (video, cluster, in_eval, leaked) and results/ape/leakage_calibration.txt.
Limit: day and IR clips of one camera can embed apart (under-merging), so leakage is under-, not over-estimated.
    python -m benchmark.nfo_vos.ape.leakage
"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np, pandas as pd, torch
from transformers import AutoImageProcessor, AutoModel
from benchmark.nfo_vos.ape import PANAF, PANAF_ROOT, SBR_ROOT, FGBG_ROOT, OUT

K, FPR = 16, 0.001


def median_frame(path):
    c = cv2.VideoCapture(str(path)); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); fs = []
    for i in np.linspace(0, max(n - 1, 0), K).astype(int):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = c.read()
        if ok:
            fs.append(cv2.cvtColor(cv2.resize(f, (448, 252), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    return np.median(np.stack(fs), 0).astype(np.uint8) if fs else None


def embed(paths, model, proc, cache=None):
    if cache is not None and cache.exists():
        return np.load(cache)
    with ThreadPoolExecutor(12) as ex:
        meds = list(ex.map(median_frame, paths))
    out = []
    for i in range(0, len(meds), 64):
        b = [np.repeat(m[..., None], 3, 2) for m in meds[i:i + 64]]
        x = proc(images=b, return_tensors='pt')['pixel_values'].cuda()
        with torch.no_grad():
            h = model(pixel_values=x).last_hidden_state
        e = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1); out.append(torch.nn.functional.normalize(e, dim=1).cpu())
        print(f'  embedded {min(i + 64, len(meds))}/{len(meds)}', flush=True)
    E = torch.cat(out).numpy()
    if cache is not None:
        np.save(cache, E)
    return E


def components(S, thr):
    n = len(S); lab = -np.ones(n, int); c = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        st = [i]; lab[i] = c
        while st:
            j = st.pop()
            for k in np.flatnonzero((S[j] > thr) & (lab < 0)):
                lab[k] = c; st.append(k)
        c += 1
    return lab


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    proc = AutoImageProcessor.from_pretrained('facebook/dinov2-small'); model = AutoModel.from_pretrained('facebook/dinov2-small').cuda().eval()
    # calibration on FGBG
    d = pd.read_csv(FGBG_ROOT / 'annotations' / 'disjoint.csv'); rng = np.random.default_rng(0)
    cams = [c for c, g in d.groupby('camera_id') if len(g) >= 3]; cams = rng.choice(cams, min(150, len(cams)), replace=False)
    cal = pd.concat([d[d.camera_id == c].sample(3, random_state=0) for c in cams])
    print(f'calibration: {len(cal)} FGBG fg clips from {len(cams)} cameras', flush=True)
    E = embed([FGBG_ROOT / 'videos' / f for f in cal.fg], model, proc, OUT / 'emb_fgbg_cal.npy'); S = E @ E.T
    cid = cal.camera_id.to_numpy(); iu = np.triu_indices(len(cid), 1); same = cid[iu[0]] == cid[iu[1]]; s = S[iu]
    thr = float(np.quantile(s[~same], 1 - FPR)); rec = float((s[same] > thr).mean())
    cal_txt = (f'same-camera pairs {same.sum()}, different {(~same).sum()}; same sim median {np.median(s[same]):.3f}, '
               f'different median {np.median(s[~same]):.3f}\nthreshold (different-camera false links {FPR:.1%}) = {thr:.3f}; '
               f'same-camera pairs caught: {rec:.3f}\n')
    print(cal_txt, flush=True)
    # PanAf600
    vids = {p.stem: p for p in sorted(PANAF_ROOT.glob('videos/*.mp4'))}
    vids.update({p.stem: p for p in sorted(SBR_ROOT.glob('videos/*.mp4')) if p.stem not in vids})
    names = list(vids); E = embed([vids[v] for v in names], model, proc, OUT / 'emb_panaf600.npy'); S = E @ E.T; np.fill_diagonal(S, -1)
    lab = components(S, thr)
    ev = set(pd.read_csv(PANAF / 'results' / 'panaf' / 'eval_subset.csv').video)
    T = pd.DataFrame(dict(video=names, cluster=lab)); T['in_eval'] = T.video.isin(ev)
    ei = np.flatnonzero(T.in_eval.to_numpy()); T['max_sim_eval'] = S[:, ei].max(1)
    T['leaked'] = (T.max_sim_eval > thr) & ~T.in_eval
    T['nn_sim'] = S.max(1); T.to_csv(OUT / 'leakage.csv', index=False)
    sz = T.groupby('cluster').size()
    summ = (f'PanAf600: {len(T)} videos, {T.cluster.nunique()} clusters (singletons {int((sz == 1).sum())}, largest {sz.max()}); '
            f'eval {int(T.in_eval.sum())}; non-eval videos directly similar to an eval video (leaked): {int(T.leaked.sum())}\n')
    print(summ, f'{time.time() - t0:.0f}s', flush=True); (OUT / 'leakage_calibration.txt').write_text(cal_txt + summ)


if __name__ == '__main__':
    main()
