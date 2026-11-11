"""Run after gait_masks fit: PYTHONPATH=. python -m benchmark.nfo_vos.eval.gait_box_control"""
# Fair box control: classifier trained on KTH GT-box rectangles (same canvas), tested on the NFO GT-box arm; plus per-class accuracy of the occ classifier.
import numpy as np, csv, json, collections
from sklearn.linear_model import LogisticRegression
from benchmark.nfo_vos.eval import gait_masks as G
kth = list(np.load(f'{G.OUT}/kth_canvases.npy', allow_pickle=True)); nfo = list(np.load(f'{G.OUT}/nfo_canvases.npy', allow_pickle=True))
cr = sorted(G.kth_crossings().items()); i = 0
for name, v in cr:
    for (s, e) in v['runs']:
        r = kth[i]; assert r['name'] == name; L = len(r['clean']); g = v['gt'][s:s + L]
        b = np.stack([g[:, 0]*160, g[:, 1]*120, (g[:, 0]+g[:, 2])*160, (g[:, 1]+g[:, 3])*120], 1)
        cx, cy, h, fl = G.trajectory(b); out = []
        for k in range(L):
            m = np.zeros((120, 160), bool); x0, y0, x1, y1 = np.round(b[k]).astype(int); m[max(y0,0):y1, max(x0,0):x1] = True
            out.append(G.canvas(m, cx[k], cy[k], h, fl))
        r['box'] = np.array(out, np.float16); i += 1
assert i == len(kth)
lab = {r['crossing']: r['label'] for r in nfo}
for k in (8, 16, 24):
    X, y, _ = G.kth_xy(kth, 'box', k); clf = LogisticRegression(C=0.1, max_iter=3000).fit(X, y)
    per = collections.defaultdict(list)
    for r in nfo:
        if r['seq'] == 'seq1': continue
        per[r['crossing']] += (clf.predict(G.windows(r['C']['box'], k)) == r['label']).tolist()
    a = {c: np.mean(v) for c, v in per.items()}
    print(f'box-trained k={k}: NFO seq2-4 box acc {np.mean(list(a.values())):.3f}  walk {np.mean([a[c] for c in a if lab[c]==0]):.3f} run {np.mean([a[c] for c in a if lab[c]==1]):.3f}')
    if k == 16: boxacc = a
res = list(csv.DictReader(open(f'{G.OUT}/gait_acc.csv')))
for kind in ('occ', 'clean'):
    A = collections.defaultdict(dict)
    for r in res:
        if r['kind'] == kind and r['k'] == '16' and not r['crossing'].startswith('seq1'): A[r['arm']][r['crossing']] = float(r['acc'])
    ours = {c: np.mean([A[a][c] for a in G.OURS]) for c in A['oracle']}
    for a, d in (('oracle', A['oracle']), ('t1-c2', A['t1-c2']), ('ours', ours)):
        print(f'{kind} k=16 {a:7s} walk {np.mean([d[c] for c in d if lab[c]==0]):.3f} run {np.mean([d[c] for c in d if lab[c]==1]):.3f}', end='')
        m, lo, hi = G.boot(np.array([d[c] - boxacc[c] for c in d])); print(f'   - fair box: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]')
