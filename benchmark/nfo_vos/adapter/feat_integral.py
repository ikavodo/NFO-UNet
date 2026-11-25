"""Learned feature-space integral image (design: docs/superpowers/specs/2026-11-19-layer-sam-feature-integral-design.md).

The integral image of frame t is built from the CACHED SAM2 encoder features of frames t−n_h..t−1 (no second encoder
pass): each key frame is shifted onto frame t by the tracker's Theil–Sen x-shift (`shift_keys`), then a per-cell softmax
over the keys plus a null key ∅ (value: the empty scene B, a learned "abstain" slot as DETR's no-object class) gives

  Ĉ^s = Σ_k α_k^s K_k^s + α_∅^s B^s,      a = 1 − α_∅^4   (evidence mass, the input `Fusion` already reads)

α is computed at stride 16 and bilinearly upsampled to strides 8 / 4 (a convex resampling: still sums to 1).
- learned: ℓ_k = h([P F_k, P F_k − P F_t, P F_k − P B, e_age(k)]), last conv zero-init, ℓ_∅ a learned scalar (init 0)
  → uniform weights at step 0; trained by `feat_loss` (stage 0) and the mask loss.
- fixed (`fixed=True`, the hand-set control): α_k = w_k G_k, α_∅ = 1 − Σ w G with recency w_k ∝ e^{−age/τ} and the
  feature-space counterpart of the pixel change gate (integral_image.py: robust z-score of the change, threshold
  max(3, q90(z))) on m_k = 1 − cos(F_k^16, B^16): a = Σ w G as today, in feature space.
Cosines at stride 16 are taken on features centred by SAM2's constant `no_mem_embed` (mu16)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDES = ('16', '8', '4')


def ccos(x, y, mu=None):
    """Cosine along the channel dim (−3) of x − mu and y − mu."""
    if mu is not None:
        x, y = x - mu, y - mu
    return F.cosine_similarity(x.float(), y.float(), dim=-3, eps=1e-6)


def shift_keys(Fk, dx1024, stride, mode='bilinear'):
    """Fk [n, k, C, h, w], dx1024 [n, k] (px in the 1024 frame; positive = content moves right).
    Returns (Fs, valid): Fs(x) = Fk(x − dx/stride) sampled bilinearly (mode='roll': shift rounded to whole cells),
    zero outside the image; valid [n, k, 1, h, w] bool = the source cell lies inside the image."""
    n, k, C, h, w = Fk.shape
    c = dx1024.float().reshape(n * k) / int(stride)
    if mode == 'roll':
        c = c.round()
    xs = torch.arange(w, device=Fk.device, dtype=torch.float32)[None] - c[:, None]          # source x per output column
    ys = torch.arange(h, device=Fk.device, dtype=torch.float32)
    gx = (2 * xs + 1) / w - 1                                                               # align_corners=False
    gy = (2 * ys + 1) / h - 1
    grid = torch.stack([gx[:, None, :].expand(-1, h, -1), gy[None, :, None].expand(n * k, -1, w)], -1)
    Fs = F.grid_sample(Fk.reshape(n * k, C, h, w), grid.to(Fk.dtype), mode='bilinear', padding_mode='zeros',
                       align_corners=False).view(n, k, C, h, w)
    ok = (xs >= -1e-4) & (xs <= w - 1 + 1e-4)                                               # [n·k, w]
    valid = ok[:, None, :].expand(-1, h, -1).reshape(n, k, 1, h, w)
    return Fs, valid


def _robust_gate(m, valid):
    """m [n, k, h, w] change score, valid [n, k, h, w]: σ(z − θ) with z the robust z-score per key over its valid
    cells and θ = max(3, q90(z)) (integral_image.py's adaptive gate)."""
    x = torch.where(valid, m, torch.full_like(m, float('nan'))).flatten(2)
    med = x.nanmedian(-1, keepdim=True).values
    mad = (x - med).abs().nanmedian(-1, keepdim=True).values
    z = (x - med) / (1.4826 * mad + 1e-6)
    zq = torch.nan_to_num(z, nan=-1e9)
    th = torch.where(valid.flatten(2).any(-1, keepdim=True), torch.nanquantile(z, 0.9, dim=-1, keepdim=True), torch.zeros_like(med))
    th = torch.nan_to_num(th, nan=3.0).clamp(min=3.0)
    return torch.sigmoid(zq - th).view_as(m) * valid


def _shift(x, dy, dx, r):
    """out[..., y, x] = x[..., y + dy, x + dx], zero outside; |dy|, |dx| ≤ r."""
    H, W = x.shape[-2:]
    return F.pad(x, (r, r, r, r))[..., r + dy:r + dy + H, r + dx:r + dx + W]


class FeatIntegral(nn.Module):
    def __init__(self, mu16, d=32, nh_max=8, nbr=0):
        """nbr = r > 0: neighbourhood keys (2r+1)² per history frame at s16: key (q, o) = frame q's aligned features
        shifted by o ∈ {−r..r}² cells (o·16/s px-cells at stride s), so the content-based softmax can pick the best-aligned
        neighbour (sub-cell / sway misalignment; design §4, NATTEN-style local attention). Offset embedding zero-init,
        offset bias 0 at the centre and −4 elsewhere (≈ the r = 0 function at init). Mass head: centre keys only."""
        super().__init__()
        self.nbr = nbr
        self.offs = [(dy, dx) for dy in range(-nbr, nbr + 1) for dx in range(-nbr, nbr + 1)]
        if nbr:
            self.e_off = nn.Embedding(len(self.offs), d); nn.init.zeros_(self.e_off.weight)
            self.b_off = nn.Parameter(torch.tensor([0. if o == (0, 0) else -4. for o in self.offs]))
        self.register_buffer('mu16', mu16.detach().float().reshape(1, -1, 1, 1).clone())
        self.P = nn.Conv2d(256, d, 1)
        self.e_age = nn.Embedding(nh_max + 1, d)
        # input: [P F_k, P F_k − P F_t, P F_k − P B, e_age, φ, |φ|], φ = the key's sub-cell shift phase at s16
        self.h = nn.Sequential(nn.Conv2d(4 * d + 2, d, 1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1), nn.GELU(),
                               nn.Conv2d(d, 1, 1))
        nn.init.zeros_(self.h[-1].weight); nn.init.zeros_(self.h[-1].bias)
        self.null = nn.Parameter(torch.zeros(1))
        # evidence mass, decoupled from the mixing weights (Task 5b rev): a = σ(logsumexp_k m_k + b_m) over valid keys, m_k
        # from its own head on the DETACHED key inputs, so mass_loss never moves the weights that build Ĉ. At init a =
        # σ(log k) = k/(k+1) (= 1 − α_∅ of the uniform softmax); all keys invalid → a = 0.
        self.hm = nn.Sequential(nn.Conv2d(4 * d + 2, d, 1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1), nn.GELU(),
                                nn.Conv2d(d, 1, 1))
        nn.init.zeros_(self.hm[-1].weight); nn.init.zeros_(self.hm[-1].bias)
        self.mbias = nn.Parameter(torch.zeros(1))

    def load_compat(self, sd):
        """Load a state dict saved before the phase input (h.0 with 4d input channels): the 2 phase channels get zero
        weights, so the loaded module computes exactly the old function."""
        sd = dict(sd)
        w = sd['h.0.weight']
        if w.shape[1] == self.h[0].in_channels - 2:
            sd['h.0.weight'] = torch.cat([w, torch.zeros_like(w[:, :2])], 1)
        for k, v in self.state_dict().items():                  # heads added later (mass head): keep their fresh init
            if k.startswith(('hm.', 'mbias', 'e_off', 'b_off')) and k not in sd:
                sd[k] = v
        self.load_state_dict(sd)

    def logits_learned(self, K16, Ft16, B16, age, phase=None):
        n, k, C, H, W = K16.shape
        # P(x − mu) = P(x) − W·mu: fold the centring into a per-channel constant (no centred copy of the keys, which
        # autograd would keep for P's weight gradient)
        wmu = F.conv2d(self.mu16.to(self.P.weight.dtype), self.P.weight)                       # [1, d, 1, 1]
        P = lambda x: self.P(x) - wmu.to(x.dtype if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype())
        pk = P(K16.reshape(n * k, C, H, W)).view(n, k, -1, H, W)
        pt, pb = P(Ft16)[:, None], P(B16)[:, None]
        ea = self.e_age(age)[..., None, None].expand(-1, -1, -1, H, W).to(pk.dtype)
        ph = torch.zeros(n, k, device=pk.device) if phase is None else phase.float()
        ph = torch.stack([ph, ph.abs()], 2)[..., None, None].expand(-1, -1, -1, H, W).to(pk.dtype)
        if self.nbr:                     # P is 1×1, so shifting P(K) = P(shifted K) (padding cells are masked invalid)
            nO, c = len(self.offs), len(self.offs) // 2
            zc = torch.cat([pk, pk - pt, pk - pb, ea, ph], 2).flatten(0, 1)
            pk = torch.stack([_shift(pk, dy, dx, self.nbr) for dy, dx in self.offs], 2).flatten(1, 2)   # [n, k·O, d, H, W]
            ea = (ea[:, :, None] + self.e_off.weight.to(ea.dtype)[None, None, :, :, None, None]).flatten(1, 2)
            ph = ph.repeat_interleave(nO, 1)
            z = torch.cat([pk, pk - pt, pk - pb, ea, ph], 2).flatten(0, 1)
            lk = self.h(z).view(n, k, nO, H, W).float() + self.b_off.float().view(1, 1, nO, 1, 1)
            self._mk = self.hm(zc.detach()).view(n, k, H, W).float()
            return lk.flatten(1, 2), self.null.float().view(1, 1, 1, 1).expand(n, 1, H, W)
        z = torch.cat([pk, pk - pt, pk - pb, ea, ph], 2).flatten(0, 1)
        lk = self.h(z).view(n, k, H, W).float()
        self._mk = self.hm(z.detach()).view(n, k, H, W).float()
        return lk, self.null.float().view(1, 1, 1, 1).expand(n, 1, H, W)

    def logits_fixed(self, K16, B16, age, valid, tau):
        w = torch.exp(-age.float() / tau)
        w = w / w.sum(1, keepdim=True)
        m = 1 - ccos(K16, B16[:, None], self.mu16)                                         # [n, k, H, W]
        G = _robust_gate(m, valid)
        wg = w[..., None, None] * G
        lk = torch.log(wg.clamp(min=1e-12))
        ln = torch.log((1 - wg.sum(1, keepdim=True)).clamp(min=1e-6))
        return lk, ln

    def forward(self, K, valid16, Ft, B, age, fixed=False, tau=3.5, phase=None):
        K16 = K['16']
        n, k = K16.shape[:2]
        v = valid16[:, :, 0]
        if fixed:
            lk, ln = self.logits_fixed(K16, B['16'], age, v, tau)
        else:
            lk, ln = self.logits_learned(K16, Ft['16'], B['16'], age, phase)
        r, nO = (0, 1) if fixed else (self.nbr, len(self.offs))       # the fixed control has no neighbourhood keys
        vk = v if not r else torch.stack([_shift(v, dy, dx, r) for dy, dx in self.offs], 2).flatten(1, 2)
        lk = lk.masked_fill(~vk, float('-inf'))
        al = torch.softmax(torch.cat([lk, ln], 1), 1)                                # float32 [n, k·O+1, H, W]
        o = {'alpha16': al}
        for s in STRIDES:
            Ks, Bs = K[s], B[s]
            a_s = al if s == '16' else F.interpolate(al, size=Ks.shape[-2:], mode='bilinear', align_corners=False)
            a_s = a_s.to(Ks.dtype)
            kn, f = k * nO, 16 // int(s)
            C = a_s[:, kn:kn + 1] * Bs
            for q in range(k):                      # accumulate per key: no [n, k, C, h, w] product (training memory)
                if not r:
                    C = C + a_s[:, q:q + 1] * Ks[:, q]
                    continue
                R = r * f; Kp = F.pad(Ks[:, q], (R, R, R, R)); h_, w_ = Ks.shape[-2:]   # one padded copy, views per offset
                for j, (dy, dx) in enumerate(self.offs):
                    i = q * nO + j
                    C = C + a_s[:, i:i + 1] * Kp[..., R + dy * f:R + dy * f + h_, R + dx * f:R + dx * f + w_]
            o[f'C{s}'] = C
            if s == '4' and fixed:
                o['a'] = (1 - a_s[:, kn].float()).clamp(0, 1)
        if not fixed:
            mk = self._mk.masked_fill(~v, float('-inf'))
            ml = torch.logsumexp(mk, 1, keepdim=True) + self.mbias.float()            # all invalid → −inf → a = 0
            o['a'] = torch.sigmoid(F.interpolate(ml.nan_to_num(neginf=-1e4), size=K['4'].shape[-2:], mode='bilinear',
                                                 align_corners=False))[:, 0]
        return o


class Refined(nn.Module):
    """Ĉ' = Ĉ + g([Ĉ − μ, B − μ]) at stride 16 (tools/feat_refine.py, findings §10.x item 15): a small residual
    refiner, distilled from the pixel route's enc(C), that approximates 'encode the averaged image' (E(Σ w X) ≠ Σ w E(X)).
    full=False: strides 8 / 4 keep the plain mix. full=True (tools/feat_refine_full.py): refiners g8 / g4 on [Ĉ, B] at
    strides 8 / 4 too (SAM2 adds no_mem_embed only at s16, so no centring there)."""
    def __init__(self, agg, full=False):
        super().__init__()
        self.agg, self.full = agg, full

        def mk(cin, hid, cout):
            g = nn.Sequential(nn.Conv2d(cin, hid, 1), nn.GELU(), nn.Conv2d(hid, hid, 3, padding=1), nn.GELU(),
                              nn.Conv2d(hid, cout, 1))
            nn.init.zeros_(g[-1].weight); nn.init.zeros_(g[-1].bias)
            return g
        self.g = mk(512, 128, 256)
        if full:
            self.g8, self.g4 = mk(128, 64, 64), mk(64, 32, 32)

    def refine_s(self, s, C, B):
        if s == '16':
            return self.refine(C, B)
        g = self.g8 if s == '8' else self.g4
        return C + g(torch.cat([C, B], 1).to(g[0].weight.dtype)).float()

    def refine(self, C, B):
        mu = self.agg.mu16
        return C + self.g(torch.cat([C - mu, B - mu], 1).to(self.g[0].weight.dtype)).float()

    def forward(self, K, valid16, Ft, B, age, phase=None, fixed=False, **kw):
        o = self.agg(K, valid16, Ft, B, age, phase=phase, fixed=fixed)
        for s in (('16', '8', '4') if self.full else ('16',)):
            o[f'C{s}'] = self.refine_s(s, o[f'C{s}'].float(), B[s].float())
        return o


def big_net(cin=6 * 256 + 2 * 6 + 2 * 256):
    """tools/feat_capacity 'big' student: unmixed keys + validity + phase + F_t + B -> C16 (5.9 M params, 0.83 ms/frame)."""
    return nn.Sequential(nn.Conv2d(cin, 512, 1), nn.GELU(), nn.Conv2d(512, 512, 3, padding=1), nn.GELU(),
                         nn.Conv2d(512, 512, 3, padding=1), nn.GELU(), nn.Conv2d(512, 256, 1))


def big_inputs(K16, v, ph, Ft16, B16, k=6):
    """K16 [n, nh, C, h, w], v [n, nh, h, w] (bool/float), ph [n, nh] -> student input; the k most recent keys
    (keys are ordered oldest first), zero-padded with valid = 0 when nh < k (synthetic nh is 4..8, NFO 6)."""
    K16, v, ph = K16[:, -k:].float(), v[:, -k:].float(), ph[:, -k:].float()
    n, nh, c, h, w = K16.shape
    if nh < k:
        K16 = torch.cat([K16.new_zeros(n, k - nh, c, h, w), K16], 1); v = torch.cat([v.new_zeros(n, k - nh, h, w), v], 1)
        ph = torch.cat([ph.new_zeros(n, k - nh), ph], 1)
    return torch.cat([(K16 * v[:, :, None]).reshape(n, k * c, h, w), v, ph[..., None, None].expand(n, k, h, w),
                      Ft16.float(), B16.float()], 1)


class BigS16(nn.Module):
    """A Refined aggregator (strides 8 / 4 and the evidence mass) whose stride-16 output is replaced by the big
    distilled student on the unmixed keys (tools/feat_bigstudent.py; findings §10.x 18-20)."""
    def __init__(self, base):
        super().__init__()
        self.base, self.net = base, big_net()
        self.mu16 = base.agg.mu16

    def forward(self, K, valid16, Ft, B, age, phase=None, fixed=False, **kw):
        o = self.base(K, valid16, Ft, B, age, phase=phase, fixed=fixed)
        x = big_inputs(K['16'], valid16[:, :, 0], phase, Ft['16'], B['16'])
        o['C16'] = self.net(x.to(self.net[0].weight.dtype) if not torch.is_autocast_enabled() else x).float()
        return o


def feat_loss(C, T, Om, mu16):
    """Mean over strides of the mean over Ω cells of 1 − cos(Ĉ^s, T^s) (stride 16 centred by mu16); 0 if Ω is empty."""
    terms = []
    for s in STRIDES:
        m = Om[s][:, 0]
        if m.any():                                  # cosine on the Ω cells only (≈ 1–3% of the map): training memory
            x = (C[f'C{s}'] if f'C{s}' in C else C[s]).permute(0, 2, 3, 1)[m]
            y = T[s].permute(0, 2, 3, 1)[m]
            mu = mu16.reshape(1, -1) if s == '16' else None
            terms.append((1 - ccos(x[..., None, None], y[..., None, None], None if mu is None else mu[..., None, None])).mean())
    if not terms:
        x = next(iter(C.values()))
        return x.sum() * 0
    return torch.stack(terms).mean()


def mass_loss(a, O):
    """BCE of the evidence mass a = 1 − α_∅ [n, 256, 256] against O = fraction of the cell where the person was visible in
    at least one aligned history frame (supervises the abstain slot; plan Task 5b)."""
    a, O = a.float().clamp(1e-4, 1 - 1e-4), O.float()               # explicit BCE: F.binary_cross_entropy refuses autocast
    return -(O * a.log() + (1 - O) * (1 - a).log()).mean()
