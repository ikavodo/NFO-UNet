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


class FeatIntegral(nn.Module):
    def __init__(self, mu16, d=32, nh_max=8):
        super().__init__()
        self.register_buffer('mu16', mu16.detach().float().reshape(1, -1, 1, 1).clone())
        self.P = nn.Conv2d(256, d, 1)
        self.e_age = nn.Embedding(nh_max + 1, d)
        # input: [P F_k, P F_k − P F_t, P F_k − P B, e_age, φ, |φ|], φ = the key's sub-cell shift phase at s16
        self.h = nn.Sequential(nn.Conv2d(4 * d + 2, d, 1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1), nn.GELU(),
                               nn.Conv2d(d, 1, 1))
        nn.init.zeros_(self.h[-1].weight); nn.init.zeros_(self.h[-1].bias)
        self.null = nn.Parameter(torch.zeros(1))

    def load_compat(self, sd):
        """Load a state dict saved before the phase input (h.0 with 4d input channels): the 2 phase channels get zero
        weights, so the loaded module computes exactly the old function."""
        w = sd['h.0.weight']
        if w.shape[1] == self.h[0].in_channels - 2:
            sd = dict(sd); sd['h.0.weight'] = torch.cat([w, torch.zeros_like(w[:, :2])], 1)
        self.load_state_dict(sd)

    def logits_learned(self, K16, Ft16, B16, age, phase=None):
        n, k, C, H, W = K16.shape
        P = lambda x: self.P(x - self.mu16.to(x.dtype))
        pk = P(K16.reshape(n * k, C, H, W)).view(n, k, -1, H, W)
        pt, pb = P(Ft16)[:, None], P(B16)[:, None]
        ea = self.e_age(age)[..., None, None].expand(-1, -1, -1, H, W).to(pk.dtype)
        ph = torch.zeros(n, k, device=pk.device) if phase is None else phase.float()
        ph = torch.stack([ph, ph.abs()], 2)[..., None, None].expand(-1, -1, -1, H, W).to(pk.dtype)
        z = torch.cat([pk, pk - pt, pk - pb, ea, ph], 2).flatten(0, 1)
        lk = self.h(z).view(n, k, H, W).float()
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
        lk = lk.masked_fill(~v, float('-inf'))
        al = torch.softmax(torch.cat([lk, ln], 1), 1)                                       # float32 [n, k+1, H, W]
        o = {'alpha16': al}
        for s in STRIDES:
            Ks, Bs = K[s], B[s]
            a_s = al if s == '16' else F.interpolate(al, size=Ks.shape[-2:], mode='bilinear', align_corners=False)
            a_s = a_s.to(Ks.dtype)
            o[f'C{s}'] = (a_s[:, :k, None] * Ks).sum(1) + a_s[:, k:k + 1] * Bs
            if s == '4':
                o['a'] = (1 - a_s[:, k].float()).clamp(0, 1)
        return o


def feat_loss(C, T, Om, mu16):
    """Mean over strides of the mean over Ω cells of 1 − cos(Ĉ^s, T^s) (stride 16 centred by mu16); 0 if Ω is empty."""
    terms = []
    for s in STRIDES:
        m = Om[s][:, 0]
        if m.any():
            terms.append((1 - ccos(C[f'C{s}'] if f'C{s}' in C else C[s], T[s], mu16 if s == '16' else None))[m].mean())
    if not terms:
        x = next(iter(C.values()))
        return x.sum() * 0
    return torch.stack(terms).mean()


def mass_loss(a, O):
    """BCE of the evidence mass a = 1 − α_∅ [n, 256, 256] against O = fraction of the cell where the person was visible in
    at least one aligned history frame (supervises the abstain slot; plan Task 5b)."""
    a, O = a.float().clamp(1e-4, 1 - 1e-4), O.float()               # explicit BCE: F.binary_cross_entropy refuses autocast
    return -(O * a.log() + (1 - O) * (1 - a).log()).mean()
