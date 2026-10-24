# GT-free cleaned integral image C (integral image gallery)

Construction of the "GT-free clean C" tile in `images/benchmark/rr_checks/composite_gallery/`.
Code: `benchmark/nfo_vos/diag_composite_diff.py`, `gallery()` (lines 306–351) and `window()` (lines 68–88);
recency weights from `benchmark/nfo_vos/a4.py`. Written 2026-10-12, documented 2026-10-14.

**Caveat on the name.** Only the *cleaning weights* are GT-free. The *alignment* uses the human GT boxes
(oracle), and the outline drawn on the first tile is the SAM2-ceiling (display only).

## 1. Setup

Let a sequence be a set of grayscale frames at native resolution

$$\mathcal{V} = \{R_k\}_{k}, \quad R_k \in [0, 255]^{H \times W}, \quad H \times W = 600 \times 800$$

For a trial window starting at frame $t_0$, the history is the $N - 1 = 6$ frames before it ($N = 7$, `a4.N`):

$$\mathcal{H} = \{t_0 - 6, \dots, t_0 - 1\}$$

Frame $t_0$ itself is **not** in the integral image.

## 2. Empty-scene background

The trial's warm-up interval $[w_0, w_1]$ contains no person. The empty scene is the per-pixel median over it:

$$B(x) = \operatorname{median}_{k \in [w_0, w_1]} R_k(x), \quad x \in \Omega$$

where $\Omega$ is the image domain. (`gallery()`, line 322)

## 3. Alignment (oracle)

Let $c_k \in \mathbb{R}^2$ be the centre of the human GT box at frame $k$, converted to native pixels
(`box_centre`, line 54). Each history frame is translated onto $t_0$:

$$\delta_k = c_{t_0} - c_k, \qquad A_k(x) = R_k(x - \delta_k)$$

with bilinear interpolation and border replication (`shift`, line 62). This is the only GT input to the integral image.

## 4. GT-free foreground weight

For each history frame, take the absolute difference from the empty scene, **before** alignment:

$$d_k(x) = \left| R_k(x) - B(x) \right|$$

and a robust z-score over the **whole frame**. MAD = median absolute deviation; 1.4826 makes it consistent with
a Gaussian standard deviation:

$$\mu_k = \operatorname{median}_{x \in \Omega} d_k(x), \qquad s_k = 1.4826 \cdot \operatorname{median}_{x \in \Omega} \left| d_k(x) - \mu_k \right|$$

$$z_k(x) = \frac{d_k(x) - \mu_k}{\max(s_k, 10^{-6})}$$

The weight is a logistic in $z$, centred at 3 robust standard deviations. It is then shifted with the same $\delta_k$:

$$g_k(x) = \sigma\!\left(z_k(x - \delta_k) - 3\right) = \frac{1}{1 + e^{-(z_k(x - \delta_k) - 3)}}$$

So $g_k \approx 1$ where frame $k$ is far from the empty scene (likely person), $g_k \approx 0$ where it matches
(likely background or leaves). (lines 324–328)

## 5. Recency weights

Ages run from 0 (frame $t_0 - 1$) to 5 (frame $t_0 - 6$), with time constant $\tau = 3.5$ (`a4.TAU_AGE`):

$$\tilde{w}_k = \exp\!\left(-\frac{t_0 - 1 - k}{\tau}\right), \qquad w_k = \frac{\tilde{w}_k}{\sum_{j \in \mathcal{H}} \tilde{w}_j}$$

(`window()`, line 87)

## 6. The integral image

$$C_{\text{GT-free}}(x) = \frac{\sum_{k \in \mathcal{H}} w_k \, g_k(x) \, A_k(x)}{\max\!\left(\sum_{k \in \mathcal{H}} w_k \, g_k(x), \; 10^{-6}\right)}$$

(line 330). Per pixel, a recency-weighted mean of the aligned history, where each frame counts only as much as
it looks unlike the empty scene.

Note: where no history frame is foreground-like ($\sum_k w_k g_k \to 0$) the numerator also goes to 0, so $C$
is near black there. It does **not** fall back to the raw integral image (unlike the oracle version below).

## 7. The comparison tiles

Raw integral image (no cleaning):

$$C_{\text{raw}}(x) = \sum_{k \in \mathcal{H}} w_k \, A_k(x)$$

Oracle-cleaned integral image: replace $g_k$ by the aligned SAM2-ceiling visibility mask $v_k(x) \in \{0, 1\}$,
falling back to raw where no history frame saw the person:

$$\bar{V}(x) = \sum_{k} w_k \, v_k(x), \qquad
C_{\text{oracle}}(x) = \begin{cases} \dfrac{\sum_k w_k \, v_k(x) \, A_k(x)}{\bar{V}(x)} & \bar{V}(x) > 10^{-6} \\[2mm] C_{\text{raw}}(x) & \text{otherwise} \end{cases}$$

(`composites()`, lines 91–96)

## 8. Why it was expected to work, and what was measured

Mixing model (`diag_composite_diff.py` docstring): a pixel at $t_0$ is a mixture of person $P$ and occluder
$O$ with visibility $v$; the integral image mixes them with history visibility $\bar{V}$:

$$R_{t_0} = v P + (1 - v) O, \qquad C = \bar{V} P + (1 - \bar{V}) \bar{O}$$

The raw integral image should separate person from occluder only where $\bar{V}$ is high; a clean integral image
($\bar{V} \to 1$ on the person) should separate everywhere.

Measured (2026-10-12, `docs/vos_benchmark/archive/nfo_vos_pilot_findings_full.md`, pre-registered fine-addition test, KILL):
- comparing frame $t$ against $C$ to decide which pixels to add: AUC 0.47–0.51 for raw, GT-free and oracle
  alike (chance);
- 51% of the "missed" pixels lie within 2 px of T1's mask, 66% within 4 px: consistent with quantization of the
  coarse SAM2-ceiling rather than missed body parts.

## 9. Known weakness

$g_k$ is built on the pixel cue $|R_k - B|$. The same cue, used as background suppression on T1, failed on
2026-10-13 (seq2–4 ΔJ −0.040): the grayscale person has nearly the same brightness as the foliage. Where person
and leaves match in brightness, $g_k \approx 0$ and the cleaning removes person pixels along with leaves. A
reuse of $C$ should compute $g_k$ on encoder features (per-cell change detection), not on pixels.
