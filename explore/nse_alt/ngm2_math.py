"""float32 mirrors of the ngm2 helpers the NSE study depends on.

Sources: include/hd_ngm2/hd_ngm2_tools.h (sigmoid, normalized_shannon_entropy),
src/inhibition/soft_wta.cpp (soft_wta_apply). Vectorised over rows: every
function takes an array of shape (..., n) and reduces/acts on the last axis.
"""
import numpy as np

F32 = np.float32
EPS = np.finfo(np.float32).eps


def sigmoid(x, steepness=0.5, transition_point=0.5):
    x = np.asarray(x, dtype=F32)
    inv = F32(1.0) / (F32(1.0) - np.power(F32(steepness), F32(0.1)))
    start = F32(-transition_point) * inv
    return (F32(1.0) / (F32(1.0) + np.exp(-(x * inv + start)))).astype(F32)


def nse(v):
    v = np.asarray(v, dtype=F32)
    n = v.shape[-1]
    s = v.sum(axis=-1, dtype=F32, keepdims=True)
    ok = np.isfinite(s) & (np.abs(s) >= np.finfo(np.float32).tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = v / s
        term = np.where(p >= EPS, p * np.log2(np.where(p > 0, p, F32(1))), F32(0))
    h = -term.sum(axis=-1, dtype=F32)
    out = np.clip(h / np.log2(F32(n)), 0.0, 1.0).astype(F32)
    return np.where(ok[..., 0], out, F32(0))


def soft_wta_factor(nse_val):
    """nse_fact in soft_wta: 1 at low NSE (sharpen), 0 at high NSE."""
    return F32(1.0) - sigmoid((np.asarray(nse_val, F32) - F32(0.8)) / F32(0.2))


def soft_wta(raw, strength=5.0):
    raw = np.asarray(raw, dtype=F32)
    mx = raw.max(axis=-1, keepdims=True)
    f = soft_wta_factor(nse(raw))[..., None]
    expo = F32(1.0) + (F32(strength) - F32(1.0)) * f
    with np.errstate(divide="ignore", invalid="ignore"):
        out = raw * np.power(raw / mx, expo)
    out = np.clip(out, 0.0, 1.0).astype(F32)
    bad = ~(np.isfinite(mx) & (mx >= np.finfo(np.float32).tiny))
    return np.where(bad, raw, out)


def max_softmax(raw, beta):
    """Mirror of inhibition::max_softmax_apply: max * exp(-beta * (max - v)), float32."""
    v = np.asarray(raw, np.float32)
    m = v.max(-1, keepdims=True)
    out = m * np.exp(np.float32(-beta) * (m - v))
    return np.where(m > np.finfo(np.float32).tiny, out, v).astype(np.float32)


def _gauss_kernel(sigma):
    """retina.cpp gaussian_kernel: radius ceil(3 sigma), renormalised."""
    r = int(np.ceil(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * x * x / (sigma * sigma)).astype(F32)
    return (k / k.sum(dtype=F32)).astype(F32)


def _blur(Y, sigma):
    """Separable blur of (n, h, w) planes, clamp-to-edge borders (retina.cpp blur)."""
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    P = np.pad(Y, ((0, 0), (0, 0), (r, r)), mode="edge")
    T = sum(k[i] * P[:, :, i:i + Y.shape[2]] for i in range(len(k))).astype(F32)
    P = np.pad(T, ((0, 0), (r, r), (0, 0)), mode="edge")
    return sum(k[i] * P[:, i:i + Y.shape[1], :] for i in range(len(k))).astype(F32)


def retina_d(X, shape, k, gain=2.0, center_sigma=1.0, surround_sigma=3.0, luminance=None):
    """Mirror of retina::contrast_filter_t::apply. X: (n, h*w*c) HWC images.
    Returns signed d, (n, h*w*out_c). luminance defaults to True for 3 channels."""
    h, w, c = shape
    X = np.asarray(X, F32).reshape(-1, h, w, c)
    lum = (c == 3) if luminance is None else luminance
    planes = [(F32(0.299) * X[..., 0] + F32(0.587) * X[..., 1] + F32(0.114) * X[..., 2])] if lum \
        else [X[..., i] for i in range(c)]
    out = []
    for Y in planes:
        mean = Y.mean(axis=(1, 2), keepdims=True, dtype=F32)
        out.append(F32(gain) * (_blur(Y, center_sigma) - F32(k) * _blur(Y, surround_sigma) - (F32(1) - F32(k)) * mean))
    return np.stack(out, -1).reshape(len(X), -1).astype(F32)


def retina_out(d, polarity, rng, noise=(0.0, 0.05)):
    """polarize + noise floor, as retina_io::process."""
    v = np.clip(d if polarity == "on" else -d, 0.0, 1.0).astype(F32)
    return np.minimum(v + rng.uniform(noise[0], noise[1], v.shape).astype(F32), F32(1)).astype(F32)
