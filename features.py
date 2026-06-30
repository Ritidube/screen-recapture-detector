"""
Feature extraction for "Spot the Fake Photo" (recapture detector).

This is the single source of truth for turning a raw image into the 194-dim
raw feature vector the trained pipeline expects. Both predict.py (CLI) and
backend/app.py (API) import this module, so the two entry points can never
drift out of sync with each other or with the training notebook.

8 signal families, all hand-engineered (no deep net):
  1. Wavelet sub-band statistics   - screens have a different frequency profile
  2. Moire / FFT peaks             - periodic pixel-grid interference
  3. LBP texture histogram         - micro-texture differs (screen vs. paper/skin/etc.)
  4. Noise residual statistics     - sensor noise vs. re-photographed noise
  5. Chromatic aberration at edges - real lenses smear color at edges, screens don't
  6. Sharpness / edge stats        - screens often show a slight defocus/moire softness
  7. Glare / highlight stats       - screens clip and reflect differently than matte objects
  8. Color-space statistics        - white balance and gamut shifts from screen emission
"""
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from skimage.feature import local_binary_pattern

# Dual-resolution extraction: texture/frequency signals need detail (384px),
# color/glare signals are stable even on a small thumbnail (200px). Running
# the color signals on the smaller image is most of what keeps latency down.
TEXTURE_LONG_SIDE = 384
STATS_LONG_SIDE = 200


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def _resize_long_side(img, target):
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > target:
        scale = target / long_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _fast_entropy(hist):
    h = hist[hist > 0]
    return float(-np.sum(h * np.log(h)))


# ── Signal 1: wavelet ───────────────────────────────────────────────
def _wavelet_features(gray):
    g = (gray / 255.0).astype(np.float32)
    coeffs = pywt.wavedec2(g, wavelet="db4", level=3)
    feats = []
    energies = []
    for level in coeffs[1:]:
        level_energy = []
        for sb in level:
            f = sb.ravel()
            e = float(np.sum(f.astype(np.float64) ** 2))
            level_energy.append(e)
            hist, _ = np.histogram(f, bins=16, density=True)
            hist = hist + 1e-12
            med = float(np.median(f))
            feats.extend([
                float(np.mean(f)), float(np.var(f)), e,
                _fast_entropy(hist), float(skew(f)), float(kurtosis(f)),
                med, float(np.median(np.abs(f - med))),
                float(np.percentile(np.abs(f), 95)),
            ])
        energies.append(level_energy)
    for i in range(len(energies) - 1):
        for a, b in zip(energies[i], energies[i + 1]):
            feats.append(float(a / (b + 1e-8)))
    return np.array(feats, dtype=np.float32)


# ── Signal 2: moire / FFT peaks ──────────────────────────────────────
def _moire_features(gray):
    g = gray / 255.0
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log(np.abs(f) + 1.0)
    H, W = mag.shape
    cy, cx = H // 2, W // 2

    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_mask = dist < (min(H, W) * 0.04)
    search = mag.copy()
    search[low_mask] = 0

    thresh = search.mean() + 3 * search.std()
    peaks = search > thresh
    peak_count = int(peaks.sum())
    peak_strength = float(search[peaks].mean()) if peak_count > 0 else 0.0

    max_r = min(H, W) / 2.0
    nbins = 12
    dist_flat = dist.ravel()
    search_flat = search.ravel()
    radial_idx = np.clip((dist_flat / max_r * nbins).astype(np.int32), 0, nbins - 1)
    radial_sum = np.bincount(radial_idx, weights=search_flat, minlength=nbins)
    radial_cnt = np.bincount(radial_idx, minlength=nbins)
    radial_energy = radial_sum / np.maximum(radial_cnt, 1)
    radial_energy = radial_energy / (radial_energy.sum() + 1e-8)

    theta = np.arctan2(yy - cy, xx - cx)
    nADirs = 8
    theta_flat = theta.ravel()
    low_flat = low_mask.ravel()
    angle_idx = np.clip(((theta_flat + np.pi) / (2 * np.pi) * nADirs).astype(np.int32), 0, nADirs - 1)
    valid = ~low_flat
    dir_sum = np.bincount(angle_idx[valid], weights=search_flat[valid], minlength=nADirs)
    dir_cnt = np.bincount(angle_idx[valid], minlength=nADirs)
    dir_energy = dir_sum / np.maximum(dir_cnt, 1)
    dir_energy = dir_energy / (dir_energy.sum() + 1e-8)
    dir_peak_to_avg = float(dir_energy.max() / (dir_energy.mean() + 1e-8))

    ys, xs = np.where(peaks)
    if len(ys) >= 2:
        coords = np.stack([ys, xs], axis=1).astype(float)
        if len(coords) > 60:
            idx = np.random.RandomState(0).choice(len(coords), 60, replace=False)
            coords = coords[idx]
        diffs = coords[:, None, :] - coords[None, :, :]
        d = np.sqrt((diffs ** 2).sum(-1))
        iu = np.triu_indices(len(coords), k=1)
        spacings = d[iu]
        spacing_std = float(np.std(spacings)) if len(spacings) else 0.0
        spacing_mean = float(np.mean(spacings)) if len(spacings) else 0.0
    else:
        spacing_std, spacing_mean = 0.0, 0.0

    periodic_energy_ratio = float(search[peaks].sum() / (search.sum() + 1e-8)) if peak_count > 0 else 0.0

    feats = [
        peak_count, peak_strength, periodic_energy_ratio,
        dir_peak_to_avg, spacing_mean, spacing_std,
        float(mag.std()), float(mag.max() / (mag.mean() + 1e-8)),
    ]
    feats.extend(radial_energy.tolist())
    feats.extend(dir_energy.tolist())
    return np.array(feats, dtype=np.float32)


# ── Signal 3: LBP ─────────────────────────────────────────────────────
def _lbp_features(gray):
    g = gray.astype(np.uint8)
    radius, npoints = 1, 8
    lbp = local_binary_pattern(g, P=npoints, R=radius, method="uniform")
    nbins = npoints + 2
    hist, _ = np.histogram(lbp, bins=nbins, range=(0, nbins), density=True)
    hist = hist + 1e-12
    feats = hist.tolist()
    feats.append(_fast_entropy(hist))
    return np.array(feats, dtype=np.float32)


# ── Signal 4: noise residual ─────────────────────────────────────────
def _noise_features(gray):
    g = (gray / 255.0).astype(np.float32)
    k = 5
    local_mean = cv2.boxFilter(g, -1, (k, k), borderType=cv2.BORDER_REFLECT)
    local_sqmean = cv2.boxFilter(g * g, -1, (k, k), borderType=cv2.BORDER_REFLECT)
    local_var = np.maximum(local_sqmean - local_mean * local_mean, 0)
    noise_power = float(local_var.mean())
    gain = local_var / (local_var + noise_power + 1e-8)
    denoised = local_mean + gain * (g - local_mean)
    noise = g - denoised
    flat = noise.ravel()

    H, W = noise.shape
    bs = 16
    Hc, Wc = (H // bs) * bs, (W // bs) * bs
    if Hc >= bs and Wc >= bs:
        blocks = noise[:Hc, :Wc].reshape(Hc // bs, bs, Wc // bs, bs)
        local_vars = blocks.var(axis=(1, 3)).ravel()
    else:
        local_vars = np.array([0.0])

    hp = cv2.Laplacian((g * 255).astype(np.uint8), cv2.CV_64F)
    hp_hist, _ = np.histogram(hp.ravel(), bins=20, density=True)
    hp_hist = hp_hist + 1e-12

    odd_var = float(np.var(noise[::2, :]))
    even_var = float(np.var(noise[1::2, :]))
    col_odd = float(np.var(noise[:, ::2]))
    col_even = float(np.var(noise[:, 1::2]))

    return np.array([
        float(noise.std()), float(kurtosis(flat)), float(skew(flat)),
        float(np.mean(np.abs(flat))), float(np.percentile(np.abs(flat), 90)),
        odd_var / (even_var + 1e-8), col_odd / (col_even + 1e-8),
        float(local_vars.mean()), float(local_vars.std()),
        float(hp.std()), _fast_entropy(hp_hist),
    ], dtype=np.float32)


# ── Signal 5: chromatic aberration ───────────────────────────────────
def _chromatic_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    edge_mask = mag > np.percentile(mag, 70)

    r = img[:, :, 2].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 0].astype(np.float32)

    feats = []
    for c1, c2 in [(r, g), (g, b), (r, b)]:
        d = np.abs(c1 - c2)[edge_mask]
        if len(d) == 0:
            feats.extend([0.0, 0.0])
        else:
            feats.extend([float(d.mean()), float(d.std())])

    H, W = gray.shape
    cs = int(min(H, W) * 0.2)
    corners = [(slice(0, cs), slice(0, cs)), (slice(0, cs), slice(W-cs, W)),
               (slice(H-cs, H), slice(0, cs)), (slice(H-cs, H), slice(W-cs, W))]
    corner_diffs = []
    for sy, sx in corners:
        d = np.abs(r[sy, sx] - b[sy, sx])
        corner_diffs.append(d.mean())
    feats.append(float(np.mean(corner_diffs)))
    feats.append(float(np.std(corner_diffs)))

    return np.array(feats, dtype=np.float32)


# ── Signal 6: blur / sharpness ───────────────────────────────────────
def _sharpness_features(gray):
    g = gray.astype(np.uint8)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    lap_var = float(lap.var())

    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2))
    sobel_energy = float(np.sum(np.sqrt(gx**2 + gy**2)))

    edges = cv2.Canny(g, 50, 150)
    edge_density = float(edges.mean() / 255.0)

    mag = np.sqrt(gx**2 + gy**2)
    strong = mag > np.percentile(mag, 90)
    edge_width = float(1.0 / (strong.mean() + 1e-8))

    return np.array([lap_var, tenengrad, sobel_energy, edge_density, edge_width], dtype=np.float32)


# ── Signal 7: reflection / glare (runs on the small STATS image) ──────
def _glare_features(img_small):
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    highlight_mask = v > 240
    highlight_ratio = float(highlight_mask.mean())

    hist, _ = np.histogram(v, bins=32, range=(0, 255), density=True)
    hist = hist + 1e-12
    highlight_entropy = _fast_entropy(hist)

    clip_ratio = float((v >= 254).mean())

    mask_u8 = (highlight_mask.astype(np.uint8)) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    glare_areas = stats[1:, cv2.CC_STAT_AREA] if n_labels > 1 else np.array([0])
    glare_mask_area = float(glare_areas.max() / (img_small.shape[0] * img_small.shape[1])) if len(glare_areas) else 0.0

    sat_low = float((s < 30).mean())

    return np.array([highlight_ratio, highlight_entropy, clip_ratio, glare_mask_area, sat_low], dtype=np.float32)


# ── Signal 8: colour-space stats (runs on the small STATS image) ──────
def _colorspace_features(img_small):
    feats = []
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2LAB).astype(np.float32)
    ycc = cv2.cvtColor(img_small, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    bgr = img_small.astype(np.float32)

    for space in [bgr, hsv, lab, ycc]:
        for c in range(3):
            ch = space[:, :, c].ravel()
            feats.extend([float(ch.mean()), float(ch.std()), float(skew(ch))])

    b, g, r = cv2.split(bgr)
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2))

    corr_rg = float(np.corrcoef(r.ravel(), g.ravel())[0, 1])
    corr_gb = float(np.corrcoef(g.ravel(), b.ravel())[0, 1])

    feats.extend([colorfulness, corr_rg, corr_gb])
    return np.array(feats, dtype=np.float32)


def extract_features(img_full):
    """img_full: the raw cv2.imread() BGR image, no resizing applied yet.

    Returns a 194-dim float32 feature vector. The trained pipeline's own
    SelectKBest step picks the 18 features it actually uses internally —
    always pass the full 194-dim vector in, never pre-select.
    """
    img_tex = _resize_long_side(img_full, TEXTURE_LONG_SIDE)
    img_stats = _resize_long_side(img_full, STATS_LONG_SIDE)
    gray_tex = cv2.cvtColor(img_tex, cv2.COLOR_BGR2GRAY).astype(np.float64)

    f1 = _wavelet_features(gray_tex)
    f2 = _moire_features(gray_tex)
    f3 = _lbp_features(gray_tex)
    f4 = _noise_features(gray_tex)
    f5 = _chromatic_features(img_tex)
    f6 = _sharpness_features(gray_tex)
    f7 = _glare_features(img_stats)
    f8 = _colorspace_features(img_stats)
    return np.concatenate([f1, f2, f3, f4, f5, f6, f7, f8]).astype(np.float32)