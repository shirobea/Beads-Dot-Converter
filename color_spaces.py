"""Color space utilities for RGB, Lab, and Oklab."""

from __future__ import annotations

import numpy as np

from cv2_utils import require_cv2 as _require_cv2


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-255) to Lab (L in 0-100)."""
    arr = np.asarray(rgb, dtype=np.float32)
    original_shape = arr.shape
    flat = arr.reshape(-1, 3)
    cv2 = _require_cv2()
    # OpenCV expects BGR; convert and scale to 0-1 for correct range.
    bgr = flat[:, ::-1] / 255.0
    lab = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).reshape(-1, 3)
    return lab.reshape(original_shape)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0-1) to linear RGB."""
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-255) to XYZ (D65, scaled to 0-100)."""
    arr = np.asarray(rgb, dtype=np.float32)
    original_shape = arr.shape
    rgb_norm = arr.reshape(-1, 3) / 255.0
    linear = srgb_to_linear(rgb_norm)

    xyz = np.empty_like(linear)
    # sRGB(D65) -> XYZ の標準行列
    xyz[:, 0] = 0.4124564 * linear[:, 0] + 0.3575761 * linear[:, 1] + 0.1804375 * linear[:, 2]
    xyz[:, 1] = 0.2126729 * linear[:, 0] + 0.7151522 * linear[:, 1] + 0.0721750 * linear[:, 2]
    xyz[:, 2] = 0.0193339 * linear[:, 0] + 0.1191920 * linear[:, 1] + 0.9503041 * linear[:, 2]
    xyz *= 100.0
    return xyz.reshape(original_shape)


def rgb_to_hunter_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-255) to Hunter Lab (D65)."""
    arr = np.asarray(rgb, dtype=np.float32)
    original_shape = arr.shape
    xyz = rgb_to_xyz(arr).reshape(-1, 3)

    x = xyz[:, 0] / 95.047
    y = xyz[:, 1] / 100.0
    z = xyz[:, 2] / 108.883
    # D65白色点で正規化してHunter Labへ変換
    sqrt_y = np.sqrt(np.maximum(y, 0.0))
    denom = np.maximum(sqrt_y, 1e-6)
    l_val = 100.0 * sqrt_y
    a_val = 175.0 * (x - y) / denom
    b_val = 70.0 * (y - z) / denom

    hunter = np.stack([l_val, a_val, b_val], axis=1)
    return hunter.reshape(original_shape)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-255) to Oklab."""
    arr = np.asarray(rgb, dtype=np.float32)
    original_shape = arr.shape
    rgb_norm = arr.reshape(-1, 3) / 255.0
    linear = srgb_to_linear(rgb_norm)

    lms = np.empty_like(linear)
    lms[:, 0] = 0.4122214708 * linear[:, 0] + 0.5363325363 * linear[:, 1] + 0.0514459929 * linear[:, 2]
    lms[:, 1] = 0.2119034982 * linear[:, 0] + 0.6806995451 * linear[:, 1] + 0.1073969566 * linear[:, 2]
    lms[:, 2] = 0.0883024619 * linear[:, 0] + 0.2817188376 * linear[:, 1] + 0.6299787005 * linear[:, 2]

    lms_cbrt = np.cbrt(lms)

    oklab = np.empty_like(linear)
    oklab[:, 0] = 0.2104542553 * lms_cbrt[:, 0] + 0.7936177850 * lms_cbrt[:, 1] - 0.0040720468 * lms_cbrt[:, 2]
    oklab[:, 1] = 1.9779984951 * lms_cbrt[:, 0] - 2.4285922050 * lms_cbrt[:, 1] + 0.4505937099 * lms_cbrt[:, 2]
    oklab[:, 2] = 0.0259040371 * lms_cbrt[:, 0] + 0.7827717662 * lms_cbrt[:, 1] - 0.8086757660 * lms_cbrt[:, 2]

    return oklab.reshape(original_shape)


def _ciede2000_matrix(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000の距離行列をベクトル化して返す。"""
    lab1 = lab1.astype(np.float64, copy=False)
    lab2 = lab2.astype(np.float64, copy=False)
    l1 = lab1[:, 0][:, None]
    a1 = lab1[:, 1][:, None]
    b1 = lab1[:, 2][:, None]
    l2 = lab2[None, :, 0]
    a2 = lab2[None, :, 1]
    b2 = lab2[None, :, 2]

    avg_l = (l1 + l2) * 0.5
    c1 = np.sqrt(a1 ** 2 + b1 ** 2)
    c2 = np.sqrt(a2 ** 2 + b2 ** 2)
    avg_c = (c1 + c2) * 0.5
    g = 0.5 * (1 - np.sqrt((avg_c ** 7) / (avg_c ** 7 + 25.0 ** 7)))
    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = np.sqrt(a1p ** 2 + b1 ** 2)
    c2p = np.sqrt(a2p ** 2 + b2 ** 2)
    avg_cp = (c1p + c2p) * 0.5

    h1p = np.degrees(np.arctan2(b1, a1p))
    h2p = np.degrees(np.arctan2(b2, a2p))
    h1p = np.where(h1p < 0, h1p + 360.0, h1p)
    h2p = np.where(h2p < 0, h2p + 360.0, h2p)

    deltahp = h2p - h1p
    deltahp = np.where(deltahp > 180.0, deltahp - 360.0, deltahp)
    deltahp = np.where(deltahp < -180.0, deltahp + 360.0, deltahp)

    delta_lp = l2 - l1
    delta_cp = c2p - c1p
    delta_hp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(deltahp) * 0.5)

    avg_hp = (h1p + h2p) * 0.5
    avg_hp = np.where(np.abs(h1p - h2p) > 180.0, avg_hp + 180.0, avg_hp)
    avg_hp = np.where(avg_hp >= 360.0, avg_hp - 360.0, avg_hp)

    t = (
        1
        - 0.17 * np.cos(np.radians(avg_hp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * avg_hp))
        + 0.32 * np.cos(np.radians(3.0 * avg_hp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * avg_hp - 63.0))
    )

    sl = 1 + (0.015 * (avg_l - 50.0) ** 2) / np.sqrt(20.0 + (avg_l - 50.0) ** 2)
    sc = 1 + 0.045 * avg_cp
    sh = 1 + 0.015 * avg_cp * t

    delta_theta = 30.0 * np.exp(-((avg_hp - 275.0) / 25.0) ** 2)
    rc = 2.0 * np.sqrt((avg_cp ** 7) / (avg_cp ** 7 + 25.0 ** 7))
    rt = -np.sin(2.0 * np.radians(delta_theta)) * rc

    return np.sqrt(
        (delta_lp / sl) ** 2
        + (delta_cp / sc) ** 2
        + (delta_hp / sh) ** 2
        + rt * (delta_cp / sc) * (delta_hp / sh)
    )


def _ciede94_matrix(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE94距離の行列計算（グラフィックアーツ標準）。"""
    lab1 = lab1.astype(np.float64, copy=False)
    lab2 = lab2.astype(np.float64, copy=False)
    l1 = lab1[:, 0][:, None]
    a1 = lab1[:, 1][:, None]
    b1 = lab1[:, 2][:, None]
    l2 = lab2[None, :, 0]
    a2 = lab2[None, :, 1]
    b2 = lab2[None, :, 2]

    delta_l = l1 - l2
    c1 = np.sqrt(a1 ** 2 + b1 ** 2)
    c2 = np.sqrt(a2 ** 2 + b2 ** 2)
    delta_c = c1 - c2

    delta_e_sq = (l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2
    delta_h_sq = np.maximum(0.0, delta_e_sq - delta_c ** 2)

    k1 = 0.045
    k2 = 0.015
    s_l = 1.0
    s_c = 1.0 + k1 * c1
    s_h = 1.0 + k2 * c1

    return np.sqrt((delta_l / s_l) ** 2 + (delta_c / s_c) ** 2 + (delta_h_sq / (s_h ** 2)))


def _ciede76_matrix(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76距離の行列計算。"""
    lab1 = lab1.astype(np.float64, copy=False)
    lab2 = lab2.astype(np.float64, copy=False)
    diff = lab2[None, :, :] - lab1[:, None, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _cmc_matrix(lab1: np.ndarray, lab2: np.ndarray, l_weight: float, c_weight: float) -> np.ndarray:
    """CMC(l:c)をサンプル×パレットで一括計算。"""
    lab1 = lab1.astype(np.float64, copy=False)
    lab2 = lab2.astype(np.float64, copy=False)
    l_w = max(l_weight, 1e-6)
    c_w = max(c_weight, 1e-6)

    l1 = lab1[:, 0][:, None]
    a1 = lab1[:, 1][:, None]
    b1 = lab1[:, 2][:, None]
    l2 = lab2[None, :, 0]
    a2 = lab2[None, :, 1]
    b2 = lab2[None, :, 2]

    c1 = np.sqrt(a1 ** 2 + b1 ** 2)
    c2 = np.sqrt(a2 ** 2 + b2 ** 2)

    delta_l = l1 - l2
    delta_c = c1 - c2
    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_h_sq = np.maximum(0.0, delta_a ** 2 + delta_b ** 2 - delta_c ** 2)

    h1 = np.degrees(np.arctan2(b1, a1))
    h1 = np.where(h1 < 0, h1 + 360.0, h1)
    t = np.where(
        (h1 >= 164.0) & (h1 <= 345.0),
        0.56 + np.abs(0.2 * np.cos(np.radians(h1 + 168.0))),
        0.36 + np.abs(0.4 * np.cos(np.radians(h1 + 35.0))),
    )

    denom_f = c1 ** 4 + 1900.0
    f = np.sqrt((c1 ** 4) / denom_f, where=denom_f > 0, out=np.zeros_like(denom_f))

    s_l = np.where(l1 < 16.0, 0.511, (0.040975 * l1) / (1 + 0.01765 * l1))
    s_c = 0.0638 * c1 / (1 + 0.0131 * c1) + 0.638
    s_h = s_c * (f * t + 1 - f)

    return np.sqrt((delta_l / (l_w * s_l)) ** 2 + (delta_c / (c_w * s_c)) ** 2 + delta_h_sq / (s_h ** 2))


def lab_distance_matrix(lab1: np.ndarray, lab2: np.ndarray, metric: str = "CIEDE2000") -> np.ndarray:
    """Lab距離の行列を返す。"""
    metric_upper = str(metric).upper()
    if metric_upper == "CIE76":
        return _ciede76_matrix(lab1, lab2)
    if metric_upper == "CIE94":
        return _ciede94_matrix(lab1, lab2)
    return _ciede2000_matrix(lab1, lab2)


def cmc_distance_matrix(
    lab1: np.ndarray, lab2: np.ndarray, l_weight: float = 2.0, c_weight: float = 1.0
) -> np.ndarray:
    """CMC(l:c)距離の行列を返す。"""
    return _cmc_matrix(lab1, lab2, l_weight, c_weight)


def cmc_delta_e(lab_sample: np.ndarray, lab_array: np.ndarray, l_weight: float = 2.0, c_weight: float = 1.0) -> np.ndarray:
    """Vectorized CMC(l:c) between one sample and many targets."""
    sample = np.asarray(lab_sample, dtype=np.float64).reshape(1, 3)
    targets = np.asarray(lab_array, dtype=np.float64)
    return cmc_distance_matrix(sample, targets, l_weight=l_weight, c_weight=c_weight).reshape(-1)


def ciede76(lab_sample: np.ndarray, lab_array: np.ndarray) -> np.ndarray:
    """CIE76のΔE。"""
    sample = np.asarray(lab_sample, dtype=np.float64).reshape(1, 3)
    targets = np.asarray(lab_array, dtype=np.float64)
    return lab_distance_matrix(sample, targets, metric="CIE76").reshape(-1)


def ciede94(lab_sample: np.ndarray, lab_array: np.ndarray) -> np.ndarray:
    """CIE94のΔE（グラフィックアーツ標準）。"""
    sample = np.asarray(lab_sample, dtype=np.float64).reshape(1, 3)
    targets = np.asarray(lab_array, dtype=np.float64)
    return lab_distance_matrix(sample, targets, metric="CIE94").reshape(-1)


def ciede2000(lab_sample: np.ndarray, lab_array: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 between one sample and many targets.

    lab_sample: shape (3,)
    lab_array: shape (N, 3)
    """
    sample = np.asarray(lab_sample, dtype=np.float64).reshape(1, 3)
    targets = np.asarray(lab_array, dtype=np.float64)
    return lab_distance_matrix(sample, targets, metric="CIEDE2000").reshape(-1)
