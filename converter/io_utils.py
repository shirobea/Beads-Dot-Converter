"""I/O系ユーティリティ: 画像読み込みとサイズ計算を集約。"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from cv2_utils import require_cv2 as _require_cv2

Size = Tuple[int, int]


def _imread_unicode(path: str, flags: int) -> np.ndarray | None:
    """Unicodeパスでも安全に画像を読み込む。"""
    try:
        cv2 = _require_cv2()
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def _load_image_rgb(input_path: str) -> np.ndarray:
    """入力パスからRGB画像を読み込む。Unicodeパスも許容。"""
    cv2 = _require_cv2()
    image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        image_bgr = _imread_unicode(input_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("入力画像を読み込めませんでした。")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _load_normal_map_rgb(input_path: str) -> np.ndarray:
    """ノーマルマップをRGBで読み込む（グレースケールでも3ch化）。"""
    rgb = _load_image_rgb(input_path)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=2)
    if rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]
    return rgb


def _load_ao_map_gray(input_path: str) -> np.ndarray:
    """AOマップを0-1のグレースケールに変換して返す。"""
    rgb = _load_image_rgb(input_path)
    if rgb.ndim == 2:
        gray = rgb.astype(np.float32)
    else:
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        rgb_f = rgb.astype(np.float32)
        gray = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    return np.clip(gray / 255.0, 0.0, 1.0)


def _load_specular_map_gray(input_path: str) -> np.ndarray:
    """Specularマップを0-1のグレースケールに変換して返す。"""
    return _load_ao_map_gray(input_path)


def _load_displacement_map_gray(input_path: str) -> np.ndarray:
    """Displacementマップを0-1のグレースケールに変換して返す。"""
    cv2 = _require_cv2()
    raw = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raw = _imread_unicode(input_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError("Displacementマップを読み込めませんでした。")
    if raw.ndim == 2:
        gray = raw.astype(np.float32)
    else:
        if raw.shape[2] > 3:
            raw = raw[:, :, :3]
        raw_f = raw.astype(np.float32)
        gray = 0.299 * raw_f[:, :, 0] + 0.587 * raw_f[:, :, 1] + 0.114 * raw_f[:, :, 2]
    if np.issubdtype(gray.dtype, np.integer):
        max_val = float(np.iinfo(gray.dtype).max)
    else:
        max_val = float(np.max(gray)) if gray.size else 1.0
    if max_val <= 0.0:
        max_val = 1.0
    return np.clip(gray / max_val, 0.0, 1.0)


def _compute_resize(
    shape: Size, target: int | Size, keep_aspect: bool
) -> Size:
    """指定サイズへリサイズし、keep_aspectがTrueなら縦横比を維持する。"""
    h, w = shape
    if isinstance(target, Iterable):
        width, height = target  # type: ignore
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("幅・高さは1以上にしてください。")
        if keep_aspect:
            scale = min(width / w, height / h)
            # 丸めの結果が0にならないよう最低1ピクセルにする
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
        else:
            new_w, new_h = width, height
        return new_w, new_h

    short = float(target)
    if short <= 0:
        raise ValueError("短辺ピクセル数は正の値にしてください。")
    if w <= h:
        new_w = int(short)
        new_h = int(round(h / w * short))
    else:
        new_h = int(short)
        new_w = int(round(w / h * short))
    return max(1, new_w), max(1, new_h)


def _compute_hybrid_size(
    shape: Size, target_size: Size, long_side_cap: int = 1600, scale_percent: float = 100.0
) -> Size:
    """ハイブリッド用の中間リサイズサイズを算出（長辺上限をユーザー指定倍率で調整）。"""
    h, w = shape
    target_w, target_h = target_size
    long_side = max(w, h)
    # ユーザー指定の縮小率（%）で上限値をスケール
    scale_percent = max(1.0, min(100.0, float(scale_percent)))
    effective_cap = max(1, int(round(long_side_cap * (scale_percent / 100.0))))
    if long_side <= effective_cap:
        return w, h
    scale = effective_cap / long_side
    new_w = max(target_w, int(round(w * scale)))
    new_h = max(target_h, int(round(h * scale)))
    return max(1, new_w), max(1, new_h)
