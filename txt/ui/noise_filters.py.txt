"""ノイズ除去フィルタの定義。"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV (cv2) が必要です。pip install opencv-python") from exc
    return cv2


def _apply_mode_filter_input(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    """入力RGB配列の各チャンネルにモードフィルタを適用する（前処理）。

    各ピクセルを近傍 kernel_size×kernel_size の最頻値で置換し、
    フラットな領域を均一化してから量子化することで色境界を明確にする。
    """
    try:
        from scipy import ndimage  # type: ignore
    except ImportError:
        return arr

    def _mode_func(values: np.ndarray) -> float:
        counts = np.bincount(values.astype(np.intp))
        return float(counts.argmax())

    result = arr.copy()
    for c in range(3):
        result[:, :, c] = ndimage.generic_filter(
            arr[:, :, c].astype(np.float64),
            _mode_func,
            size=kernel_size,
            mode="nearest",
        ).astype(np.uint8)
    return result


def _apply_island_removal_input(arr: np.ndarray, min_area: int) -> np.ndarray:
    """入力RGB配列から孤立した小さな色領域を除去する（前処理）。

    粗い量子化でカラーインデックス化し、面積 < min_area の連結領域を
    周囲の平均色で置換する。
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return arr

    # 各チャンネルを5ビット(32階調)に粗量子化してインデックス化
    quantized = (arr >> 3).astype(np.uint32)
    index_2d = (quantized[:, :, 0] << 10) | (quantized[:, :, 1] << 5) | quantized[:, :, 2]
    unique_indices = np.unique(index_2d)
    result = arr.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)

    for idx in unique_indices:
        mask = (index_2d == idx).astype(np.uint8)
        if not mask.any():
            continue
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label_id in range(1, n_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                island_mask = labels == label_id
                dilated = cv2.dilate(island_mask.astype(np.uint8), kernel)
                border = dilated.astype(bool) & ~island_mask
                if border.any():
                    replace_color = arr[border].mean(axis=0).astype(np.uint8)
                    result[island_mask] = replace_color
    return result


def build_noise_filter_registry(size: int, sigma: int = 75) -> dict[str, Callable[[Image.Image], Image.Image]]:
    cv2 = _require_cv2()
    return {
        "メディアン": lambda img: Image.fromarray(
            cv2.medianBlur(np.asarray(img.convert("RGB"), dtype=np.uint8), size)
        ),
        "ガウシアン": lambda img: Image.fromarray(
            cv2.GaussianBlur(np.asarray(img.convert("RGB"), dtype=np.uint8), (size, size), 0)
        ),
        # sigmaColor/sigmaSpace は両方同じ sigma 値を使用する（スライダーで調整可能）
        "バイラテラル": lambda img: Image.fromarray(
            cv2.bilateralFilter(
                np.asarray(img.convert("RGB"), dtype=np.uint8),
                size,
                sigmaColor=float(sigma),
                sigmaSpace=float(sigma),
            )
        ),
        "非局所的平均": lambda img: Image.fromarray(
            cv2.fastNlMeansDenoisingColored(
                np.asarray(img.convert("RGB"), dtype=np.uint8),
                None,
                h=8.0,
                hColor=10.0,
                templateWindowSize=size,
                searchWindowSize=max(7, size * 3),
            )
        ),
        # AI生成イラスト向け: 各チャンネル最頻値フィルタでフラット領域を均一化
        "モードフィルタ": lambda img: Image.fromarray(
            _apply_mode_filter_input(np.asarray(img.convert("RGB"), dtype=np.uint8), size)
        ),
        # AI生成イラスト向け: 粗量子化で色領域を分割し小アイランドを除去
        "アイランド除去": lambda img: Image.fromarray(
            _apply_island_removal_input(np.asarray(img.convert("RGB"), dtype=np.uint8), size * size)
        ),
    }
