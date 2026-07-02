"""パレットへの最短距離マッピングを担う軽量ヘルパー。"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple
import threading

import numpy as np

from color_spaces import (
    cmc_distance_matrix,
    lab_distance_matrix,
    rgb_to_hunter_lab,
    rgb_to_lab,
    rgb_to_oklab,
)
from palette import BeadPalette

ProgressCb = Callable[[float], None]
CancelEvent = threading.Event
CHUNK_SIZE = 2048  # 距離行列のメモリ肥大化を防ぐチャンクサイズ


def _report(progress_callback: ProgressCb | None, value: float, cancel_event: CancelEvent | None = None) -> None:
    """進捗更新とキャンセル判定をまとめて行う。"""
    if cancel_event and cancel_event.is_set():
        from . import ConversionCancelled  # 遅延インポートで循環を回避
        raise ConversionCancelled()
    if progress_callback:
        progress_callback(value)


def _chunk_ranges(length: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    """0-length区間を返さないチャンク分割のイテレータ。"""
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        yield start, end


def _compute_lab_distances(
    centers_lab: np.ndarray, palette_lab: np.ndarray, metric: str
) -> np.ndarray:
    """Lab系距離をチャンク単位で計算し、最小インデックスを返す。"""
    distances = lab_distance_matrix(centers_lab, palette_lab, metric)
    return np.argmin(distances, axis=1)


def _compute_cmc_distances(
    centers_lab: np.ndarray, palette_lab: np.ndarray, l_weight: float, c_weight: float
) -> np.ndarray:
    """CMC(l:c)距離をチャンク単位で計算し、最小インデックスを返す。"""
    distances = cmc_distance_matrix(centers_lab, palette_lab, l_weight, c_weight)
    return np.argmin(distances, axis=1)


def _map_centers_to_palette(
    centers: np.ndarray,
    palette: BeadPalette,
    mode: str,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.5, 0.8),
    cancel_event: CancelEvent | None = None,
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lab_metric: str = "CIEDE2000",
) -> np.ndarray:
    """中心色をビーズパレットへ最近傍対応付けする。"""
    mode_upper = mode.upper()
    mapping = np.zeros(len(centers), dtype=np.int32)
    total = len(centers)
    start, end = progress_range
    span = max(0.0, end - start)

    if mode_upper == "RGB":
        w_r, w_g, w_b = (float(w) for w in rgb_weights)
        weights = np.array([max(w_r, 1e-6), max(w_g, 1e-6), max(w_b, 1e-6)], dtype=np.float32)
        for idx0, idx1 in _chunk_ranges(total, CHUNK_SIZE):
            chunk = centers[idx0:idx1].astype(np.float32, copy=False)
            diff = chunk[:, None, :] - palette.rgb_array[None, :, :]
            weighted = diff * weights[None, None, :]
            distances = np.sum(weighted ** 2, axis=2)
            mapping[idx0:idx1] = np.argmin(distances, axis=1)
            _report(progress_callback, start + span * (idx1 / max(1, total)), cancel_event)
        return mapping
    elif mode_upper == "OKLAB":
        for idx0, idx1 in _chunk_ranges(total, CHUNK_SIZE):
            chunk = centers[idx0:idx1].astype(np.float32, copy=False)
            center_oklab = rgb_to_oklab(chunk)
            diff = center_oklab[:, None, :] - palette.oklab_array[None, :, :]
            distances = np.sum(diff ** 2, axis=2)
            mapping[idx0:idx1] = np.argmin(distances, axis=1)
            _report(progress_callback, start + span * (idx1 / max(1, total)), cancel_event)
        return mapping
    elif mode_upper in {"HUNTER LAB", "HUNTERLAB", "HUNTER"}:
        for idx0, idx1 in _chunk_ranges(total, CHUNK_SIZE):
            chunk = centers[idx0:idx1].astype(np.float32, copy=False)
            center_hunter = rgb_to_hunter_lab(chunk)
            diff = center_hunter[:, None, :] - palette.hunter_lab_array[None, :, :]
            distances = np.sum(diff ** 2, axis=2)
            mapping[idx0:idx1] = np.argmin(distances, axis=1)
            _report(progress_callback, start + span * (idx1 / max(1, total)), cancel_event)
        return mapping
    elif mode_upper.startswith("CMC"):
        center_lab = rgb_to_lab(centers)
        l_weight = max(float(cmc_l), 1e-6)
        c_weight = max(float(cmc_c), 1e-6)
        for idx0, idx1 in _chunk_ranges(total, CHUNK_SIZE):
            chunk = center_lab[idx0:idx1]
            mapping[idx0:idx1] = _compute_cmc_distances(chunk, palette.lab_array, l_weight, c_weight)
            _report(progress_callback, start + span * (idx1 / max(1, total)), cancel_event)
        return mapping
    elif mode_upper.startswith("LAB"):
        center_lab = rgb_to_lab(centers)
        metric = lab_metric.upper()
        for idx0, idx1 in _chunk_ranges(total, CHUNK_SIZE):
            chunk = center_lab[idx0:idx1]
            mapping[idx0:idx1] = _compute_lab_distances(chunk, palette.lab_array, metric)
            _report(progress_callback, start + span * (idx1 / max(1, total)), cancel_event)
        return mapping
    else:  # Lab + CIEDE2000 (後方互換)
        center_lab = rgb_to_lab(centers)
        for idx0, idx1 in _chunk_ranges(total, CHUNK_SIZE):
            chunk = center_lab[idx0:idx1]
            mapping[idx0:idx1] = _compute_lab_distances(chunk, palette.lab_array, "CIEDE2000")
            _report(progress_callback, start + span * (idx1 / max(1, total)), cancel_event)
        return mapping
