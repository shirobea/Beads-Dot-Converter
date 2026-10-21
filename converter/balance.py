"""色バランス補正：パレット置換で生じる明るさ・色味のズレを、入力側を修正して打ち消す。

最近傍置換は画素ごとには最適でも、パレットの色の偏りによって画像全体が暗く・黄色く・
くすんだりする。そこで「仮置換 → 結果を測定 → 目標の Lab を修正 → 再置換」を数回
繰り返し、置換結果の統計が入力に近づくようにする。修正量は元の入力から一定範囲に
制限するので、個々の画素の色が大きく変わることはない。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from color_spaces import lab_to_rgb_uint8, rgb_to_lab
from cv2_utils import require_cv2 as _require_cv2
from palette import BeadPalette
from .models import BalanceConfig

# (目標RGB, 進捗範囲) → パレットインデックス (H×W)
MapFn = Callable[[np.ndarray, tuple[float, float]], np.ndarray]

# 元の入力からの修正量の上限
MAX_L_OFFSET = 8.0
MAX_AB_OFFSET = 6.0
# 平均彩度の比率の制限
CHROMA_RATIO_RANGE = (0.8, 1.25)
# 平均彩度がこれ未満の画像（ほぼ無彩色）では彩度の比率を使わない
MIN_MEAN_CHROMA = 1.0
# 前回から変わった画素の割合がこれ未満なら収束とみなす
CONVERGE_RATIO = 0.005


def _global_step(
    target: np.ndarray, in_lab: np.ndarray, out_lab: np.ndarray, strength: float
) -> np.ndarray:
    """平均 L*・平均彩度・平均 a*b* のズレの strength 倍だけ目標を動かす。"""
    target = target.copy()
    in_flat = in_lab.reshape(-1, 3)
    out_flat = out_lab.reshape(-1, 3)
    target[..., 0] += strength * float(in_flat[:, 0].mean() - out_flat[:, 0].mean())

    in_chroma = float(np.hypot(in_flat[:, 1], in_flat[:, 2]).mean())
    out_chroma = float(np.hypot(out_flat[:, 1], out_flat[:, 2]).mean())
    ratio = 1.0
    if in_chroma >= MIN_MEAN_CHROMA and out_chroma >= MIN_MEAN_CHROMA:
        ratio = float(np.clip(in_chroma / out_chroma, *CHROMA_RATIO_RANGE))
    # 彩度の拡大で出力の平均 ab も ratio 倍に動くと見込み、その残りを色の偏りとして戻す。
    # 単純に両方足すと、全体が同じ色相の画像では同じズレを二重に補正してしまう
    cast = in_flat[:, 1:].mean(axis=0) - ratio * out_flat[:, 1:].mean(axis=0)
    target[..., 1:] = target[..., 1:] * (1.0 + strength * (ratio - 1.0)) + strength * cast
    return target


def _local_step(
    target: np.ndarray,
    in_lab: np.ndarray,
    out_lab: np.ndarray,
    strength: float,
    cfg: BalanceConfig,
) -> np.ndarray:
    """「自分 − 近傍平均」の L* 差が入力と出力でそろうよう目標 L* を動かす。"""
    cv2 = _require_cv2()
    sigma = max(0.1, float(cfg.local_sigma))
    in_l = np.ascontiguousarray(in_lab[..., 0], dtype=np.float32)
    out_l = np.ascontiguousarray(out_lab[..., 0], dtype=np.float32)
    in_detail = in_l - cv2.GaussianBlur(in_l, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    out_detail = out_l - cv2.GaussianBlur(out_l, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    # 入力側の差がノイズ程度の画素は、出力の段差を消す方向にも広げる方向にも触らない
    mask = np.abs(in_detail) >= float(cfg.noise_threshold)
    target = target.copy()
    target[..., 0] += np.where(mask, strength * (in_detail - out_detail), 0.0)
    return target


def _clamp_offset(target: np.ndarray, in_lab: np.ndarray) -> np.ndarray:
    """元の入力からの修正量を L* ±8、a*/b* ±6 に収める。"""
    offset = target - in_lab
    offset[..., 0] = np.clip(offset[..., 0], -MAX_L_OFFSET, MAX_L_OFFSET)
    offset[..., 1:] = np.clip(offset[..., 1:], -MAX_AB_OFFSET, MAX_AB_OFFSET)
    target = in_lab + offset
    target[..., 0] = np.clip(target[..., 0], 0.0, 100.0)
    return target


def balanced_index(
    base_rgb: np.ndarray,
    palette: BeadPalette,
    map_fn: MapFn,
    cfg: BalanceConfig,
    progress_range: tuple[float, float],
) -> np.ndarray:
    """補正ループを回し、最後の置換結果のパレットインデックス (H×W) を返す。

    最大 iterations 回置換する。前回から変わった画素が 0.5% 未満になったら打ち切る。
    """
    iterations = max(1, int(cfg.iterations))
    global_strength = max(0.0, min(1.0, float(cfg.global_strength)))
    local_strength = max(0.0, min(1.0, float(cfg.local_strength)))
    start, end = progress_range
    span = (end - start) / iterations

    in_lab = rgb_to_lab(base_rgb).astype(np.float32)
    target = in_lab.copy()
    prev: np.ndarray | None = None
    idx = None
    for i in range(iterations):
        # 進捗は回数で等分する。途中で打ち切った残りは呼び出し側の完了報告で埋まる
        idx = map_fn(lab_to_rgb_uint8(target), (start + span * i, start + span * (i + 1)))
        if prev is not None and float((idx != prev).mean()) < CONVERGE_RATIO:
            break
        if i == iterations - 1:
            break
        prev = idx
        out_lab = palette.lab_array[idx].astype(np.float32)
        if global_strength > 0:
            target = _global_step(target, in_lab, out_lab, global_strength)
        if local_strength > 0:
            target = _local_step(target, in_lab, out_lab, local_strength, cfg)
        target = _clamp_offset(target, in_lab)
    return idx
