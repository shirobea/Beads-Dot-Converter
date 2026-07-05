"""ディザリング手法を集約したモジュール。将来の手法追加はここだけ変更すればよい。"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from palette import BeadPalette
from .quantize import _report

ProgressCb = Callable[[float], None]
CancelEvent = threading.Event

DITHER_NONE     = "none"
DITHER_FS       = "floyd_steinberg"
DITHER_ATKINSON = "atkinson"
DITHER_BAYER4   = "bayer4"
DITHER_BAYER8   = "bayer8"

DITHER_SPECS = [
    {"label": "なし",            "method": DITHER_NONE},
    {"label": "Floyd-Steinberg", "method": DITHER_FS},
    {"label": "Atkinson",        "method": DITHER_ATKINSON},
    {"label": "Bayer 4x4",       "method": DITHER_BAYER4},
    {"label": "Bayer 8x8",       "method": DITHER_BAYER8},
]

# ---------------------------------------------------------------------------
# Bayer 行列（正規化済み: 0.0 〜 1.0）
# ---------------------------------------------------------------------------

_BAYER4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
], dtype=np.float64) / 16.0

_BAYER8 = np.array([
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float64) / 64.0


# ---------------------------------------------------------------------------
# 公開エントリポイント
# ---------------------------------------------------------------------------

def apply_dither(
    image_rgb: np.ndarray,
    palette: BeadPalette,
    method: str,
    strength: float = 1.0,
    mode: str = "Lab",
    lab_metric: str = "CIEDE2000",
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.3, 1.0),
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """methodに応じてディザリング処理を分岐する。量子化も内包する。

    strength : 誤差伝播の強度 (0.0=伝播なし / 1.0=通常)
    mode     : 変換モード文字列（"RGB" / "Oklab" / "Hunter" / "Lab" / "CMC(l:c)" 等）
               モードに応じた色空間で誤差拡散を行うため、モード変更が結果に反映される。
    """
    s = max(0.0, min(1.0, float(strength)))
    kw = dict(
        strength=s,
        mode=mode,
        lab_metric=lab_metric,
        rgb_weights=rgb_weights,
        cmc_l=cmc_l,
        cmc_c=cmc_c,
        progress_callback=progress_callback,
        progress_range=progress_range,
        cancel_event=cancel_event,
    )
    if method == DITHER_FS:
        return _floyd_steinberg(image_rgb, palette, **kw)
    if method == DITHER_ATKINSON:
        return _atkinson(image_rgb, palette, **kw)
    if method == DITHER_BAYER4:
        return _bayer(image_rgb, palette, matrix=_BAYER4, **kw)
    if method == DITHER_BAYER8:
        return _bayer(image_rgb, palette, matrix=_BAYER8, **kw)
    return image_rgb


# ---------------------------------------------------------------------------
# 内部ヘルパー：モード対応色空間への変換
# ---------------------------------------------------------------------------

def _prepare_dither_state(
    image_rgb: np.ndarray,
    palette: BeadPalette,
    mode: str,
    rgb_weights: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, str]:
    """入力画像とパレットをモード対応の色空間に変換する。

    戻り値:
      img_buf   : (H, W, 3) float64  誤差蓄積用作業バッファ
      pal_cs    : (N, 3)   float64  同一色空間のパレット
      space_tag : str  "rgb" | "oklab" | "hunter" | "lab"

    モードと色空間の対応：
      RGB        → RGB 空間（重み付きユークリッド）
      Oklab      → Oklab 空間（ユークリッド）
      Hunter     → Hunter Lab 空間（ユークリッド）
      Lab/CMC 系 → CIE Lab 空間（CIE76 ユークリッド）
    """
    H, W = image_rgb.shape[:2]
    flat = image_rgb.reshape(-1, 3).astype(np.float32)
    mu = mode.upper()

    if mu == "RGB":
        img_buf = flat.reshape(H, W, 3).astype(np.float64)
        pal_cs  = palette.rgb_array.astype(np.float64)
        return img_buf, pal_cs, "rgb"

    if mu == "OKLAB":
        from color_spaces import rgb_to_oklab
        img_buf = rgb_to_oklab(flat).reshape(H, W, 3).astype(np.float64)
        pal_cs  = palette.oklab_array.astype(np.float64)
        return img_buf, pal_cs, "oklab"

    if mu in {"HUNTER LAB", "HUNTERLAB", "HUNTER"}:
        from color_spaces import rgb_to_hunter_lab
        img_buf = rgb_to_hunter_lab(flat).reshape(H, W, 3).astype(np.float64)
        pal_cs  = palette.hunter_lab_array.astype(np.float64)
        return img_buf, pal_cs, "hunter"

    # Lab 系（CIE76 / CIE94 / CIEDE2000）および CMC → CIE Lab + CIE76
    # 誤差拡散ループ内では CIE76 ユークリッド距離を使用（速度優先）
    from color_spaces import rgb_to_lab
    img_buf = rgb_to_lab(flat).reshape(H, W, 3).astype(np.float64)
    pal_cs  = palette.lab_array.astype(np.float64)
    return img_buf, pal_cs, "lab"


def _metric_weights(space_tag: str, rgb_weights: tuple[float, float, float]) -> np.ndarray | None:
    """距離計算用のチャンネル重みを返す（RGB以外は重みなし）。"""
    if space_tag != "rgb":
        return None
    return np.array([max(float(rgb_weights[0]), 1e-6),
                     max(float(rgb_weights[1]), 1e-6),
                     max(float(rgb_weights[2]), 1e-6)], dtype=np.float64)


# ---------------------------------------------------------------------------
# 誤差拡散エンジン（波面分解によるベクトル化）
# ---------------------------------------------------------------------------
#
# 誤差拡散は各画素が左・上近傍の結果に依存する逐次処理だが、依存先は
# いずれも波面インデックス t = 2y + x が小さい側にある。したがって
# 同一 t の画素同士は独立で、波面単位でまとめてベクトル処理できる。
# （FS/Atkinson 両カーネルの全依存元について t の減少を確認済み）

def _diffuse_error_wavefront(
    img_buf: np.ndarray,
    pal_cs: np.ndarray,
    space_tag: str,
    rgb_weights: tuple[float, float, float],
    kernel: tuple[tuple[int, int], ...],
    kernel_weights: tuple[float, ...],
    err_scale: float,
    progress_callback: ProgressCb | None,
    progress_range: tuple[float, float],
    cancel_event: CancelEvent | None,
) -> np.ndarray:
    """波面分解で誤差拡散を実行し、パレットインデックス配列 (H×W) を返す。

    kernel         : 誤差の拡散先オフセット (dy, dx) の列（dy >= 0）
    kernel_weights : 各拡散先に掛ける係数
    err_scale      : 量子化誤差に掛ける全体係数（FS: strength / Atkinson: 1.0）
    """
    H, W = img_buf.shape[:2]
    w = _metric_weights(space_tag, rgb_weights)

    # 距離のargminは |c-p|² = |c|² - 2c·p + |p|² の |c|² を除いた
    # スコア c·p - ½Σp² の argmax と等価。matmul化で高速だが丸めが変わるため、
    # 整数入力×重みで厳密な等距離タイが起こり得るRGBモードのみ元の距離式を使い、
    # 従来実装とのタイブレーク（小さいインデックス優先）を厳密に保つ。
    pal_t = np.ascontiguousarray(pal_cs.T)  # (3, N)
    pal_half_sq = 0.5 * np.einsum("ij,ij->i", pal_cs, pal_cs)  # (N,)

    # 波面ごとの画素座標を前計算（t順の安定ソート → 波面内はy昇順）
    ys_all, xs_all = np.mgrid[0:H, 0:W]
    ys_all = ys_all.ravel()
    xs_all = xs_all.ravel()
    t_all = 2 * ys_all + xs_all
    order = np.argsort(t_all, kind="stable")
    ys_all = ys_all[order]
    xs_all = xs_all[order] + 1  # 誤差バッファのxオフセット（左パディング分）
    n_waves = 2 * (H - 1) + (W - 1) + 1
    bounds = np.searchsorted(t_all[order], np.arange(n_waves + 1))

    # 誤差バッファは拡散オフセットの最大範囲（dy≦2, -1≦dx≦2）分パディングし、
    # 境界チェックなしで散布できるようにする（パディング部への書き込みは捨てる）
    error = np.zeros((H + 2, W + 3, 3), np.float64)
    out_idx = np.empty((H, W), np.int32)
    start, end = progress_range
    span = max(0.0, end - start)

    for t in range(n_waves):
        i0, i1 = bounds[t], bounds[t + 1]
        ys = ys_all[i0:i1]
        xs = xs_all[i0:i1]
        cur = img_buf[ys, xs - 1] + error[ys, xs]  # (B, 3)

        # 波面内の全画素×全パレットの最近傍を一括計算
        if w is None:
            scores = cur @ pal_t  # (B, N)
            scores -= pal_half_sq
            idx = np.argmax(scores, axis=1)
        else:
            diff = (pal_cs[None, :, :] - cur[:, None, :]) * w[None, None, :]
            idx = np.argmin(np.einsum("bij,bij->bi", diff, diff), axis=1)
        out_idx[ys, xs - 1] = idx

        err = (cur - pal_cs[idx]) * err_scale  # (B, 3)
        for (dy, dx), kw in zip(kernel, kernel_weights):
            # 同一波面・同一オフセットの拡散先は重複しないため通常の加算で安全
            error[ys + dy, xs + dx] += err * kw

        if (t & 31) == 0:
            _report(progress_callback, start + span * (t + 1) / n_waves, cancel_event)

    _report(progress_callback, end, cancel_event)
    return out_idx


# ---------------------------------------------------------------------------
# Floyd-Steinberg
# ---------------------------------------------------------------------------

def _floyd_steinberg(
    image_rgb: np.ndarray,
    palette: BeadPalette,
    strength: float = 1.0,
    mode: str = "Lab",
    lab_metric: str = "CIEDE2000",
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.3, 1.0),
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """モード対応色空間でFloyd-Steinberg誤差拡散を実行する。

    誤差伝播カーネル（/ 16）：
           [ *  7 ]
       [ 3  5  1 ]

    strength で誤差量を全体スケール。0.0 → 最近傍量子化のみ、1.0 → 通常FS。
    """
    _report(progress_callback, progress_range[0], cancel_event)
    img_buf, pal_cs, space_tag = _prepare_dither_state(image_rgb, palette, mode, rgb_weights)
    out_idx = _diffuse_error_wavefront(
        img_buf, pal_cs, space_tag, rgb_weights,
        kernel=((0, 1), (1, -1), (1, 0), (1, 1)),
        kernel_weights=(7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0),
        err_scale=float(strength),
        progress_callback=progress_callback,
        progress_range=progress_range,
        cancel_event=cancel_event,
    )
    return palette.rgb_uint8[out_idx]


# ---------------------------------------------------------------------------
# Bayer（組織的ディザリング）
# ---------------------------------------------------------------------------

def _bayer(
    image_rgb: np.ndarray,
    palette: BeadPalette,
    matrix: np.ndarray,
    strength: float = 1.0,
    mode: str = "Lab",
    lab_metric: str = "CIEDE2000",
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.3, 1.0),
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """Bayer組織的ディザリング。

    誤差拡散ではなくしきい値比較によるオーダーディザー。
    ピクセル座標 (x % n, y % n) に対応する Bayer 行列値（0〜1）を
    strength でスケールしてピクセル値に加算し、パレット最近傍色を選択する。
    誤差拡散のような逐次性がないため、全画素を一括でベクトル化して処理する。

    strength : オフセット量の倍率 (0.0=ディザなし / 1.0=通常)
    """
    start, end = progress_range
    _report(progress_callback, start, cancel_event)

    H, W = image_rgb.shape[:2]
    n = matrix.shape[0]  # 4 or 8

    img_buf, pal_cs, space_tag = _prepare_dither_state(image_rgb, palette, mode, rgb_weights)

    # Bayer 行列を (H, W) 全面にタイリングし、RGB 各 ch に適用できる形に拡張
    # オフセットの値域は色空間依存で異なるため、色空間ごとにスケールする
    # RGB:      0〜255 → 最大オフセットを 32 程度（strength=1.0）
    # Lab:      L 0〜100, ab ±128 程度 → 最大 16 程度
    # Oklab:    0〜1 程度 → 最大 0.08 程度
    # Hunter:   L 0〜100 程度 → Lab と同スケール
    _SCALE = {
        "rgb":    32.0,
        "lab":    16.0,
        "oklab":  0.08,
        "hunter": 16.0,
    }
    scale = _SCALE.get(space_tag, 16.0) * float(strength)

    # (H, W) タイル → (H, W, 1) にブロードキャスト
    tile_y = np.arange(H) % n
    tile_x = np.arange(W) % n
    bayer_tile = matrix[np.ix_(tile_y, tile_x)][:, :, np.newaxis]  # (H, W, 1)

    # オフセットを -0.5〜+0.5 中心に変換してスケール
    offset = (bayer_tile - 0.5) * scale  # (H, W, 1) ブロードキャスト

    # オフセット済みの全画素をチャンク単位で最近傍量子化する
    flat = (img_buf + offset).reshape(-1, 3)
    if space_tag == "rgb":
        w = np.array([max(float(rgb_weights[0]), 1e-6),
                      max(float(rgb_weights[1]), 1e-6),
                      max(float(rgb_weights[2]), 1e-6)], dtype=np.float64)
        flat = flat * w
        pal_cs = pal_cs * w
    total = flat.shape[0]
    out_idx = np.empty(total, np.int32)
    span = max(0.0, end - start)
    chunk_size = 4096
    for idx0 in range(0, total, chunk_size):
        idx1 = min(total, idx0 + chunk_size)
        diff = flat[idx0:idx1, None, :] - pal_cs[None, :, :]
        out_idx[idx0:idx1] = np.argmin(np.einsum("ijk,ijk->ij", diff, diff), axis=1)
        _report(progress_callback, start + span * (idx1 / max(1, total)), cancel_event)

    _report(progress_callback, end, cancel_event)
    return palette.rgb_uint8[out_idx.reshape(H, W)]


# ---------------------------------------------------------------------------
# Atkinson
# ---------------------------------------------------------------------------

def _atkinson(
    image_rgb: np.ndarray,
    palette: BeadPalette,
    strength: float = 1.0,
    mode: str = "Lab",
    lab_metric: str = "CIEDE2000",
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.3, 1.0),
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """モード対応色空間でAtkinson誤差拡散を実行する。

    誤差伝播カーネル（各 1/8、合計 6/8 = 75% を拡散）：
           [ *  1  1 ]
       [ 1  1  1     ]
           [ 1       ]  ← 2行下

    Floyd-Steinberg との違い：
      - 誤差を 75% しか拡散しない（残り 25% は捨てる）
      - 拡散先が 6 ピクセルに広がる
      → べた塗り部分がきれいに保たれ、グラデーション部分だけにパターンが出る
    """
    _report(progress_callback, progress_range[0], cancel_event)
    img_buf, pal_cs, space_tag = _prepare_dither_state(image_rgb, palette, mode, rgb_weights)
    frac = float(strength) / 8.0  # 各拡散先の係数
    out_idx = _diffuse_error_wavefront(
        img_buf, pal_cs, space_tag, rgb_weights,
        kernel=((0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)),
        kernel_weights=(frac,) * 6,
        err_scale=1.0,
        progress_callback=progress_callback,
        progress_range=progress_range,
        cancel_event=cancel_event,
    )
    return palette.rgb_uint8[out_idx]
