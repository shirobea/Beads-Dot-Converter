"""拡大・再圧縮されたドット絵を元のドット解像度へ戻す（逆変換）。

仕様: 「拡大済みドット絵の逆変換（元解像度への復元）仕様 v2」

方針:
- 閾値やクラスタ数は固定値で持たず、画像から測ったノイズ量（σ_bg / σ_cell）・ピッチ・
  色の分布を基準に決める。固定で持つのは RestoreParams の無次元係数だけ。
- 重い前半（Step 0〜5: 判定・ピッチ・サンプリング・初期クラスタ）を analyze()、
  軽い後半（Step 6〜10: 統合・補正・切り抜き・検証）を finalize() に分けている。
  GUI で統合係数を動かしたときは finalize() だけをやり直せば済む。

仕様からの変更点:
- Step 8（混ざったマスの再判定）は Step 6-1 の直後に行う。中央サンプリングの取り違えを
  先に直してから、にじみ判定（6-2）と内部の穴（7）を判断するほうが筋がよいため。
- Step 7 の内部の小さな穴は「色を残して不透明にする」を既定にした（白い背景の画像で、目の
  ハイライトや線に囲まれた白が周りの色で塗りつぶされないように）。仕様どおり周囲の色で
  埋める動作は hole_mode="fill" で選べる。
- Step 3 のピッチの選択は、評価関数 E の代わりに「マスの中央をサンプリングして並べ直した
  ときの誤差」で行う。E はマスが小さいほど有利で、JPEG では半分のピッチを選びやすく、
  ぼやけた画像では差が出にくいため。E は各候補について計算してログに残す。
  半分のピッチの取り違えは、境界の並び（区切り番号 mod m ごとのエッジの強さ）で確かめて直す。
- 統合閾値 T は σ_cell（中央値）ではなく、前景のマス内ばらつきの二乗平均を基準にする。
  JPEG は平らな面がほぼ無傷で中央値が小さく出る一方、境界の近くに強いノイズが乗るため。
- 線形補間で拡大されたとみなした画像では Step 8（混ざったマスの多数決）を行わない。
  それ以外でも、多数決の色が今の色の 2 倍以上の面積を占め、統合閾値以上に違うときだけ直す。
- Step 6-2（にじみ）は YCbCr で判定する。JPEG は輝度をほぼ保ったまま色差だけをにじませるため。
  輝度が隣の色 A と一致し、色差が A と別の隣の色 B の間にあるマスを A に戻す。高彩度の色に
  接する低彩度のマスも調べる対象にする。補間（ぼかし）で拡大された画像は全チャンネルが
  混ざるので、仕様どおり Lab の線分で判定する。
- 8 近傍がすべて同じ色 A に囲まれ、自分の色が A と 2T 以内の孤立マスは A にする。
- マス数が前景の 0.5% 未満の極小クラスタは、最も近い色への統合候補として知らせる（自動では統合しない）。
- Step 6-3 の手動統合は、統合元の色のマスを統合先の色に置き換える（Step 7 の後に適用）。
- Step 10 の誤差は、マスの内側ではなく代表色を取ったサンプリング範囲で計算する。
- バイリニア補間・ぼかしで境界がはっきりしない画像も、警告を出したうえで処理を続ける。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from cv2_utils import require_cv2

ProgressFn = Optional[Callable[[float, str], None]]

SQRT3 = math.sqrt(3.0)
LOSSLESS_MAX_COLORS = 256  # Step 0: これ以下のユニーク色数なら可逆とみなす
FG_FLOOR = 10.0  # Step 1: 前景判定の下限（RGB ユークリッド距離）
LAB_FLOOR = 1.0  # σ が極端に小さい非可逆画像でも閾値が 0 にならないための下限（Lab）
REGION_MAX = 1024  # ピッチ探索に使う中央領域の最大辺（px）。大きい画像のメモリと時間を抑える
CELL_CAP = 40  # ログ用の評価関数 E で 1 軸あたりに使うマス数の上限（間引き）
MIN_GRID_STRENGTH = 0.1  # これ未満ならグリッドが見つからない（既に等倍）とみなす
BLUR_WARN_STRENGTH = 0.3  # これ未満なら境界がぼやけている旨を警告する
REFINE_MAX = 4  # Step 3: ピッチを詰める候補の数
PERIODICITY_RATIO = 0.5  # Step 3: 周期性の突出度が最大値のこの割合未満の候補は除く
LINE_COUNT = 32  # Step 3: ピッチ・原点の評価に使う行（列）の数
EDGE_SPLIT_RATIO = 0.45  # Step 3: 境界の並びで半分・倍のピッチを見分けるときの強さの比


@dataclass(frozen=True)
class RestoreParams:
    """逆変換の係数と上書き値。係数はすべて画像から測った値に掛けて使う。"""

    fg_coef: float = 3.0  # Step 1: σ_bg の何倍を前景とみなすか
    half_pitch_tol: float = 0.15  # Step 3: E_min に対する許容
    inner_margin: float = 0.2  # Step 3: マス内側の除外幅（ピッチ比）
    sample_ratio: float = 0.35  # Step 4: 代表色のサンプリング範囲（ピッチ比）
    merge_coef: float = 3.0  # Step 6-1: σ_cell の何倍未満を同じ色とみなすか
    minor_ratio: float = 0.02  # Step 6-2: にじみ判定で少数色とみなす割合
    hole_ratio: float = 0.01  # Step 7: 内部の穴とみなす最大面積（前景比）
    mixed_coef: float = 3.0  # Step 8: σ_cell の何倍を混ざったマスとみなすか
    island_ratio: float = 0.1  # Step 8: 前景と離れた小さな成分の最大面積（前景比）
    tiny_ratio: float = 0.005  # 統合候補として知らせる極小クラスタのマス数（前景比）
    verify_coef: float = 4.0  # Step 10: σ_cell の何倍を怪しいマスとみなすか
    max_clusters: int = 32  # Step 5: 初期クラスタ数の上限
    hole_mode: str = "keep"  # Step 7: keep（色を残して不透明にする）/ fill（周囲の色で埋める）
    background: str = "auto"  # Step 1: auto / on / off
    pitch: Optional[float] = None  # 上書き：ピッチ（px）
    origin_x: Optional[float] = None  # 上書き：グリッド原点（px）
    origin_y: Optional[float] = None
    # 手動統合: ((統合元RGB), (統合先RGB)) の並び
    manual_merges: tuple = ()

    def analysis_key(self) -> tuple:
        """analyze() の結果に影響する値。これが同じなら finalize() だけやり直せばよい。"""
        return tuple(getattr(self, name) for name in ANALYSIS_PARAM_NAMES)


# analyze()（Step 0〜5）に影響する RestoreParams の項目
ANALYSIS_PARAM_NAMES = (
    "fg_coef",
    "half_pitch_tol",
    "inner_margin",
    "sample_ratio",
    "max_clusters",
    "background",
    "pitch",
    "origin_x",
    "origin_y",
)


@dataclass
class Analysis:
    """Step 0〜5 の結果。"""

    rgb: np.ndarray
    lab: np.ndarray
    params: RestoreParams
    lossless: bool
    unique_colors: int
    bg_rgb: Optional[np.ndarray]
    bg_used: bool
    bg_confidence: float
    sigma_bg: float
    sigma_bg_lab: float
    fg_threshold: float
    bbox: tuple
    candidates: list
    pitch: float
    origin: tuple
    grid_strength: float
    e_min: float
    sigma_cell: float
    sigma_cell_lab: float
    sigma_noise_lab: float
    sigma_y: float  # 輝度 Y のマス内ばらつき（RMS、0-255）
    sigma_c: float  # 色差 Cb/Cr のマス内ばらつき（RMS、0-255）
    x_lo: np.ndarray
    x_hi: np.ndarray
    y_lo: np.ndarray
    y_hi: np.ndarray
    x_in: tuple
    x_s: tuple  # 代表色のサンプリング範囲（Step 4）
    y_s: tuple
    y_in: tuple
    rep_rgb: np.ndarray
    rep_lab: np.ndarray
    full_std_lab: np.ndarray
    labels0: np.ndarray
    centers0: np.ndarray
    next_candidate: Optional[tuple]
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    retry: Optional[dict] = None
    model: str = "nearest"  # 拡大の種類の推定（nearest / bilinear）

    @property
    def grid_shape(self) -> tuple:
        return (len(self.y_lo), len(self.x_lo))


@dataclass
class RestoreResult:
    """Step 6〜10 の結果。座標はすべて切り抜き後のマス単位。"""

    pixels: np.ndarray  # (h, w, 3) uint8 等倍
    alpha: np.ndarray  # (h, w) uint8 255=不透明
    labels: np.ndarray  # (h, w) palette の添字
    palette: list  # [{"rgb": (r,g,b), "count": n, "background": bool}]
    changes: list  # [{"x","y","reason","from","to"}]
    suspicious: list  # [(x, y)]
    cell_error: np.ndarray  # (h, w) Lab 平均距離
    mean_error: float
    merge_threshold: float
    crop: tuple  # 切り抜き左上のマス位置（グリッド座標、余白で -1 になることがある）
    preview_scale: int
    warnings: list

    @property
    def size(self) -> tuple:
        return (self.pixels.shape[1], self.pixels.shape[0])


# ---------------------------------------------------------------- 基本ユーティリティ


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB(0-255) → CIELAB（L: 0-100）。形はそのまま返す。"""
    cv2 = require_cv2()
    arr = np.asarray(rgb, dtype=np.float32)
    shape = arr.shape
    lab = cv2.cvtColor((arr.reshape(-1, 1, 3) / 255.0).astype(np.float32), cv2.COLOR_RGB2Lab)
    return lab.reshape(shape).astype(np.float32)


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """RGB(0-255) → YCbCr（JPEG と同じ BT.601 フルレンジ、0-255）。形はそのまま返す。"""
    a = np.asarray(rgb, dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def flatten_alpha(rgba: np.ndarray) -> np.ndarray:
    """透過付き画像を、絵に使われていない色の上に合成して RGB にする（透過＝背景）。"""
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    opaque = rgb[alpha[..., 0] > 0.5]
    keys = np.array(
        [(255, 0, 255), (0, 255, 0), (0, 255, 255), (255, 255, 255), (0, 0, 0), (255, 255, 0)],
        dtype=np.float32,
    )
    if opaque.size:
        sample = opaque[:: max(1, len(opaque) // 20000)]
        dist = np.array([np.min(np.linalg.norm(sample - k, axis=1)) for k in keys])
        key = keys[int(np.argmax(dist))]
    else:
        key = keys[0]
    out = rgb * alpha + key * (1.0 - alpha)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _rms_std(values: np.ndarray) -> float:
    """各チャンネルの分散の平均の平方根（チャンネルあたりの標準偏差）。"""
    if len(values) < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.var(values.astype(np.float64), axis=0))))


def _count_unique(rgb: np.ndarray) -> int:
    flat = rgb.reshape(-1, 3).astype(np.uint32)
    return int(np.unique((flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]).size)


def _nearest(points: np.ndarray, centers: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.empty(len(points), dtype=np.int64)
    for s in range(0, len(points), chunk):
        p = points[s : s + chunk]
        d = ((p[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        out[s : s + chunk] = np.argmin(d, axis=1)
    return out


def _report(progress: ProgressFn, value: float, message: str) -> None:
    if progress is not None:
        progress(value, message)


# ---------------------------------------------------------------- Step 1: 背景とノイズ量


def _estimate_background(rgb: np.ndarray, params: RestoreParams) -> dict:
    border = np.concatenate([rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]]).astype(np.float32)
    # 粗く量子化して最頻のビンを背景候補にする
    q = (border // 32).astype(np.int32)
    key = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    color = np.median(border[key == vals[int(np.argmax(counts))]], axis=0)
    near = np.linalg.norm(border - color, axis=1) < 40.0
    sigma = _rms_std(border[near])
    thr = max(params.fg_coef * sigma * SQRT3, FG_FLOOR)
    for _ in range(2):
        matched = np.linalg.norm(border - color, axis=1) < thr
        if not matched.any():
            break
        color = np.median(border[matched], axis=0)
        sigma = _rms_std(border[matched])
        thr = max(params.fg_coef * sigma * SQRT3, FG_FLOOR)
    matched = np.linalg.norm(border - color, axis=1) < thr
    confidence = float(matched.mean())
    sigma_lab = _rms_std(rgb_to_lab(border[matched])) if matched.sum() > 1 else 0.0
    if params.background == "on":
        used = True
    elif params.background == "off":
        used = False
    else:
        used = confidence >= 0.5
    return {
        "color": np.round(color).astype(np.float32),
        "confidence": confidence,
        "sigma": sigma,
        "sigma_lab": sigma_lab,
        "threshold": thr,
        "used": used,
    }


def _foreground_bbox(rgb: np.ndarray, bg: dict) -> Optional[tuple]:
    h, w = rgb.shape[:2]
    if not bg["used"]:
        return (0, 0, w, h)
    cv2 = require_cv2()
    diff = rgb.astype(np.float32) - bg["color"]
    fg = (np.sqrt((diff * diff).sum(axis=2)) > bg["threshold"]).astype(np.uint8)
    # 圧縮ノイズで散った 1px の点を除く（2×2 の開閉なのでピッチ 2 の線は残る）
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    rows = np.nonzero(fg.any(axis=1))[0]
    cols = np.nonzero(fg.any(axis=0))[0]
    if rows.size == 0:
        return None
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


# ---------------------------------------------------------------- Step 2: ピッチ候補


def _autocorr(profile: np.ndarray) -> np.ndarray:
    x = profile - profile.mean()
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / ac[0] if ac[0] > 0 else ac


def _comb_strength(profile: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """周期 p での「エッジが p ごとに並んでいる度合い」（非整数周波数の DFT 振幅）。"""
    w = profile - profile.mean()
    norm = np.linalg.norm(w)
    if norm <= 0:
        return np.zeros(len(periods))
    x = np.arange(len(w))
    phase = np.exp(-2j * np.pi * np.outer(1.0 / periods, x))
    return np.abs(phase @ w) / norm


def periodicity(profiles: tuple, periods: np.ndarray) -> np.ndarray:
    """縦横のエッジ列（1 次差分・2 次差分）がピッチ p ごとに並んでいる度合いの和。

    2 次差分は、線形補間で拡大された画像（1 次差分が階段状になり周期が出にくい）で
    ドットの中心の折れ目に鋭いピークが出る。
    """
    return sum(_comb_strength(prof, periods) for prof in profiles)


def prominence(profiles: tuple, periods: np.ndarray) -> np.ndarray:
    """周期性を、前後 5〜18% ずれた周期での周期性の中央値で割った値（格子の線の鋭さ）。"""
    rel = np.concatenate([np.linspace(0.82, 0.95, 6), np.linspace(1.05, 1.18, 6)])
    peak = periodicity(profiles, periods)
    around = periodicity(profiles, (periods[:, None] * rel[None, :]).ravel()).reshape(len(periods), -1)
    return peak / np.maximum(np.median(around, axis=1), 1e-9)


def _edge_profiles(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """隣接画素差を列ごと・行ごとに合計したエッジ強度プロファイル P_x, P_y。

    P_x[j] は列 j と j+1 の間の境界の強さ。
    """
    r = region.astype(np.int16)
    px = np.abs(np.diff(r, axis=1)).sum(axis=(0, 2)).astype(np.float64)
    py = np.abs(np.diff(r, axis=0)).sum(axis=(1, 2)).astype(np.float64)
    return px, py


def _curvature_profiles(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2 次差分の絶対値を列ごと・行ごとに合計したプロファイル。"""
    r = region.astype(np.int16)
    qx = np.abs(np.diff(r, n=2, axis=1)).sum(axis=(0, 2)).astype(np.float64)
    qy = np.abs(np.diff(r, n=2, axis=0)).sum(axis=(1, 2)).astype(np.float64)
    return qx, qy


def _pitch_candidates(profiles: tuple, max_lag: float) -> list[float]:
    base: list[float] = []
    for prof in profiles:
        if len(prof) < 8:
            continue
        ac = _autocorr(prof)
        hi = int(min(max_lag, len(ac) - 2))
        peaks = [
            lag for lag in range(2, hi + 1) if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0
        ]
        peaks.sort(key=lambda lag: ac[lag], reverse=True)
        for lag in peaks[:6]:
            a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den < 0 else 0.0
            base.append(lag + float(np.clip(off, -0.5, 0.5)))
    # 周期性（エッジ列の非整数周波数 DFT）のピークも候補にする。ぼやけた画像では
    # 自己相関のピークが鈍るが、基本周期の DFT 振幅は残りやすい
    if max_lag > 2.0:
        periods = np.geomspace(2.0, max_lag, int(np.ceil(np.log(max_lag / 2.0) / np.log(1.004))) + 2)
        score = periodicity(profiles, periods)
        # 絵柄そのものの低い周波数の成分は幅が広く、ドットの格子は鋭い線になるので、
        # 近くの周期の中央値で割った「突出度」でピークを選ぶ
        half = 24
        padded = np.pad(score, half, mode="edge")
        local = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)
        score = score / np.maximum(local, 1e-9)
        peaks = [i for i in range(1, len(score) - 1) if score[i] > score[i - 1] and score[i] >= score[i + 1]]
        peaks.sort(key=lambda i: score[i], reverse=True)
        base.extend(float(periods[i]) for i in peaks[:8])
    expanded: set[float] = set()
    for c in base:
        for k in (1, 2, 3, 4):
            expanded.add(c / k)
        expanded.add(c * 2)
        expanded.add(c * 3)
    cands = sorted(c for c in expanded if 2.0 <= c <= max_lag)
    # エッジ列の周期性が最大になる位置へ細かく寄せる（小数ピッチを正確に求める）
    refined = []
    rel = np.linspace(-0.03, 0.03, 61)
    for c in cands:
        periods = c * (1.0 + rel)
        periods = periods[periods >= 2.0]
        if periods.size == 0:
            continue
        score = periodicity(profiles, periods)
        i = int(np.argmax(score))
        refined.append((float(periods[i]), float(score[i])))
    # 近い候補はまとめ、周期性の最も強いものを残す
    out: list[tuple[float, float]] = []
    for c, sc in sorted(refined):
        if out and abs(c - out[-1][0]) / out[-1][0] <= 0.015:
            if sc > out[-1][1]:
                out[-1] = (c, sc)
        else:
            out.append((c, sc))
    return [c for c, _ in out]


# ---------------------------------------------------------------- Step 3: ピッチと原点の決定


class _RegionStats:
    """積分画像で任意の矩形の RGB 標準偏差を O(1) で求める。"""

    def __init__(self, img: np.ndarray) -> None:
        a = img.astype(np.float64)
        self.h, self.w = a.shape[:2]
        self.s1 = np.zeros((self.h + 1, self.w + 1, 3))
        self.s2 = np.zeros((self.h + 1, self.w + 1, 3))
        self.s1[1:, 1:] = a.cumsum(0).cumsum(1)
        self.s2[1:, 1:] = (a * a).cumsum(0).cumsum(1)

    def std(self, ylo, yhi, xlo, xhi) -> np.ndarray:
        def box(s):
            return s[np.ix_(yhi, xhi)] - s[np.ix_(ylo, xhi)] - s[np.ix_(yhi, xlo)] + s[np.ix_(ylo, xlo)]

        n = ((yhi - ylo)[:, None] * (xhi - xlo)[None, :])[..., None].astype(np.float64)
        m = box(self.s1) / n
        v = box(self.s2) / n - m * m
        # 不偏分散（画素数の違うマス同士を比べるため）
        v = v * (n / np.maximum(n - 1, 1))
        return np.sqrt(np.clip(v, 0, None).mean(axis=2))


def _axis_inner(p: float, o: float, length: int, margin: float, cap: Optional[int]):
    """領域に完全に収まるマスの内側範囲（整数）。"""
    n = int(np.floor((length - o) / p + 1e-9))
    if n < 1:
        return None
    idx = np.arange(n)
    if cap is not None and n > cap:
        idx = np.unique(np.round(np.linspace(0, n - 1, cap)).astype(np.int64))
    start = o + idx * p
    lo = np.ceil(start + margin - 1e-6).astype(np.int64)
    hi = np.floor(start + p - margin + 1e-6).astype(np.int64)
    # 内側が 1px だと標準偏差が常に 0 になり評価にならないので、小さいピッチはマス全体を使う
    small = hi - lo < 2
    lo = np.where(small, np.floor(start + 0.5).astype(np.int64), lo)
    hi = np.where(small, np.floor(start + p + 0.5).astype(np.int64), hi)
    lo = np.clip(lo, 0, length - 1)
    hi = np.clip(np.maximum(hi, lo + 1), 1, length)
    return lo, hi


def _grid_E(stats: _RegionStats, p: float, ox: float, oy: float, margin_ratio: float, cap=CELL_CAP) -> float:
    m = p * margin_ratio
    bx = _axis_inner(p, ox, stats.w, m, cap)
    by = _axis_inner(p, oy, stats.h, m, cap)
    if bx is None or by is None:
        return float("inf")
    return float(stats.std(by[0], by[1], bx[0], bx[1]).mean())


def _pick_lines(region: np.ndarray, n: int = LINE_COUNT) -> tuple[np.ndarray, np.ndarray]:
    """エッジの多い行と列を選ぶ。格子は縦横で独立なので、原点とピッチは 1 次元で評価できる。

    戻り値の列は (列数, 高さ, 3) に転置してあり、行と同じ関数で扱える。
    """
    r = region.astype(np.float32)
    row_e = np.abs(np.diff(r, axis=1)).sum(axis=(1, 2))
    col_e = np.abs(np.diff(r, axis=0)).sum(axis=(0, 2))
    rows = np.sort(np.argsort(row_e)[::-1][: min(n, len(row_e))])
    cols = np.sort(np.argsort(col_e)[::-1][: min(n, len(col_e))])
    return np.ascontiguousarray(r[rows]), np.ascontiguousarray(r[:, cols].transpose(1, 0, 2))


def _recon_error(lines: np.ndarray, p: float, o: float, ratio: float) -> tuple[float, float]:
    """マスの中央をサンプリングして並べ直した線と元の線の平均差（ニアレスト / 線形補間）。

    区切りがドット境界と一致すると、ニアレストで拡大された画像はニアレストで、
    補間で拡大された（ぼやけた）画像は線形補間でほぼ元どおりに戻る。
    半分のピッチでもほぼ同じ誤差になり、倍のピッチでは大きくなる。
    """
    length = lines.shape[1]
    n = int(np.floor((length - o) / p + 1e-9))
    if n < 3:
        return float("inf"), float("inf")
    centers = o + (np.arange(n) + 0.5) * p
    hw = int(round(p * ratio / 2.0)) if p >= 4.0 else 0
    cs = np.clip(np.floor(centers).astype(np.int64), 0, length - 1)
    if hw > 0:
        idx = np.clip(cs[:, None] + np.arange(-hw, hw + 1)[None, :], 0, length - 1)
        samples = np.median(lines[:, idx], axis=2)
    else:
        samples = lines[:, cs]
    # 線形補間のモデルでは、ドットの中心（折れ目）の値を左右それぞれの傾きから外挿して平均する。
    # 折れ目をまたいで内挿すると、山や谷の頂点が削れてしまうため
    i1 = np.clip(np.floor(centers - 0.5).astype(np.int64), 1, length - 3)
    dl = (centers - (i1 + 0.5))[None, :, None]
    left = lines[:, i1] + dl * (lines[:, i1] - lines[:, i1 - 1])
    right = lines[:, i1 + 1] - (1.0 - dl) * (lines[:, i1 + 2] - lines[:, i1 + 1])
    knots = (left + right) * 0.5
    x = np.arange(max(0, int(np.ceil(centers[0] - 0.5))), min(length, int(np.floor(centers[-1] - 0.5)) + 1))
    pos = x + 0.5
    ref = lines[:, x]
    ci = np.clip(np.floor((pos - o) / p).astype(np.int64), 0, n - 1)
    err_nn = float(np.abs(ref - samples[:, ci]).mean())
    t = (pos - centers[0]) / p
    k = np.clip(np.floor(t).astype(np.int64), 0, n - 2)
    f = np.clip(t - k, 0.0, 1.0)[None, :, None]
    rec = knots[:, k] * (1.0 - f) + knots[:, k + 1] * f
    err_bl = float(np.abs(ref - rec).mean())
    return err_nn, err_bl


def _fit_axis(lines: np.ndarray, p: float, ratio: float, fixed=None, around=None) -> tuple:
    """1 軸の原点を粗→細の 2 段階で探す。(誤差, 原点, 補間の種類, 構造の強さ) を返す。"""

    def ev(o: float) -> tuple[float, str]:
        a, b = _recon_error(lines, p, o % p, ratio)
        return (a, "nearest") if a <= b else (b, "bilinear")

    if fixed is not None:
        e, model = ev(fixed)
        return e, fixed % p, model, float("nan")
    med = None
    if around is None:
        step = max(1.0, p / 15.0)
        offs = np.arange(0.0, p - 1e-9, step)
        errs = np.array([ev(o)[0] for o in offs])
        if not np.isfinite(errs).any():
            return float("inf"), 0.0, "nearest", 0.0
        center, span = float(offs[int(np.argmin(errs))]), step
        med = float(np.median(errs[np.isfinite(errs)]))
    else:
        center, span = around, 0.5
    best = None
    for o in center + np.arange(-span, span + 1e-9, 0.25):
        e, model = ev(o)
        if best is None or e < best[0] - 1e-9:
            best = (e, o % p, model)
    strength = 1.0 - best[0] / med if med else float("nan")
    return best[0], best[1], best[2], strength


def _fit_pitch(lx, ly, p: float, ratio: float, fixed=None, around=None) -> dict:
    ex, ox, mx, sx = _fit_axis(lx, p, ratio, None if fixed is None else fixed[0], None if around is None else around[0])
    ey, oy, my, sy = _fit_axis(ly, p, ratio, None if fixed is None else fixed[1], None if around is None else around[1])
    # 縦横で判定が分かれたら、誤差の大きい軸の判定を採る
    model = mx if mx == my else (mx if ex >= ey else my)
    vals = [v for v in (sx, sy) if np.isfinite(v)]
    strength = float(np.mean(vals)) if vals else float("nan")
    return {"pitch": float(p), "err": (ex + ey) / 2.0, "origin_r": (ox, oy), "model": model, "strength": strength}


def _refine_pitch(lx, ly, cand: dict, ratio: float, fixed=None) -> dict:
    """候補のピッチを ±0.6% → ±0.12% の 2 段階で詰める（小数ピッチのずれの蓄積を防ぐ）。"""
    best = cand
    for span, steps in ((0.006, 9), (0.0015, 7)):
        base = best["pitch"]
        for p in base * (1.0 + np.linspace(-span, span, steps)):
            c = _fit_pitch(lx, ly, p, ratio, fixed=fixed, around=None if fixed is not None else best["origin_r"])
            if c["err"] < best["err"] - 1e-9:
                best = c
    best = dict(best)
    best["strength"] = cand["strength"]
    return best


def _edge_classes(prof: np.ndarray, q: float, o: float, m: int) -> Optional[np.ndarray]:
    """周期 q・原点 o の区切り位置のエッジ強度を、区切り番号 mod m ごとに平均する。

    区切り位置が小数のときはプロファイルを線形補間して読む（窓の幅を変えると、幅の広いほうが
    最大値を拾いやすくなり偏るため）。大きいピッチでは前後 r 画素の最大値をとってずれを吸収する。
    強度はプロファイルの下位 10%（ドットの内側＝ノイズの強さ）を差し引いた値。
    """
    length = len(prof)
    floor = float(np.percentile(prof, 10))
    i = np.arange(int((length + 1 - o) / q) + 1)
    pos = o + i * q - 1.0  # 区切り b は列 b-1 と b の間 → プロファイルの添字 b-1
    r = int(q * 0.1)
    keep = (pos - r >= 0) & (pos + r <= length - 1)
    i, pos = i[keep], pos[keep]
    if len(i) < 2 * m:
        return None
    vals = []
    for k in range(-r, r + 1):
        x = pos + k
        x0 = np.clip(np.floor(x).astype(np.int64), 0, length - 1)
        x1 = np.minimum(x0 + 1, length - 1)
        f = x - x0
        vals.append(prof[x0] * (1.0 - f) + prof[x1] * f)
    e = np.clip(np.max(vals, axis=0) - floor, 0, None)
    cls = i % m
    return np.array([e[cls == k].mean() for k in range(m)])


def _edge_fit(prof: np.ndarray, q: float) -> tuple[float, float]:
    """区切り位置のエッジ強度の平均が最大になるピッチ（q の ±0.4%）と原点。

    ピッチがわずかにずれていると、遠くの区切りほど境界から外れて判定が鈍るため、ピッチも合わせる。
    """
    best = (q, 0.0, -1.0)
    for qq in q * (1.0 + np.linspace(-0.004, 0.004, 9)):
        for o in np.arange(0.0, qq - 1e-9, 0.25):
            c = _edge_classes(prof, qq, o, 1)
            if c is not None and c[0] > best[2]:
                best = (float(qq), float(o), float(c[0]))
    return best[0], best[1]


def _edge_upgrade(profiles: tuple, p: float, max_pitch: float, notes: list) -> float:
    """境界が m 個おきの区切りにしか無ければ、本当のピッチは m·p（仕様 Step 3 の 2×2 確認の一般化）。

    1 次差分（境界）と 2 次差分（線形補間の折れ目）のどちらかで、縦横そろって偏っていれば上げる。
    """

    def split_ratio(pair: tuple, m: int) -> Optional[float]:
        ratios = []
        for prof in pair:
            qq, o = _edge_fit(prof, p)
            c = _edge_classes(prof, qq, o, m)
            if c is None or c.max() <= 0:
                return None
            k = int(np.argmax(c))
            ratios.append(np.delete(c, k).max() / c[k])
        return max(ratios)

    changed = True
    while changed:
        changed = False
        for m in range(2, 9):
            if p * m > max_pitch:
                continue
            ratios = [split_ratio(profiles[:2], m), split_ratio(profiles[2:], m)]
            ratios = [r for r in ratios if r is not None]
            if ratios and min(ratios) < EDGE_SPLIT_RATIO:
                notes.append(f"境界が {m} マスおきにしか無かったため、ピッチを {p:.3f} → {p * m:.3f}px にしました。")
                p *= m
                changed = True
                break
    return p


def _related(a: float, b: float) -> bool:
    """一方が他方のほぼ整数倍（半分・倍の関係）か。"""
    r = max(a, b) / min(a, b)
    return abs(r - round(r)) < 0.03


def _select_pitch(cands: list[dict], tol: float) -> tuple[Optional[dict], Optional[dict]]:
    """大きいピッチから順に、誤差 ≤ 最小誤差 × (1 + 許容) を満たす最大のものを選ぶ。

    次点は、採用したピッチと半分・倍の関係にない候補から選ぶ（Step 10 のやり直し用）。
    """
    valid = [c for c in cands if np.isfinite(c["err"])]
    if not valid:
        return None, None
    e_min = min(c["err"] for c in valid)
    limit = e_min * (1.0 + tol) + 0.05
    ok = sorted((c for c in valid if c["err"] <= limit), key=lambda c: c["pitch"], reverse=True)
    chosen = ok[0]
    # JPEG では細かい格子ほど圧縮ノイズまで再現して誤差が下がる。整数倍の大きいピッチの方が
    # 格子の線がはっきり鋭く（突出度が高く）、誤差もそれほど悪くなければ、そちらが本当のピッチ
    for c in sorted(valid, key=lambda c: c["pitch"], reverse=True):
        if (
            c["pitch"] > chosen["pitch"]
            and _related(c["pitch"], chosen["pitch"])
            and c.get("periodicity", 0.0) >= 1.3 * chosen.get("periodicity", 0.0)
            and c["err"] <= 1.5 * chosen["err"] + 0.05
        ):
            chosen = c
            break
    others = sorted(
        (c for c in valid if c is not chosen and not _related(c["pitch"], chosen["pitch"])), key=lambda c: c["err"]
    )
    return chosen, (others[0] if others else None)


# ---------------------------------------------------------------- Step 4: マスの区切りとサンプリング


def _grid_axis(p: float, o: float, length: int):
    """画像全体を覆うマスの区切り。端の欠けたマスは半分以上見えていれば含める。"""
    idx = np.arange(-1, int(np.ceil((length - o) / p)) + 1)
    start = o + idx * p
    end = start + p
    vis = np.minimum(end, length) - np.maximum(start, 0)
    keep = vis >= 0.5 * p - 1e-6
    start, end = start[keep], end[keep]
    lo = np.clip(np.round(start), 0, length).astype(np.int64)
    hi = np.clip(np.round(end), 0, length).astype(np.int64)
    hi = np.maximum(hi, np.minimum(lo + 1, length))
    lo = np.minimum(lo, hi - 1)
    return start, end, lo, hi


def _inner_bounds(start, end, length: int, margin: float):
    vs, ve = np.maximum(start, 0), np.minimum(end, length)
    lo = np.ceil(start + margin - 1e-6).astype(np.int64)
    hi = np.floor(end - margin + 1e-6).astype(np.int64)
    lo, hi = np.maximum(lo, np.floor(vs).astype(np.int64)), np.minimum(hi, np.ceil(ve).astype(np.int64))
    # 内側が 2px 未満になる小さいピッチは、マス全体を使う（1px では標準偏差が 0 になる）
    full_lo = np.clip(np.round(vs).astype(np.int64), 0, length - 1)
    full_hi = np.clip(np.round(ve).astype(np.int64), 1, length)
    small = hi - lo < 2
    lo = np.where(small, full_lo, lo)
    hi = np.where(small, np.maximum(full_hi, full_lo + 1), hi)
    return np.clip(lo, 0, length - 1), np.clip(hi, 1, length)


def _sample_bounds(start, end, length: int, p: float, ratio: float, full_lo, full_hi):
    if p < 4.0:
        return full_lo, full_hi
    s = max(1, int(round(p * ratio)))
    s = min(s, length)
    center = (np.maximum(start, 0) + np.minimum(end, length)) / 2.0
    lo = np.clip(np.round(center - s / 2.0).astype(np.int64), 0, length - s)
    return lo, lo + s


def _window_index(lo: np.ndarray, hi: np.ndarray):
    size = int(max(1, (hi - lo).max()))
    idx = lo[:, None] + np.arange(size)[None, :]
    valid = idx < hi[:, None]
    return np.minimum(idx, (hi - 1)[:, None]), valid


def _iter_windows(img, ylo, yhi, xlo, xhi, budget: int = 6_000_000):
    """マス単位の窓を行ごとにまとめて取り出す（メモリを抑えるため分割）。"""
    iy, vy = _window_index(ylo, yhi)
    ix, vx = _window_index(xlo, xhi)
    per_row = len(xlo) * iy.shape[1] * ix.shape[1] * img.shape[2]
    rows = max(1, budget // max(1, per_row))
    for r0 in range(0, len(ylo), rows):
        r1 = min(len(ylo), r0 + rows)
        win = img[iy[r0:r1][:, None, :, None], ix[None, :, None, :]]
        valid = vy[r0:r1][:, None, :, None] & vx[None, :, None, :]
        yield r0, r1, win, valid


def _cell_median(img, ylo, yhi, xlo, xhi) -> np.ndarray:
    out = np.empty((len(ylo), len(xlo), img.shape[2]), dtype=np.float32)
    for r0, r1, win, valid in _iter_windows(img, ylo, yhi, xlo, xhi):
        w = win.astype(np.float32)
        n, m = w.shape[0], w.shape[1]
        if valid.all():
            out[r0:r1] = np.median(w.reshape(n, m, -1, w.shape[-1]), axis=2)
        else:
            w = np.where(valid[..., None], w, np.nan)
            out[r0:r1] = np.nanmedian(w.reshape(n, m, -1, w.shape[-1]), axis=2)
    return out


def _cell_std(img, ylo, yhi, xlo, xhi) -> np.ndarray:
    out = np.empty((len(ylo), len(xlo)), dtype=np.float32)
    for r0, r1, win, valid in _iter_windows(img, ylo, yhi, xlo, xhi):
        w = win.astype(np.float32)
        v = valid[..., None].astype(np.float32)
        n = v.sum(axis=(2, 3))
        mean = (w * v).sum(axis=(2, 3)) / n
        var = (((w - mean[:, :, None, None, :]) ** 2) * v).sum(axis=(2, 3)) / n
        out[r0:r1] = np.sqrt(var.mean(axis=2))
    return out


def _cell_mean_dist(img, ylo, yhi, xlo, xhi, colors: np.ndarray) -> np.ndarray:
    """各マスの画素と colors[y, x] との平均距離。"""
    out = np.empty((len(ylo), len(xlo)), dtype=np.float32)
    for r0, r1, win, valid in _iter_windows(img, ylo, yhi, xlo, xhi):
        c = colors[r0:r1][:, :, None, None, :]
        d = np.sqrt(((win.astype(np.float32) - c) ** 2).sum(axis=-1))
        v = valid.astype(np.float32)
        out[r0:r1] = (d * v).sum(axis=(2, 3)) / v.sum(axis=(2, 3))
    return out


# ---------------------------------------------------------------- analyze（Step 0〜5）


def analyze(rgb: np.ndarray, params: RestoreParams = RestoreParams(), progress: ProgressFn = None) -> Analysis:
    """Step 0〜5: 入力の判定・背景・ピッチと原点・代表色・初期クラスタ。"""
    cv2 = require_cv2()
    rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    h, w = rgb.shape[:2]
    if h < 2 or w < 2:
        raise ValueError("画像が小さすぎます。")
    warnings: list[str] = []

    # Step 0: 可逆か非可逆か
    _report(progress, 0.02, "入力の種類を判定しています")
    unique = _count_unique(rgb)
    lossless = unique <= LOSSLESS_MAX_COLORS
    lab = rgb_to_lab(rgb)

    # Step 1: 背景とノイズ量
    _report(progress, 0.06, "背景色とノイズ量を推定しています")
    bg = _estimate_background(rgb, params)
    if params.background == "auto" and not bg["used"]:
        warnings.append(f"外周の背景色の一致率が低いため（{bg['confidence']:.0%}）、背景なしとして扱いました。")
    bbox = _foreground_bbox(rgb, bg)
    if bbox is None:
        bg["used"] = False
        bbox = (0, 0, w, h)
        warnings.append("背景と異なる部分が見つからなかったため、背景なしとして扱いました。")

    # Step 2〜3: ピッチと原点（前景の中央領域で探す）
    _report(progress, 0.12, "ドットのピッチを探しています")
    x0, y0, x1, y1 = bbox
    pad = 2
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    rx0 = x0 + max(0, (x1 - x0 - REGION_MAX) // 2)
    ry0 = y0 + max(0, (y1 - y0 - REGION_MAX) // 2)
    rx1, ry1 = min(x1, rx0 + REGION_MAX), min(y1, ry0 + REGION_MAX)
    region = rgb[ry0:ry1, rx0:rx1]
    fg_w, fg_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    max_lag = max(2.0, min(fg_w, fg_h) / 4.0)
    max_lag = min(max_lag, min(region.shape[0], region.shape[1]) / 3.0)
    max_pitch = min(fg_w, fg_h) / 2.0
    ratio = params.sample_ratio

    def to_region(ox: float, oy: float, p: float) -> tuple:
        return ((ox - rx0) % p, (oy - ry0) % p)

    def to_global(ox: float, oy: float, p: float) -> tuple:
        return ((ox + rx0) % p, (oy + ry0) % p)

    def fixed_for(p: float):
        if params.origin_x is None or params.origin_y is None:
            return None
        return to_region(float(params.origin_x), float(params.origin_y), p)

    px, py = _edge_profiles(region)
    profiles = (px, py, *_curvature_profiles(region))
    lx, ly = _pick_lines(region)
    if params.pitch is not None:
        pitch_list = [float(params.pitch)]
    else:
        pitch_list = _pitch_candidates(profiles, max_lag) if max_lag >= 2.0 else []
    # 周期性の弱い候補（倍のピッチや、ぼやけた画像での半分のピッチ、ノイズ周期）を先に除く。
    # シャープな画像では p, p/2, p/3 の周期性がほぼ等しく、ぼやけた画像では高調波（p/2 など）が
    # 弱まるので、どちらでも基本周期 p は残る
    per = prominence(profiles, np.array(pitch_list)) if pitch_list else np.zeros(0)
    if params.pitch is None and len(per):
        keep = per >= PERIODICITY_RATIO * per.max()
        pitch_list, per = [p for p, k in zip(pitch_list, keep) if k], per[keep]
    candidates: list[dict] = []
    for idx, p in enumerate(pitch_list):
        _report(progress, 0.12 + 0.35 * idx / max(1, len(pitch_list)), f"ピッチ候補 {p:.2f}px を評価しています")
        c = _fit_pitch(lx, ly, p, ratio, fixed=fixed_for(p))
        c["periodicity"] = float(per[idx])
        candidates.append(c)

    # 構造の弱い候補（原点を変えても誤差がほとんど変わらないもの）を除いてから選ぶ
    strong = candidates
    if params.pitch is None and candidates:
        top = max((c["strength"] for c in candidates if np.isfinite(c["strength"])), default=0.0)
        strong = [c for c in candidates if not np.isfinite(c["strength"]) or c["strength"] >= 0.5 * top]
        # 誤差の小さい候補だけピッチを詰める（小数ピッチのずれを防ぐ）
        finite = sorted(c["err"] for c in strong if np.isfinite(c["err"]))
        if finite:
            todo = sorted((c for c in strong if c["err"] <= 2.0 * finite[0] + 0.5), key=lambda c: c["err"])[:REFINE_MAX]
            for n_done, c in enumerate(todo):
                _report(progress, 0.47 + 0.15 * n_done / len(todo), "ピッチを詰めています")
                c.update(_refine_pitch(lx, ly, c, ratio, fixed_for(c["pitch"])))
    chosen, runner_up = _select_pitch(strong, params.half_pitch_tol)
    notes: list[str] = []
    if params.pitch is None and max((c["strength"] for c in candidates), default=0.0) < MIN_GRID_STRENGTH:
        chosen = None
    # 可逆画像を拡大したものなら、正しい格子では元どおりに戻る（構造がはっきり出る）はず
    if chosen is not None and params.pitch is None and lossless and chosen["strength"] < BLUR_WARN_STRENGTH:
        chosen = None
    if chosen is not None and params.pitch is None:
        # 半分のピッチを選んでいないか、境界の並び方で確かめる
        p_up = _edge_upgrade(profiles, chosen["pitch"], max_pitch, notes)
        if p_up != chosen["pitch"]:
            c = _fit_pitch(lx, ly, p_up, ratio, fixed=fixed_for(p_up))
            c = _refine_pitch(lx, ly, c, ratio, fixed_for(p_up))
            c["periodicity"] = float(prominence(profiles, np.array([c["pitch"]]))[0])
            candidates.append(c)
            chosen = c

    stats = _RegionStats(region)
    for c in candidates:
        c["E"] = _grid_E(stats, c["pitch"], *c["origin_r"], params.inner_margin)
        c["origin"] = to_global(*c["origin_r"], c["pitch"])
    del stats

    if chosen is None:
        pitch, origin, e_min, strength, model = 1.0, (0.0, 0.0), 0.0, 0.0, "nearest"
        if params.pitch is None:
            warnings.append("ドットの区切りが見つからないため、既に等倍として色の整理だけ行いました。")
    else:
        pitch, origin, e_min = chosen["pitch"], chosen["origin"], chosen["E"]
        strength, model = chosen["strength"], chosen["model"]
        if model == "bilinear":
            notes.append("補間（ぼかし）で拡大された画像とみなしました。")
        if params.pitch is None and np.isfinite(strength) and strength < BLUR_WARN_STRENGTH:
            warnings.append(
                "ドットの境界がはっきりしません（補間拡大やぼかしの可能性）。比較画像で結果を確認してください。"
            )

    def build_cells(p: float, ox: float, oy: float) -> dict:
        xs, xe, xlo, xhi = _grid_axis(p, ox, w)
        ys, ye, ylo, yhi = _grid_axis(p, oy, h)
        sxlo, sxhi = _sample_bounds(xs, xe, w, p, params.sample_ratio, xlo, xhi)
        sylo, syhi = _sample_bounds(ys, ye, h, p, params.sample_ratio, ylo, yhi)
        rep_rgb = _cell_median(rgb, sylo, syhi, sxlo, sxhi)
        m = p * params.inner_margin
        x_in = _inner_bounds(xs, xe, w, m)
        y_in = _inner_bounds(ys, ye, h, m)
        return {
            "x": (xlo, xhi),
            "y": (ylo, yhi),
            "x_in": x_in,
            "y_in": y_in,
            "x_s": (sxlo, sxhi),
            "y_s": (sylo, syhi),
            "rep_rgb": rep_rgb,
            "rep_lab": rgb_to_lab(rep_rgb),
        }

    # Step 4: 代表色
    _report(progress, 0.65, "マスの代表色を取り出しています")
    cells = build_cells(pitch, *origin)

    # σ_cell: 採用グリッドでのマス内標準偏差の中央値。背景のマスは白飛びなどでノイズが
    # 0 になりやすく、数が多いと中央値を引き下げてしまうので、前景のマスだけで測る
    in_std_rgb = _cell_std(rgb, *cells["y_in"], *cells["x_in"])
    in_std_lab = _cell_std(lab, *cells["y_in"], *cells["x_in"])
    measure = np.ones(in_std_rgb.shape, dtype=bool)
    if bg["used"]:
        diff = cells["rep_rgb"] - bg["color"]
        measure = np.sqrt((diff * diff).sum(axis=-1)) > bg["threshold"]
        if measure.sum() < 4:
            measure[:] = True
    sigma_cell = float(np.median(in_std_rgb[measure]))
    sigma_cell_lab = float(np.median(in_std_lab[measure]))
    # 統合の閾値には、マス内のばらつきの二乗平均（RMS）を使う。JPEG は平らな面がほぼ無傷で
    # 中央値が小さく出る一方、境界の近くのマスにだけ強いノイズとにじみが乗るため
    sigma_noise_lab = float(np.sqrt(np.mean(in_std_lab[measure].astype(np.float64) ** 2)))
    full_std_lab = _cell_std(lab, *cells["y"], *cells["x"])
    # にじみ判定（Step 6-2）用に、輝度と色差のノイズ量を別々に測る。JPEG は色差を強く間引く
    ycc = rgb_to_ycbcr(rgb)
    in_std_y = _cell_std(ycc[..., :1], *cells["y_in"], *cells["x_in"])[measure]
    in_std_c = _cell_std(ycc[..., 1:], *cells["y_in"], *cells["x_in"])[measure]
    del ycc
    sigma_y = float(np.sqrt(np.mean(in_std_y.astype(np.float64) ** 2)))
    sigma_c = float(np.sqrt(np.mean(in_std_c.astype(np.float64) ** 2)))

    # Step 5: 色のクラスタリング
    _report(progress, 0.8, "色をクラスタリングしています")
    rep_rgb = cells["rep_rgb"]
    rep_lab = cells["rep_lab"]
    flat_rgb = np.clip(np.round(rep_rgb), 0, 255).astype(np.uint8).reshape(-1, 3)
    uniq_rgb, inverse = np.unique(flat_rgb, axis=0, return_inverse=True)
    if lossless or len(uniq_rgb) <= 1:
        labels0 = inverse.reshape(rep_rgb.shape[:2])
        centers0 = rgb_to_lab(uniq_rgb.astype(np.float32))
    else:
        data = rep_lab.reshape(-1, 3).astype(np.float32)
        k = int(min(params.max_clusters, len(uniq_rgb)))
        rng = np.random.default_rng(0)
        sample = data if len(data) <= 60000 else data[rng.choice(len(data), 60000, replace=False)]
        cv2.setRNGSeed(12345)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.05)
        _, _, centers0 = cv2.kmeans(sample, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
        labels0 = _nearest(data, centers0).reshape(rep_rgb.shape[:2])
    _report(progress, 0.9, "仕上げています")

    # 次点は最終的なピッチと半分・倍の関係にない候補から（半分ピッチの補正後に選び直す）
    next_cand = None
    if runner_up is not None and pitch > 1.0 and _related(runner_up["pitch"], pitch):
        others = [c for c in strong if np.isfinite(c["err"]) and not _related(c["pitch"], pitch)]
        runner_up = min(others, key=lambda c: c["err"]) if others else None
    if runner_up is not None:
        next_cand = (runner_up["pitch"], *runner_up["origin"])
    return Analysis(
        rgb=rgb,
        lab=lab,
        params=params,
        lossless=lossless,
        unique_colors=unique,
        bg_rgb=bg["color"] if bg["used"] else None,
        bg_used=bool(bg["used"]),
        bg_confidence=bg["confidence"],
        sigma_bg=bg["sigma"],
        sigma_bg_lab=bg["sigma_lab"],
        fg_threshold=bg["threshold"],
        bbox=bbox,
        candidates=candidates,
        pitch=float(pitch),
        origin=(float(origin[0]), float(origin[1])),
        grid_strength=float(strength),
        e_min=float(e_min),
        sigma_cell=sigma_cell,
        sigma_cell_lab=sigma_cell_lab,
        sigma_noise_lab=sigma_noise_lab,
        sigma_y=sigma_y,
        sigma_c=sigma_c,
        x_lo=cells["x"][0],
        x_hi=cells["x"][1],
        y_lo=cells["y"][0],
        y_hi=cells["y"][1],
        x_in=cells["x_in"],
        x_s=cells["x_s"],
        y_s=cells["y_s"],
        y_in=cells["y_in"],
        rep_rgb=rep_rgb,
        rep_lab=rep_lab,
        full_std_lab=full_std_lab,
        labels0=np.asarray(labels0, dtype=np.int64),
        centers0=np.asarray(centers0, dtype=np.float32),
        next_candidate=next_cand,
        warnings=warnings,
        notes=notes,
        model=model,
    )


# ---------------------------------------------------------------- finalize（Step 6〜10）


def _merge_clusters(labels: np.ndarray, centers: np.ndarray, threshold: float) -> np.ndarray:
    """Lab 距離が閾値未満のクラスタ対を近い順に統合する（中心はマス数で重み付けして再計算）。"""
    k = len(centers)
    counts = np.bincount(labels.ravel(), minlength=k).astype(np.float64)
    alive = counts > 0
    c = centers.astype(np.float64).copy()
    parent = np.arange(k)
    if threshold > 0:
        while True:
            idx = np.nonzero(alive)[0]
            if len(idx) < 2:
                break
            sub = c[idx]
            d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=2))
            np.fill_diagonal(d, np.inf)
            a, b = np.unravel_index(int(np.argmin(d)), d.shape)
            if d[a, b] >= threshold:
                break
            i, j = idx[a], idx[b]
            c[i] = (c[i] * counts[i] + c[j] * counts[j]) / (counts[i] + counts[j])
            counts[i] += counts[j]
            counts[j] = 0
            alive[j] = False
            parent[j] = i
    roots = np.arange(k)
    for x in range(k):
        r = x
        while parent[r] != r:
            r = parent[r]
        roots[x] = r
    return roots[labels]


def _cluster_colors(labels: np.ndarray, rep_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ラベルを 0.. に詰め直し、各色の RGB（メンバーの代表色の中央値）を返す。"""
    used, compact = np.unique(labels, return_inverse=True)
    compact = compact.reshape(labels.shape)
    flat = rep_rgb.reshape(-1, 3)
    lab_flat = compact.ravel()
    colors = np.zeros((len(used), 3), dtype=np.float32)
    order = np.argsort(lab_flat, kind="stable")
    bounds = np.searchsorted(lab_flat[order], np.arange(len(used) + 1))
    for i in range(len(used)):
        colors[i] = np.median(flat[order[bounds[i] : bounds[i + 1]]], axis=0)
    return compact, np.clip(np.round(colors), 0, 255)


def _neighbors8(labels: np.ndarray) -> np.ndarray:
    """8 近傍のラベル（画像の外は -1）。(ny, nx, 8)"""
    pad = np.pad(labels, 1, constant_values=-1)
    h, w = labels.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    return np.stack([pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy, dx in offs], axis=-1)


def _neighbors_r2(labels: np.ndarray) -> np.ndarray:
    """2 マス先までの近傍（5×5 から自分を除く 24 マス）のラベル。画像の外は -1。"""
    pad = np.pad(labels, 2, constant_values=-1)
    h, w = labels.shape
    offs = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy, dx) != (0, 0)]
    return np.stack([pad[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w] for dy, dx in offs], axis=-1)


def _fix_blend(params, labels, centers, threshold, bg_id, record) -> np.ndarray:
    """Step 6-2（補間で拡大された画像）: 隣り合う 2 色 A, B の混色を近いほうの色に戻す。

    補間のぼけは全チャンネルを線形に混ぜるので、Lab で A と B を結ぶ線分の近く（T 未満）にある
    少数色・孤立したマスを対象にする（仕様どおりの判定）。
    """
    k = len(centers)
    if threshold <= 0 or not 3 <= k <= 64:
        return labels
    c = centers.astype(np.float64)
    a, b, x = c[None, :, None, :], c[None, None, :, :], c[:, None, None, :]
    ab = b - a
    den = (ab * ab).sum(axis=-1)
    t = np.where(den > 0, ((x - a) * ab).sum(axis=-1) / np.where(den > 0, den, 1), 0.0)
    seg = np.sqrt(((x - (a + np.clip(t, 0.0, 1.0)[..., None] * ab)) ** 2).sum(axis=-1))
    counts = np.bincount(labels.ravel(), minlength=k)
    minor = counts < params.minor_ratio * labels.size
    n4 = _neighbors8(labels)[..., [1, 3, 4, 6]]
    isolated = ~(n4 == labels[..., None]).any(axis=-1)
    cand = minor[labels] | isolated
    if bg_id is not None:
        cand &= labels != bg_id
    new = labels.copy()
    for y, xx in np.argwhere(cand):
        cur = labels[y, xx]
        around = sorted({int(v) for v in n4[y, xx] if v >= 0 and v != cur})
        best = None
        for i, ia in enumerate(around):
            for ib in around[i + 1 :]:
                d = seg[cur, ia, ib]
                if d < threshold and (best is None or d < best[0]):
                    best = (d, ia, ib)
        if best is not None:
            _, ia, ib = best
            da = np.linalg.norm(c[cur] - c[ia])
            db = np.linalg.norm(c[cur] - c[ib])
            new[y, xx] = ia if da <= db else ib
    moved = np.argwhere(new != labels)
    record(moved, labels[moved[:, 0], moved[:, 1]], new[moved[:, 0], moved[:, 1]], "smudge")
    return new


def _fix_chroma_bleed(an, params, labels, colors, bg_id, record) -> np.ndarray:
    """Step 6-2: 色差のにじみを YCbCr で見つけて、輝度の一致する隣の色に戻す。

    条件（どちらもノイズ量に係数を掛けた範囲で判定する）:
    - マスの輝度 Y が、隣のクラスタ A の Y と一致する
    - マスの色差 (Cb, Cr) が、A と別のクラスタ B を結ぶ線分の近く、しかも A 寄り（中点より手前）にあり、
      A からのずれが主に B の方向を向いている（横へのずれより B へ向かうずれのほうが大きい）
    - B は A より鮮やか（にじみの元）。この向きの条件が無いと、正しい色のマスを
      「にじんだ色と別の色の間にある」とみなして、にじんだ色の側へ倒してしまう
    A は 8 近傍から、にじみの元 B は 2 マス先まで（5×5）から探す。JPEG の色差は 2×2 画素に
    間引かれたうえ 8×8 のブロックで量子化されるので、1 マス先より遠くまで流れ込むことがある。
    調べるのは、少数色・4 近傍に同じ色が無いマス・2 マス以内に自分より鮮やかな色があるマス
    （黄色などの高彩度の色の色差が、隣の低彩度のマスへ流れ込むのが典型）。
    判定は各回の開始時のラベルで行い、3 回まで繰り返す（にじみの奥のマスは、手前のマスが
    直ってから判定できる）。1 つのマスを直すのは 1 回だけにして、行ったり来たりを防ぐ。
    """
    k = len(colors)
    if k < 2 or an.lossless:
        return labels
    ycc = rgb_to_ycbcr(colors).astype(np.float64)
    cell_ycc = rgb_to_ycbcr(an.rep_rgb).astype(np.float64)
    y_tol = max(params.merge_coef * an.sigma_y, 2.0)
    c_tol = max(params.merge_coef * an.sigma_c, 2.0)
    sat = np.hypot(ycc[:, 1] - 128.0, ycc[:, 2] - 128.0)
    total = labels.size
    original = labels.copy()
    fixed = np.zeros(labels.shape, dtype=bool)
    for _ in range(3):
        counts = np.bincount(labels.ravel(), minlength=k)
        minor = counts < params.minor_ratio * total
        neigh = _neighbors8(labels)
        far = _neighbors_r2(labels)
        n4 = neigh[..., [1, 3, 4, 6]]
        isolated = ~(n4 == labels[..., None]).any(axis=-1)
        nsat = np.where(far >= 0, sat[np.maximum(far, 0)], -1.0).max(axis=-1)
        vivid_next = nsat > sat[labels] + c_tol
        cand = (minor[labels] | isolated | vivid_next) & ~fixed
        if bg_id is not None:
            cand &= labels != bg_id
        new = labels.copy()
        for y, x in np.argwhere(cand):
            c = labels[y, x]
            cy = cell_ycc[y, x, 0]
            cc = cell_ycc[y, x, 1:]
            around = sorted({int(v) for v in neigh[y, x] if v >= 0 and v != c})
            sources = sorted({int(v) for v in far[y, x] if v >= 0 and v != c})
            best = None
            for a in around:
                dy = abs(cy - ycc[a, 0])
                if dy > y_tol:
                    continue
                pa = ycc[a, 1:]
                for b in sources:
                    if b == a or sat[b] <= sat[a] + c_tol:
                        continue
                    ab = ycc[b, 1:] - pa
                    den = float(ab @ ab)
                    t = float(np.clip(((cc - pa) @ ab) / den, 0.0, 1.0)) if den > 0 else 0.0
                    dc = float(np.hypot(*(cc - (pa + t * ab))))
                    along = t * np.sqrt(den)
                    if dc > c_tol or t > 0.5 or along < dc:
                        continue
                    score = dy / y_tol + dc / c_tol
                    if best is None or score < best[0]:
                        best = (score, a)
            if best is not None:
                new[y, x] = best[1]
        changed = new != labels
        if not changed.any():
            break
        fixed |= changed
        labels = new
    moved = np.argwhere(labels != original)
    record(moved, original[moved[:, 0], moved[:, 1]], labels[moved[:, 0], moved[:, 1]], "smudge")
    return labels


def _cell_votes(an: Analysis, cells: np.ndarray, centers_lab: np.ndarray) -> np.ndarray:
    """マス全体の画素を最も近い色に分類し、色ごとの画素数を返す（Step 8）。"""
    k = len(centers_lab)
    out = np.zeros((len(cells), k), dtype=np.int64)
    if len(cells) == 0:
        return out
    ylo, yhi = an.y_lo[cells[:, 0]], an.y_hi[cells[:, 0]]
    xlo, xhi = an.x_lo[cells[:, 1]], an.x_hi[cells[:, 1]]
    iy, vy = _window_index(ylo, yhi)
    ix, vx = _window_index(xlo, xhi)
    per = iy.shape[1] * ix.shape[1] * max(1, k)
    step = max(1, 4_000_000 // per)
    for s in range(0, len(cells), step):
        e = min(len(cells), s + step)
        win = an.lab[iy[s:e][:, :, None], ix[s:e][:, None, :]]
        valid = vy[s:e][:, :, None] & vx[s:e][:, None, :]
        d = ((win[..., None, :] - centers_lab[None, None, None, :, :]) ** 2).sum(axis=-1)
        near = np.argmin(d, axis=-1)
        n = e - s
        flat = (np.arange(n)[:, None, None] * k + near)[valid]
        out[s:e] = np.bincount(flat, minlength=n * k).reshape(n, k)
    return out


def finalize(an: Analysis, params: Optional[RestoreParams] = None) -> RestoreResult:
    """Step 6〜10: 統合・混ざったマス・にじみ・内部の穴・切り抜き・検証。"""
    cv2 = require_cv2()
    params = params or an.params
    ny, nx = an.grid_shape
    warnings = list(an.warnings)
    changes: list[dict] = []

    # Step 6-1: ノイズ由来の分裂を統合
    sigma = an.sigma_cell_lab
    threshold = params.merge_coef * max(sigma, an.sigma_noise_lab)
    if not an.lossless:
        threshold = max(threshold, LAB_FLOOR)
    labels = _merge_clusters(an.labels0, an.centers0, threshold)
    labels, colors = _cluster_colors(labels, an.rep_rgb)
    centers = rgb_to_lab(colors)

    bg_id = None
    if an.bg_used and an.bg_rgb is not None:
        bg_lab = rgb_to_lab(an.bg_rgb[None, :])[0]
        d = np.sqrt(((centers - bg_lab) ** 2).sum(axis=1))
        if d.min() < max(2.0 * threshold, 15.0):
            bg_id = int(np.argmin(d))

    def record(cells: np.ndarray, before: np.ndarray, after: np.ndarray, reason: str) -> None:
        for (y, x), a, b in zip(cells, before, after):
            if a != b:
                changes.append({"gx": int(x), "gy": int(y), "reason": reason, "from": int(a), "to": int(b)})

    # Step 8: 混ざったマス（マス全体の多数決で決め直す）
    mixed_thr = max(params.mixed_coef * sigma, 2.0)
    target = an.full_std_lab > mixed_thr
    if bg_id is not None:
        fg_mask = (labels != bg_id).astype(np.uint8)
        n_comp, comp, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
        if n_comp > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            main = 1 + int(np.argmax(areas))
            small = np.zeros(n_comp, dtype=bool)
            small[1:] = areas < params.island_ratio * fg_mask.sum()
            small[main] = False
            target |= small[comp]
    # 補間（ぼかし）で拡大された画像はマスのほとんどが混色なので、多数決はかえって隣の色を拾う
    if an.pitch > 1.0 and an.model != "bilinear" and target.any():
        cells = np.argwhere(target)
        before = labels[cells[:, 0], cells[:, 1]].copy()
        votes = _cell_votes(an, cells, centers)
        win = np.argmax(votes, axis=1)
        rows = np.arange(len(cells))
        # 中央のサンプルが明らかに外れたときだけ直す：多数決の色が今の色の 2 倍以上の面積を
        # 占め、しかも今の色と統合閾値以上に違う（似た色どうしの入れ替えはノイズなので行わない）
        gap = np.sqrt(((centers[win] - centers[before]) ** 2).sum(axis=1))
        accept = (votes[rows, win] >= 2 * votes[rows, before]) & (gap >= threshold)
        after = np.where(accept, win, before)
        labels[cells[:, 0], cells[:, 1]] = after
        record(cells, before, after, "mixed")

    # Step 6-2: にじみ。JPEG は輝度をほぼ保ったまま色差だけを間引いてにじませるので、YCbCr で
    # 「輝度は隣の色 A とノイズの範囲で一致し、色差は A と別の隣の色 B の間にある」マスを A に戻す
    if an.model == "bilinear":
        labels = _fix_blend(params, labels, centers, threshold, bg_id, record)
    else:
        labels = _fix_chroma_bleed(an, params, labels, colors, bg_id, record)

    # 孤立マス: 8 近傍がすべて同じ色 A で、自分の色が A と統合閾値の 2 倍以内なら A にする
    neigh = _neighbors8(labels)
    same = (neigh == neigh[..., :1]).all(axis=-1) & (neigh[..., 0] >= 0) & (neigh[..., 0] != labels)
    if same.any():
        cells = np.argwhere(same)
        before = labels[cells[:, 0], cells[:, 1]]
        after = neigh[cells[:, 0], cells[:, 1], 0]
        close = np.sqrt(((centers[before] - centers[after]) ** 2).sum(axis=1)) < 2.0 * threshold
        cells, before, after = cells[close], before[close], after[close]
        labels[cells[:, 0], cells[:, 1]] = after
        record(cells, before, after, "isolated")

    # Step 7: 内部の背景色マス
    transparent = np.zeros((ny, nx), dtype=bool)
    if bg_id is not None:
        mode = params.hole_mode
        bg_mask = (labels == bg_id).astype(np.uint8)
        n_comp, comp, stats, _ = cv2.connectedComponentsWithStats(bg_mask, connectivity=4)
        fg_area = int((labels != bg_id).sum())
        sx, sy = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
        sw, sh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
        area = stats[:, cv2.CC_STAT_AREA]
        touches = (sx == 0) | (sy == 0) | (sx + sw == nx) | (sy + sh == ny)
        real = touches | (area >= params.hole_ratio * fg_area)
        real[0] = False
        transparent = real[comp] & (bg_mask > 0)
        if mode == "fill":
            cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
            for i in np.nonzero(~real)[0]:
                if i == 0:
                    continue
                ya, yb = max(0, sy[i] - 1), min(ny, sy[i] + sh[i] + 1)
                xa, xb = max(0, sx[i] - 1), min(nx, sx[i] + sw[i] + 1)
                hole = (comp[ya:yb, xa:xb] == i).astype(np.uint8)
                ring = (cv2.dilate(hole, cross) > 0) & (hole == 0)
                vals = labels[ya:yb, xa:xb][ring]
                vals = vals[vals != bg_id]
                if vals.size == 0:
                    continue
                to = int(np.bincount(vals).argmax())
                cells = np.argwhere(hole > 0) + [ya, xa]
                labels[cells[:, 0], cells[:, 1]] = to
                record(cells, np.full(len(cells), bg_id), np.full(len(cells), to), "hole")

    # 手動統合（GUI で選んだ 2 色）: 統合元の色のマスを統合先の色にする
    for src, dst in params.manual_merges:
        pair = rgb_to_lab(np.array([src, dst], dtype=np.float32))
        cs = int(_nearest(pair[:1], centers)[0])
        cd = int(_nearest(pair[1:], centers)[0])
        if cs != cd:
            moved = np.argwhere(labels == cs)
            labels[labels == cs] = cd
            record(moved, np.full(len(moved), cs), np.full(len(moved), cd), "manual")
            if bg_id is not None and cd == bg_id:
                transparent[moved[:, 0], moved[:, 1]] = True

    # Step 9: 背景でないマスの範囲 + 1 マスの余白で切り抜く
    opaque = ~transparent
    if bg_id is not None and opaque.any() and transparent.any():
        rows = np.nonzero(opaque.any(axis=1))[0]
        cols = np.nonzero(opaque.any(axis=0))[0]
        gy0, gy1 = rows[0] - 1, rows[-1] + 2
        gx0, gx1 = cols[0] - 1, cols[-1] + 2
    else:
        gy0, gy1, gx0, gx1 = 0, ny, 0, nx
    pad_lab = np.pad(labels, 1, constant_values=bg_id if bg_id is not None else 0)
    pad_tr = np.pad(transparent, 1, constant_values=bg_id is not None)
    crop_labels = pad_lab[gy0 + 1 : gy1 + 1, gx0 + 1 : gx1 + 1]
    crop_tr = pad_tr[gy0 + 1 : gy1 + 1, gx0 + 1 : gx1 + 1]

    # Step 10: 検証（マス内側の画素と復元色の平均 Lab 距離）
    final_lab = centers[labels]
    # 境界付近のにじみを数えないよう、代表色を取ったサンプリング範囲だけで比べる
    err = _cell_mean_dist(an.lab, an.y_s[0], an.y_s[1], an.x_s[0], an.x_s[1], final_lab)
    crop_err = np.pad(err, 1)[gy0 + 1 : gy1 + 1, gx0 + 1 : gx1 + 1]
    in_grid = np.zeros((ny + 2, nx + 2), dtype=bool)
    in_grid[1:-1, 1:-1] = True
    crop_in = in_grid[gy0 + 1 : gy1 + 1, gx0 + 1 : gx1 + 1]
    judged = crop_in & ~crop_tr
    mean_error = float(crop_err[judged].mean()) if judged.any() else 0.0
    # 統合閾値 T 以内の差は「同じ色」とみなしているので、それより小さい差は怪しいとしない
    sus_thr = max(params.verify_coef * max(sigma, an.sigma_noise_lab), 1.5 * threshold, 3.0)
    sus = np.argwhere((crop_err > sus_thr) & judged)

    # パレットを詰め直す（使われている色だけ）
    used, final = np.unique(crop_labels, return_inverse=True)
    final = final.reshape(crop_labels.shape)
    remap = {int(u): i for i, u in enumerate(used)}
    pal_rgb = colors[used].astype(np.uint8)
    pixels = pal_rgb[final]
    alpha = np.where(crop_tr, 0, 255).astype(np.uint8)
    palette = []
    for i, u in enumerate(used):
        palette.append(
            {
                "rgb": tuple(int(v) for v in pal_rgb[i]),
                "count": int(((final == i) & ~crop_tr).sum()),
                "background": bg_id is not None and int(u) == bg_id,
                "merge_to": None,
            }
        )
    # 極小クラスタ（にじみ・孤立マスの処理のあとも残ったごく少ない色）は、最も近い色への
    # 統合候補として知らせる。本物の小さな模様の色かもしれないので自動では統合しない
    fg_total = sum(e["count"] for e in palette if not e["background"])
    pal_lab = rgb_to_lab(pal_rgb.astype(np.float32))
    others = [j for j, e in enumerate(palette) if not e["background"]]
    for i, entry in enumerate(palette):
        if entry["background"] or entry["count"] >= params.tiny_ratio * fg_total or len(others) < 2:
            continue
        cand = [j for j in others if j != i]
        d = np.sqrt(((pal_lab[cand] - pal_lab[i]) ** 2).sum(axis=1))
        entry["merge_to"] = int(cand[int(np.argmin(d))])
    # 同じマスが複数の段階で変わったら 1 件にまとめる（最初の色 → 最後の色。戻ったものは除く）
    per_cell: dict[tuple, dict] = {}
    for ch in changes:
        key = (ch["gx"], ch["gy"])
        if key in per_cell:
            per_cell[key].update(to=ch["to"], reason=ch["reason"])
        else:
            per_cell[key] = dict(ch)
    out_changes = []
    for ch in per_cell.values():
        if ch["from"] == ch["to"]:
            continue
        x, y = ch["gx"] - gx0, ch["gy"] - gy0
        if 0 <= x < final.shape[1] and 0 <= y < final.shape[0]:
            out_changes.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "reason": ch["reason"],
                    "from": tuple(int(v) for v in colors[ch["from"]]),
                    "to": tuple(int(v) for v in colors[ch["to"]]),
                }
            )
    h, w = final.shape
    scale = max(1, 1024 // max(w, h))
    result = RestoreResult(
        pixels=pixels,
        alpha=alpha,
        labels=final,
        palette=palette,
        changes=out_changes,
        suspicious=[(int(x), int(y)) for y, x in sus],
        cell_error=crop_err,
        mean_error=mean_error,
        merge_threshold=float(threshold),
        crop=(int(gx0), int(gy0)),
        preview_scale=int(scale),
        warnings=warnings,
    )
    return result


def restore(rgb: np.ndarray, params: RestoreParams = RestoreParams(), progress: ProgressFn = None):
    """analyze → finalize。全体の差が大きければ次点のピッチでもやり直して比べる（Step 10-3）。"""
    an = analyze(rgb, params, progress)
    res = finalize(an, params)
    noise = an.sigma_bg_lab if an.bg_used else an.sigma_cell_lab
    limit = max(4.0 * noise, 3.0)
    if params.pitch is None and an.next_candidate is not None and an.pitch > 1.0 and res.mean_error > limit:
        _report(progress, 0.92, "差が大きいため次点のピッチでも試しています")
        p2, ox2, oy2 = an.next_candidate
        an2 = analyze(rgb, replace(params, pitch=p2, origin_x=ox2, origin_y=oy2))
        an2.candidates = an.candidates
        an2.warnings = [w for w in an.warnings if "ピッチ" not in w]
        an2.params = params
        res2 = finalize(an2, params)
        info = {"first_pitch": an.pitch, "first_error": res.mean_error, "second_pitch": p2, "second_error": res2.mean_error}
        if res2.mean_error < res.mean_error * 0.8:
            info["adopted"] = "second"
            an, res = an2, res2
            res.warnings.append(f"採用したピッチの誤差が大きかったため、次点の {p2:.2f}px に切り替えました。")
        else:
            info["adopted"] = "first"
        an.retry = info
    _report(progress, 1.0, "完了")
    return an, res


# ---------------------------------------------------------------- Step 9: 出力（ログ JSON は書き出さない）


def upscale(pixels: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)


def rgba(res: RestoreResult) -> np.ndarray:
    return np.dstack([res.pixels, res.alpha])


def comparison_image(an: Analysis, res: RestoreResult, gap: int = 8) -> np.ndarray:
    """元画像のグリッド範囲と、復元結果を元のピッチで拡大したものを左右に並べる。"""
    ny, nx = an.grid_shape
    h, w = res.labels.shape
    gx0, gy0 = res.crop
    cx = np.clip(np.arange(gx0, gx0 + w), 0, nx - 1)
    cy = np.clip(np.arange(gy0, gy0 + h), 0, ny - 1)
    xa, xb = int(an.x_lo[cx[0]]), int(an.x_hi[cx[-1]])
    ya, yb = int(an.y_lo[cy[0]]), int(an.y_hi[cy[-1]])
    src = an.rgb[ya:yb, xa:xb]
    col = np.searchsorted(an.x_hi, np.arange(xa, xb), side="right")
    row = np.searchsorted(an.y_hi, np.arange(ya, yb), side="right")
    col = np.clip(col - gx0, 0, w - 1)
    row = np.clip(row - gy0, 0, h - 1)
    recon = res.pixels[row[:, None], col[None, :]]
    sep = np.full((src.shape[0], gap, 3), 128, dtype=np.uint8)
    return np.concatenate([src, sep, recon], axis=1)


# 個別に保存できる出力の種類: キー → (表示名, 既定のファイル名の末尾, 拡張子)
EXPORTS = {
    "transparent": ("ドット絵（透過 PNG）", "_dot", ".png"),
    "opaque": ("ドット絵（背景あり PNG）", "_dot_bg", ".png"),
    "preview": ("拡大プレビュー PNG", "_preview", ".png"),
    "compare": ("比較画像 PNG", "_compare", ".png"),
    "palette": ("パレット一覧 CSV", "_palette", ".csv"),
}


def palette_csv(res: RestoreResult) -> str:
    lines = ["index,R,G,B,hex,count,background"]
    for i, p in enumerate(res.palette):
        r, g, b = p["rgb"]
        lines.append(f"{i + 1},{r},{g},{b},#{r:02X}{g:02X}{b:02X},{p['count']},{int(p['background'])}")
    return "\n".join(lines) + "\n"


def save_export(an: Analysis, res: RestoreResult, kind: str, path: str | Path) -> Path:
    """EXPORTS の 1 種類だけを保存する。"""
    from PIL import Image

    path = Path(path)
    if kind == "transparent":
        Image.fromarray(rgba(res), "RGBA").save(path)
    elif kind == "opaque":
        Image.fromarray(res.pixels, "RGB").save(path)
    elif kind == "preview":
        Image.fromarray(upscale(res.pixels, res.preview_scale), "RGB").save(path)
    elif kind == "compare":
        Image.fromarray(comparison_image(an, res), "RGB").save(path)
    elif kind == "palette":
        path.write_text(palette_csv(res), encoding="utf-8-sig")
    else:
        raise ValueError(f"unknown export: {kind}")
    return path
