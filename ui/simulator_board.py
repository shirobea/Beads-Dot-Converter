"""立体アイロンビーズの盤面データ（OpenGL 非依存）。

ビーズを (layer, row, col) の3次元グリッドで保持する。値はパレットindex、
-1 は空。描画・保存・設計図出力はいずれもこのグリッドだけを入力とするため、
GLコンテキスト無しで単体確認できる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cv2_utils import require_cv2 as _require_cv2

EMPTY = -1
MAX_LAYERS = 64  # 1作品あたりの層数上限（メモリとUIの都合で頭打ちにする）
PLATE_PEGS = 29  # 1プレートあたりのペグ数（29x29）
MAX_PLATES = 100  # 立体設計を開けるプレート枚数の上限（描画負荷の頭打ち）


def plate_count(height: int, width: int) -> int:
    """画像サイズに必要なプレート枚数（29x29単位で切り上げ）。"""
    if height <= 0 or width <= 0:
        return 0
    return max(1, math.ceil(height / PLATE_PEGS)) * max(1, math.ceil(width / PLATE_PEGS))

# 自動ドームの外殻カーブ
DOME_PROFILES = ("hemisphere", "parabola", "cone")
DOME_EDGE_HEIGHT = 2  # 背景に接する縁の高さ（最下層から2層目）
# 自動ドームの背景判定: 四隅の色 / ユーザー指定の色 / 1層目の最多色
BACKGROUND_MODES = ("corners", "manual", "dominant")
# キャラに囲まれた背景色のかたまりを背景とみなす最小面積（キャラ全体の面積に対する比）
DEFAULT_HOLE_RATIO = 0.02


class BeadBoard:
    """立体作品のビーズ配置。"""

    def __init__(self, height: int, width: int, layers: int = 1) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("盤面サイズが不正です。")
        layers = max(1, min(int(layers), MAX_LAYERS))
        self.grid = np.full((layers, int(height), int(width)), EMPTY, dtype=np.int16)

    @classmethod
    def from_target(cls, target: np.ndarray) -> "BeadBoard":
        """目標グリッド (H, W) と同じ平面サイズの空盤面を作る。"""
        return cls(target.shape[0], target.shape[1])

    # --- 形状 ---

    @property
    def layer_count(self) -> int:
        return int(self.grid.shape[0])

    @property
    def height(self) -> int:
        return int(self.grid.shape[1])

    @property
    def width(self) -> int:
        return int(self.grid.shape[2])

    def in_plane(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def ensure_layer(self, layer: int) -> bool:
        """指定の層を使える状態にする。上限を超える場合は False。"""
        if layer < 0 or layer >= MAX_LAYERS:
            return False
        if layer < self.layer_count:
            return True
        extra = layer + 1 - self.layer_count
        pad = np.full((extra, self.height, self.width), EMPTY, dtype=np.int16)
        self.grid = np.concatenate([self.grid, pad], axis=0)
        return True

    # --- 配置操作 ---

    def get(self, layer: int, row: int, col: int) -> int:
        if not (0 <= layer < self.layer_count) or not self.in_plane(row, col):
            return EMPTY
        return int(self.grid[layer, row, col])

    def place(self, layer: int, row: int, col: int, index: int) -> bool:
        """ビーズを置く。実際に変化したときだけ True。"""
        if index < 0 or not self.in_plane(row, col):
            return False
        if not self.ensure_layer(layer):
            return False
        if self.grid[layer, row, col] == index:
            return False
        self.grid[layer, row, col] = index
        return True

    def remove(self, layer: int, row: int, col: int) -> bool:
        """ビーズを取り除く。実際に変化したときだけ True。"""
        if not (0 <= layer < self.layer_count) or not self.in_plane(row, col):
            return False
        if self.grid[layer, row, col] < 0:
            return False
        self.grid[layer, row, col] = EMPTY
        return True

    def clear(self) -> None:
        self.grid.fill(EMPTY)

    def clear_layer(self, layer: int) -> bool:
        if not (0 <= layer < self.layer_count):
            return False
        if not (self.grid[layer] >= 0).any():
            return False
        self.grid[layer] = EMPTY
        return True

    def fill_layer(self, layer: int, plane: np.ndarray) -> bool:
        """層を2Dグリッドで丸ごと置き換える（1層目の敷き詰めに使う）。"""
        if plane.shape != (self.height, self.width):
            return False
        if not self.ensure_layer(layer):
            return False
        self.grid[layer] = plane.astype(np.int16, copy=False)
        return True

    # --- 集計 ---

    def top_layer(self) -> int:
        """ビーズが存在する最上層。全て空なら -1。"""
        occupied = np.nonzero((self.grid >= 0).any(axis=(1, 2)))[0]
        return int(occupied[-1]) if occupied.size else -1

    def total(self) -> int:
        return int((self.grid >= 0).sum())

    def layer_total(self, layer: int) -> int:
        if not (0 <= layer < self.layer_count):
            return 0
        return int((self.grid[layer] >= 0).sum())

    def counts(self) -> dict[int, int]:
        """パレットindexごとの使用数（全層合計）。"""
        used = self.grid[self.grid >= 0]
        if used.size == 0:
            return {}
        uniq, counts = np.unique(used, return_counts=True)
        return {int(k): int(v) for k, v in zip(uniq, counts)}

    def has_beads_above(self, layer: int) -> bool:
        """指定層より上にビーズがあるか（作品を破棄する前の確認に使う）。"""
        if layer + 1 >= self.layer_count:
            return False
        return bool((self.grid[layer + 1 :] >= 0).any())

    def dominant_color(self, layer: int = 0) -> int:
        """指定層で最も多く使われている色。空なら -1。"""
        if not (0 <= layer < self.layer_count):
            return EMPTY
        used = self.grid[layer]
        used = used[used >= 0]
        if used.size == 0:
            return EMPTY
        uniq, counts = np.unique(used, return_counts=True)
        return int(uniq[int(np.argmax(counts))])

    def corner_colors(self, layer: int = 0) -> list[int]:
        """指定層の四隅の色（重複なし）。

        盤面はプレート単位で切り上げた空セルを含むため、ビーズが置かれた範囲
        （元画像の範囲）の四隅を見る。
        """
        if not (0 <= layer < self.layer_count):
            return []
        plane = self.grid[layer]
        rows = np.nonzero((plane >= 0).any(axis=1))[0]
        cols = np.nonzero((plane >= 0).any(axis=0))[0]
        if rows.size == 0:
            return []
        colors: list[int] = []
        for r in (rows[0], rows[-1]):
            for c in (cols[0], cols[-1]):
                value = int(plane[r, c])
                if value >= 0 and value not in colors:
                    colors.append(value)
        return colors

    def background_colors(self, mode: str, manual: Optional[list[int]] = None) -> list[int]:
        """自動ドームで背景として扱う色。判定できなければ空。"""
        if mode == "manual":
            return [int(pi) for pi in (manual or []) if int(pi) >= 0]
        if mode == "dominant":
            background = self.dominant_color(0)
            return [background] if background >= 0 else []
        return self.corner_colors(0)

    def top_map(self) -> np.ndarray:
        """各列の最上位の層番号 (H, W)。ビーズが無い列は -1。"""
        occupied = self.grid >= 0
        # 上の層から見て最初に埋まっている位置を探す
        top = self.layer_count - 1 - np.argmax(occupied[::-1], axis=0)
        return np.where(occupied.any(axis=0), top, EMPTY).astype(np.int16)

    def apply_column_heights(self, heights: np.ndarray) -> int:
        """heights > 0 の列を、その列の1層目の色でその高さちょうどに整える。

        上にはみ出した既存ビーズは取り除く（パラメータを変えて実行し直したときに
        高さが積み上がらず、設定どおりの形に置き換わるようにするため）。
        1層目だけは決して消さない。戻り値は変化したセル数。
        """
        if heights.shape != (self.height, self.width):
            return 0
        heights = np.asarray(heights, dtype=np.int32)
        region = heights > 0
        if not region.any():
            return 0
        if not self.ensure_layer(int(heights.max()) - 1):
            return 0
        before = self.grid.copy()
        idx = np.arange(self.layer_count, dtype=np.int32)[:, None, None]
        fill = (idx < heights) & region
        drop = (idx >= heights) & region & (idx > 0)
        base_colors = np.broadcast_to(self.grid[0], self.grid.shape)
        self.grid[fill] = base_colors[fill]
        self.grid[drop] = EMPTY
        return int((self.grid != before).sum())

    def raise_non_background(self, background: int) -> int:
        """背景色以外の列の一番上へ、同じ色のビーズをもう1段積む。

        戻り値は積んだ列の数。層数の上限に達した列は積まずに数えない。
        """
        top = self.top_map()
        rows, cols = np.nonzero(top >= 0)
        if rows.size == 0:
            return 0
        tops = top[rows, cols].astype(np.int64)
        colors = self.grid[tops, rows, cols]
        sel = (colors != background) & (tops + 1 < MAX_LAYERS)
        if not sel.any():
            return 0
        rows, cols, tops, colors = rows[sel], cols[sel], tops[sel], colors[sel]
        self.ensure_layer(int(tops.max()) + 1)
        self.grid[tops + 1, rows, cols] = colors
        return int(rows.size)


@dataclass(frozen=True)
class DomePlan:
    """自動ドームの計算結果（盤面へ反映する前の目標の高さ）。"""

    heights: np.ndarray  # (H, W) 各列の目標の高さ。0 は対象外
    islands: int         # 独立した領域（島）の数
    top_layer: int       # 最高層（0始まり）
    total_beads: int     # 反映後に対象領域が使うビーズ総数


def _dome_profile_z(t: np.ndarray, profile: str) -> np.ndarray:
    """正規化距離 t（縁=0, 中心=1）を 0〜1 の高さへ変換する。"""
    if profile == "parabola":
        return t * (2.0 - t)
    if profile == "cone":
        return t
    # 半球: 極座標の円弧。縁が最も急峻で頂上が平らになる
    return np.sqrt(np.clip(t * (2.0 - t), 0.0, None))


def enclosed_background_states(
    plane: np.ndarray,
    backgrounds: list[int],
    min_ratio: float = DEFAULT_HOLE_RATIO,
    overrides: Optional[dict[tuple[int, int], bool]] = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, bool]]:
    """背景色のセルを「外周の背景」と「囲まれた背景」に分ける。

    外周（プレート単位の空セルを含む）へ4近傍でつながる背景色だけが無条件に
    背景になる。キャラに囲まれた背景色のかたまりは、面積がキャラ全体
    （囲まれた部分を含む）の min_ratio 以上なら背景（腕と胴のすき間など）、
    未満ならキャラの一部（目のハイライトなど）とみなす。
    overrides は {かたまり内の任意のセル: 背景にするか} の手動指定で、面積判定より優先する。

    戻り値は (外周の背景マスク, 囲まれた背景のラベル（0は対象外）, {ラベル: 背景にするか})。
    """
    cv2 = _require_cv2()
    plane = np.asarray(plane)
    candidate = np.isin(plane, backgrounds) | (plane < 0)
    if not candidate.any():
        return np.zeros(plane.shape, dtype=bool), np.zeros(plane.shape, dtype=np.int32), {}
    # 斜めでつながると線画の隙間から内側へ漏れるので4近傍で分ける
    _count, labels = cv2.connectedComponents(
        np.ascontiguousarray(candidate.astype(np.uint8)), connectivity=4
    )
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    is_outer = np.isin(labels, np.unique(edge[edge > 0]))
    outer = is_outer & (plane >= 0)
    enclosed = np.where(candidate & ~is_outer & (plane >= 0), labels, 0).astype(np.int32)

    ids, areas = np.unique(enclosed[enclosed > 0], return_counts=True)
    figure_area = int(((plane >= 0) & ~outer).sum())
    states = {
        int(label): bool(area >= min_ratio * figure_area) for label, area in zip(ids, areas)
    }
    for (row, col), is_background in (overrides or {}).items():
        if 0 <= row < plane.shape[0] and 0 <= col < plane.shape[1]:
            label = int(enclosed[row, col])
            if label > 0:
                states[label] = bool(is_background)
    return outer, enclosed, states


def dome_background_mask(
    plane: np.ndarray,
    backgrounds: list[int],
    min_ratio: float = DEFAULT_HOLE_RATIO,
    overrides: Optional[dict[tuple[int, int], bool]] = None,
) -> np.ndarray:
    """自動ドームで背景として扱うセル（外周の背景 + 背景と判定した囲まれた背景）。"""
    outer, enclosed, states = enclosed_background_states(plane, backgrounds, min_ratio, overrides)
    background_ids = [label for label, is_background in states.items() if is_background]
    if not background_ids:
        return outer
    return outer | np.isin(enclosed, background_ids)


def _fill_boundary_heights(
    heights: np.ndarray, boundary: np.ndarray, base: int
) -> None:
    """境界（線画）セルの高さを、隣接するパーツの高さのうち最も低いものに合わせる。

    太い線の内側はパーツに直接接しないので、確定したセルから最小値を
    8近傍で1セルずつ内側へ広げる。どのパーツにも接しない線は base にする。
    """
    cv2 = _require_cv2()
    unknown = 1e9
    work = np.where(heights > 0, heights, unknown).astype(np.float32)
    work[boundary] = unknown
    pending = boundary.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    while pending.any():
        # erode は近傍の最小値。背景や未確定セルは unknown のままなので影響しない
        neighbor_min = cv2.erode(work, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=unknown)
        reached = pending & (neighbor_min < unknown)
        if not reached.any():
            break
        work[reached] = neighbor_min[reached]
        pending &= ~reached
    heights[boundary] = np.where(work[boundary] < unknown, work[boundary], base).astype(np.int32)


def dome_heights(
    mask: np.ndarray,
    profile: str = "hemisphere",
    scale: float = 1.0,
    base: int = DOME_EDGE_HEIGHT,
    per_island: bool = True,
    max_height: int = MAX_LAYERS,
    boundary: Optional[np.ndarray] = None,
) -> DomePlan:
    """マスク領域をドーム状に盛り上げるための、列ごとの高さを求める。

    各セルから背景までの最短距離（距離変換）を使う。距離の最大値は楕円の
    短軸半径に一致するため、縦横の最長直線を測る必要はなく、楕円でない形でも
    そのままシルエットに沿ったドームになる。

    正規化は t = (距離 - 1) / (最大距離 - 1) とし、縁で t=0、中心で t=1 になる。
    これにより縁は必ず base 層、頂点は base + 短軸半径 × scale 層になる。

    boundary（線画の境界セル）を渡すと、境界も背景と同じく距離0として扱い、
    線で区切られたパーツごとに別のドームを作る。境界セル自体は隣接パーツの
    最も低い高さに合わせ、溝や突起にならないようにする。
    """
    cv2 = _require_cv2()
    figure = mask.astype(bool)
    line = None
    if boundary is not None:
        line = boundary.astype(bool) & figure
        if not line.any():
            line = None
    parts = figure & ~line if line is not None else figure
    mask_u8 = np.ascontiguousarray(parts.astype(np.uint8))
    heights = np.zeros(mask_u8.shape, dtype=np.int32)
    if not figure.any():
        return DomePlan(heights, 0, -1, 0)
    if not mask_u8.any():
        # 線画だけで塗りのパーツが無い。盛り上げる対象なしとして扱う
        return DomePlan(heights, 0, -1, 0)

    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if per_island:
        count, labels = cv2.connectedComponents(mask_u8)
        island_ids = range(1, count)
    else:
        labels = mask_u8.astype(np.int32)
        island_ids = (1,)

    islands = 0
    for label in island_ids:
        sel = labels == label
        if not sel.any():
            continue
        islands += 1
        d = dist[sel]
        span = float(d.max()) - 1.0
        if span <= 1e-6:
            # 1セル幅の島は膨らませようがないので縁の高さで揃える
            heights[sel] = base
            continue
        t = np.clip((d - 1.0) / span, 0.0, 1.0)
        z = _dome_profile_z(t, profile)
        column = base + np.rint(span * float(scale) * z).astype(np.int32)
        heights[sel] = np.clip(column, 1, max_height)

    if line is not None and islands:
        _fill_boundary_heights(heights, line, min(base, max_height))

    top =int(heights.max()) - 1 if islands else -1
    return DomePlan(heights, islands, top, int(heights.sum()))


class BoardHistory:
    """盤面のアンドゥ/リドゥ履歴。

    立体作品は想定規模（1プレート・数十層）でも数十KB程度なので、
    差分ではなくグリッドのスナップショットをそのまま積む。
    """

    def __init__(self, limit: int = 50) -> None:
        self._limit = max(1, int(limit))
        self._stack: list[np.ndarray] = []
        self._index = -1

    def reset(self, grid: np.ndarray) -> None:
        """現在の状態を履歴の起点にする。"""
        self._stack = [grid.copy()]
        self._index = 0

    def push(self, grid: np.ndarray) -> None:
        """操作後の状態を積む。やり直し分は捨てる。"""
        if self._index < len(self._stack) - 1:
            del self._stack[self._index + 1 :]
        self._stack.append(grid.copy())
        if len(self._stack) > self._limit:
            del self._stack[0]
        self._index = len(self._stack) - 1

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._stack) - 1

    def undo(self) -> Optional[np.ndarray]:
        if not self.can_undo:
            return None
        self._index -= 1
        return self._stack[self._index].copy()

    def redo(self) -> Optional[np.ndarray]:
        if not self.can_redo:
            return None
        self._index += 1
        return self._stack[self._index].copy()
