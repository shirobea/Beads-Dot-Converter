"""立体アイロンビーズの盤面データ（OpenGL 非依存）。

ビーズを (layer, row, col) の3次元グリッドで保持する。値はパレットindex、
-1 は空。描画・保存・設計図出力はいずれもこのグリッドだけを入力とするため、
GLコンテキスト無しで単体確認できる。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

EMPTY = -1
MAX_LAYERS = 64  # 1作品あたりの層数上限（メモリとUIの都合で頭打ちにする）


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

    def top_map(self) -> np.ndarray:
        """各列の最上位の層番号 (H, W)。ビーズが無い列は -1。"""
        occupied = self.grid >= 0
        # 上の層から見て最初に埋まっている位置を探す
        top = self.layer_count - 1 - np.argmax(occupied[::-1], axis=0)
        return np.where(occupied.any(axis=0), top, EMPTY).astype(np.int16)

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
