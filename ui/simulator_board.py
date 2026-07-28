"""立体アイロンビーズの盤面データ（OpenGL 非依存）。

ビーズを (layer, row, col) の3次元グリッドで保持する。値はパレットindex、
-1 は空。描画・保存・設計図出力はいずれもこのグリッドだけを入力とするため、
GLコンテキスト無しで単体確認できる。
"""

from __future__ import annotations

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
