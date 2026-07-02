"""Palette loader for bead colors."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class BeadColor:
    """Represents a single bead color."""

    color_id: str
    name: str
    rgb: tuple[float, float, float]
    lab: tuple[float, float, float]
    oklab: tuple[float, float, float]
    hunter_lab: tuple[float, float, float]


class BeadPalette:
    """Container for bead colors with convenient numpy views."""

    def __init__(self, colors: List[BeadColor]) -> None:
        if not colors:
            raise ValueError("パレットが空です。")
        self.colors = colors
        self.rgb_array = np.array([c.rgb for c in colors], dtype=np.float32)
        # 出力画像の生成には丸め済みのuint8を使う（astypeの切り捨てだと
        # 色使用一覧側のroundと不整合になり色がリストから漏れる）
        self.rgb_uint8 = np.rint(self.rgb_array).astype(np.uint8)
        self.lab_array = np.array([c.lab for c in colors], dtype=np.float32)
        self.oklab_array = np.array([c.oklab for c in colors], dtype=np.float32)
        # Hunter LabはCSVの値をそのまま保持する
        self.hunter_lab_array = np.array([c.hunter_lab for c in colors], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.colors)

    def __iter__(self):
        return iter(self.colors)


def _extract_csv_text(raw_text: str) -> str:
    """Handle embedded triple-quoted CSV payloads."""
    if '"""' in raw_text:
        start = raw_text.find('"""')
        end = raw_text.rfind('"""')
        if start != -1 and end != -1 and end > start:
            return raw_text[start + 3 : end]
    return raw_text


def load_palette(csv_path: str | Path) -> BeadPalette:
    """Load bead colors from CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"{csv_path} が見つかりません。")

    raw_text = path.read_text(encoding="utf-8")
    payload = _extract_csv_text(raw_text)

    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    if not lines:
        raise ValueError("CSVが空です。")

    reader = csv.reader(lines)
    header = None
    colors: List[BeadColor] = []

    for row in reader:
        if not row:
            continue
        if header is None:
            # detect header row
            if "色番号" in row[0] or "color" in row[0].lower():
                header = row
                continue
            header = row
            continue
        if len(row) < 14:
            continue
        try:
            color = BeadColor(
                color_id=row[0].strip(),
                name=row[1].strip(),
                rgb=(float(row[2]), float(row[3]), float(row[4])),
                lab=(float(row[5]), float(row[6]), float(row[7])),
                oklab=(float(row[8]), float(row[9]), float(row[10])),
                hunter_lab=(float(row[11]), float(row[12]), float(row[13])),
            )
            colors.append(color)
        except ValueError:
            continue

    if not colors:
        raise ValueError("CSVから色データを読み込めませんでした。")

    return BeadPalette(colors)
