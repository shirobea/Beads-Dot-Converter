"""Palette loader for bead colors."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


# パレットCSVを置くフォルダ名。ここにCSVを追加するだけで選択肢に並ぶ
PALETTE_DIR_NAME = "palettes"


def _base_dirs() -> List[Path]:
    """実行環境ごとの配置揺れを吸収した探索ベースディレクトリ一覧。"""
    dirs = [
        Path.cwd(),
        Path(__file__).resolve().parent,
    ]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        dirs.append(exe_dir)
        dirs.append(Path(getattr(sys, "_MEIPASS", exe_dir)))
    # 重複を除きつつ順序は維持する
    unique: List[Path] = []
    for path in dirs:
        if path not in unique:
            unique.append(path)
    return unique


def palette_dir_candidates() -> List[Path]:
    """パレットフォルダの候補パス（存在有無は問わない）。"""
    return [base / PALETTE_DIR_NAME for base in _base_dirs()]


def resolve_palette_dir() -> Path:
    """CSVが1つ以上入っているパレットフォルダを返す。"""
    for path in palette_dir_candidates():
        if path.is_dir() and any(path.glob("*.csv")):
            return path
    raise FileNotFoundError(
        f"{PALETTE_DIR_NAME} フォルダにカラーパレットCSVが見つかりません。"
    )


@dataclass(frozen=True)
class PaletteEntry:
    """選択肢1件（コンボボックスの表示名とCSVの実パス）。"""

    label: str  # 拡張子を除いたファイル名をそのまま表示名にする
    filename: str
    path: Path


def discover_palettes() -> List[PaletteEntry]:
    """パレットフォルダ内のCSVを名前順に列挙する。

    フォルダを走査するだけなので、CSVを追加・改名・削除しても
    コード側の変更は不要。
    """
    try:
        directory = resolve_palette_dir()
    except FileNotFoundError:
        return []
    entries = [
        PaletteEntry(label=path.stem, filename=path.name, path=path)
        for path in sorted(directory.glob("*.csv"), key=lambda p: p.name.lower())
        if path.is_file()
    ]
    return entries


def find_palette_entry(entries: List[PaletteEntry], key: str | None) -> PaletteEntry | None:
    """ファイル名または表示名から選択肢を引く（見つからなければ None）。"""
    if not key:
        return None
    for entry in entries:
        if key in (entry.filename, entry.label):
            return entry
    return None


def resolve_palette_path(filename: str) -> Path:
    """ファイル名（または表示名）からCSVの実パスを引く。"""
    entry = find_palette_entry(discover_palettes(), filename)
    if entry is None:
        raise FileNotFoundError(f"{filename} が {PALETTE_DIR_NAME} フォルダに見つかりません。")
    return entry.path


def load_default_palette() -> tuple["BeadPalette", str]:
    """先頭のパレットを読み込み、(パレット, ファイル名) を返す。"""
    entries = discover_palettes()
    if not entries:
        searched = " / ".join(str(p) for p in palette_dir_candidates())
        raise FileNotFoundError(
            f"カラーパレットCSVが見つかりません。探索先: {searched}"
        )
    last_error: Exception | None = None
    for entry in entries:
        # 壊れたCSVが先頭にあっても起動できるよう、読める最初の1件を採用する
        try:
            return load_palette(entry.path), entry.filename
        except Exception as exc:
            last_error = exc
    raise ValueError(f"カラーパレットCSVを読み込めませんでした。\n{last_error}")


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
