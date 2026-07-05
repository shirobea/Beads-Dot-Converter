"""settings.json で使用するキー名の定数定義。

app.py（復元側）と actions.py（保存側）の両方で使うため一元管理する。
キー名を変更する場合はここだけ直せば両側が追従する。
"""

from __future__ import annotations

WIDTH = "幅"
HEIGHT = "高さ"
MODE = "モード"
LAB_METRIC = "Lab距離式"
CMC_L = "CMC l"
CMC_C = "CMC c"
RESIZE_METHOD = "リサイズ方式"
RGB_WEIGHTS = "RGB重み"

NORMAL_ENABLED = "ノーマル有効"
NORMAL_INVERT_Y = "ノーマルY反転"
NORMAL_STRENGTH = "ノーマル強さ"
NORMAL_AMBIENT = "ノーマル環境光"
NORMAL_GAMMA = "ノーマルガンマ"
NORMAL_LIGHT_DIR = "ノーマル光方向"
NORMAL_MAP = "ノーマルマップ"

AO_ENABLED = "AO有効"
AO_STRENGTH = "AO強さ"
AO_MAP = "AOマップ"

SPECULAR_ENABLED = "Specular有効"
SPECULAR_STRENGTH = "Specular強さ"
SPECULAR_SHININESS = "Specular鋭さ"
SPECULAR_MAP = "Specularマップ"

DISPLACEMENT_ENABLED = "Displacement有効"
DISPLACEMENT_STRENGTH = "Displacement強さ"
DISPLACEMENT_MIDPOINT = "Displacement中心"
DISPLACEMENT_INVERT = "Displacement反転"
DISPLACEMENT_MAP = "Displacementマップ"

PSEUDO_GRADIENT = "グラデーション強度"
DITHER = "ディザリング"
DITHER_STRENGTH = "ディザリング強度"
BILATERAL_SIGMA = "バイラテラルσ"
USE_SUPER_SAMPLING = "use_super_sampling"
