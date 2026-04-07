"""UIで使用する入力パラメータのデータモデル定義。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConversionRequest:
    """UIから取得した変換パラメータ一式。"""

    width: int
    height: int
    mode: str
    lab_metric: str
    cmc_l: float
    cmc_c: float
    keep_aspect: bool
    resize_method: str
    rgb_weights: tuple[float, float, float]
    normal_map_path: str | None
    normal_enabled: bool
    normal_invert_y: bool
    normal_light_dir: tuple[float, float, float]
    normal_strength: float
    normal_ambient: float
    normal_gamma: float
    ao_map_path: str | None
    ao_enabled: bool
    ao_strength: float
    specular_map_path: str | None
    specular_enabled: bool
    specular_strength: float
    specular_shininess: float
    displacement_map_path: str | None
    displacement_enabled: bool
    displacement_strength: float
    displacement_midpoint: float
    displacement_invert: bool
    pseudo_gradient_strength: float
    use_super_sampling: bool = False
    dither_method: str = "none"
    dither_strength: float = 1.0
