"""UIで使用する入力パラメータのデータモデル定義。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShadingConfig:
    """シェーディングマップに関するパラメータ一式。"""

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


@dataclass(frozen=True)
class DitherConfig:
    """ディザリングに関するパラメータ一式。"""

    dither_method: str = "none"
    dither_strength: float = 1.0


@dataclass(frozen=True)
class PostFilterConfig:
    """変換後処理フィルタのパラメータ一式。"""

    post_mode_filter_size: int = 0
    post_island_min_area: int = 0


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
    use_super_sampling: bool
    shading: ShadingConfig
    dither: DitherConfig
    post_filter: PostFilterConfig = field(default_factory=PostFilterConfig)
