"""変換パラメータのデータモデル定義。

UI層とconverter層の両方から参照されるため、converter側に置く。
すべてfrozenで、生成後の変更は不可。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShadingConfig:
    """シェーディングマップに関するパラメータ一式。デフォルトは全機能オフ。"""

    normal_map_path: str | None = None
    normal_enabled: bool = False
    normal_invert_y: bool = False
    normal_light_dir: tuple[float, float, float] = (0.2, -0.2, 0.95)
    normal_strength: float = 0.6
    normal_ambient: float = 0.25
    normal_gamma: float = 1.0
    ao_map_path: str | None = None
    ao_enabled: bool = False
    ao_strength: float = 0.6
    specular_map_path: str | None = None
    specular_enabled: bool = False
    specular_strength: float = 0.6
    specular_shininess: float = 24.0
    displacement_map_path: str | None = None
    displacement_enabled: bool = False
    displacement_strength: float = 0.6
    displacement_midpoint: float = 0.5
    displacement_invert: bool = False
    pseudo_gradient_strength: float = 0.0

    @property
    def any_map_enabled(self) -> bool:
        """ノーマル/AO/Specular/Displacementのいずれかが有効か。"""
        return bool(
            self.normal_enabled
            or self.ao_enabled
            or self.specular_enabled
            or self.displacement_enabled
        )


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

    @property
    def enabled(self) -> bool:
        return self.post_mode_filter_size > 0 or self.post_island_min_area > 0
