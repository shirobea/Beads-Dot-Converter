"""UIで使用する入力パラメータのデータモデル定義。

ShadingConfig等の変換パラメータ本体は converter.models に移動した。
既存コードのために再エクスポートする。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from converter.models import ShadingConfig, DitherConfig, PostFilterConfig

__all__ = ["ShadingConfig", "DitherConfig", "PostFilterConfig", "ConversionRequest"]


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
