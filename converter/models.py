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
    # 量子化直前にL*へかけるアンシャープマスク（局所コントラスト強調）
    local_contrast_amount: float = 0.0
    local_contrast_sigma: float = 1.2

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
class BalanceConfig:
    """色バランス補正（置換結果の明るさ・色味のズレを入力側で打ち消す）のパラメータ一式。"""

    enabled: bool = False
    # 全体補正の強さ（平均 L*・平均 a*b*・平均彩度のズレを戻す）。0〜1
    global_strength: float = 0.7
    # 局所補正の強さ（近傍との L* 差を入力に合わせる）。0〜1
    local_strength: float = 0.0
    # 局所補正の近傍の広さ（ガウシアンの σ、ビーズ数）
    local_sigma: float = 1.5
    # 入力の「自分 − 近傍平均」の L* 差がこれ未満の画素はノイズとみなして局所補正しない
    noise_threshold: float = 2.0
    # 仮置換 → 測定 → 修正の最大回数
    iterations: int = 4

    @property
    def active(self) -> bool:
        return self.enabled and (self.global_strength > 0 or self.local_strength > 0)


@dataclass(frozen=True)
class PostFilterConfig:
    """変換後処理フィルタのパラメータ一式。"""

    post_mode_filter_size: int = 0
    post_island_min_area: int = 0
    # ディテール保護の強さ。0.0=積極的に置換 / 1.0=ほぼ置換しない
    post_protect_strength: float = 0.5

    @property
    def enabled(self) -> bool:
        return self.post_mode_filter_size > 0 or self.post_island_min_area > 0
