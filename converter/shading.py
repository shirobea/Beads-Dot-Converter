"""シェーディングエフェクト: ノーマル/AO/Specular/Displacement 処理。"""

from __future__ import annotations

import numpy as np

from cv2_utils import require_cv2 as _require_cv2
from .io_utils import (
    _load_normal_map_rgb,
    _load_ao_map_gray,
    _load_specular_map_gray,
    _load_displacement_map_gray,
)


def _normalize_normals(normal_rgb: np.ndarray, invert_y: bool) -> np.ndarray:
    """ノーマルマップを[-1, 1]に変換して正規化する。"""
    normals = normal_rgb.astype(np.float32) / 255.0
    normals = normals * 2.0 - 1.0
    if invert_y:
        normals[:, :, 1] *= -1.0
    length = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.clip(length, 1e-6, None)
    return normals


def _apply_normal_shading(
    image_rgb: np.ndarray,
    normal_rgb: np.ndarray,
    light_dir: tuple[float, float, float],
    strength: float,
    ambient: float,
    gamma: float,
    invert_y: bool,
    ao_gray: np.ndarray | None,
    ao_strength: float,
) -> np.ndarray:
    """ノーマルマップ由来の陰影を明度に反映する。"""
    cv2 = _require_cv2()
    light = np.array(light_dir, dtype=np.float32)
    if np.linalg.norm(light) < 1e-6:
        light = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    light = light / np.linalg.norm(light)

    normals = _normalize_normals(normal_rgb, invert_y)
    shade = np.sum(normals * light[None, None, :], axis=2)
    shade = np.clip(shade, 0.0, 1.0)
    shade = float(ambient) + (1.0 - float(ambient)) * shade
    if ao_gray is not None:
        ao_strength = max(0.0, min(1.0, float(ao_strength)))
        shade = shade * ((1.0 - ao_strength) + ao_strength * ao_gray)
    shade = shade ** max(float(gamma), 1e-3)

    # LabのLだけ補正して色味を保つ
    bgr = image_rgb[:, :, ::-1].astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l = lab[:, :, 0]
    strength = max(0.0, float(strength))
    l = np.clip(l * (1.0 + strength * (shade - 0.5)), 0.0, 100.0)
    lab[:, :, 0] = l
    bgr_out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    rgb_out = np.clip(bgr_out[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    return rgb_out


def _apply_ao_shading(image_rgb: np.ndarray, ao_gray: np.ndarray, ao_strength: float) -> np.ndarray:
    """AOマップで明度だけを調整する。"""
    cv2 = _require_cv2()
    ao_strength = max(0.0, min(1.0, float(ao_strength)))
    bgr = image_rgb[:, :, ::-1].astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l = lab[:, :, 0]
    l = np.clip(l * ((1.0 - ao_strength) + ao_strength * ao_gray), 0.0, 100.0)
    lab[:, :, 0] = l
    bgr_out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    rgb_out = np.clip(bgr_out[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    return rgb_out


def _apply_pseudo_gradient(
    image_rgb: np.ndarray,
    blur_sigma: float = 20.0,
    strength: float = 10.0,
) -> np.ndarray:
    """画像本来の明暗分布に沿った疑似グラデーション揺らぎをLチャンネルに加える。
    リサイズ後・量子化前に適用する。strength=0 で処理をスキップ。
    """
    if strength <= 0:
        return image_rgb
    cv2 = _require_cv2()
    bgr = image_rgb[:, :, ::-1].astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    L_blur = cv2.GaussianBlur(L, (0, 0), float(blur_sigma))
    L_norm = L_blur - L_blur.mean()
    L_norm /= (float(L_norm.std()) + 1e-6)
    L_modified = np.clip(L + L_norm * float(strength), 0.0, 100.0)
    lab_out = cv2.merge([L_modified, a, b])
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)
    return np.clip(bgr_out[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)


def _apply_displacement_shading(
    image_rgb: np.ndarray,
    displacement_gray: np.ndarray,
    strength: float,
    midpoint: float,
    invert: bool,
) -> np.ndarray:
    """Displacementマップで明度を押し出すように補正する。"""
    cv2 = _require_cv2()
    strength = max(0.0, float(strength))
    midpoint = max(0.0, min(1.0, float(midpoint)))
    height = displacement_gray
    if invert:
        height = 1.0 - height
    offset = (height - midpoint) * 2.0
    factor = 1.0 + strength * offset
    factor = np.clip(factor, 0.0, 2.0)
    bgr = image_rgb[:, :, ::-1].astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l = lab[:, :, 0]
    l = np.clip(l * factor, 0.0, 100.0)
    lab[:, :, 0] = l
    bgr_out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    rgb_out = np.clip(bgr_out[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    return rgb_out


def _apply_specular_highlight(
    image_rgb: np.ndarray,
    normal_rgb: np.ndarray | None,
    light_dir: tuple[float, float, float],
    strength: float,
    shininess: float,
    invert_y: bool,
    specular_gray: np.ndarray,
) -> np.ndarray:
    """Specularマップと法線でハイライトを加える。"""
    cv2 = _require_cv2()
    strength = max(0.0, float(strength))
    shininess = max(1.0, float(shininess))

    if normal_rgb is None:
        normals = np.zeros((*specular_gray.shape, 3), dtype=np.float32)
        normals[:, :, 2] = 1.0
    else:
        normals = _normalize_normals(normal_rgb, invert_y)

    light = np.array(light_dir, dtype=np.float32)
    if np.linalg.norm(light) < 1e-6:
        light = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    light = light / np.linalg.norm(light)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half = light + view
    if np.linalg.norm(half) < 1e-6:
        half = view
    half = half / np.linalg.norm(half)

    dot = np.sum(normals * half[None, None, :], axis=2)
    spec = np.clip(dot, 0.0, 1.0) ** shininess
    spec_factor = np.clip(spec * specular_gray * strength, 0.0, 1.0)

    bgr = image_rgb[:, :, ::-1].astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l = lab[:, :, 0]
    l = np.clip(l + (100.0 - l) * spec_factor, 0.0, 100.0)
    lab[:, :, 0] = l
    bgr_out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    rgb_out = np.clip(bgr_out[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    return rgb_out


def _apply_shading_suite(
    base: np.ndarray,
    width: int,
    height: int,
    normal_map_path: str | None,
    normal_enabled: bool,
    normal_invert_y: bool,
    normal_light_dir: tuple[float, float, float],
    normal_strength: float,
    normal_ambient: float,
    normal_gamma: float,
    ao_map_path: str | None,
    ao_enabled: bool,
    ao_strength: float,
    specular_map_path: str | None,
    specular_enabled: bool,
    specular_strength: float,
    specular_shininess: float,
    displacement_map_path: str | None,
    displacement_enabled: bool,
    displacement_strength: float,
    displacement_midpoint: float,
    displacement_invert: bool,
) -> np.ndarray:
    """AO/Normal/Specular/Displacement の各マップをロード・リサイズして適用する共通処理。"""
    cv2 = _require_cv2()

    ao_gray = None
    if ao_enabled and ao_map_path:
        ao_gray = _load_ao_map_gray(ao_map_path)
        ao_gray = cv2.resize(ao_gray, (width, height), interpolation=cv2.INTER_LINEAR)

    normal_rgb = None
    if (normal_enabled or specular_enabled) and normal_map_path:
        normal_rgb = _load_normal_map_rgb(normal_map_path)
        normal_rgb = cv2.resize(normal_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    if normal_enabled and normal_rgb is not None:
        base = _apply_normal_shading(
            base,
            normal_rgb,
            light_dir=normal_light_dir,
            strength=normal_strength,
            ambient=normal_ambient,
            gamma=normal_gamma,
            invert_y=normal_invert_y,
            ao_gray=ao_gray,
            ao_strength=ao_strength,
        )
    elif ao_gray is not None:
        base = _apply_ao_shading(base, ao_gray, ao_strength)

    specular_gray = None
    if specular_enabled and specular_map_path:
        specular_gray = _load_specular_map_gray(specular_map_path)
        specular_gray = cv2.resize(specular_gray, (width, height), interpolation=cv2.INTER_LINEAR)
    if specular_gray is not None and specular_enabled:
        base = _apply_specular_highlight(
            base,
            normal_rgb,
            light_dir=normal_light_dir,
            strength=specular_strength,
            shininess=specular_shininess,
            invert_y=normal_invert_y,
            specular_gray=specular_gray,
        )

    if displacement_enabled and displacement_map_path:
        disp_gray = _load_displacement_map_gray(displacement_map_path)
        disp_gray = cv2.resize(disp_gray, (width, height), interpolation=cv2.INTER_LINEAR)
        base = _apply_displacement_shading(
            base,
            disp_gray,
            strength=displacement_strength,
            midpoint=displacement_midpoint,
            invert=displacement_invert,
        )

    return base


def apply_shading_preview(
    image_rgb: np.ndarray,
    normal_map_path: str | None = None,
    normal_enabled: bool = False,
    normal_invert_y: bool = False,
    normal_light_dir: tuple[float, float, float] = (0.2, -0.2, 0.95),
    normal_strength: float = 0.6,
    normal_ambient: float = 0.25,
    normal_gamma: float = 1.0,
    ao_map_path: str | None = None,
    ao_enabled: bool = False,
    ao_strength: float = 0.6,
    specular_map_path: str | None = None,
    specular_enabled: bool = False,
    specular_strength: float = 0.6,
    specular_shininess: float = 24.0,
    displacement_map_path: str | None = None,
    displacement_enabled: bool = False,
    displacement_strength: float = 0.6,
    displacement_midpoint: float = 0.5,
    displacement_invert: bool = False,
) -> np.ndarray:
    """入力画像にノーマル/AO/Specular/Displacementの明度補正だけを適用する（プレビュー用）。"""
    if image_rgb is None:
        raise ValueError("image_rgb が必要です。")
    if not (normal_enabled or ao_enabled or specular_enabled or displacement_enabled):
        return np.asarray(image_rgb, dtype=np.uint8)
    base = np.asarray(image_rgb, dtype=np.uint8)
    height, width = base.shape[:2]
    return _apply_shading_suite(
        base, width, height,
        normal_map_path=normal_map_path,
        normal_enabled=normal_enabled,
        normal_invert_y=normal_invert_y,
        normal_light_dir=normal_light_dir,
        normal_strength=normal_strength,
        normal_ambient=normal_ambient,
        normal_gamma=normal_gamma,
        ao_map_path=ao_map_path,
        ao_enabled=ao_enabled,
        ao_strength=ao_strength,
        specular_map_path=specular_map_path,
        specular_enabled=specular_enabled,
        specular_strength=specular_strength,
        specular_shininess=specular_shininess,
        displacement_map_path=displacement_map_path,
        displacement_enabled=displacement_enabled,
        displacement_strength=displacement_strength,
        displacement_midpoint=displacement_midpoint,
        displacement_invert=displacement_invert,
    )
