"""変換パイプライン（リサイズ + パレット写像）。"""

from __future__ import annotations

from typing import Callable, Tuple
import threading

import numpy as np

from cv2_utils import require_cv2 as _require_cv2
from palette import BeadPalette
from .io_utils import _compute_resize, _load_image_rgb
from .quantize import _map_centers_to_palette, _report
from .shading import _apply_shading_suite, _apply_pseudo_gradient, apply_shading_preview  # noqa: F401

ProgressCb = Callable[[float], None]
CancelEvent = threading.Event
Size = Tuple[int, int]
PACKED_UNIQUE_THRESHOLD = 2_000_000

# リサイズ手法名 → cv2 属性名のマッピング。
# INTER_AREA はダウンスケール専用の面積平均法。拡大時はOpenCVが自動でINTER_LINEARにフォールバックする。
_RESIZE_METHOD_ATTR: dict[str, str] = {
    "nearest": "INTER_NEAREST",
    "bilinear": "INTER_LINEAR",
    "area": "INTER_AREA",
}

ALL_MODE_SPECS = [
    {"label": "なし", "mode": "none"},
    {"label": "RGB", "mode": "RGB"},
    {"label": "Lab2000", "mode": "Lab", "lab_metric": "CIEDE2000"},
    {"label": "Lab94", "mode": "Lab", "lab_metric": "CIE94"},
    {"label": "Lab76", "mode": "Lab", "lab_metric": "CIE76"},
    {"label": "Hunter", "mode": "Hunter"},
    {"label": "Oklab", "mode": "Oklab"},
    {"label": "CMC", "mode": "CMC(l:c)"},
]


class ConversionCancelled(Exception):
    """ユーザーによる中断を示す例外。"""


def _map_image_to_palette_index(
    image_rgb: np.ndarray,
    palette: BeadPalette,
    mode: str,
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lab_metric: str = "CIEDE2000",
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.3, 1.0),
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """各画素を最短距離のパレット色へ写像し、インデックス配列 (H×W) を返す。"""
    start, end = progress_range
    _report(progress_callback, start, cancel_event)

    flat = image_rgb.reshape(-1, 3)
    total_pixels = flat.shape[0]
    if total_pixels > PACKED_UNIQUE_THRESHOLD:
        flat_u32 = flat.astype(np.uint32, copy=False)
        codes = (flat_u32[:, 0] << 16) | (flat_u32[:, 1] << 8) | flat_u32[:, 2]
        unique_codes, inv = np.unique(codes, return_inverse=True)
        centers = np.stack(
            (
                (unique_codes >> 16) & 0xFF,
                (unique_codes >> 8) & 0xFF,
                unique_codes & 0xFF,
            ),
            axis=1,
        ).astype(np.float32)
    else:
        centers, inv = np.unique(flat, axis=0, return_inverse=True)
        centers = centers.astype(np.float32, copy=False)
    mapping = _map_centers_to_palette(
        centers,
        palette,
        mode,
        progress_callback=progress_callback,
        progress_range=(start, end),
        cancel_event=cancel_event,
        cmc_l=cmc_l,
        cmc_c=cmc_c,
        rgb_weights=rgb_weights,
        lab_metric=lab_metric,
    )
    _report(progress_callback, end, cancel_event)
    return mapping[inv].reshape(image_rgb.shape[:2])


def _map_image_to_palette(
    image_rgb: np.ndarray,
    palette: BeadPalette,
    mode: str,
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lab_metric: str = "CIEDE2000",
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    progress_callback: ProgressCb | None = None,
    progress_range: tuple[float, float] = (0.3, 1.0),
    cancel_event: CancelEvent | None = None,
) -> np.ndarray:
    """減色せず、各画素を最短距離のパレット色へ写像する。"""
    idx = _map_image_to_palette_index(
        image_rgb, palette, mode,
        rgb_weights=rgb_weights, lab_metric=lab_metric,
        cmc_l=cmc_l, cmc_c=cmc_c,
        progress_callback=progress_callback, progress_range=progress_range,
        cancel_event=cancel_event,
    )
    return palette.rgb_uint8[idx]


def _palette_rgb_to_index(rgb_arr: np.ndarray, palette: BeadPalette) -> np.ndarray:
    """パレット色のみを含むRGB配列をインデックス配列 (H×W) に変換する。"""
    pal = palette.rgb_uint8.astype(np.uint32)
    codes_pal = (pal[:, 0] << 16) | (pal[:, 1] << 8) | pal[:, 2]
    sort_order = np.argsort(codes_pal)
    sorted_codes = codes_pal[sort_order]
    flat = rgb_arr.reshape(-1, 3).astype(np.uint32)
    codes_flat = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    pos = np.searchsorted(sorted_codes, codes_flat)
    pos = np.clip(pos, 0, len(sort_order) - 1)
    idx = sort_order[pos]
    # searchsortedは完全一致が前提。パレット外の色が混入した場合は
    # 黙って誤マップせず、RGB最近傍で救済する（通常は到達しない防御）
    mismatch = sorted_codes[pos] != codes_flat
    if mismatch.any():
        bad = flat[mismatch].astype(np.float32)
        diff = bad[:, None, :] - palette.rgb_array[None, :, :]
        idx = idx.copy()
        idx[mismatch] = np.argmin(np.sum(diff ** 2, axis=2), axis=1)
    return idx.reshape(rgb_arr.shape[:2])


def _apply_mode_filter_index(index_arr: np.ndarray, kernel_size: int) -> np.ndarray:
    """パレットインデックス配列にモードフィルタを適用する（後処理）。

    各ピクセルを近傍 kernel_size×kernel_size の最頻インデックスで置換し、
    変換後のビーズ色を均一化する。

    各インデックスの出現マスクを箱フィルタで数え、最大のものを選ぶ。
    同数の場合は小さいインデックスを優先（bincount.argmaxと同じ挙動）。
    """
    cv2 = _require_cv2()
    ksize = (kernel_size, kernel_size)
    best_count = np.full(index_arr.shape, -1.0, dtype=np.float32)
    best_idx = index_arr.copy()
    for idx in np.unique(index_arr):
        mask = (index_arr == idx).astype(np.float32)
        counts = cv2.boxFilter(
            mask, -1, ksize, normalize=False, borderType=cv2.BORDER_REPLICATE
        )
        better = counts > best_count
        best_count[better] = counts[better]
        best_idx[better] = idx
    return best_idx


def _apply_island_removal_index(index_arr: np.ndarray, min_area: int) -> np.ndarray:
    """パレットインデックス配列から孤立した小さな色領域を除去する（後処理）。

    面積 < min_area の連結領域を周囲の最頻インデックスで置換する。
    """
    cv2 = _require_cv2()
    result = index_arr.copy()
    unique_indices = np.unique(index_arr)
    kernel = np.ones((3, 3), dtype=np.uint8)

    for idx in unique_indices:
        mask = (index_arr == idx).astype(np.uint8)
        if not mask.any():
            continue
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label_id in range(1, n_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                island_mask = labels == label_id
                dilated = cv2.dilate(island_mask.astype(np.uint8), kernel)
                border = dilated.astype(bool) & ~island_mask
                if border.any():
                    border_indices = index_arr[border]
                    counts = np.bincount(border_indices.astype(np.intp))
                    result[island_mask] = int(counts.argmax())
    return result


def convert_image(
    input_path: str | None,
    output_size: int | Tuple[int, int],
    mode: str,
    palette: BeadPalette,
    keep_aspect: bool = True,
    resize_method: str = "nearest",
    lab_metric: str = "CIEDE2000",
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
    input_image: np.ndarray | None = None,
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
    pseudo_gradient_strength: float = 0.0,
    use_super_sampling: bool = False,
    dither_method: str = "none",
    dither_strength: float = 1.0,
    post_mode_filter_size: int = 0,
    post_island_min_area: int = 0,
) -> np.ndarray:
    """入力画像を指定サイズへリサイズし、パレットへ写像して返す。"""
    _report(progress_callback, 0.0, cancel_event)
    cv2 = _require_cv2()
    if input_image is not None:
        # 事前ノイズ除去などで渡されたRGB配列を優先する
        image_rgb = np.asarray(input_image, dtype=np.uint8)
    else:
        if input_path is None:
            raise ValueError("input_image または input_path を指定してください。")
        image_rgb = _load_image_rgb(input_path)
    orig_h, orig_w = image_rgb.shape[:2]
    target_w, target_h = _compute_resize((orig_h, orig_w), output_size, keep_aspect)

    interp = getattr(cv2, _RESIZE_METHOD_ATTR.get(resize_method.lower(), "INTER_NEAREST"))

    if use_super_sampling and interp == cv2.INTER_AREA:
        inter_w = target_w * 2
        inter_h = target_h * 2
        if inter_w < orig_w and inter_h < orig_h:
            image_rgb = cv2.resize(image_rgb, (inter_w, inter_h), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=interp)
    _report(progress_callback, 0.3, cancel_event)

    mode_lower = mode.lower()
    base = _apply_shading_suite(
        resized, target_w, target_h,
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
    base = _apply_pseudo_gradient(base, strength=pseudo_gradient_strength)

    if mode_lower in {"none", "なし"}:
        _report(progress_callback, 1.0, cancel_event)
        return base

    if dither_method != "none":
        from .dither import apply_dither
        result = apply_dither(
            base,
            palette,
            dither_method,
            strength=dither_strength,
            mode=mode,
            lab_metric=lab_metric,
            rgb_weights=rgb_weights,
            cmc_l=cmc_l,
            cmc_c=cmc_c,
            progress_callback=progress_callback,
            progress_range=(0.3, 1.0),
            cancel_event=cancel_event,
        )
        if post_mode_filter_size > 0 or post_island_min_area > 0:
            idx = _palette_rgb_to_index(result, palette)
            if post_mode_filter_size > 0:
                idx = _apply_mode_filter_index(idx, post_mode_filter_size)
            if post_island_min_area > 0:
                idx = _apply_island_removal_index(idx, post_island_min_area)
            result = palette.rgb_uint8[idx]
        _report(progress_callback, 1.0, cancel_event)
        return result

    if post_mode_filter_size > 0 or post_island_min_area > 0:
        idx = _map_image_to_palette_index(
            base, palette, mode,
            rgb_weights=rgb_weights, lab_metric=lab_metric,
            cmc_l=cmc_l, cmc_c=cmc_c,
            progress_callback=progress_callback, progress_range=(0.3, 1.0),
            cancel_event=cancel_event,
        )
        if post_mode_filter_size > 0:
            idx = _apply_mode_filter_index(idx, post_mode_filter_size)
        if post_island_min_area > 0:
            idx = _apply_island_removal_index(idx, post_island_min_area)
        mapped = palette.rgb_uint8[idx]
    else:
        mapped = _map_image_to_palette(
            base, palette, mode,
            rgb_weights=rgb_weights, lab_metric=lab_metric,
            cmc_l=cmc_l, cmc_c=cmc_c,
            progress_callback=progress_callback, progress_range=(0.3, 1.0),
            cancel_event=cancel_event,
        )
    _report(progress_callback, 1.0, cancel_event)
    return mapped


def convert_all_modes(
    input_path: str | None,
    output_size: int | Tuple[int, int],
    palette: BeadPalette,
    keep_aspect: bool = True,
    resize_method: str = "nearest",
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
    input_image: np.ndarray | None = None,
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
    pseudo_gradient_strength: float = 0.0,
    use_super_sampling: bool = False,
    dither_method: str = "none",
    dither_strength: float = 1.0,
    post_mode_filter_size: int = 0,
    post_island_min_area: int = 0,
) -> list[dict[str, np.ndarray]]:
    """全ての変換モードで処理した結果を順番に返す。"""
    _report(progress_callback, 0.0, cancel_event)
    cv2 = _require_cv2()
    if input_image is not None:
        # 事前ノイズ除去などで渡されたRGB配列を優先する
        image_rgb = np.asarray(input_image, dtype=np.uint8)
    else:
        if input_path is None:
            raise ValueError("input_image または input_path を指定してください。")
        image_rgb = _load_image_rgb(input_path)
    orig_h, orig_w = image_rgb.shape[:2]
    target_w, target_h = _compute_resize((orig_h, orig_w), output_size, keep_aspect)

    interp = getattr(cv2, _RESIZE_METHOD_ATTR.get(resize_method.lower(), "INTER_NEAREST"))

    if use_super_sampling and interp == cv2.INTER_AREA:
        inter_w = target_w * 2
        inter_h = target_h * 2
        if inter_w < orig_w and inter_h < orig_h:
            image_rgb = cv2.resize(image_rgb, (inter_w, inter_h), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=interp)
    _report(progress_callback, 0.2, cancel_event)

    base = _apply_shading_suite(
        resized, target_w, target_h,
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
    base = _apply_pseudo_gradient(base, strength=pseudo_gradient_strength)

    results: list[dict[str, np.ndarray]] = []
    total = len(ALL_MODE_SPECS)
    span = 0.8 / max(1, total)
    for idx, spec in enumerate(ALL_MODE_SPECS):
        start = 0.2 + span * idx
        end = start + span
        mode = str(spec.get("mode", ""))
        label = str(spec.get("label", ""))
        if mode.lower() in {"none", "なし"}:
            # 変換なしはリサイズ結果をそのまま使う
            _report(progress_callback, start, cancel_event)
            results.append({"label": label, "image": base.copy()})
            _report(progress_callback, end, cancel_event)
            continue
        if dither_method != "none":
            from .dither import apply_dither
            _report(progress_callback, start, cancel_event)
            mapped = apply_dither(
                base,
                palette,
                dither_method,
                strength=dither_strength,
                mode=mode,
                lab_metric=str(spec.get("lab_metric", "CIEDE2000")),
                rgb_weights=rgb_weights,
                cmc_l=cmc_l,
                cmc_c=cmc_c,
                progress_callback=progress_callback,
                progress_range=(start, end),
                cancel_event=cancel_event,
            )
            if post_mode_filter_size > 0 or post_island_min_area > 0:
                pidx = _palette_rgb_to_index(mapped, palette)
                if post_mode_filter_size > 0:
                    pidx = _apply_mode_filter_index(pidx, post_mode_filter_size)
                if post_island_min_area > 0:
                    pidx = _apply_island_removal_index(pidx, post_island_min_area)
                mapped = palette.rgb_uint8[pidx]
        elif post_mode_filter_size > 0 or post_island_min_area > 0:
            pidx = _map_image_to_palette_index(
                base, palette, mode,
                rgb_weights=rgb_weights,
                lab_metric=str(spec.get("lab_metric", "CIEDE2000")),
                cmc_l=cmc_l, cmc_c=cmc_c,
                progress_callback=progress_callback,
                progress_range=(start, end),
                cancel_event=cancel_event,
            )
            if post_mode_filter_size > 0:
                pidx = _apply_mode_filter_index(pidx, post_mode_filter_size)
            if post_island_min_area > 0:
                pidx = _apply_island_removal_index(pidx, post_island_min_area)
            mapped = palette.rgb_uint8[pidx]
        else:
            mapped = _map_image_to_palette(
                base,
                palette,
                mode,
                rgb_weights=rgb_weights,
                lab_metric=str(spec.get("lab_metric", "CIEDE2000")),
                cmc_l=cmc_l,
                cmc_c=cmc_c,
                progress_callback=progress_callback,
                progress_range=(start, end),
                cancel_event=cancel_event,
            )
        results.append({"label": label, "image": mapped})

    _report(progress_callback, 1.0, cancel_event)
    return results
