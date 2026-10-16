"""変換パイプライン（リサイズ + パレット写像）。"""

from __future__ import annotations

from typing import Callable, Tuple
import threading

import numpy as np

from cv2_utils import require_cv2 as _require_cv2
from palette import BeadPalette
from .io_utils import _compute_resize, _load_image_rgb
from .models import ShadingConfig, DitherConfig, PostFilterConfig
from .quantize import _map_centers_to_palette, _report
from .shading import (  # noqa: F401
    _apply_shading_suite,
    _apply_pseudo_gradient,
    _apply_local_contrast,
    apply_shading_preview,
)

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


def _lab_distance_limit(protect: float) -> float:
    """置換を許容する Lab 距離の上限。保護が強いほど厳しくなる。"""
    return 20.0 * (1.0 - protect) + 5.0


def _circular_kernel(kernel_size: int, protect: float) -> np.ndarray:
    """円形カーネルを作る。中心には保護強度に応じた重みを乗せる。

    正方形カーネルは角のピクセルまで等票なので角を削り落とす。円形にして
    さらに中心票を重くすることで、細い線や 1 ビーズのディテールが
    周囲の多数決で簡単に潰されないようにする。
    """
    radius = kernel_size // 2
    offsets = np.arange(kernel_size, dtype=np.float32) - radius
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    kernel = ((dx ** 2 + dy ** 2) <= radius ** 2 + 1e-6).astype(np.float32)
    # 中心重みはカーネルサイズに依存させない。k に比例させると k=3 では
    # 近傍票が中心を上回れなくなり、フィルタがまったく効かなくなる
    kernel[radius, radius] = 1.0 + 2.0 * protect
    return kernel


def _apply_mode_filter_index(
    index_arr: np.ndarray,
    palette: BeadPalette,
    kernel_size: int,
    protect: float,
) -> np.ndarray:
    """パレットインデックス配列にモードフィルタを適用する（後処理）。

    近傍の最頻インデックスへ置換して平坦部のちらつきをならすが、
    エッジと小さなディテールは残す。従来の単純多数決との違いは 3 点：

    - 円形カーネル＋中心重み（`_circular_kernel`）で角落ちを防ぐ
    - 最頻票が過半数閾値を超えたピクセルだけ置換する
    - 元色から Lab 距離で離れすぎた色には置換しない

    同数票は Lab 距離が近い側を選ぶ（従来はパレットの並び順依存だった）。
    """
    cv2 = _require_cv2()
    kernel = _circular_kernel(kernel_size, protect)
    total_weight = float(kernel.sum())
    threshold = total_weight * (0.5 + 0.3 * protect)

    lab = palette.lab_array.astype(np.float32)
    orig_lab = lab[index_arr]

    best_count = np.full(index_arr.shape, -1.0, dtype=np.float32)
    best_dist = np.full(index_arr.shape, np.inf, dtype=np.float32)
    best_idx = index_arr.copy()
    for idx in np.unique(index_arr):
        mask = (index_arr == idx).astype(np.float32)
        counts = cv2.filter2D(
            mask, -1, kernel, borderType=cv2.BORDER_REPLICATE
        )
        dist = np.linalg.norm(orig_lab - lab[idx], axis=-1).astype(np.float32)
        better = (counts > best_count) | ((counts == best_count) & (dist < best_dist))
        best_count[better] = counts[better]
        best_dist[better] = dist[better]
        best_idx[better] = idx

    replace = (best_count > threshold) & (best_dist <= _lab_distance_limit(protect))
    return np.where(replace, best_idx, index_arr)


def _pick_border_index(
    border_indices: np.ndarray,
    origin_idx: int,
    palette: BeadPalette,
    protect: float,
) -> int | None:
    """島の境界に最も長く接する色を選ぶ。同数は Lab 距離で解決する。

    元色から離れすぎた色しか候補にない場合は None を返して置換を見送る。
    """
    counts = np.bincount(border_indices.astype(np.intp))
    candidates = np.flatnonzero(counts == counts.max())
    lab = palette.lab_array
    dists = np.linalg.norm(lab[candidates] - lab[origin_idx], axis=-1)
    nearest = int(candidates[int(np.argmin(dists))])
    if float(dists.min()) > _lab_distance_limit(protect):
        return None
    return nearest


def _apply_island_removal_index(
    index_arr: np.ndarray,
    palette: BeadPalette,
    min_area: int,
    protect: float,
) -> np.ndarray:
    """パレットインデックス配列から細切れのディザノイズを除去する（後処理）。

    面積だけで判定すると目やハイライトのような意図的な小ディテールまで
    消えてしまうため、`周囲長 / 面積` が高い（＝細切れの）領域だけを対象にする。
    面積の小さい順に、逐次更新した結果配列を参照しながら処理するので、
    隣接する島同士が互いの消える前の色を参照して連鎖崩壊することはない。
    """
    cv2 = _require_cv2()
    result = index_arr.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    shape_limit = 1.5 + protect

    height, width = index_arr.shape
    # 島ごとに全画面マスクを持つと島の数×画素数でメモリと時間が膨らむため、
    # 外接矩形を 1px 広げた範囲（境界判定に必要な最小範囲）だけを切り出して保持する
    islands: list[tuple[int, int, int, np.ndarray]] = []
    for idx in np.unique(index_arr):
        mask = (index_arr == idx).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label_id in range(1, n_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                x = int(stats[label_id, cv2.CC_STAT_LEFT])
                y = int(stats[label_id, cv2.CC_STAT_TOP])
                w = int(stats[label_id, cv2.CC_STAT_WIDTH])
                h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
                y0, x0 = max(y - 1, 0), max(x - 1, 0)
                y1, x1 = min(y + h + 1, height), min(x + w + 1, width)
                islands.append((area, y0, x0, labels[y0:y1, x0:x1] == label_id))
    islands.sort(key=lambda item: item[0])

    for area, y0, x0, island_mask in islands:
        dilated = cv2.dilate(
            island_mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        border = dilated.astype(bool) & ~island_mask
        if not border.any():
            continue
        # 周囲長 / 面積。塊状なら小さく、細切れなら大きくなる
        if area > 1 and border.sum() / area <= shape_limit:
            continue
        region = result[y0:y0 + island_mask.shape[0], x0:x0 + island_mask.shape[1]]
        origin_idx = int(region[island_mask][0])
        replacement = _pick_border_index(region[border], origin_idx, palette, protect)
        if replacement is not None:
            region[island_mask] = replacement
    return result


def _apply_post_filters_index(
    index_arr: np.ndarray,
    palette: BeadPalette,
    post_filter: PostFilterConfig,
) -> np.ndarray:
    """モードフィルタ・島除去をインデックス配列へ順に適用する。"""
    protect = max(0.0, min(1.0, float(post_filter.post_protect_strength)))
    if post_filter.post_mode_filter_size > 0:
        index_arr = _apply_mode_filter_index(
            index_arr, palette, post_filter.post_mode_filter_size, protect
        )
    if post_filter.post_island_min_area > 0:
        index_arr = _apply_island_removal_index(
            index_arr, palette, post_filter.post_island_min_area, protect
        )
    return index_arr


def _prepare_base_image(
    input_path: str | None,
    input_image: np.ndarray | None,
    output_size: int | Tuple[int, int],
    keep_aspect: bool,
    resize_method: str,
    use_super_sampling: bool,
    shading: ShadingConfig,
    progress_callback: ProgressCb | None,
    cancel_event: CancelEvent | None,
    resize_progress: float,
) -> np.ndarray:
    """画像の読み込み・リサイズ・シェーディング適用までの共通前処理。"""
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
        # INTER_AREA時は2倍サイズを経由して細部を残す
        inter_w = target_w * 2
        inter_h = target_h * 2
        if inter_w < orig_w and inter_h < orig_h:
            image_rgb = cv2.resize(image_rgb, (inter_w, inter_h), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=interp)
    _report(progress_callback, resize_progress, cancel_event)

    base = _apply_shading_suite(resized, target_w, target_h, shading)
    base = _apply_pseudo_gradient(base, strength=shading.pseudo_gradient_strength)
    # 局所コントラスト強調は縮小で効果が消えないよう、必ずリサイズ後・量子化直前に置く
    return _apply_local_contrast(
        base,
        amount=shading.local_contrast_amount,
        sigma=shading.local_contrast_sigma,
    )


def _convert_single_mode(
    base: np.ndarray,
    palette: BeadPalette,
    mode: str,
    lab_metric: str,
    cmc_l: float,
    cmc_c: float,
    rgb_weights: tuple[float, float, float],
    dither: DitherConfig,
    post_filter: PostFilterConfig,
    progress_callback: ProgressCb | None,
    progress_range: tuple[float, float],
    cancel_event: CancelEvent | None,
) -> np.ndarray:
    """前処理済み画像を1つのモードでパレットへ写像する。"""
    if mode.lower() in {"none", "なし"}:
        # 変換なしは前処理結果をそのまま返す
        _report(progress_callback, progress_range[1], cancel_event)
        return base.copy()

    if dither.dither_method != "none":
        from .dither import apply_dither
        result = apply_dither(
            base, palette, dither.dither_method,
            strength=dither.dither_strength,
            mode=mode, lab_metric=lab_metric,
            rgb_weights=rgb_weights, cmc_l=cmc_l, cmc_c=cmc_c,
            progress_callback=progress_callback,
            progress_range=progress_range,
            cancel_event=cancel_event,
        )
        if post_filter.enabled:
            idx = _palette_rgb_to_index(result, palette)
            result = palette.rgb_uint8[_apply_post_filters_index(idx, palette, post_filter)]
        return result

    if post_filter.enabled:
        idx = _map_image_to_palette_index(
            base, palette, mode,
            rgb_weights=rgb_weights, lab_metric=lab_metric,
            cmc_l=cmc_l, cmc_c=cmc_c,
            progress_callback=progress_callback,
            progress_range=progress_range,
            cancel_event=cancel_event,
        )
        return palette.rgb_uint8[_apply_post_filters_index(idx, palette, post_filter)]

    return _map_image_to_palette(
        base, palette, mode,
        rgb_weights=rgb_weights, lab_metric=lab_metric,
        cmc_l=cmc_l, cmc_c=cmc_c,
        progress_callback=progress_callback,
        progress_range=progress_range,
        cancel_event=cancel_event,
    )


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
    use_super_sampling: bool = False,
    shading: ShadingConfig | None = None,
    dither: DitherConfig | None = None,
    post_filter: PostFilterConfig | None = None,
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
    input_image: np.ndarray | None = None,
) -> np.ndarray:
    """入力画像を指定サイズへリサイズし、パレットへ写像して返す。"""
    _report(progress_callback, 0.0, cancel_event)
    shading = shading if shading is not None else ShadingConfig()
    dither = dither if dither is not None else DitherConfig()
    post_filter = post_filter if post_filter is not None else PostFilterConfig()

    base = _prepare_base_image(
        input_path, input_image, output_size, keep_aspect, resize_method,
        use_super_sampling, shading, progress_callback, cancel_event,
        resize_progress=0.3,
    )
    result = _convert_single_mode(
        base, palette, mode, lab_metric, cmc_l, cmc_c, rgb_weights,
        dither, post_filter, progress_callback, (0.3, 1.0), cancel_event,
    )
    _report(progress_callback, 1.0, cancel_event)
    return result


def convert_all_modes(
    input_path: str | None,
    output_size: int | Tuple[int, int],
    palette: BeadPalette,
    keep_aspect: bool = True,
    resize_method: str = "nearest",
    cmc_l: float = 2.0,
    cmc_c: float = 1.0,
    rgb_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_super_sampling: bool = False,
    shading: ShadingConfig | None = None,
    dither: DitherConfig | None = None,
    post_filter: PostFilterConfig | None = None,
    progress_callback: ProgressCb | None = None,
    cancel_event: CancelEvent | None = None,
    input_image: np.ndarray | None = None,
) -> list[dict[str, np.ndarray]]:
    """全ての変換モードで処理した結果を順番に返す。"""
    _report(progress_callback, 0.0, cancel_event)
    shading = shading if shading is not None else ShadingConfig()
    dither = dither if dither is not None else DitherConfig()
    post_filter = post_filter if post_filter is not None else PostFilterConfig()

    base = _prepare_base_image(
        input_path, input_image, output_size, keep_aspect, resize_method,
        use_super_sampling, shading, progress_callback, cancel_event,
        resize_progress=0.2,
    )

    results: list[dict[str, np.ndarray]] = []
    total = len(ALL_MODE_SPECS)
    span = 0.8 / max(1, total)
    for idx, spec in enumerate(ALL_MODE_SPECS):
        start = 0.2 + span * idx
        _report(progress_callback, start, cancel_event)
        mapped = _convert_single_mode(
            base, palette,
            mode=str(spec.get("mode", "")),
            lab_metric=str(spec.get("lab_metric", "CIEDE2000")),
            cmc_l=cmc_l, cmc_c=cmc_c, rgb_weights=rgb_weights,
            dither=dither, post_filter=post_filter,
            progress_callback=progress_callback,
            progress_range=(start, start + span),
            cancel_event=cancel_event,
        )
        results.append({"label": str(spec.get("label", "")), "image": mapped})

    _report(progress_callback, 1.0, cancel_event)
    return results
