"""Preview panel controller for color usage window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from .scale_utils import calc_scale_value_from_pointer


@dataclass(frozen=True)
class _PreviewBuffer:
    """描画済みバッファ。拡大画像座標系の [px0, px1) x [py0, py1) をグリッド込みで保持する。"""

    image_id: int
    scale: float
    grid_key: Optional[tuple]
    px0: int
    py0: int
    px1: int
    py1: int


@dataclass(frozen=True)
class _PreviewView:
    """現在のズーム・中心から求めた表示配置。"""

    scale: float
    origin_x: int
    origin_y: int
    scaled_w: int
    scaled_h: int
    vis_x0: int
    vis_y0: int
    vis_x1: int
    vis_y1: int
    margin_x: int
    margin_y: int


class ColorUsagePreviewController:
    # 表示範囲の外側に余分に描画しておく幅（ボックス寸法比）。
    # 余白が残っている間のパンは画像アイテムを動かすだけで済み、再描画もグリッドの消灯も起きない。
    _BUFFER_MARGIN_RATIO = 0.4
    _BUFFER_MARGIN_MIN = 64
    # 細グリッド線を描くビーズ1粒あたりの最小ピクセル数。これ未満だと線で画像が埋まってしまう。
    _MINOR_GRID_MIN_SCALE = 3.0
    _PLATE_SIZE = 28

    def __init__(
        self,
        canvas: tk.Canvas,
        grid_var: tk.BooleanVar,
        on_select: Callable[[Optional[tuple[int, int, int]]], None],
        dim_var: Optional[tk.DoubleVar] = None,
        empty_message: str = "",
    ) -> None:
        self._canvas = canvas
        self._grid_var = grid_var
        self._dim_var = dim_var
        self._on_select = on_select
        self._empty_message = empty_message

        self._selected_rgb: Optional[tuple[int, int, int]] = None
        self._preview_base_image: Optional[Image.Image] = None
        self._preview_source_image: Optional[Image.Image] = None
        self._preview_array: Optional[np.ndarray] = None
        self._preview_photo: Optional[ImageTk.PhotoImage] = None
        self._preview_buffer: Optional[_PreviewBuffer] = None
        self._preview_image_item: Optional[int] = None
        self._preview_text_item: Optional[int] = None
        self._preview_zoom = 1.0
        self._preview_zoom_step = 0.01
        # ビーズ1粒あたりの表示ピクセル数と、キャンバス上での拡大画像の原点
        self._preview_scale = 1.0
        self._preview_origin = (0, 0)
        # 表示中心（ビーズ座標。小数を許すので1粒未満の滑らかなパンができる）
        self._preview_center: Optional[tuple[float, float]] = None
        self._preview_pan_anchor: Optional[tuple[float, float]] = None
        self._preview_pan_center: Optional[tuple[float, float]] = None
        self._preview_zoom_anchor: Optional[tuple[float, float, float, float]] = None
        self._preview_pan_job: Optional[str] = None
        self._preview_zoom_job: Optional[str] = None
        self._preview_idle_job: Optional[str] = None
        self._preview_request_job: Optional[str] = None
        self._preview_idle_delay_ms = 150
        self._preview_box_size = (0, 0)

        self._canvas.bind("<Configure>", self._on_preview_resize)
        self._canvas.bind("<MouseWheel>", self._on_preview_wheel)
        self._canvas.bind("<Button-4>", self._on_preview_wheel)
        self._canvas.bind("<Button-5>", self._on_preview_wheel)
        self._canvas.bind("<Button-1>", self._on_preview_click)
        self._canvas.bind("<ButtonPress-3>", self._on_preview_pan_start)
        self._canvas.bind("<B3-Motion>", self._on_preview_pan_move)
        self._canvas.bind("<ButtonRelease-3>", self._on_preview_pan_end)

    def bind_shortcuts(self, window: tk.Toplevel) -> None:
        window.bind("<KeyPress>", self._on_preview_key, add="+")

    def on_grid_toggle(self) -> None:
        self._render_preview()

    def on_tone_pointer(self, event: tk.Event) -> str:
        new_val = calc_scale_value_from_pointer(event)
        if self._dim_var is not None:
            self._dim_var.set(new_val)
        return "break"

    def set_selected_rgb(self, rgb: Optional[tuple[int, int, int]]) -> None:
        self._selected_rgb = rgb
        self._request_render()

    def set_preview_image(self, image: Optional[Image.Image], source_image: Optional[Image.Image] = None) -> None:
        prev_center = self._preview_center
        prev_size = self._preview_base_image.size if self._preview_base_image is not None else None
        new_size = image.size if image is not None else None
        keep_center = prev_center is not None and prev_size is not None and new_size == prev_size
        if image is not self._preview_base_image:
            self._preview_array = None
            self._preview_buffer = None
            self._preview_center = prev_center if keep_center else None
            self._preview_pan_anchor = None
            self._preview_pan_center = None
            self._cancel_job("_preview_pan_job")
        self._preview_base_image = image
        self._preview_source_image = source_image if source_image is not None else image
        # 選択変更では set_selected_rgb と続けて呼ばれるので、描画は1回にまとめる
        self._request_render()

    # ------------------------------------------------------------------
    # 入力ハンドラ
    # ------------------------------------------------------------------
    def _on_preview_key(self, event: tk.Event) -> Optional[str]:
        keysym = str(getattr(event, "keysym", "")).lower()
        if keysym in {"w", "a", "s", "d"}:
            self._move_preview_by_key(keysym)
            return "break"
        if keysym in {"q", "e"}:
            self._zoom_preview_by_key(keysym)
            return "break"
        if keysym == "f":
            self._grid_var.set(not self._grid_var.get())
            self.on_grid_toggle()
            return "break"
        return None

    def _move_preview_by_key(self, key: str) -> None:
        if self._preview_base_image is None:
            return
        box_w, box_h = self._get_preview_box_size()
        step = max(12, int(min(box_w, box_h) * 0.07))
        dx = 0
        dy = 0
        if key == "w":
            dy = -step
        elif key == "s":
            dy = step
        elif key == "a":
            dx = -step
        elif key == "d":
            dx = step
        self._move_preview_by_pixels(dx, dy)

    def _move_preview_by_pixels(self, dx: int, dy: int) -> None:
        if self._preview_base_image is None or not self._can_pan():
            return
        scale = float(self._preview_scale or 1.0)
        img_w, img_h = self._preview_base_image.size
        center_x, center_y = self._preview_center or (img_w / 2, img_h / 2)
        self._preview_center = self._clamp_preview_center(center_x + dx / scale, center_y + dy / scale, scale)
        self._schedule_preview_pan_render()

    def _zoom_preview_by_key(self, key: str) -> None:
        if self._preview_base_image is None:
            return
        box_w, box_h = self._get_preview_box_size()
        delta = -120 if key == "q" else 120
        self._apply_preview_zoom(delta, box_w / 2, box_h / 2)

    def _on_preview_resize(self, _event: tk.Event) -> None:
        self._preview_box_size = (max(1, self._canvas.winfo_width()), max(1, self._canvas.winfo_height()))
        self._render_preview()

    def _on_preview_pan_start(self, event: tk.Event) -> None:
        if self._preview_base_image is None:
            return
        self._preview_pan_anchor = (float(event.x), float(event.y))
        img_w, img_h = self._preview_base_image.size
        if self._preview_center is None:
            self._preview_center = (img_w / 2, img_h / 2)
        self._preview_pan_center = self._preview_center

    def _on_preview_pan_move(self, event: tk.Event) -> None:
        if self._preview_base_image is None:
            return
        if self._preview_pan_anchor is None or self._preview_pan_center is None:
            return
        if not self._can_pan():
            return
        scale = float(self._preview_scale or 1.0)
        dx = float(event.x) - self._preview_pan_anchor[0]
        dy = float(event.y) - self._preview_pan_anchor[1]
        center_x = self._preview_pan_center[0] - dx / scale
        center_y = self._preview_pan_center[1] - dy / scale
        self._preview_center = self._clamp_preview_center(center_x, center_y, scale)
        self._schedule_preview_pan_render()

    def _on_preview_pan_end(self, _event: tk.Event) -> None:
        self._preview_pan_anchor = None
        self._preview_pan_center = None

    def _schedule_preview_pan_render(self) -> None:
        if self._preview_pan_job is not None:
            return
        self._preview_pan_job = self._canvas.after(16, self._flush_preview_pan)

    def _flush_preview_pan(self) -> None:
        self._preview_pan_job = None
        # 余白が残っていれば画像アイテムを動かすだけ。尽きたときだけバッファを作り直す
        self._render_preview()
        self._schedule_idle_refresh()

    def _on_preview_wheel(self, event: tk.Event) -> None:
        if self._preview_base_image is None:
            return
        delta = 0
        if getattr(event, "delta", 0):
            delta = int(event.delta)
        elif getattr(event, "num", None) == 4:
            delta = 120
        elif getattr(event, "num", None) == 5:
            delta = -120
        if delta == 0:
            return
        self._apply_preview_zoom(delta, float(event.x), float(event.y))

    def _apply_preview_zoom(self, delta: int, anchor_x: float, anchor_y: float) -> None:
        if self._preview_base_image is None:
            return
        box_w, box_h = self._get_preview_box_size()
        img_w, img_h = self._preview_base_image.size
        scale = max(float(self._preview_scale or 1.0), 1e-6)
        origin_x, origin_y = self._preview_origin
        cursor_x = float(self._canvas.canvasx(anchor_x))
        cursor_y = float(self._canvas.canvasy(anchor_y))
        base_x = (cursor_x - origin_x) / scale
        base_y = (cursor_y - origin_y) / scale
        if base_x < 0 or base_y < 0 or base_x >= img_w or base_y >= img_h:
            # 画像の外ではボックス中央を基準にする
            cursor_x = box_w / 2
            cursor_y = box_h / 2
            base_x = (cursor_x - origin_x) / scale
            base_y = (cursor_y - origin_y) / scale
        base_x = min(max(base_x, 0.0), float(img_w))
        base_y = min(max(base_y, 0.0), float(img_h))

        old_zoom = self._preview_zoom
        factor = 1.1
        if delta > 0:
            new_zoom = old_zoom * factor
        else:
            new_zoom = old_zoom / factor
        step = max(self._preview_zoom_step, 0.001)
        new_zoom = max(1.0, round(max(new_zoom, 1.0) / step) * step)
        if new_zoom == old_zoom:
            return
        self._preview_zoom = new_zoom
        if self._preview_zoom_anchor is None:
            self._preview_zoom_anchor = (base_x, base_y, cursor_x, cursor_y)
        if self._preview_zoom_job is None:
            self._preview_zoom_job = self._canvas.after(16, self._flush_preview_zoom)

    def _flush_preview_zoom(self) -> None:
        self._preview_zoom_job = None
        anchor = self._preview_zoom_anchor
        self._preview_zoom_anchor = None
        if self._preview_base_image is not None and anchor:
            box_w, box_h = self._get_preview_box_size()
            scale = self._compute_scale()
            base_x, base_y, cursor_x, cursor_y = anchor
            # 基準点 (base_x, base_y) がカーソル位置に留まるよう中心を求める
            center_x = base_x + (box_w / 2 - cursor_x) / scale
            center_y = base_y + (box_h / 2 - cursor_y) / scale
            self._preview_center = self._clamp_preview_center(center_x, center_y, scale)
        # 連続ホイール中は表示範囲だけ描き、止まってから余白付きで描き直す
        self._render_preview(with_margin=False)
        self._schedule_idle_refresh()

    def _on_preview_click(self, event: tk.Event) -> Optional[str]:
        if self._preview_source_image is None or self._preview_base_image is None:
            return None
        img_w, img_h = self._preview_base_image.size
        scale = float(self._preview_scale or 1.0)
        origin_x, origin_y = self._preview_origin
        local_x = int(np.floor(float(self._canvas.canvasx(event.x)) - origin_x))
        local_y = int(np.floor(float(self._canvas.canvasy(event.y)) - origin_y))
        bounds_x = self._bounds(img_w, scale)
        bounds_y = self._bounds(img_h, scale)
        if local_x < 0 or local_y < 0 or local_x >= bounds_x[-1] or local_y >= bounds_y[-1]:
            return None
        img_x = int(np.searchsorted(bounds_x, local_x, side="right")) - 1
        img_y = int(np.searchsorted(bounds_y, local_y, side="right")) - 1
        src_w, src_h = self._preview_source_image.size
        if img_x < 0 or img_y < 0 or img_x >= src_w or img_y >= src_h:
            return None
        r, g, b = self._preview_source_image.getpixel((img_x, img_y))[:3]
        rgb = (int(r), int(g), int(b))
        if self._selected_rgb == rgb:
            self._on_select(None)
        else:
            self._on_select(rgb)
        return "break"

    # ------------------------------------------------------------------
    # ジョブ管理
    # ------------------------------------------------------------------
    def _cancel_job(self, attr: str) -> None:
        job = getattr(self, attr)
        if job is None:
            return
        try:
            self._canvas.after_cancel(job)
        except Exception:
            pass
        setattr(self, attr, None)

    def _request_render(self) -> None:
        if self._preview_request_job is not None:
            return
        self._preview_request_job = self._canvas.after_idle(self._flush_requested_render)

    def _flush_requested_render(self) -> None:
        self._preview_request_job = None
        try:
            self._render_preview()
        except tk.TclError:
            pass

    def _schedule_idle_refresh(self) -> None:
        self._cancel_job("_preview_idle_job")
        self._preview_idle_job = self._canvas.after(self._preview_idle_delay_ms, self._refresh_buffer_margin)

    def _refresh_buffer_margin(self) -> None:
        """操作が止まったら、表示位置を中心に余白付きのバッファを作り直して次のパンに備える。"""
        self._preview_idle_job = None
        if self._preview_base_image is None or self._preview_buffer is None:
            return
        if self._preview_pan_job is not None or self._preview_zoom_job is not None:
            return
        view = self._compute_view()
        buf = self._preview_buffer
        want_x = view.margin_x // 2
        want_y = view.margin_y // 2
        short = (
            buf.px0 > max(0, view.vis_x0 - want_x)
            or buf.py0 > max(0, view.vis_y0 - want_y)
            or buf.px1 < min(view.scaled_w, view.vis_x1 + want_x)
            or buf.py1 < min(view.scaled_h, view.vis_y1 + want_y)
        )
        if short:
            self._preview_buffer = None
            self._render_preview()

    # ------------------------------------------------------------------
    # 座標計算
    # ------------------------------------------------------------------
    @staticmethod
    def _bounds(count: int, scale: float) -> np.ndarray:
        """ビーズ境界 i (0..count) の拡大画像上のピクセル位置。全描画・クリック判定で共通に使う。"""
        return np.floor(np.arange(count + 1, dtype=np.float64) * scale + 0.5).astype(np.int64)

    @staticmethod
    def _scaled_size(count: int, scale: float) -> int:
        return max(1, int(np.floor(count * scale + 0.5)))

    def _compute_scale(self) -> float:
        box_w, box_h = self._get_preview_box_size()
        img_w, img_h = self._preview_base_image.size
        base_scale = max(min(box_w / img_w, box_h / img_h), 0.01)
        return base_scale * max(self._preview_zoom, 1.0)

    def _can_pan(self) -> bool:
        if self._preview_base_image is None:
            return False
        box_w, box_h = self._get_preview_box_size()
        img_w, img_h = self._preview_base_image.size
        scale = float(self._preview_scale or 1.0)
        return self._scaled_size(img_w, scale) > box_w or self._scaled_size(img_h, scale) > box_h

    def _clamp_preview_center(self, center_x: float, center_y: float, scale: float) -> tuple[float, float]:
        box_w, box_h = self._get_preview_box_size()
        img_w, img_h = self._preview_base_image.size
        return (
            self._clamp_axis(center_x, scale, img_w, box_w),
            self._clamp_axis(center_y, scale, img_h, box_h),
        )

    def _clamp_axis(self, center: float, scale: float, count: int, box: int) -> float:
        if self._scaled_size(count, scale) <= box:
            return count / 2
        half = box / (2 * scale)
        return min(max(center, half), count - half)

    @staticmethod
    def _axis_origin(center: float, scale: float, scaled: int, box: int) -> int:
        if scaled <= box:
            return (box - scaled) // 2
        view0 = int(round(center * scale - box / 2))
        view0 = max(0, min(view0, scaled - box))
        return -view0

    def _compute_view(self) -> _PreviewView:
        box_w, box_h = self._get_preview_box_size()
        img_w, img_h = self._preview_base_image.size
        scale = self._compute_scale()
        scaled_w = self._scaled_size(img_w, scale)
        scaled_h = self._scaled_size(img_h, scale)
        center_x, center_y = self._preview_center or (img_w / 2, img_h / 2)
        center_x, center_y = self._clamp_preview_center(center_x, center_y, scale)
        self._preview_center = (center_x, center_y)
        origin_x = self._axis_origin(center_x, scale, scaled_w, box_w)
        origin_y = self._axis_origin(center_y, scale, scaled_h, box_h)
        return _PreviewView(
            scale=scale,
            origin_x=origin_x,
            origin_y=origin_y,
            scaled_w=scaled_w,
            scaled_h=scaled_h,
            vis_x0=max(0, -origin_x),
            vis_y0=max(0, -origin_y),
            vis_x1=min(scaled_w, box_w - origin_x),
            vis_y1=min(scaled_h, box_h - origin_y),
            margin_x=max(self._BUFFER_MARGIN_MIN, int(box_w * self._BUFFER_MARGIN_RATIO)),
            margin_y=max(self._BUFFER_MARGIN_MIN, int(box_h * self._BUFFER_MARGIN_RATIO)),
        )

    def _get_preview_box_size(self) -> tuple[int, int]:
        box_w, box_h = self._preview_box_size
        if box_w > 0 and box_h > 0:
            return box_w, box_h
        box_w = self._canvas.winfo_width() or self._canvas.winfo_reqwidth() or 200
        box_h = self._canvas.winfo_height() or self._canvas.winfo_reqheight() or 200
        self._preview_box_size = (box_w, box_h)
        return box_w, box_h

    # ------------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------------
    def _get_grid_line_color(self) -> tuple[int, int, int, int]:
        if self._selected_rgb is None:
            return (0, 0, 0, 90)
        r, g, b = (int(self._selected_rgb[0]), int(self._selected_rgb[1]), int(self._selected_rgb[2]))
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        tone = int(round(255 - luminance))
        tone = max(0, min(255, tone))
        return (tone, tone, tone, 120)

    def _grid_key(self, scale: float) -> Optional[tuple]:
        if not self._grid_var.get():
            return None
        r, g, b, a = self._get_grid_line_color()
        return ((r, g, b), a, scale >= self._MINOR_GRID_MIN_SCALE)

    def _get_preview_array(self) -> np.ndarray:
        if self._preview_array is None:
            self._preview_array = np.asarray(self._preview_base_image.convert("RGB"))
        return self._preview_array

    def _grid_positions(
        self, bounds: np.ndarray, b0: int, b1: int, origin_px: int, size: int, draw_minor: bool
    ) -> np.ndarray:
        """バッファ内でグリッド線を引くピクセル列（行）。プレート境界は2px幅。"""
        idx = np.arange(max(1, b0), b1)
        if idx.size == 0:
            return np.empty(0, dtype=np.int64)
        major = (idx % self._PLATE_SIZE) == 0
        if not draw_minor:
            idx = idx[major]
            major = major[major]
        pos = bounds[idx] - origin_px
        major_pos = pos[major]
        extra = np.where(major_pos > 0, major_pos - 1, major_pos + 1)
        lines = np.unique(np.concatenate([pos, extra]))
        return lines[(lines >= 0) & (lines < size)]

    @staticmethod
    def _blend(pixels: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
        src = pixels.astype(np.float32)
        return (src + (color - src) * alpha + 0.5).astype(np.uint8)

    def _build_buffer(self, view: _PreviewView, grid_key: Optional[tuple], with_margin: bool) -> _PreviewBuffer:
        arr = self._get_preview_array()
        img_h, img_w = arr.shape[:2]
        scale = view.scale
        bounds_x = self._bounds(img_w, scale)
        bounds_y = self._bounds(img_h, scale)
        mx = view.margin_x if with_margin else 0
        my = view.margin_y if with_margin else 0
        px0 = max(0, view.vis_x0 - mx)
        py0 = max(0, view.vis_y0 - my)
        px1 = min(int(bounds_x[-1]), view.vis_x1 + mx)
        py1 = min(int(bounds_y[-1]), view.vis_y1 + my)
        # ビーズ境界まで広げて、端のビーズが欠けないようにする
        bx0 = max(0, int(np.searchsorted(bounds_x, px0, side="right")) - 1)
        by0 = max(0, int(np.searchsorted(bounds_y, py0, side="right")) - 1)
        bx1 = min(img_w, max(bx0 + 1, int(np.searchsorted(bounds_x, px1, side="left"))))
        by1 = min(img_h, max(by0 + 1, int(np.searchsorted(bounds_y, py1, side="left"))))
        px0, px1 = int(bounds_x[bx0]), int(bounds_x[bx1])
        py0, py1 = int(bounds_y[by0]), int(bounds_y[by1])

        # 最近傍拡大を np.repeat で行う。境界は _bounds と一致するのでグリッド線とずれない
        widths = np.diff(bounds_x[bx0 : bx1 + 1])
        heights = np.diff(bounds_y[by0 : by1 + 1])
        out = np.repeat(np.repeat(arr[by0:by1, bx0:bx1], heights, axis=0), widths, axis=1)

        if grid_key is not None and out.size:
            rgb, alpha, draw_minor = grid_key
            color = np.asarray(rgb, dtype=np.float32)
            a = alpha / 255.0
            out_h, out_w = out.shape[:2]
            rows = self._grid_positions(bounds_y, by0, by1, py0, out_h, draw_minor)
            cols = self._grid_positions(bounds_x, bx0, bx1, px0, out_w, draw_minor)
            if rows.size:
                out[rows] = self._blend(out[rows], color, a)
            if cols.size:
                # 交点は行側で塗り済みなので二重に濃くしない
                keep = np.ones(out_h, dtype=bool)
                keep[rows] = False
                keep_rows = np.nonzero(keep)[0]
                if keep_rows.size:
                    sel = np.ix_(keep_rows, cols)
                    out[sel] = self._blend(out[sel], color, a)

        self._preview_photo = ImageTk.PhotoImage(Image.fromarray(out, "RGB"))
        return _PreviewBuffer(
            image_id=id(self._preview_base_image),
            scale=scale,
            grid_key=grid_key,
            px0=px0,
            py0=py0,
            px1=px1,
            py1=py1,
        )

    def _buffer_covers(self, buf: Optional[_PreviewBuffer], view: _PreviewView, grid_key: Optional[tuple]) -> bool:
        return (
            buf is not None
            and buf.image_id == id(self._preview_base_image)
            and buf.scale == view.scale
            and buf.grid_key == grid_key
            and buf.px0 <= view.vis_x0
            and buf.py0 <= view.vis_y0
            and buf.px1 >= view.vis_x1
            and buf.py1 >= view.vis_y1
        )

    def _render_preview(self, with_margin: bool = True) -> None:
        box_w, box_h = self._get_preview_box_size()
        if self._preview_base_image is None:
            if self._preview_image_item is not None:
                self._canvas.delete(self._preview_image_item)
                self._preview_image_item = None
            self._preview_photo = None
            self._preview_buffer = None
            if self._preview_text_item is None:
                self._preview_text_item = self._canvas.create_text(
                    box_w / 2,
                    box_h / 2,
                    text=self._empty_message,
                )
            else:
                self._canvas.itemconfig(self._preview_text_item, text=self._empty_message)
                self._canvas.coords(self._preview_text_item, box_w / 2, box_h / 2)
            self._preview_scale = 1.0
            self._preview_origin = (0, 0)
            self._preview_center = None
            return
        if self._preview_text_item is not None:
            self._canvas.delete(self._preview_text_item)
            self._preview_text_item = None

        view = self._compute_view()
        self._preview_scale = view.scale
        self._preview_origin = (view.origin_x, view.origin_y)
        grid_key = self._grid_key(view.scale)
        buf = self._preview_buffer
        rebuilt = False
        if not self._buffer_covers(buf, view, grid_key):
            buf = self._build_buffer(view, grid_key, with_margin)
            self._preview_buffer = buf
            rebuilt = True

        pos_x = view.origin_x + buf.px0
        pos_y = view.origin_y + buf.py0
        if self._preview_image_item is None:
            self._preview_image_item = self._canvas.create_image(
                pos_x,
                pos_y,
                anchor="nw",
                image=self._preview_photo,
            )
        else:
            if rebuilt:
                self._canvas.itemconfig(self._preview_image_item, image=self._preview_photo)
            self._canvas.coords(self._preview_image_item, pos_x, pos_y)
