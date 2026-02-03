"""Preview panel controller for color usage window."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

from .scale_utils import calc_scale_value_from_pointer


class ColorUsagePreviewController:
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
        self._preview_photo: Optional[ImageTk.PhotoImage] = None
        self._preview_zoom = 1.0
        self._preview_image_item: Optional[int] = None
        self._preview_text_item: Optional[int] = None
        self._preview_scale = 1.0
        self._preview_image_pos = (0.0, 0.0)
        self._preview_region_size = (0, 0)
        self._preview_view_origin = (0.0, 0.0)
        self._preview_view_size = (0, 0)
        self._preview_center: Optional[tuple[float, float]] = None
        self._preview_pan_anchor: Optional[tuple[float, float]] = None
        self._preview_pan_center: Optional[tuple[float, float]] = None
        self._preview_render_job: Optional[str] = None
        self._preview_pan_job: Optional[str] = None
        self._preview_zoom_anchor: Optional[tuple[float, float, float, float]] = None
        self._preview_cache: OrderedDict[tuple, ImageTk.PhotoImage] = OrderedDict()
        self._preview_cache_limit = 8
        self._preview_grid_suspended = False
        self._preview_grid_deferred_job: Optional[str] = None
        self._preview_grid_delay_ms = 120
        self._preview_box_size = (0, 0)
        self._preview_zoom_step = 0.01

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
        self._preview_grid_suspended = False
        if self._preview_grid_deferred_job is not None:
            try:
                self._canvas.after_cancel(self._preview_grid_deferred_job)
            except Exception:
                pass
            self._preview_grid_deferred_job = None
        self._render_preview()

    def on_tone_pointer(self, event: tk.Event) -> str:
        new_val = calc_scale_value_from_pointer(event)
        if self._dim_var is not None:
            self._dim_var.set(new_val)
        return "break"

    def set_selected_rgb(self, rgb: Optional[tuple[int, int, int]]) -> None:
        self._selected_rgb = rgb
        self._render_preview()

    def set_preview_image(self, image: Optional[Image.Image], source_image: Optional[Image.Image] = None) -> None:
        prev_center = self._preview_center
        prev_size = self._preview_base_image.size if self._preview_base_image is not None else None
        new_size = image.size if image is not None else None
        keep_center = prev_center is not None and prev_size is not None and new_size == prev_size
        if image is not self._preview_base_image:
            self._preview_cache.clear()
            self._preview_grid_suspended = False
            if self._preview_grid_deferred_job is not None:
                try:
                    self._canvas.after_cancel(self._preview_grid_deferred_job)
                except Exception:
                    pass
                self._preview_grid_deferred_job = None
            self._preview_center = prev_center if keep_center else None
            self._preview_pan_anchor = None
            self._preview_pan_center = None
            if self._preview_pan_job is not None:
                try:
                    self._canvas.after_cancel(self._preview_pan_job)
                except Exception:
                    pass
                self._preview_pan_job = None
        self._preview_base_image = image
        self._preview_source_image = source_image if source_image is not None else image
        self._render_preview()

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
        img_w, img_h = self._preview_base_image.size
        scale = float(self._preview_scale or 1.0)
        view_w, view_h = self._get_preview_view_size(scale, box_w, box_h, img_w, img_h)
        if view_w >= img_w and view_h >= img_h:
            return
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
        if self._preview_base_image is None:
            return
        box_w, box_h = self._get_preview_box_size()
        img_w, img_h = self._preview_base_image.size
        scale = float(self._preview_scale or 1.0)
        if scale <= 0:
            return
        view_w, view_h = self._get_preview_view_size(scale, box_w, box_h, img_w, img_h)
        if view_w >= img_w and view_h >= img_h:
            return
        center_x, center_y = self._preview_center or (img_w / 2, img_h / 2)
        center_x += dx / scale
        center_y += dy / scale
        self._preview_center = self._clamp_preview_center(center_x, center_y, view_w, view_h, img_w, img_h)
        if self._grid_var.get():
            self._preview_grid_suspended = True
            if self._preview_grid_deferred_job is not None:
                try:
                    self._canvas.after_cancel(self._preview_grid_deferred_job)
                except Exception:
                    pass
            self._preview_grid_deferred_job = self._canvas.after(
                self._preview_grid_delay_ms, self._flush_grid_overlay
            )
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
        scale = float(self._preview_scale or 1.0)
        if scale <= 0:
            return
        box_w, box_h = self._get_preview_box_size()
        img_w, img_h = self._preview_base_image.size
        view_w, view_h = self._get_preview_view_size(scale, box_w, box_h, img_w, img_h)
        if view_w >= img_w and view_h >= img_h:
            return
        dx = float(event.x) - self._preview_pan_anchor[0]
        dy = float(event.y) - self._preview_pan_anchor[1]
        center_x = self._preview_pan_center[0] - dx / scale
        center_y = self._preview_pan_center[1] - dy / scale
        self._preview_center = self._clamp_preview_center(center_x, center_y, view_w, view_h, img_w, img_h)
        if self._grid_var.get():
            self._preview_grid_suspended = True
            if self._preview_grid_deferred_job is not None:
                try:
                    self._canvas.after_cancel(self._preview_grid_deferred_job)
                except Exception:
                    pass
            self._preview_grid_deferred_job = self._canvas.after(
                self._preview_grid_delay_ms, self._flush_grid_overlay
            )
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
        self._render_preview()

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
        base_scale = min(box_w / img_w, box_h / img_h)
        base_scale = max(base_scale, 0.01)
        old_zoom = self._preview_zoom
        cursor_canvas_x = float(self._canvas.canvasx(anchor_x))
        cursor_canvas_y = float(self._canvas.canvasy(anchor_y))
        pos_x, pos_y = self._preview_image_pos
        draw_w, draw_h = self._preview_region_size
        view_x0, view_y0 = self._preview_view_origin
        scale = float(self._preview_scale or base_scale)
        local_x = cursor_canvas_x - pos_x
        local_y = cursor_canvas_y - pos_y
        if local_x < 0 or local_y < 0 or local_x >= draw_w or local_y >= draw_h:
            base_x = view_x0 + (draw_w / 2) / max(scale, 1e-6)
            base_y = view_y0 + (draw_h / 2) / max(scale, 1e-6)
        else:
            base_x = view_x0 + local_x / max(scale, 1e-6)
            base_y = view_y0 + local_y / max(scale, 1e-6)
        base_x = min(max(base_x, 0.0), float(img_w))
        base_y = min(max(base_y, 0.0), float(img_h))

        factor = 1.1
        if delta > 0:
            new_zoom = old_zoom * factor
        else:
            new_zoom = old_zoom / factor
        if new_zoom < 1.0:
            new_zoom = 1.0
        step = max(self._preview_zoom_step, 0.001)
        new_zoom = round(new_zoom / step) * step
        if new_zoom < 1.0:
            new_zoom = 1.0
        if new_zoom == old_zoom:
            return
        self._preview_zoom = new_zoom
        self._preview_zoom_anchor = (base_x, base_y, float(anchor_x), float(anchor_y))
        if self._grid_var.get():
            self._preview_grid_suspended = True
            if self._preview_grid_deferred_job is not None:
                try:
                    self._canvas.after_cancel(self._preview_grid_deferred_job)
                except Exception:
                    pass
            self._preview_grid_deferred_job = self._canvas.after(
                self._preview_grid_delay_ms, self._flush_grid_overlay
            )
        if self._preview_render_job is not None:
            try:
                self._canvas.after_cancel(self._preview_render_job)
            except Exception:
                pass
        self._preview_render_job = self._canvas.after(16, self._flush_preview_zoom)

    def _on_preview_click(self, event: tk.Event) -> Optional[str]:
        if self._preview_source_image is None or self._preview_base_image is None:
            return None
        scale = float(self._preview_scale or 1.0)
        pos_x, pos_y = self._preview_image_pos
        draw_w, draw_h = self._preview_region_size
        view_x0, view_y0 = self._preview_view_origin
        canvas_x = float(self._canvas.canvasx(event.x))
        canvas_y = float(self._canvas.canvasy(event.y))
        local_x = canvas_x - pos_x
        local_y = canvas_y - pos_y
        if local_x < 0 or local_y < 0 or local_x >= draw_w or local_y >= draw_h:
            return None
        img_x = view_x0 + local_x / max(scale, 1e-6)
        img_y = view_y0 + local_y / max(scale, 1e-6)
        img_w, img_h = self._preview_source_image.size
        if img_x < 0 or img_y < 0 or img_x >= img_w or img_y >= img_h:
            return None
        r, g, b = self._preview_source_image.getpixel((int(img_x), int(img_y)))
        rgb = (int(r), int(g), int(b))
        if self._selected_rgb == rgb:
            self._on_select(None)
        else:
            self._on_select(rgb)
        return "break"

    def _flush_preview_zoom(self) -> None:
        self._preview_render_job = None
        anchor = self._preview_zoom_anchor
        if self._preview_base_image is not None and anchor:
            box_w, box_h = self._get_preview_box_size()
            img_w, img_h = self._preview_base_image.size
            base_scale = min(box_w / img_w, box_h / img_h)
            base_scale = max(base_scale, 0.01)
            scale = base_scale * max(self._preview_zoom, 1.0)
            scaled_w = max(1, int(round(img_w * scale)))
            scaled_h = max(1, int(round(img_h * scale)))
            view_w, view_h = self._get_preview_view_size(scale, box_w, box_h, img_w, img_h)
            crop_x = scaled_w > box_w
            crop_y = scaled_h > box_h
            draw_w = box_w if crop_x else scaled_w
            draw_h = box_h if crop_y else scaled_h
            pos_x = 0.0 if crop_x else (box_w - draw_w) / 2
            pos_y = 0.0 if crop_y else (box_h - draw_h) / 2
            base_x, base_y, event_x, event_y = anchor
            if event_x < pos_x or event_x > pos_x + draw_w:
                event_x = pos_x + draw_w / 2
            if event_y < pos_y or event_y > pos_y + draw_h:
                event_y = pos_y + draw_h / 2
            scale_x = draw_w / float(view_w)
            scale_y = draw_h / float(view_h)
            center_x = self._preview_center[0] if self._preview_center else img_w / 2
            center_y = self._preview_center[1] if self._preview_center else img_h / 2
            if view_w < img_w:
                view_x0 = base_x - (event_x - pos_x) / max(scale_x, 1e-6)
                center_x = view_x0 + view_w / 2
            else:
                center_x = img_w / 2
            if view_h < img_h:
                view_y0 = base_y - (event_y - pos_y) / max(scale_y, 1e-6)
                center_y = view_y0 + view_h / 2
            else:
                center_y = img_h / 2
            self._preview_center = self._clamp_preview_center(center_x, center_y, view_w, view_h, img_w, img_h)
        self._preview_zoom_anchor = None
        self._render_preview()

    def _flush_grid_overlay(self) -> None:
        self._preview_grid_deferred_job = None
        if not self._grid_var.get():
            self._preview_grid_suspended = False
            return
        self._preview_grid_suspended = False
        self._render_preview()

    def _get_preview_view_size(
        self, scale: float, box_w: int, box_h: int, img_w: int, img_h: int
    ) -> tuple[int, int]:
        if scale <= 0:
            return (img_w, img_h)
        view_w = max(1, int(round(box_w / scale)))
        view_h = max(1, int(round(box_h / scale)))
        view_w = min(view_w, img_w)
        view_h = min(view_h, img_h)
        return view_w, view_h

    def _clamp_preview_center(
        self,
        center_x: float,
        center_y: float,
        view_w: int,
        view_h: int,
        img_w: int,
        img_h: int,
    ) -> tuple[float, float]:
        if view_w >= img_w:
            center_x = img_w / 2
        else:
            half_w = view_w / 2
            center_x = min(max(center_x, half_w), img_w - half_w)
        if view_h >= img_h:
            center_y = img_h / 2
        else:
            half_h = view_h / 2
            center_y = min(max(center_y, half_h), img_h - half_h)
        return center_x, center_y

    def _get_preview_box_size(self) -> tuple[int, int]:
        box_w, box_h = self._preview_box_size
        if box_w > 0 and box_h > 0:
            return box_w, box_h
        box_w = self._canvas.winfo_width() or self._canvas.winfo_reqwidth() or 200
        box_h = self._canvas.winfo_height() or self._canvas.winfo_reqheight() or 200
        self._preview_box_size = (box_w, box_h)
        return box_w, box_h

    def _get_grid_line_color(self) -> tuple[int, int, int, int]:
        if self._selected_rgb is None:
            return (0, 0, 0, 90)
        r, g, b = (int(self._selected_rgb[0]), int(self._selected_rgb[1]), int(self._selected_rgb[2]))
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        tone = int(round(255 - luminance))
        tone = max(0, min(255, tone))
        return (tone, tone, tone, 120)

    def _apply_grid_overlay(
        self,
        image: Image.Image,
        source_size: tuple[int, int],
        source_offset: tuple[int, int] = (0, 0),
        plate_size: int = 28,
    ) -> Image.Image:
        src_w, src_h = source_size
        off_x, off_y = source_offset
        if src_w <= 1 or src_h <= 1:
            return image
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        line_color = self._get_grid_line_color()
        dst_w, dst_h = base.size
        step_x = dst_w / src_w
        step_y = dst_h / src_h
        last_x = None
        for i in range(1, src_w):
            x = int(round(i * step_x))
            if x <= 0 or x >= dst_w or x == last_x:
                continue
            is_major = plate_size > 0 and ((i + off_x) % plate_size == 0)
            width = 2 if is_major else 1
            draw.line([(x, 0), (x, dst_h)], fill=line_color, width=width)
            last_x = x
        last_y = None
        for i in range(1, src_h):
            y = int(round(i * step_y))
            if y <= 0 or y >= dst_h or y == last_y:
                continue
            is_major = plate_size > 0 and ((i + off_y) % plate_size == 0)
            width = 2 if is_major else 1
            draw.line([(0, y), (dst_w, y)], fill=line_color, width=width)
            last_y = y
        return Image.alpha_composite(base, overlay)

    def _get_cached_preview_photo(self, key: tuple) -> Optional[ImageTk.PhotoImage]:
        photo = self._preview_cache.get(key)
        if photo is not None:
            self._preview_cache.move_to_end(key)
        return photo

    def _store_preview_cache(self, key: tuple, photo: ImageTk.PhotoImage) -> None:
        self._preview_cache[key] = photo
        self._preview_cache.move_to_end(key)
        while len(self._preview_cache) > self._preview_cache_limit:
            self._preview_cache.popitem(last=False)

    def _render_preview(self) -> None:
        box_w, box_h = self._get_preview_box_size()
        if self._preview_base_image is None:
            if self._preview_image_item is not None:
                self._canvas.delete(self._preview_image_item)
                self._preview_image_item = None
            self._preview_photo = None
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
            self._preview_image_pos = (0.0, 0.0)
            self._preview_region_size = (0, 0)
            self._preview_view_origin = (0.0, 0.0)
            self._preview_view_size = (0, 0)
            self._preview_center = None
            return
        if self._preview_text_item is not None:
            self._canvas.delete(self._preview_text_item)
            self._preview_text_item = None
        img_w, img_h = self._preview_base_image.size
        base_scale = min(box_w / img_w, box_h / img_h)
        base_scale = max(base_scale, 0.01)
        scale = base_scale * max(self._preview_zoom, 1.0)
        scaled_w = max(1, int(round(img_w * scale)))
        scaled_h = max(1, int(round(img_h * scale)))
        grid_on = self._grid_var.get() and not self._preview_grid_suspended
        line_color = self._get_grid_line_color() if grid_on else None
        if self._preview_center is None:
            self._preview_center = (img_w / 2, img_h / 2)

        view_w, view_h = self._get_preview_view_size(scale, box_w, box_h, img_w, img_h)
        crop_x = view_w < img_w
        crop_y = view_h < img_h
        if not crop_x and not crop_y:
            view_x0 = 0
            view_y0 = 0
            draw_w = scaled_w
            draw_h = scaled_h
            pos_x = (box_w - draw_w) / 2
            pos_y = (box_h - draw_h) / 2
            self._preview_scale = draw_w / float(img_w)
            cache_key = (
                id(self._preview_base_image),
                view_x0,
                view_y0,
                view_w,
                view_h,
                draw_w,
                draw_h,
                grid_on,
                line_color,
            )
            photo = self._get_cached_preview_photo(cache_key)
            if photo is None:
                resized = self._preview_base_image.resize((draw_w, draw_h), Image.Resampling.NEAREST)
                if grid_on:
                    resized = self._apply_grid_overlay(resized, (img_w, img_h))
                photo = ImageTk.PhotoImage(resized)
                self._store_preview_cache(cache_key, photo)
        else:
            center_x, center_y = self._preview_center
            center_x, center_y = self._clamp_preview_center(center_x, center_y, view_w, view_h, img_w, img_h)
            self._preview_center = (center_x, center_y)
            view_x0 = int(round(center_x - view_w / 2))
            view_y0 = int(round(center_y - view_h / 2))
            view_x0 = max(0, min(view_x0, img_w - view_w))
            view_y0 = max(0, min(view_y0, img_h - view_h))
            view_x1 = view_x0 + view_w
            view_y1 = view_y0 + view_h
            draw_w = box_w if crop_x else scaled_w
            draw_h = box_h if crop_y else scaled_h
            pos_x = 0.0 if crop_x else (box_w - draw_w) / 2
            pos_y = 0.0 if crop_y else (box_h - draw_h) / 2
            self._preview_scale = draw_w / float(view_w)
            cache_key = (
                id(self._preview_base_image),
                view_x0,
                view_y0,
                view_w,
                view_h,
                draw_w,
                draw_h,
                grid_on,
                line_color,
            )
            photo = self._get_cached_preview_photo(cache_key)
            if photo is None:
                cropped = self._preview_base_image.crop((view_x0, view_y0, view_x1, view_y1))
                resized = cropped.resize((draw_w, draw_h), Image.Resampling.NEAREST)
                if grid_on:
                    resized = self._apply_grid_overlay(
                        resized,
                        (view_w, view_h),
                        source_offset=(view_x0, view_y0),
                    )
                photo = ImageTk.PhotoImage(resized)
                self._store_preview_cache(cache_key, photo)

        self._preview_view_origin = (float(view_x0), float(view_y0))
        self._preview_view_size = (view_w, view_h)
        self._preview_image_pos = (pos_x, pos_y)
        self._preview_region_size = (draw_w, draw_h)
        self._preview_photo = photo
        if self._preview_image_item is None:
            self._preview_image_item = self._canvas.create_image(
                pos_x,
                pos_y,
                anchor="nw",
                image=self._preview_photo,
            )
        else:
            self._canvas.itemconfig(self._preview_image_item, image=self._preview_photo)
            self._canvas.coords(self._preview_image_item, pos_x, pos_y)
