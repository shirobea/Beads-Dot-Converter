"""プレビュー描画まわりのシンプルなMixin。"""

from __future__ import annotations

import tkinter as tk
from typing import Optional, TYPE_CHECKING, Callable, Tuple
from PIL import Image, ImageTk

if TYPE_CHECKING:
    from .app import BeadsApp


class PreviewMixin:
    """入力/出力画像のプレビュー更新だけを扱う。"""

    def _can_toggle_input_overlay(self: "BeadsApp") -> bool:
        """入力プレビューで比較表示できるかを判定する。"""
        if getattr(self, "_noise_busy", False):
            return False
        return bool(self.input_pil and self.input_original_pil)

    def _on_input_press(self: "BeadsApp", _event: tk.Event) -> None:
        """入力プレビューを長押しでノイズ除去前後を表示切替する。"""
        if getattr(self, "_noise_busy", False):
            return "break"
        if self._can_toggle_input_overlay():
            self._showing_input_overlay = True
            self._refresh_previews()
        return None

    def _on_input_release(self: "BeadsApp", _event: tk.Event) -> None:
        """長押しを離したら通常表示に戻す。"""
        if getattr(self, "_noise_busy", False):
            return "break"
        if self._showing_input_overlay:
            self._showing_input_overlay = False
            self._refresh_previews()
        return None

    def _on_output_press(self: "BeadsApp", _event: tk.Event) -> None:
        """出力プレビューを長押しで一つ前の出力に重ね表示する。"""
        if getattr(self, "_all_mode_results", None):
            return
        if self.prev_output_pil:
            self._showing_prev = True
            self._refresh_previews()

    def _on_output_release(self: "BeadsApp", _event: tk.Event) -> None:
        """長押しを離したら最新出力に戻す。"""
        if getattr(self, "_all_mode_results", None):
            return
        if self._showing_prev:
            self._showing_prev = False
            self._refresh_previews()
            return

    def _on_preview_resize(self: "BeadsApp", _event: tk.Event) -> None:
        self._refresh_previews()

    def _update_preview_layout(self: "BeadsApp", landscape: bool) -> None:
        """横長/縦長に応じてプレビューパネルの配置を切り替える。"""
        prev_landscape = getattr(self, "_preview_landscape", None)
        if prev_landscape == landscape:
            return
        self._preview_landscape = landscape
        preview_frame = getattr(self, "preview_frame", None)
        input_canvas = getattr(self, "input_canvas", None)
        output_container = getattr(self, "output_container", None)
        if not preview_frame or not input_canvas or not output_container:
            return
        if landscape:
            # 上下並び: 入力を上(row=1), 出力を下(row=2)
            preview_frame.rowconfigure(1, weight=1)
            preview_frame.rowconfigure(2, weight=1)
            input_canvas.grid_configure(row=1, column=0, columnspan=2)
            output_container.grid_configure(row=2, column=0, columnspan=2)
        else:
            # 横並び: 入力を左(col=0), 出力を右(col=1)
            preview_frame.rowconfigure(2, weight=0)
            input_canvas.grid_configure(row=1, column=0, columnspan=1)
            output_container.grid_configure(row=1, column=1, columnspan=1)

    def _refresh_previews(
        self: "BeadsApp", progress_cb: Optional[Callable[[str, float], None]] = None
    ) -> None:
        show_all = bool(getattr(self, "_all_mode_results", None))

        # アスペクト比に応じてレイアウトを先に決定する
        landscape = False
        if not show_all and getattr(self, "input_pil", None):
            img_w, img_h = self.input_pil.size
            landscape = img_w > img_h
        self._update_preview_layout(landscape)

        self._set_input_visible(not show_all)
        self.root.update_idletasks()
        frame_w = self.preview_frame.winfo_width() or self.preview_frame.winfo_reqwidth() or 400
        frame_h = self.preview_frame.winfo_height() or self.preview_frame.winfo_reqheight() or 300
        if landscape and not show_all:
            cell_w = max(1, frame_w - 20)
            cell_h = max(1, (frame_h - 20) // 2)
        else:
            cell_w = max(1, (frame_w - 20) // 2)
            cell_h = max(1, frame_h - 20)
        # 進捗の初期位置を通知する
        if progress_cb:
            progress_cb("resize", 0.0)
            progress_cb("draw", 0.0)

        input_display: Optional[Image.Image] = None
        if not show_all:
            input_caption = ""
            if self.input_pil:
                input_display = self.input_pil
                if self._showing_input_overlay and self.input_original_pil:
                    input_display = self.input_original_pil
                    input_caption = "元画像を表示中"
                if (
                    not self._showing_input_overlay
                    and input_display is self.input_pil
                    and getattr(self, "_input_shaded_pil", None) is not None
                ):
                    input_display = self._input_shaded_pil

            if input_display:
                photo = self._resize_to_box(
                    input_display,
                    cell_w,
                    cell_h,
                    progress_cb=(lambda v: progress_cb("resize", 0.4 * v) if progress_cb else None),
                )
                if photo:
                    self._input_photo = photo
                    self.input_canvas.configure(image=self._input_photo, text=input_caption)
            else:
                self.input_canvas.configure(image="", text="入力画像")
        if progress_cb and not show_all:
            # 入力プレビュー更新完了
            progress_cb("resize", 0.4)
            self.root.update_idletasks()

        if getattr(self, "_all_mode_results", None):
            self._set_output_grid_visible(True)
            self._refresh_all_mode_grid(frame_w, frame_h, progress_cb=progress_cb)
            return

        self._set_output_grid_visible(False)
        display_pil: Optional[Image.Image] = None
        if self._showing_prev and self.prev_output_pil:
            # 前回画像はサイズを変えずにそのまま表示する
            display_pil = self.prev_output_pil
        else:
            display_pil = self.output_pil

        if display_pil:
            photo = self._resize_to_box(
                display_pil,
                cell_w,
                cell_h,
                progress_cb=(lambda v: progress_cb("resize", 0.4 + 0.6 * v) if progress_cb else None),
            )
            if photo:
                self._output_photo = photo
                caption = "前回出力との比較表示" if self._showing_prev and self.prev_output_pil else ""
                self.output_canvas.configure(image=self._output_photo, text=caption)
        else:
            self.output_canvas.configure(image="", text="変換後")
        if progress_cb:
            # 出力プレビュー更新完了
            progress_cb("resize", 1.0)
            progress_cb("draw", 1.0)
            self.root.update_idletasks()

    def _set_output_grid_visible(self: "BeadsApp", visible: bool) -> None:
        grid_frame = getattr(self, "output_grid_frame", None)
        output_canvas = getattr(self, "output_canvas", None)
        if not grid_frame or not output_canvas:
            return
        if visible:
            output_canvas.grid_remove()
            grid_frame.grid()
        else:
            grid_frame.grid_remove()
            output_canvas.grid()

    def _refresh_all_mode_grid(
        self: "BeadsApp",
        col_w: int,
        col_h: int,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        results = getattr(self, "_all_mode_results", None) or []
        base_w = 0
        base_h = 0
        for entry in results:
            pil = entry.get("pil")
            if isinstance(pil, Image.Image):
                base_w, base_h = pil.size
                break
        rows, cols = self._get_all_mode_grid_shape(base_w, base_h)
        # 行列数に合わせてグリッドを並べ替える
        grid_frame = getattr(self, "output_grid_frame", None)
        if grid_frame:
            for row in range(4):
                grid_frame.rowconfigure(row, weight=0)
            for col in range(4):
                grid_frame.columnconfigure(col, weight=0)
            for row in range(rows):
                grid_frame.rowconfigure(row, weight=1)
            for col in range(cols):
                grid_frame.columnconfigure(col, weight=1)
            for idx, cell in enumerate(getattr(self, "output_grid_cells", [])):
                frame = cell.get("frame")
                if frame is None:
                    continue
                row = idx // cols
                col = idx % cols
                frame.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        # 行列数に合わせて表示領域を計算
        out_w = getattr(self, "output_container", None)
        container_w = col_w
        container_h = col_h
        if out_w:
            measured_w = out_w.winfo_width()
            measured_h = out_w.winfo_height()
            if measured_w and measured_w >= col_w * 0.9:
                container_w = measured_w
            if measured_h and measured_h >= col_h * 0.9:
                container_h = measured_h
        cell_w = max(1, (container_w - 12) // cols)
        cell_h = max(1, (container_h - 12) // rows)
        caption_h = 18
        image_h = max(1, cell_h - caption_h)
        self._output_grid_photos = []
        cells = list(getattr(self, "output_grid_cells", []))
        total_cells = max(1, len(cells))
        for idx, cell in enumerate(cells):
            image_label = cell.get("image")
            caption_label = cell.get("caption")
            if image_label is None or caption_label is None:
                continue
            if idx < len(results):
                entry = results[idx]
                caption = str(entry.get("label", ""))
                caption_label.configure(text=caption, wraplength=cell_w)
                pil = entry.get("pil")
                if pil is None and "image" in entry:
                    pil = Image.fromarray(entry["image"])
                if pil:
                    def _grid_progress(value: float) -> None:
                        if not progress_cb:
                            return
                        base = idx / total_cells
                        span = 1.0 / total_cells
                        progress_cb("resize", min(1.0, base + span * max(0.0, min(1.0, value))))

                    photo = self._resize_to_box(pil, cell_w, image_h, progress_cb=_grid_progress)
                    if photo:
                        self._output_grid_photos.append(photo)
                        image_label.configure(image=photo, text="")
                else:
                    image_label.configure(image="", text="")
            else:
                caption_label.configure(text="")
                image_label.configure(image="", text="")
            if progress_cb:
                # グリッドの更新に合わせて進捗を進める
                progress_cb("draw", min(1.0, (idx + 1) / total_cells))
                self.root.update_idletasks()

    def _resize_to_box(
        self: "BeadsApp",
        image: Image.Image,
        box_w: int,
        box_h: int,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> Optional[ImageTk.PhotoImage]:
        img_w, img_h = image.size
        scale = min(box_w / img_w, box_h / img_h)
        scale = max(scale, 0.01)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        if progress_cb:
            progress_cb(0.0)
        resized = self._resize_with_steps(image, new_size, progress_cb)
        if progress_cb:
            progress_cb(1.0)
        return ImageTk.PhotoImage(resized)

    def _resize_with_steps(
        self: "BeadsApp",
        image: Image.Image,
        target_size: Tuple[int, int],
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> Image.Image:
        """プレビュー用の縮小を段階的に行い、進捗を細かく刻む。"""
        target_w, target_h = target_size
        src_w, src_h = image.size
        if target_w >= src_w and target_h >= src_h:
            return image.resize(target_size, Image.Resampling.NEAREST)
        # 大きく縮小する場合は小刻みに縮小して進捗を出す
        scale_step = 0.8  # 1回で縮小しすぎない
        min_steps = 5
        current = image
        w, h = current.size
        steps = 0
        while w * scale_step >= target_w and h * scale_step >= target_h:
            w = max(target_w, int(round(w * scale_step)))
            h = max(target_h, int(round(h * scale_step)))
            steps += 1
            if w == target_w and h == target_h:
                break
        steps = max(min_steps, steps + 1)  # 最終リサイズ分も含める

        for step in range(steps - 1):
            # 目標に近づくほど縮小幅を小さくして滑らかにする
            remain = steps - step
            blend = 1.0 / max(2, remain)
            next_w = max(target_w, int(round(current.size[0] * (1.0 - blend))))
            next_h = max(target_h, int(round(current.size[1] * (1.0 - blend))))
            if next_w == current.size[0] and next_h == current.size[1]:
                next_w = max(target_w, current.size[0] - 1)
                next_h = max(target_h, current.size[1] - 1)
            current = current.resize((next_w, next_h), Image.Resampling.NEAREST)
            if progress_cb:
                progress_cb((step + 1) / steps)
                self.root.update_idletasks()
            if current.size[0] <= target_w and current.size[1] <= target_h:
                break

        if current.size != target_size:
            current = current.resize(target_size, Image.Resampling.NEAREST)
        if progress_cb:
            progress_cb(1.0)
        return current

    def _set_input_visible(self: "BeadsApp", visible: bool) -> None:
        input_canvas = getattr(self, "input_canvas", None)
        output_container = getattr(self, "output_container", None)
        if not input_canvas or not output_container:
            return
        landscape = getattr(self, "_preview_landscape", False)
        if visible:
            input_canvas.grid()
            if landscape:
                # 上下並び: 出力は row=2
                output_container.grid_configure(row=2, column=0, columnspan=2)
            else:
                # 横並び: 出力は row=1, col=1
                output_container.grid_configure(row=1, column=1, columnspan=1)
        else:
            input_canvas.grid_remove()
            output_container.grid_configure(row=1, column=0, columnspan=2)
