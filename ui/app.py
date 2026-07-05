"""Tkinter UI for beads palette conversion (simplified)."""

from __future__ import annotations

import threading
import time
import json
from pathlib import Path
from typing import Optional
from math import ceil

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from palette import BeadPalette
from . import settings_keys as SK
from .controller import ConversionRunner
from .models import ConversionRequest
from .layout import LayoutMixin
from .actions import ActionsMixin
from .state import StateMixin
from .preview import PreviewMixin
from .scale_utils import bind_scale_click_jump


class BeadsApp(LayoutMixin, ActionsMixin, StateMixin, PreviewMixin):
    """Main application window (必要機能のみに絞った版)。"""

    PIXEL_SIZE_MM = 2.6
    PLATE_PIXELS = 28
    MAX_PLATES = 5

    def __init__(self, root: tk.Tk, palette: BeadPalette) -> None:
        self.root = root
        self.palette = palette
        bind_scale_click_jump(root)
        self._window_state_path = Path(__file__).resolve().parent / "window_state.json"
        self._settings_path = Path(__file__).resolve().parent / "settings.json"
        self._saved_mode: Optional[str] = None  # 前回選択したモードを保持
        self._restored_geometry = self._load_window_state()
        self._last_geometry: Optional[tuple[int, int, int, int]] = None
        self._last_normal_geometry: Optional[tuple[int, int, int, int]] = None
        self._last_window_state: str = "normal"
        self.input_image_path: Optional[Path] = None
        self.output_image: Optional[np.ndarray] = None
        self.output_path: Optional[Path] = None
        self.input_original_pil: Optional[Image.Image] = None
        self.input_pil: Optional[Image.Image] = None
        self.input_filtered_pil: Optional[Image.Image] = None
        self._input_using_filtered = False
        self.output_pil: Optional[Image.Image] = None
        self._input_photo: Optional[ImageTk.PhotoImage] = None
        self._output_photo: Optional[ImageTk.PhotoImage] = None
        self._output_grid_photos: list[ImageTk.PhotoImage] = []
        self.prev_output_pil: Optional[Image.Image] = None
        self.prev_settings: Optional[dict] = None
        self.last_settings: Optional[dict] = None
        self._pending_settings: Optional[dict] = None
        self._all_mode_results: Optional[list[dict]] = None
        self.color_usage: list[dict] = []
        self._color_usage_base_image: Optional[np.ndarray] = None
        self._color_usage_window = None
        self._preview_3d_window = None  # 3Dプレビューウィンドウの参照
        self._color_usage_selected_rgb: Optional[tuple[int, int, int]] = None
        self.color_usage_tone_var = tk.DoubleVar(value=0.85)
        self.color_usage_tone_display = tk.StringVar(value="")
        self.diff_var = tk.StringVar(value="")
        self.physical_size_var = tk.StringVar(value="完成サイズ: 幅・高さを入力してください")
        self.plate_requirement_var = tk.StringVar(value="28×28プレート: 幅・高さを入力してください")
        self.original_size: Optional[tuple[int, int]] = None
        self._start_time: Optional[float] = None
        # ノイズ除去の疑似進捗表示に使う状態
        self._noise_progress_start: Optional[float] = None
        self._noise_progress_value: float = 0.0
        self._noise_progress_after_id: Optional[str] = None
        self._progress_style_default = "Horizontal.TProgressbar"
        self._progress_style_noise = "Noise.Horizontal.TProgressbar"
        # 進捗を「変換」と「表示準備」に分けて扱う
        self._conversion_progress_range = (0.0, 0.85)
        self._ui_progress_range = (0.85, 1.0)
        self._conversion_progress_last = 0.0
        self.status_var = tk.StringVar(value="")
        self.width_var = tk.StringVar(value="")
        self.height_var = tk.StringVar(value="")
        self.resize_method_var = tk.StringVar(value="INTER_AREA")
        self.lock_aspect_var = tk.BooleanVar(value=True)
        self.cmc_l_var = tk.DoubleVar(value=2.0)
        self.cmc_c_var = tk.DoubleVar(value=1.0)
        self.cmc_l_display = tk.StringVar(value="2.0")
        self.cmc_c_display = tk.StringVar(value="1.0")
        self.rgb_r_weight_var = tk.DoubleVar(value=1.0)
        self.rgb_g_weight_var = tk.DoubleVar(value=1.0)
        self.rgb_b_weight_var = tk.DoubleVar(value=1.0)
        self.rgb_r_display = tk.StringVar(value="1.0")
        self.rgb_g_display = tk.StringVar(value="1.0")
        self.rgb_b_display = tk.StringVar(value="1.0")
        self.rgb_log_var = tk.StringVar(value="")
        self.lab_metric_var = tk.StringVar(value="CIEDE2000")
        self.noise_filter_var = tk.StringVar(value="メディアン")
        self.noise_filter_size_var = tk.IntVar(value=3)
        self.bilateral_sigma_var = tk.IntVar(value=75)
        self.post_mode_filter_enabled_var = tk.BooleanVar(value=False)
        self.post_mode_filter_size_var = tk.IntVar(value=3)
        self.post_island_enabled_var = tk.BooleanVar(value=False)
        self.post_island_min_area_var = tk.IntVar(value=4)
        self.normal_map_path: Optional[Path] = None
        self.normal_map_label = tk.StringVar(value="未選択")
        self.normal_enabled_var = tk.BooleanVar(value=False)
        self.normal_invert_y_var = tk.BooleanVar(value=False)
        self.normal_strength_var = tk.DoubleVar(value=0.6)
        self.normal_ambient_var = tk.DoubleVar(value=0.25)
        self.normal_gamma_var = tk.DoubleVar(value=1.0)
        self.normal_light_x_var = tk.DoubleVar(value=0.2)
        self.normal_light_y_var = tk.DoubleVar(value=-0.2)
        self.normal_light_z_var = tk.DoubleVar(value=0.95)
        self.normal_light_pad_canvas: Optional[tk.Canvas] = None
        self._light_pad_center = (0.0, 0.0)
        self._light_pad_radius = 0.0
        self._light_pad_handle: Optional[int] = None
        self.ao_map_path: Optional[Path] = None
        self.ao_map_label = tk.StringVar(value="未選択")
        self.ao_enabled_var = tk.BooleanVar(value=False)
        self.ao_strength_var = tk.DoubleVar(value=0.6)
        self.specular_map_path: Optional[Path] = None
        self.specular_map_label = tk.StringVar(value="未選択")
        self.specular_enabled_var = tk.BooleanVar(value=False)
        self.specular_strength_var = tk.DoubleVar(value=0.6)
        self.specular_shininess_var = tk.DoubleVar(value=24.0)
        self.displacement_map_path: Optional[Path] = None
        self.displacement_map_label = tk.StringVar(value="未選択")
        self.displacement_enabled_var = tk.BooleanVar(value=False)
        self.displacement_strength_var = tk.DoubleVar(value=0.6)
        self.displacement_midpoint_var = tk.DoubleVar(value=0.5)
        self.displacement_invert_var = tk.BooleanVar(value=False)
        self.pseudo_gradient_var = tk.DoubleVar(value=0.0)
        self.super_sampling_var = tk.BooleanVar(value=False)
        self.dither_var = tk.StringVar(value="なし")
        self.dither_strength_var = tk.DoubleVar(value=1.0)
        self.dither_strength_display = tk.StringVar(value="1.00")
        self.normal_detail_var = tk.BooleanVar(value=False)
        self.map_detail_var = tk.BooleanVar(value=True)
        self._input_shaded_pil: Optional[Image.Image] = None
        self._input_shading_after_id: Optional[str] = None
        self._updating_size_fields = False
        self._noise_busy = False
        self._noise_job_id = 0  # 実行中ノイズ除去の世代トークン（入力差し替えで無効化）
        self._closing = False  # 終了処理中フラグ
        self._showing_prev: bool = False
        self._showing_input_overlay: bool = False
        self._runner = ConversionRunner(self._schedule_on_ui, lambda: self._closing)
        self.color_usage_tone_var.trace_add("write", lambda *_: self._on_color_usage_tone_change())
        self._on_color_usage_tone_change()
        self._setup_shading_watchers()
        self.normal_light_x_var.trace_add("write", lambda *_: self._update_light_pad_from_vars())
        self.normal_light_y_var.trace_add("write", lambda *_: self._update_light_pad_from_vars())

        self._load_settings()
        self.normal_detail_var.trace_add("write", lambda *_: self._save_settings())
        self.map_detail_var.trace_add("write", lambda *_: self._save_settings())
        self._build_layout()
        self._apply_saved_settings()

    def _setup_shading_watchers(self) -> None:
        """ノーマル/AO/Specular/Displacementの変更を入力プレビューへ即時反映する。"""
        for var in (
            self.normal_enabled_var,
            self.normal_invert_y_var,
            self.normal_strength_var,
            self.normal_ambient_var,
            self.normal_gamma_var,
            self.normal_light_x_var,
            self.normal_light_y_var,
            self.normal_light_z_var,
            self.ao_enabled_var,
            self.ao_strength_var,
            self.specular_enabled_var,
            self.specular_strength_var,
            self.specular_shininess_var,
            self.displacement_enabled_var,
            self.displacement_strength_var,
            self.displacement_midpoint_var,
            self.displacement_invert_var,
        ):
            try:
                var.trace_add("write", lambda *_: self._request_input_shading_update())
            except Exception:
                pass

    def _init_light_direction_pad(self) -> None:
        """方向パッドのベース描画と初期位置を更新する。"""
        if self.normal_light_pad_canvas is None:
            return
        self._draw_light_pad_base()
        self._update_light_pad_from_vars()

    def _draw_light_pad_base(self) -> None:
        """方向パッドの円とガイド線を描く。"""
        canvas = self.normal_light_pad_canvas
        if canvas is None:
            return
        canvas.delete("all")
        size = int(canvas.cget("width")) or 110
        center = size / 2.0
        pad = 6.0
        radius = max(10.0, center - pad)
        self._light_pad_center = (center, center)
        self._light_pad_radius = radius
        # 十字と円で方向を分かりやすくする
        canvas.create_oval(
            center - radius,
            center - radius,
            center + radius,
            center + radius,
            outline="#888",
            fill="#f8f8f8",
        )
        canvas.create_line(center - radius, center, center + radius, center, fill="#bbb")
        canvas.create_line(center, center - radius, center, center + radius, fill="#bbb")
        self._light_pad_handle = canvas.create_oval(
            center - 4,
            center - 4,
            center + 4,
            center + 4,
            fill="#ff9800",
            outline="#cc7a00",
        )

    def _update_light_pad_from_vars(self) -> None:
        """X/Yの数値から方向パッドのつまみ位置を更新する。"""
        canvas = self.normal_light_pad_canvas
        if canvas is None or self._light_pad_radius <= 0.0:
            return
        try:
            x_val = float(self.normal_light_x_var.get())
            y_val = float(self.normal_light_y_var.get())
        except Exception:
            return
        x_val = max(-1.0, min(1.0, x_val))
        y_val = max(-1.0, min(1.0, y_val))
        # 単位円から外れる場合は縮めて表示する
        length = (x_val * x_val + y_val * y_val) ** 0.5
        if length > 1.0:
            x_val /= length
            y_val /= length
        cx, cy = self._light_pad_center
        px = cx + x_val * self._light_pad_radius
        py = cy - y_val * self._light_pad_radius
        r = 4
        if self._light_pad_handle is not None:
            canvas.coords(self._light_pad_handle, px - r, py - r, px + r, py + r)

    def _on_light_pad_drag(self, event: tk.Event) -> str:
        """方向パッド上のドラッグで光の向きを更新する。"""
        canvas = self.normal_light_pad_canvas
        if canvas is None:
            return "break"
        if self._light_pad_radius <= 0.0:
            self._draw_light_pad_base()
        cx, cy = self._light_pad_center
        dx = event.x - cx
        dy = event.y - cy
        length = (dx * dx + dy * dy) ** 0.5
        if length > self._light_pad_radius and length > 0.0:
            scale = self._light_pad_radius / length
            dx *= scale
            dy *= scale
        x_val = dx / self._light_pad_radius
        y_val = -dy / self._light_pad_radius
        # 現在のZ符号を保ちつつ、球面上でZを補完する
        try:
            current_z = float(self.normal_light_z_var.get())
        except Exception:
            current_z = 1.0
        sign = -1.0 if current_z < 0.0 else 1.0
        z_val = sign * max(0.0, 1.0 - x_val * x_val - y_val * y_val) ** 0.5
        self.normal_light_x_var.set(round(x_val, 3))
        self.normal_light_y_var.set(round(y_val, 3))
        self.normal_light_z_var.set(round(z_val, 3))
        self._update_light_pad_from_vars()
        return "break"

    # --- 設定復元と差分表示 ---

    # 復元可能な選択肢（コンボボックスの値と一致させる）
    _ALLOWED_RESIZE = {"ニアレストネイバー", "バイリニア", "INTER_AREA"}
    _ALLOWED_LAB_METRICS = {"CIEDE2000", "CIE76", "CIE94"}
    _ALLOWED_MODES = {"全て", "なし", "RGB", "Lab", "Hunter Lab", "Oklab", "CMC(l:c)"}

    # スカラー設定のテーブル: (設定キー, tk変数の属性名, キャスト, クランプ範囲)
    _SCALAR_SETTINGS: tuple = (
        (SK.NORMAL_INVERT_Y, "normal_invert_y_var", bool, None),
        (SK.NORMAL_STRENGTH, "normal_strength_var", float, None),
        (SK.NORMAL_AMBIENT, "normal_ambient_var", float, None),
        (SK.NORMAL_GAMMA, "normal_gamma_var", float, None),
        (SK.AO_STRENGTH, "ao_strength_var", float, None),
        (SK.SPECULAR_STRENGTH, "specular_strength_var", float, None),
        (SK.SPECULAR_SHININESS, "specular_shininess_var", float, None),
        (SK.DISPLACEMENT_STRENGTH, "displacement_strength_var", float, None),
        (SK.DISPLACEMENT_MIDPOINT, "displacement_midpoint_var", float, None),
        (SK.DISPLACEMENT_INVERT, "displacement_invert_var", bool, None),
        (SK.BILATERAL_SIGMA, "bilateral_sigma_var", int, None),
        (SK.PSEUDO_GRADIENT, "pseudo_gradient_var", float, (0.0, 20.0)),
    )

    # マップパス設定のテーブル: (設定キー, パス属性名, ラベル変数の属性名)
    _MAP_PATH_SETTINGS: tuple = (
        (SK.NORMAL_MAP, "normal_map_path", "normal_map_label"),
        (SK.AO_MAP, "ao_map_path", "ao_map_label"),
        (SK.SPECULAR_MAP, "specular_map_path", "specular_map_label"),
        (SK.DISPLACEMENT_MAP, "displacement_map_path", "displacement_map_label"),
    )

    def _get_saved(self, key: str):
        """last_settings から保存値を取得する（未保存なら None）。"""
        return self.last_settings.get(key) if self.last_settings else None

    @staticmethod
    def _sanitize_choice(value: Optional[str], allowed: set[str], fallback: str) -> str:
        if value in allowed:
            return value  # type: ignore[return-value]
        return fallback

    def _apply_saved_settings(self) -> None:
        self.width_var.set("")
        self.height_var.set("")

        # 最後に選択したモードを優先して復元
        saved_mode = self._sanitize_choice(self._saved_mode, self._ALLOWED_MODES, "")
        if saved_mode:
            self.mode_var.set(saved_mode)
        else:
            mode = self._get_saved(SK.MODE)
            if mode:
                self.mode_var.set(self._sanitize_choice(mode, self._ALLOWED_MODES, self.mode_var.get()))

        resize_label = self._sanitize_choice(
            self._get_saved(SK.RESIZE_METHOD), self._ALLOWED_RESIZE, self.resize_method_var.get()
        )
        self.resize_method_var.set(resize_label)
        try:
            self.super_sampling_var.set(bool(self._get_saved(SK.USE_SUPER_SAMPLING) or False))
        except Exception:
            pass
        if hasattr(self, "super_sampling_check"):
            if resize_label == "INTER_AREA":
                self.super_sampling_check.grid()
            else:
                self.super_sampling_check.grid_remove()
                self.super_sampling_var.set(False)

        self.lab_metric_var.set(
            self._sanitize_choice(self._get_saved(SK.LAB_METRIC), self._ALLOWED_LAB_METRICS, self.lab_metric_var.get())
        )

        # CMC重み（表示ラベルはクランプ済みで整形）
        for key, var, disp in (
            (SK.CMC_L, self.cmc_l_var, self.cmc_l_display),
            (SK.CMC_C, self.cmc_c_var, self.cmc_c_display),
        ):
            raw = self._get_saved(key)
            if raw is None:
                continue
            try:
                val = float(raw)
            except Exception:
                continue
            var.set(val)
            disp.set(f"{max(0.5, min(3.0, val)):.1f}")

        # RGB重み
        rgb_w = self._get_saved(SK.RGB_WEIGHTS)
        if isinstance(rgb_w, (list, tuple)) and len(rgb_w) == 3:
            try:
                vals = [float(x) for x in rgb_w]
            except Exception:
                vals = None
            if vals is not None:
                for val, var, disp in zip(
                    vals,
                    (self.rgb_r_weight_var, self.rgb_g_weight_var, self.rgb_b_weight_var),
                    (self.rgb_r_display, self.rgb_g_display, self.rgb_b_display),
                ):
                    var.set(val)
                    disp.set(f"{max(0.5, min(2.0, val)):.1f}")

        # スカラー設定（テーブル駆動）
        for key, attr, cast, clamp in self._SCALAR_SETTINGS:
            raw = self._get_saved(key)
            if raw is None:
                continue
            try:
                val = cast(raw)
                if clamp is not None:
                    val = max(clamp[0], min(clamp[1], val))
                getattr(self, attr).set(val)
            except Exception:
                pass

        # 光方向ベクトル
        light = self._get_saved(SK.NORMAL_LIGHT_DIR)
        if isinstance(light, (list, tuple)) and len(light) == 3:
            try:
                self.normal_light_x_var.set(float(light[0]))
                self.normal_light_y_var.set(float(light[1]))
                self.normal_light_z_var.set(float(light[2]))
            except Exception:
                pass

        # マップパス（存在するファイルのみ復元）
        for key, path_attr, label_attr in self._MAP_PATH_SETTINGS:
            raw = self._get_saved(key)
            if isinstance(raw, str) and raw:
                try:
                    path = Path(raw)
                    if path.exists():
                        setattr(self, path_attr, path)
                        getattr(self, label_attr).set(path.name)
                except Exception:
                    pass

        # 有効化チェックは起動時は常にオフにする
        for var in (
            self.normal_enabled_var,
            self.ao_enabled_var,
            self.specular_enabled_var,
            self.displacement_enabled_var,
        ):
            var.set(False)

        # ディザリング
        try:
            from converter.dither import DITHER_SPECS
            dither_label = self._get_saved(SK.DITHER)
            if dither_label in [s["label"] for s in DITHER_SPECS]:
                self.dither_var.set(dither_label)
        except Exception:
            pass
        raw = self._get_saved(SK.DITHER_STRENGTH)
        if raw is not None:
            try:
                val = max(0.0, min(1.0, float(raw)))
                self.dither_strength_var.set(val)
                self.dither_strength_display.set(f"{val:.2f}")
            except Exception:
                pass

        self._sanitize_last_settings()
        self._update_mode_frames()
        if hasattr(self, "_request_input_shading_update"):
            self._request_input_shading_update()

    def _sanitize_last_settings(self) -> None:
        """last_settings の不足キーを現在のUI値で補完する。"""
        if self.last_settings is None:
            return
        s = dict(self.last_settings)
        try:
            s.setdefault(SK.CMC_L, f"{float(self.cmc_l_var.get()):.1f}")
            s.setdefault(SK.CMC_C, f"{float(self.cmc_c_var.get()):.1f}")
            s.setdefault(
                SK.RGB_WEIGHTS,
                [
                    float(self.rgb_r_weight_var.get()),
                    float(self.rgb_g_weight_var.get()),
                    float(self.rgb_b_weight_var.get()),
                ],
            )
            s.setdefault(
                SK.NORMAL_LIGHT_DIR,
                [
                    float(self.normal_light_x_var.get()),
                    float(self.normal_light_y_var.get()),
                    float(self.normal_light_z_var.get()),
                ],
            )
        except Exception:
            pass
        # 有効化フラグと拡大方式は常に現在値で上書きする（起動時オフを反映）
        s[SK.NORMAL_ENABLED] = bool(self.normal_enabled_var.get())
        s[SK.AO_ENABLED] = bool(self.ao_enabled_var.get())
        s[SK.SPECULAR_ENABLED] = bool(self.specular_enabled_var.get())
        s[SK.DISPLACEMENT_ENABLED] = bool(self.displacement_enabled_var.get())
        s[SK.USE_SUPER_SAMPLING] = bool(self.super_sampling_var.get())
        # スカラー設定はテーブルから補完
        for key, attr, cast, _clamp in self._SCALAR_SETTINGS:
            try:
                s.setdefault(key, cast(getattr(self, attr).get()))
            except Exception:
                pass
        # マップパスは選択済みのもののみ補完
        for key, path_attr, _label_attr in self._MAP_PATH_SETTINGS:
            path = getattr(self, path_attr)
            if path:
                s.setdefault(key, str(path))
        s.setdefault(SK.RESIZE_METHOD, self.resize_method_var.get())
        s.setdefault(SK.MODE, self.mode_var.get())
        s.setdefault(SK.LAB_METRIC, self.lab_metric_var.get())
        self.last_settings = s

    def _build_diff_overlay(self) -> str:
        if not self.last_settings or not self.prev_settings:
            return "変更された設定: なし"
        diffs: list[str] = []
        for key, prev_val in self.prev_settings.items():
            last_val = self.last_settings.get(key)
            if last_val != prev_val:
                diffs.append(f"{key}: {prev_val} → {last_val}")
        if not diffs:
            return "変更された設定: なし"
        return "変更された設定: " + " / " .join(diffs)

    # --- サイズ計算系 ---
    def _parse_int(self, value: str) -> Optional[int]:
        try:
            return int(value)
        except Exception:
            return None

    def _set_size_fields(self, width: int, height: int) -> None:
        self._updating_size_fields = True
        self.width_var.set(str(width))
        self.height_var.set(str(height))
        self._updating_size_fields = False
        self._update_physical_size_display()

    def _update_physical_size_display(self) -> None:
        width = self._parse_int(self.width_var.get())
        height = self._parse_int(self.height_var.get())
        if width and height:
            mm_w = width * self.PIXEL_SIZE_MM
            mm_h = height * self.PIXEL_SIZE_MM
            text = f"完成サイズ: 約 {mm_w/10:.1f}cm × {mm_h/10:.1f}cm"
            plates_x = ceil(width / self.PLATE_PIXELS)
            plates_y = ceil(height / self.PLATE_PIXELS)
            plates_total = plates_x * plates_y
            plate_text = f"28×28プレート: {plates_x} × {plates_y} 枚 (合計 {plates_total} 枚)"
        else:
            text = "完成サイズ: 幅・高さを入力してください"
            plate_text = "28×28プレート: 幅・高さを入力してください"
        self.physical_size_var.set(text)
        self.plate_requirement_var.set(plate_text)

    def _set_height_from_width(self, width: int) -> None:
        if not self.original_size:
            return
        orig_w, orig_h = self.original_size
        new_h = max(1, int(round(orig_h / orig_w * width)))
        self._set_size_fields(width, new_h)

    def _set_width_from_height(self, height: int) -> None:
        if not self.original_size:
            return
        orig_w, orig_h = self.original_size
        new_w = max(1, int(round(orig_w / orig_h * height)))
        self._set_size_fields(new_w, height)

    def _on_width_changed(self) -> None:
        if self._updating_size_fields or not self.lock_aspect_var.get():
            return
        width = self._parse_int(self.width_var.get())
        if width and width > 0:
            self._set_height_from_width(width)

    def _on_height_changed(self) -> None:
        if self._updating_size_fields or not self.lock_aspect_var.get():
            return
        height = self._parse_int(self.height_var.get())
        if height and height > 0:
            self._set_width_from_height(height)

    def _on_aspect_toggle(self) -> None:
        if not self.lock_aspect_var.get() or not self.original_size:
            return
        width = self._parse_int(self.width_var.get())
        height = self._parse_int(self.height_var.get())
        if width and width > 0:
            self._set_height_from_width(width)
        elif height and height > 0:
            self._set_width_from_height(height)

    def _set_initial_target_size(self, image: Image.Image) -> None:
        img_w, img_h = image.size
        self._set_size_fields(img_w, img_h)

    def _halve_size(self) -> None:
        width = self._parse_int(self.width_var.get())
        height = self._parse_int(self.height_var.get())
        if width is None or height is None:
            self.status_var.set("幅と高さは整数で入力してください。")
            return
        new_w = max(1, width // 2)
        new_h = max(1, height // 2)
        self._set_size_fields(new_w, new_h)
        if self.lock_aspect_var.get() and self.original_size:
            self._set_height_from_width(new_w)

    def _reset_size(self) -> None:
        if not self.original_size:
            self.status_var.set("先に入力画像を選択してください。")
            return
        orig_w, orig_h = self.original_size
        self._set_size_fields(orig_w, orig_h)

    def _fit_size_to_plate_limit(self) -> None:
        """5×5プレート内に収まるようサイズを調整する。"""
        if self.original_size:
            base_w, base_h = self.original_size
        else:
            width = self._parse_int(self.width_var.get())
            height = self._parse_int(self.height_var.get())
            if not width or not height:
                self.status_var.set("先に入力画像を選択するか、幅・高さを入力してください。")
                return
            base_w, base_h = width, height
        if base_w <= 0 or base_h <= 0:
            self.status_var.set("幅・高さは1以上で指定してください。")
            return
        max_w = self.PLATE_PIXELS * self.MAX_PLATES
        max_h = self.PLATE_PIXELS * self.MAX_PLATES
        # 5×5の最大範囲に収める倍率を計算する
        scale = min(max_w / base_w, max_h / base_h, 1.0)
        new_w = max(1, int(round(base_w * scale)))
        new_h = max(1, int(round(base_h * scale)))
        self._set_size_fields(new_w, new_h)
        self.status_var.set("5×5プレートに収まるようサイズを調整しました。")

    # --- キーボード/プレビュー ---
    def _on_preview_resize(self, _event: tk.Event) -> None:
        self._refresh_previews()

    def _on_space_key(self, event: tk.Event) -> str:
        return super()._on_space_key(event)

    def _cancel_worker_safely(self, timeout: float = 2.0) -> None:
        """変換スレッドをキャンセルし、短時間だけ待機する。"""
        self._runner.cancel_and_wait(timeout=timeout)

    # --- 終了時 ---
    def _on_close(self) -> None:
        self._closing = True
        self._cancel_worker_safely()
        self._remember_mode_selection()
        self._save_window_state()
        self.root.destroy()
