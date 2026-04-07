"""3Dプレビューの最小試作（出力画像のビーズ配置対応）。"""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from typing import Callable, Optional

import numpy as np
from PIL import Image
from pyopengltk import OpenGLFrame
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ARRAY,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_EXTENSIONS,
    GL_FLAT,
    GL_FLOAT,
    GL_MAX_TEXTURE_SIZE,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_RGB,
    GL_NEAREST,
    GL_TRIANGLE_FAN,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_COORD_ARRAY,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_ARRAY,
    GL_VERSION,
    GL_SMOOTH,
    glBegin,
    glBindTexture,
    glClear,
    glClearColor,
    glColor3f,
    glColorPointer,
    glDisable,
    glDisableClientState,
    glDrawArrays,
    glEnable,
    glEnableClientState,
    glEnd,
    glGenTextures,
    glGetIntegerv,
    glGetString,
    glLoadIdentity,
    glMatrixMode,
    glPixelStorei,
    glRotatef,
    glShadeModel,
    glTexCoord2f,
    glTexCoordPointer,
    glTexImage2D,
    glTexParameteri,
    glTranslatef,
    glVertex3f,
    glVertexPointer,
    glViewport,
)
from OpenGL.GLU import gluLookAt, gluPerspective


class BeadsPreview3DWindow(tk.Toplevel):
    """3Dプレビュー用の簡易ウィンドウ。"""

    def __init__(self, master: tk.Misc, on_close: Optional[Callable[[], None]] = None) -> None:
        super().__init__(master)
        self._window_state_path = Path(__file__).resolve().parent / "preview_3d_state.json"
        self._last_geometry: Optional[tuple[int, int, int, int]] = None
        self._last_normal_geometry: Optional[tuple[int, int, int, int]] = None
        self._last_window_state: str = "normal"
        self.title("3Dプレビュー（試作）")
        if not self._load_window_state():
            self.geometry("560x420")
        self._on_close_callback = on_close
        self._gl_frame = BeadsPreviewGL(self)
        self._gl_frame.pack(fill="both", expand=True)
        self._gl_frame.animate = 0  # 操作時のみ再描画する
        self._gl_frame.focus_set()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Configure>", self._on_window_configure)

    def _on_close(self) -> None:
        # 閉じる前に現在の位置・サイズを保存する
        self._save_window_state()
        if self._on_close_callback:
            self._on_close_callback()
        self.destroy()

    def set_image(self, image: np.ndarray) -> None:
        """出力画像からビーズ配置を更新する。"""
        self._gl_frame.set_image(image)

    def _on_window_configure(self, event: tk.Event) -> None:
        """ウィンドウ位置・サイズの変更を記録する。"""
        if event.widget is not self:
            return
        try:
            current_state = self.state()
        except Exception:
            current_state = "normal"
        self._last_window_state = current_state
        geometry = (
            self.winfo_width(),
            self.winfo_height(),
            self.winfo_x(),
            self.winfo_y(),
        )
        self._last_geometry = geometry
        if current_state == "normal":
            self._last_normal_geometry = geometry

    def _load_window_state(self) -> bool:
        """前回のウィンドウ配置を復元する。"""
        try:
            if not self._window_state_path.exists():
                return False
            data = json.loads(self._window_state_path.read_text(encoding="utf-8"))
            width = int(data.get("width", 0))
            height = int(data.get("height", 0))
            x = int(data.get("x", 0))
            y = int(data.get("y", 0))
            state = data.get("state", "normal")
            if state not in ("normal", "zoomed"):
                state = "normal"
            if width > 0 and height > 0:
                self.geometry(f"{width}x{height}+{x}+{y}")
                self._last_geometry = (width, height, x, y)
                self._last_normal_geometry = (width, height, x, y)
                self._last_window_state = state
                if state == "zoomed":
                    try:
                        self.state("zoomed")
                    except tk.TclError:
                        pass
                return True
        except Exception:
            return False
        return False

    def _save_window_state(self) -> None:
        """直近のウィンドウ配置を保存する。"""
        if self._last_geometry is None:
            self._last_geometry = (
                self.winfo_width(),
                self.winfo_height(),
                self.winfo_x(),
                self.winfo_y(),
            )
        try:
            current_state = self.state()
        except Exception:
            current_state = "normal"
        if current_state not in ("normal", "zoomed"):
            current_state = "normal"
        geometry = self._last_geometry
        if current_state == "zoomed" and self._last_normal_geometry:
            geometry = self._last_normal_geometry
        if geometry is None:
            geometry = (
                self.winfo_width(),
                self.winfo_height(),
                self.winfo_x(),
                self.winfo_y(),
            )
        width, height, x, y = geometry
        payload = {
            "width": int(width),
            "height": int(height),
            "x": int(x),
            "y": int(y),
            "state": current_state,
        }
        try:
            self._window_state_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass


class BeadsPreviewGL(OpenGLFrame):
    """出力画像に合わせたビーズ配置を回転表示するOpenGLフレーム。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._angle = 0.0
        self._angle_speed = 0.6
        self._auto_rotate = False
        self._grid_size = (0, 0)
        self._spacing = 0.3
        self._bead_radius = 0.45
        self._bead_height = 0.3
        self._max_beads = None  # 上限なし
        self._max_texture_size: Optional[int] = None
        self._supports_npot: Optional[bool] = None
        self._texture_id: Optional[int] = None
        self._texture_image: Optional[Image.Image] = None
        self._texture_dirty = False
        self._source_rgb: Optional[np.ndarray] = None
        self._texture_size = (0, 0)
        self._plane_size = (1.0, 1.0)
        self._box_depth_ratio = 1.0  # 箱の厚み比率（ビーズ間隔に対して）
        self._box_depth = 0.6
        self._side_wave_segments_per_bead = 4
        self._side_wave_depth_ratio = 0.45
        self._side_wave_shade_strength = 0.12
        self._side_wave_face_tint = {
            "front": 0.92,
            "back": 0.8,
            "left": 0.88,
            "right": 0.88,
        }
        self._side_wave_mesh: Optional[dict[str, np.ndarray]] = None
        self._side_wave_dirty = False
        self._side_valley_strength = 0.35
        self._side_valley_power = 1.6
        self._side_min_shade = 0.65
        self._bead_px_base = 6
        self._bead_px = self._bead_px_base
        self._bead_radius_ratio = 0.57
        self._hole_radius_ratio = 0.3
        self._hole_strength = 0.8
        self._gap_color = (25, 25, 25)
        self._hole_color = (0, 0, 0)
        self._gap_color_dark = (25, 25, 25)
        self._hole_color_dark = (0, 0, 0)
        self._gap_color_light = (255, 255, 255)
        self._hole_color_light = (255, 255, 255)
        # 見た目のプリセットを切り替えるための設定
        self._appearance_mode = "normal"
        self._appearance_profiles = {
            "normal": {
                "bead_radius_ratio": self._bead_radius_ratio,
                "hole_radius_ratio": self._hole_radius_ratio,
                "hole_strength": self._hole_strength,
            },
            "ironed": {
                "bead_radius_ratio": 0.62,
                "hole_radius_ratio": 0.18,
                "hole_strength": 0.8,
            },
        }
        self._preserve_view_on_rebuild = False
        self._hole_gap_is_white = False
        self._base_camera_distance = 5.0
        self._camera_distance = 5.0
        self._rot_x = 20.0
        self._rot_y = 0.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._last_left: Optional[tuple[int, int]] = None
        self._last_right: Optional[tuple[int, int]] = None
        self._keys_held: set[str] = set()
        self._redraw_pending = False
        self._auto_rotate_job: Optional[str] = None
        self._auto_rotate_interval_ms = 16
        self._background_is_black = True

        # マウス操作のバインド
        self.bind("<Enter>", lambda _event: self.focus_set())
        self.bind("<ButtonPress-1>", self._on_left_press)
        self.bind("<B1-Motion>", self._on_left_drag)
        self.bind("<ButtonRelease-1>", self._on_left_release)
        self.bind("<ButtonPress-3>", self._on_right_press)
        self.bind("<B3-Motion>", self._on_right_drag)
        self.bind("<ButtonRelease-3>", self._on_right_release)
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", self._on_mouse_wheel_linux)
        self.bind("<Button-5>", self._on_mouse_wheel_linux)
        self.bind("<KeyPress-space>", self._on_space)
        self.bind("<KeyPress-z>", self._on_toggle_background)
        self.bind("<KeyPress-Z>", self._on_toggle_background)
        self.bind("<KeyPress-x>", self._on_toggle_hole_gap)
        self.bind("<KeyPress-X>", self._on_toggle_hole_gap)
        self.bind("<KeyPress-c>", self._on_toggle_appearance)
        self.bind("<KeyPress-C>", self._on_toggle_appearance)
        self.bind("<KeyRelease-z>", self._on_toggle_key_release)
        self.bind("<KeyRelease-Z>", self._on_toggle_key_release)
        self.bind("<KeyRelease-x>", self._on_toggle_key_release)
        self.bind("<KeyRelease-X>", self._on_toggle_key_release)
        self.bind("<KeyRelease-c>", self._on_toggle_key_release)
        self.bind("<KeyRelease-C>", self._on_toggle_key_release)

    def tkMap(self, event: tk.Event) -> None:
        super().tkMap(event)
        # 初回表示時にも描画を要求する
        self._request_redraw()

    def tkResize(self, event: tk.Event) -> None:
        super().tkResize(event)
        # リサイズ後に再描画する
        self._request_redraw()

    def set_image(self, image: np.ndarray) -> None:
        """画像のRGB値からビーズの位置と色を作る。"""
        if not isinstance(image, np.ndarray):
            return
        if image.ndim < 2:
            return
        rgb = image
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=2)
        if rgb.shape[2] >= 4:
            rgb = rgb[:, :, :3]
        rgb = self._downsample_if_needed(rgb)
        # OpenGLの最大テクスチャサイズが確定するまで元画像を保持する
        self._source_rgb = rgb
        self._texture_image = None
        self._texture_size = (0, 0)
        self._texture_dirty = True
        self._side_wave_dirty = True
        if self._max_texture_size is not None:
            self._build_texture_from_source()
        self._request_redraw()

    def _request_redraw(self) -> None:
        """描画要求をまとめて1回だけ実行する。"""
        if self._redraw_pending:
            return
        self._redraw_pending = True
        self.after_idle(self._on_redraw_idle)

    def _on_redraw_idle(self) -> None:
        self._redraw_pending = False
        if not self.winfo_exists() or not self.context_created:
            return
        if not self.winfo_ismapped():
            return
        self._display()

    def _start_auto_rotate(self) -> None:
        if self._auto_rotate_job is not None:
            return
        self._auto_rotate_job = self.after(self._auto_rotate_interval_ms, self._auto_rotate_tick)

    def _stop_auto_rotate(self) -> None:
        if self._auto_rotate_job is None:
            return
        try:
            self.after_cancel(self._auto_rotate_job)
        except Exception:
            pass
        self._auto_rotate_job = None

    def _auto_rotate_tick(self) -> None:
        self._auto_rotate_job = None
        if not self._auto_rotate or not self.winfo_exists():
            return
        if self.context_created and self.winfo_ismapped():
            self._display()
        self._auto_rotate_job = self.after(self._auto_rotate_interval_ms, self._auto_rotate_tick)

    def _downsample_if_needed(self, rgb: np.ndarray) -> np.ndarray:
        """描画負荷を下げるため上限を超える場合は縮小する。"""
        height, width = rgb.shape[:2]
        total = height * width
        limit = self._max_beads
        if limit is None:
            return rgb
        if total <= limit:
            return rgb
        scale = (limit / total) ** 0.5
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        small = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.NEAREST)
        return np.asarray(small, dtype=np.uint8)

    def _downsample_to_texture_limit(self, rgb: np.ndarray) -> np.ndarray:
        """テクスチャ上限を超える場合は画像を縮小する。"""
        height, width = rgb.shape[:2]
        max_size = self._get_max_texture_size()
        max_w = max_size // self._bead_px
        max_h = max_size // self._bead_px
        if width <= max_w and height <= max_h:
            return rgb
        scale = min(max_w / width, max_h / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        small = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.NEAREST)
        return np.asarray(small, dtype=np.uint8)

    def _get_max_texture_size(self) -> int:
        if self._max_texture_size is not None:
            return self._max_texture_size
        try:
            max_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
            if isinstance(max_size, (list, tuple, np.ndarray)):
                max_size = max_size[0]
            self._max_texture_size = int(max_size)
            return self._max_texture_size
        except Exception:
            # 取得できない場合は仮の値にする（後で再取得する）
            return 4096

    def _supports_npot_texture(self) -> bool:
        """NPOTテクスチャが使えるか判定する。"""
        if self._supports_npot is not None:
            return self._supports_npot
        try:
            version = glGetString(GL_VERSION)
            if isinstance(version, bytes):
                version = version.decode("ascii", errors="ignore")
            if version:
                major = int(str(version).split(".")[0])
                self._supports_npot = major >= 2
                return self._supports_npot
            extensions = glGetString(GL_EXTENSIONS)
            if isinstance(extensions, bytes):
                extensions = extensions.decode("ascii", errors="ignore")
            self._supports_npot = "GL_ARB_texture_non_power_of_two" in str(extensions)
        except Exception:
            self._supports_npot = False
        return bool(self._supports_npot)

    def _nearest_power_of_two(self, value: int) -> int:
        """近い2のべき乗へ丸める（最小は1）。"""
        if value <= 1:
            return 1
        power = 1
        while power * 2 <= value:
            power *= 2
        return power

    def _fit_texture_image(self, image: Image.Image) -> Image.Image:
        """OpenGLに合わせてテクスチャサイズを調整する。"""
        max_size = self._get_max_texture_size()
        width, height = image.size
        if width > max_size or height > max_size:
            scale = min(max_size / width, max_size / height)
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))
            image = image.resize((width, height), Image.Resampling.NEAREST)
        if not self._supports_npot_texture():
            pot_w = self._nearest_power_of_two(width)
            pot_h = self._nearest_power_of_two(height)
            if pot_w != width or pot_h != height:
                image = image.resize((pot_w, pot_h), Image.Resampling.NEAREST)
        return image

    def _build_texture_from_source(self) -> bool:
        """元画像からテクスチャを組み立て直す。"""
        if self._source_rgb is None:
            return False
        # 画像サイズに合わせてビーズ解像度を切り替える
        src_h, src_w = self._source_rgb.shape[:2]
        self._bead_px = self._get_dynamic_bead_px(src_h, src_w)
        rgb = self._downsample_to_texture_limit(self._source_rgb)
        height, width = rgb.shape[:2]
        self._grid_size = (width, height)
        self._plane_size = (width * self._spacing, height * self._spacing)
        self._box_depth = max(self._bead_height * 1.6, self._spacing * self._box_depth_ratio)
        texture = self._build_texture_image(rgb)
        self._texture_image = self._fit_texture_image(texture)
        self._texture_size = self._texture_image.size if self._texture_image else (0, 0)
        self._build_side_wave_mesh(rgb)
        self._side_wave_dirty = False

        size = max(width, height)
        self._base_camera_distance = max(5.0, size * self._spacing * 0.9 + 2.0)
        preserve_view = self._preserve_view_on_rebuild
        self._preserve_view_on_rebuild = False
        if not preserve_view:
            self._camera_distance = self._base_camera_distance
            self._pan_x = 0.0
            self._pan_y = 0.0
        return True

    def _build_texture_image(self, rgb: np.ndarray) -> Image.Image:
        """出力画像から穴付きのビーズテクスチャを作る。"""
        height, width = rgb.shape[:2]
        bead_px = self._bead_px
        base = Image.fromarray(rgb, "RGB")
        expanded = base.resize((width * bead_px, height * bead_px), Image.Resampling.NEAREST)
        expanded_arr = np.asarray(expanded, dtype=np.float32)

        # 円形ビーズと穴のマスクを作ってタイル状に並べる
        bead_tile = self._make_circle_mask(bead_px, self._bead_radius_ratio, feather=1.0)
        hole_tile = self._make_circle_mask(bead_px, self._hole_radius_ratio, feather=0.6)
        bead_mask = np.tile(bead_tile, (height, width))[:, :, None]
        hole_mask = np.tile(hole_tile, (height, width))[:, :, None]
        hole_mask = hole_mask * bead_mask

        gap_color = np.array(self._gap_color, dtype=np.float32)[None, None, :]
        result = gap_color + (expanded_arr - gap_color) * bead_mask
        hole_color = np.array(self._hole_color, dtype=np.float32)[None, None, :]
        hole_alpha = hole_mask * self._hole_strength
        result = result * (1.0 - hole_alpha) + hole_color * hole_alpha

        texture = np.clip(result, 0, 255).astype(np.uint8)
        # OpenGLの座標系に合わせて上下反転
        image = Image.fromarray(texture, "RGB").transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def _build_side_wave_mesh(self, rgb: np.ndarray) -> None:
        """側面を連続カーブでつないだ波型メッシュを作る。"""
        height, width = rgb.shape[:2]
        if height <= 0 or width <= 0:
            self._side_wave_mesh = None
            return

        edges = {
            "front": rgb[height - 1, :, :],
            "right": rgb[::-1, width - 1, :],
            "back": rgb[0, ::-1, :],
            "left": rgb[:, 0, :],
        }
        half_w = self._plane_size[0] / 2.0
        half_h = self._plane_size[1] / 2.0

        segments = self._get_dynamic_side_wave_segments(height, width)
        spacing = float(self._spacing)
        depth = spacing * float(self._side_wave_depth_ratio)
        step = spacing / segments

        base_phase = np.arange(segments, dtype=np.float32) / segments
        if depth <= 0.0:
            curve = np.zeros_like(base_phase)
        else:
            half_span = spacing / 2.0
            radius = (depth * depth + half_span * half_span) / (2.0 * depth)
            x = (base_phase - 0.5) * spacing
            inside = np.clip(radius * radius - x * x, 0.0, None)
            sagitta = radius - np.sqrt(inside)
            curve = np.clip(sagitta / depth, 0.0, 1.0)

        # 角と側面の位相を合わせる
        phase_offset = 0
        if segments > 1:
            phase_offset = int(np.argmax(curve))

        def build_side(
            count: int,
            axis_start: float,
            axis_step: float,
            normal_base: float,
            normal_sign: float,
            axis_is_x: bool,
            phase_offset: int,
            edge_colors: np.ndarray,
            tint: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            samples = count * segments + 1
            idx = np.arange(samples, dtype=np.int32)
            phase = (idx + phase_offset) % segments
            offset = depth * curve[phase]
            shade = 1.0 - float(self._side_wave_shade_strength) * curve[phase]
            bead_idx = np.minimum((idx // segments).astype(np.int32), count - 1)
            base_colors = edge_colors[bead_idx].astype(np.float32) / 255.0
            colors = np.clip(base_colors * shade[:, None] * tint, 0.0, 1.0)
            axis = axis_start + axis_step * idx.astype(np.float32)
            normal = normal_base + normal_sign * offset
            if axis_is_x:
                x_vals = axis
                y_vals = normal
            else:
                x_vals = normal
                y_vals = axis
            return x_vals.astype(np.float32), y_vals.astype(np.float32), colors.astype(np.float32)

        front = build_side(
            width,
            -half_w,
            step,
            -half_h,
            1.0,
            True,
            phase_offset,
            edges["front"],
            float(self._side_wave_face_tint["front"]),
        )
        right = build_side(
            height,
            -half_h,
            step,
            half_w,
            -1.0,
            False,
            phase_offset,
            edges["right"],
            float(self._side_wave_face_tint["right"]),
        )
        back = build_side(
            width,
            half_w,
            -step,
            half_h,
            -1.0,
            True,
            phase_offset,
            edges["back"],
            float(self._side_wave_face_tint["back"]),
        )
        left = build_side(
            height,
            half_h,
            -step,
            -half_w,
            1.0,
            False,
            phase_offset,
            edges["left"],
            float(self._side_wave_face_tint["left"]),
        )

        # 分割数と同じ値で角も分割する
        corner_segments = segments
        ring_x: list[np.ndarray] = []
        ring_y: list[np.ndarray] = []
        ring_c: list[np.ndarray] = []

        def append_segment(x_vals: np.ndarray, y_vals: np.ndarray, colors: np.ndarray, drop_last: bool) -> None:
            if drop_last:
                x_vals = x_vals[:-1]
                y_vals = y_vals[:-1]
                colors = colors[:-1]
            ring_x.append(x_vals)
            ring_y.append(y_vals)
            ring_c.append(colors)

        def append_corner(
            center_x: float,
            center_y: float,
            start_angle: float,
            end_angle: float,
            start_color: np.ndarray,
            end_color: np.ndarray,
        ) -> None:
            if corner_segments <= 1:
                return
            angles = np.linspace(start_angle, end_angle, corner_segments + 1, dtype=np.float32)
            angles = angles[1:-1]
            if angles.size == 0:
                return
            x_vals = center_x + depth * np.cos(angles)
            y_vals = center_y + depth * np.sin(angles)
            t = np.linspace(0.0, 1.0, angles.size, dtype=np.float32)
            colors = start_color[None, :] * (1.0 - t[:, None]) + end_color[None, :] * t[:, None]
            ring_x.append(x_vals.astype(np.float32))
            ring_y.append(y_vals.astype(np.float32))
            ring_c.append(colors.astype(np.float32))

        front_x, front_y, front_c = front
        right_x, right_y, right_c = right
        back_x, back_y, back_c = back
        left_x, left_y, left_c = left

        append_segment(front_x, front_y, front_c, drop_last=False)
        append_corner(
            half_w - depth,
            -half_h + depth,
            0.0,
            -0.5 * np.pi,
            front_c[-1],
            right_c[0],
        )
        append_segment(right_x, right_y, right_c, drop_last=False)
        append_corner(
            half_w - depth,
            half_h - depth,
            0.5 * np.pi,
            0.0,
            right_c[-1],
            back_c[0],
        )
        append_segment(back_x, back_y, back_c, drop_last=False)
        append_corner(
            -half_w + depth,
            half_h - depth,
            np.pi,
            0.5 * np.pi,
            back_c[-1],
            left_c[0],
        )
        append_segment(left_x, left_y, left_c, drop_last=False)
        append_corner(
            -half_w + depth,
            -half_h + depth,
            1.5 * np.pi,
            np.pi,
            left_c[-1],
            front_c[0],
        )

        ring = np.column_stack([np.concatenate(ring_x), np.concatenate(ring_y)]).astype(np.float32)
        colors = np.concatenate(ring_c).astype(np.float32)
        if ring.shape[0] == 0:
            self._side_wave_mesh = None
        else:
            ring_uv = np.zeros_like(ring, dtype=np.float32)
            if half_w > 0.0 and half_h > 0.0:
                # 上下面の波形描画用にUVも持たせる
                ring_uv[:, 0] = (ring[:, 0] + half_w) / (2.0 * half_w)
                ring_uv[:, 1] = (ring[:, 1] + half_h) / (2.0 * half_h)
            # GL描画用にクワッドの頂点・色配列をあらかじめ構築する（描画時のPythonループを排除）
            count = ring.shape[0]
            next_ring = np.roll(ring, -1, axis=0)
            half_d = np.float32(max(self._box_depth, 0.05) / 2.0)
            quad_verts = np.empty((count * 4, 3), dtype=np.float32)
            quad_verts[0::4, :2] = ring
            quad_verts[0::4, 2] = half_d
            quad_verts[1::4, :2] = ring
            quad_verts[1::4, 2] = -half_d
            quad_verts[2::4, :2] = next_ring
            quad_verts[2::4, 2] = -half_d
            quad_verts[3::4, :2] = next_ring
            quad_verts[3::4, 2] = half_d
            quad_colors = np.empty((count * 4, 3), dtype=np.float32)
            quad_colors[0::4] = colors
            quad_colors[1::4] = colors
            quad_colors[2::4] = colors
            quad_colors[3::4] = colors
            # GL_TRIANGLE_FAN用配列: center + ring + closing vertex = count+2頂点
            fan_count = count + 2
            fan_verts_top = np.empty((fan_count, 3), dtype=np.float32)
            fan_verts_top[0] = (0.0, 0.0, half_d)
            fan_verts_top[1:count + 1, :2] = ring
            fan_verts_top[1:count + 1, 2] = half_d
            fan_verts_top[count + 1] = fan_verts_top[1]
            fan_verts_bot = fan_verts_top.copy()
            fan_verts_bot[:, 2] = -half_d
            fan_uv = np.empty((fan_count, 2), dtype=np.float32)
            fan_uv[0] = (0.5, 0.5)
            fan_uv[1:count + 1] = ring_uv
            fan_uv[count + 1] = fan_uv[1]
            fan_uv_flipped = fan_uv.copy()
            fan_uv_flipped[:, 0] = 1.0 - fan_uv_flipped[:, 0]
            self._side_wave_mesh = {
                "ring": ring,
                "colors": colors,
                "ring_uv": ring_uv,
                "quad_verts": quad_verts,
                "quad_colors": quad_colors,
                "fan_verts_top": fan_verts_top,
                "fan_verts_bot": fan_verts_bot,
                "fan_uv": fan_uv,
                "fan_uv_flipped": fan_uv_flipped,
            }

    def _get_dynamic_side_wave_segments(self, height: int, width: int) -> int:
        """画像サイズに応じて波型分割数を減らす。"""
        base = max(1, int(self._side_wave_segments_per_bead))
        total = int(height) * int(width)
        # 画像が大きいほど分割数を抑えるが、少しだけ上限を緩める
        if total >= 100000:
            return min(base, 1)
        if total > 80000:
            return min(base, 2)
        if total > 40000:
            return min(base, 3)
        return base

    def _get_dynamic_bead_px(self, height: int, width: int) -> int:
        """画像サイズに応じてビーズのテクスチャ解像度(px)を切り替える。"""
        base = max(1, int(self._bead_px_base))
        total = int(height) * int(width)
        # 段階を細かくして滑らかに解像度を上げる
        if total >= 100000:
            return max(2, min(base, 3))
        if total > 80000:
            return max(2, min(base, 4))
        if total > 60000:
            return max(2, min(base, 5))
        if total > 40000:
            return max(2, min(base, 6))
        if total > 30000:
            return max(2, min(base + 2, 7))
        if total > 20000:
            return max(2, min(base + 3, 10))
        if total > 12000:
            return max(2, min(base + 5, 20))
        if total > 6000:
            return max(2, min(base + 7, 40))
        return min(base + 10, 100)

    def _make_circle_mask(self, size: int, radius_ratio: float, feather: float = 0.0) -> np.ndarray:
        """円形のソフトマスクを作る。"""
        center = (size - 1) / 2.0
        radius = size * radius_ratio
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        if feather <= 0.0:
            return (dist <= radius).astype(np.float32)
        # 端だけ少しぼかしてギザギザを減らす
        return np.clip((radius - dist) / feather, 0.0, 1.0).astype(np.float32)

    def _on_left_press(self, event: tk.Event) -> None:
        # 左ドラッグで回転
        self._last_left = (event.x, event.y)

    def _on_left_drag(self, event: tk.Event) -> None:
        if self._last_left is None:
            return
        dx = event.x - self._last_left[0]
        dy = event.y - self._last_left[1]
        self._rot_y = (self._rot_y + dx * 0.3) % 360.0
        self._rot_x = max(-80.0, min(80.0, self._rot_x + dy * 0.2))
        self._last_left = (event.x, event.y)
        self._request_redraw()

    def _on_left_release(self, _event: tk.Event) -> None:
        self._last_left = None

    def _on_right_press(self, event: tk.Event) -> None:
        # 右ドラッグで平行移動
        self._last_right = (event.x, event.y)

    def _on_right_drag(self, event: tk.Event) -> None:
        if self._last_right is None:
            return
        dx = event.x - self._last_right[0]
        dy = event.y - self._last_right[1]
        scale = 0.03
        self._pan_x += dx * scale
        self._pan_y -= dy * scale
        self._last_right = (event.x, event.y)
        self._request_redraw()

    def _on_right_release(self, _event: tk.Event) -> None:
        self._last_right = None

    def _apply_zoom(self, steps: float) -> None:
        # ズームはカメラ距離を変更
        self._camera_distance = max(2.0, min(120.0, self._camera_distance - steps * 0.8))
        self._request_redraw()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if getattr(event, "delta", 0) == 0:
            return
        steps = event.delta / 120.0
        self._apply_zoom(steps)

    def _on_mouse_wheel_linux(self, event: tk.Event) -> None:
        if event.num == 4:
            self._apply_zoom(1.0)
        elif event.num == 5:
            self._apply_zoom(-1.0)

    def _on_space(self, _event: tk.Event) -> str:
        # スペースで自動回転を切り替える
        self._auto_rotate = not self._auto_rotate
        if self._auto_rotate:
            self._start_auto_rotate()
        else:
            self._stop_auto_rotate()
        self._request_redraw()
        return "break"

    def _on_toggle_key_release(self, event: tk.Event) -> None:
        self._keys_held.discard(event.keysym.lower())

    def _on_toggle_background(self, event: tk.Event) -> str:
        # Zキーで背景色を白/黒に切り替える（長押し無効）
        key = event.keysym.lower()
        if key in self._keys_held:
            return "break"
        self._keys_held.add(key)
        self._background_is_black = not self._background_is_black
        if self.context_created:
            self._apply_background_color()
        self._request_redraw()
        return "break"

    def _on_toggle_hole_gap(self, event: tk.Event) -> str:
        # Xキーで穴と角の色を白/黒に切り替える（長押し無効）
        key = event.keysym.lower()
        if key in self._keys_held:
            return "break"
        self._keys_held.add(key)
        self._hole_gap_is_white = not self._hole_gap_is_white
        self._apply_appearance_profile(self._appearance_mode)
        return "break"

    def _on_toggle_appearance(self, event: tk.Event) -> str:
        # Cキーでビーズの穴と隙間の見た目を切り替える（長押し無効）
        key = event.keysym.lower()
        if key in self._keys_held:
            return "break"
        self._keys_held.add(key)
        self._appearance_mode = "ironed" if self._appearance_mode == "normal" else "normal"
        self._apply_appearance_profile(self._appearance_mode)
        return "break"

    def _apply_appearance_profile(self, profile_key: str) -> None:
        # 見た目プリセットを反映してテクスチャを作り直す
        profile = self._appearance_profiles.get(profile_key)
        if not profile:
            return
        self._bead_radius_ratio = float(profile.get("bead_radius_ratio", self._bead_radius_ratio))
        self._hole_radius_ratio = float(profile.get("hole_radius_ratio", self._hole_radius_ratio))
        self._hole_strength = float(profile.get("hole_strength", self._hole_strength))
        if self._hole_gap_is_white:
            self._hole_color = self._hole_color_light
            self._gap_color = self._gap_color_light
        else:
            self._hole_color = self._hole_color_dark
            self._gap_color = self._gap_color_dark
        self._preserve_view_on_rebuild = True
        self._texture_image = None
        self._texture_dirty = True
        self._side_wave_dirty = True
        self._request_redraw()

    def _apply_background_color(self) -> None:
        """現在の設定に合わせて背景色を反映する。"""
        if self._background_is_black:
            glClearColor(0.0, 0.0, 0.0, 1.0)
        else:
            glClearColor(1.0, 1.0, 1.0, 1.0)

    def initgl(self) -> None:
        # 画面の初期化（ライティングは使わない）
        self._apply_background_color()
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)

    def redraw(self) -> None:
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / height
        gluPerspective(45.0, aspect, 0.1, 200.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, 0.0, self._camera_distance, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        glTranslatef(self._pan_x, self._pan_y, 0.0)
        glRotatef(self._rot_x, 1.0, 0.0, 0.0)
        glRotatef(self._rot_y + self._angle, 0.0, 1.0, 0.0)

        if self._texture_dirty:
            self._upload_texture()
        if self._texture_id is not None and self._texture_size != (0, 0):
            glEnable(GL_TEXTURE_2D)
            self._draw_textured_box()
        else:
            glDisable(GL_TEXTURE_2D)

        # 自動回転が有効な場合のみ更新
        if self._auto_rotate:
            self._angle = (self._angle + self._angle_speed) % 360.0

    def _upload_texture(self) -> None:
        """OpenGLにテクスチャを送る。"""
        max_size = self._get_max_texture_size()
        if self._texture_image is None or (
            self._texture_image.size[0] > max_size or self._texture_image.size[1] > max_size
        ):
            if not self._build_texture_from_source():
                self._texture_dirty = False
                return
        if self._texture_image is None:
            self._texture_dirty = False
            return
        if self._texture_id is None:
            tex_ids = glGenTextures(1)
            if isinstance(tex_ids, (list, tuple, np.ndarray)):
                tex_ids = tex_ids[0]
            self._texture_id = int(tex_ids)

        image = self._texture_image
        width, height = image.size
        data = image.tobytes("raw", "RGB")
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
        self._texture_dirty = False

    def _draw_wavy_face(self, z: float, tint: float, flip_u: bool) -> None:
        """波型の上下面を描画する。"""
        mesh = self._side_wave_mesh
        if not mesh:
            return
        fan_verts = mesh.get("fan_verts_top" if z >= 0.0 else "fan_verts_bot")
        fan_uv = mesh.get("fan_uv_flipped" if flip_u else "fan_uv")
        if fan_verts is None or fan_uv is None or fan_verts.shape[0] < 3:
            return
        glColor3f(tint, tint, tint)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, fan_verts)
        glTexCoordPointer(2, GL_FLOAT, 0, fan_uv)
        glDrawArrays(GL_TRIANGLE_FAN, 0, fan_verts.shape[0])
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def _draw_textured_box(self) -> None:
        """テクスチャ付きの箱を描く。"""
        half_w = self._plane_size[0] / 2.0
        half_h = self._plane_size[1] / 2.0
        half_d = max(self._box_depth, 0.05) / 2.0
        top_tex = self._texture_id
        if top_tex is None:
            return

        if self._side_wave_dirty and self._source_rgb is not None:
            # テクスチャ更新に追従して波型メッシュも更新する
            rgb = self._downsample_to_texture_limit(self._source_rgb)
            self._build_side_wave_mesh(rgb)
            self._side_wave_dirty = False

        use_wave_face = self._side_wave_mesh is not None

        # 面ごとに明度を変えて厚みを見せる
        glBindTexture(GL_TEXTURE_2D, top_tex)
        if use_wave_face:
            self._draw_wavy_face(half_d, 1.0, False)
        else:
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 0.0)
            glVertex3f(-half_w, -half_h, half_d)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(half_w, -half_h, half_d)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(half_w, half_h, half_d)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(-half_w, half_h, half_d)
            glEnd()

        glBindTexture(GL_TEXTURE_2D, top_tex)
        if use_wave_face:
            self._draw_wavy_face(-half_d, 0.85, False)
        else:
            glColor3f(0.85, 0.85, 0.85)
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 0.0)
            glVertex3f(-half_w, -half_h, -half_d)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(half_w, -half_h, -half_d)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(half_w, half_h, -half_d)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(-half_w, half_h, -half_d)
            glEnd()

        glDisable(GL_TEXTURE_2D)
        if self._side_wave_mesh:
            self._draw_side_wave_mesh()

    def _draw_side_wave_mesh(self) -> None:
        """連続カーブの波型メッシュを描画する。"""
        mesh = self._side_wave_mesh
        if not mesh:
            return
        quad_verts = mesh.get("quad_verts")
        quad_colors = mesh.get("quad_colors")
        if quad_verts is None or quad_colors is None or quad_verts.shape[0] == 0:
            return
        glShadeModel(GL_FLAT)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, quad_verts)
        glColorPointer(3, GL_FLOAT, 0, quad_colors)
        glDrawArrays(GL_QUADS, 0, quad_verts.shape[0])
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glShadeModel(GL_SMOOTH)
