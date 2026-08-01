"""立体アイロンビーズ設計（試作）。

変換後画像を1層目として3Dペグボードに敷き詰め、その上へ層を重ねて
立体作品を組み上げるエディタ。ビーズは「作業層」の高さの平面へ
マウス追従のピンセットで1粒ずつ置く。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

import numpy as np
from pyopengltk import OpenGLFrame
from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_COMPILE,
    GL_CONSTANT_ALPHA,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FLOAT,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_MODELVIEW,
    GL_MODELVIEW_MATRIX,
    GL_NORMAL_ARRAY,
    GL_NORMALIZE,
    GL_ONE_MINUS_CONSTANT_ALPHA,
    GL_POSITION,
    GL_PROJECTION,
    GL_PROJECTION_MATRIX,
    GL_TRIANGLES,
    GL_VERTEX_ARRAY,
    GL_VIEWPORT,
    glBlendColor,
    glBlendFunc,
    glCallList,
    glClear,
    glClearColor,
    glColor3f,
    glColorMaterial,
    glDeleteLists,
    glDisable,
    glDisableClientState,
    glDrawArrays,
    glEnable,
    glEnableClientState,
    glEndList,
    glGenLists,
    glGetDoublev,
    glGetIntegerv,
    glLightfv,
    glLoadIdentity,
    glMatrixMode,
    glNewList,
    glNormalPointer,
    glPopMatrix,
    glPushMatrix,
    glTranslatef,
    glVertexPointer,
    glViewport,
)
from OpenGL.GLU import gluLookAt, gluPerspective, gluUnProject

from converter.pipeline import _palette_rgb_to_index
from palette import BeadPalette

from .simulator_board import (
    MAX_LAYERS,
    BeadBoard,
    BoardHistory,
    DomePlan,
    dome_heights,
)

# --- 寸法定数（ペグ間隔 = 1.0 のグリッド単位） ---
PLATE_PEGS = 29          # 1プレートあたりのペグ数（29x29）
BEAD_OUTER_R = 0.47      # ビーズ外半径（直径0.94。実物同様、隣とわずかに隙間ができる）
BEAD_INNER_R = 0.19      # ビーズ穴半径
BEAD_HEIGHT = 0.94       # ビーズ高さ。実物のビーズは直径:高さ≒1:1なので直径に合わせる
BEAD_SEGMENTS = 12
PEG_RADIUS = 0.13
PEG_HEIGHT = 0.42
PEG_SEGMENTS = 8
PLATE_THICKNESS = 0.25
GHOST_ALPHA = 0.35       # ゴーストビーズの不透明度
GHOST_ALPHA_STEP = 0.05  # A/Dキーでの濃度調整幅
MAX_BRUSH = 9            # ブラシの一辺の最大値（N×Nセルをまとめて塗る）
REDRAW_INTERVAL_MS = 16  # 描画の遅延上限（約60fps）。after_idle が走れない時の保険
BEAD_HOLD_Y = BEAD_HEIGHT * 1.45  # ピンセットで持ったビーズの浮遊高さ（作業層の面から）
BEAD_LIFT_Y = 0.012      # プレート上面との同一平面Zファイティングを避ける微小な浮かせ量

_SLAB_COLOR = (0.80, 0.81, 0.84)
_PEG_COLOR = (0.70, 0.71, 0.75)
_TWEEZER_COLOR = (0.72, 0.74, 0.78)


# 自動ドームの表示名（内部名 → 日本語ラベル）
_DOME_PROFILE_LABELS = {
    "hemisphere": "半球",
    "parabola": "放物線",
    "cone": "円錐",
}
_DOME_LABEL_TO_PROFILE = {v: k for k, v in _DOME_PROFILE_LABELS.items()}
DOME_SCALE_MIN = 0.2
DOME_SCALE_MAX = 1.5


def layer_y(layer: int) -> float:
    """層の底面の高さ。ビーズは下の層のビーズにそのまま載る。"""
    return layer * BEAD_HEIGHT


# --- メッシュ生成（純numpy、GL_TRIANGLES用の頂点/法線配列を返す） ---

def _band_triangles(
    bottom: np.ndarray,
    top: np.ndarray,
    n_bottom: np.ndarray,
    n_top: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """リング2本（各 (N+1,3)）を三角形帯に変換する。"""
    v0, v1 = bottom[:-1], bottom[1:]
    v2, v3 = top[1:], top[:-1]
    m0, m1 = n_bottom[:-1], n_bottom[1:]
    m2, m3 = n_top[1:], n_top[:-1]
    verts = np.stack([v0, v1, v2, v0, v2, v3], axis=1).reshape(-1, 3)
    norms = np.stack([m0, m1, m2, m0, m2, m3], axis=1).reshape(-1, 3)
    return verts, norms


def _ring(radius: float, y: float, segments: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    return np.stack(
        [radius * np.cos(theta), np.full_like(theta, y), radius * np.sin(theta)],
        axis=1,
    )


def _ring_normals(segments: int, direction: tuple[float, float, float] | None = None,
                  inward: bool = False) -> np.ndarray:
    """円筒壁の放射法線、または固定方向の法線を作る。"""
    theta = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    if direction is not None:
        return np.tile(np.asarray(direction, dtype=np.float64), (segments + 1, 1))
    sign = -1.0 if inward else 1.0
    return np.stack(
        [sign * np.cos(theta), np.zeros_like(theta), sign * np.sin(theta)], axis=1
    )


def build_bead_mesh() -> tuple[np.ndarray, np.ndarray]:
    """穴あき円筒のビーズメッシュ（底面 y=0、外壁/上面/内壁/底面）。"""
    n = BEAD_SEGMENTS
    ob = _ring(BEAD_OUTER_R, 0.0, n)
    ot = _ring(BEAD_OUTER_R, BEAD_HEIGHT, n)
    ib = _ring(BEAD_INNER_R, 0.0, n)
    it = _ring(BEAD_INNER_R, BEAD_HEIGHT, n)
    out_n = _ring_normals(n)
    in_n = _ring_normals(n, inward=True)
    up_n = _ring_normals(n, direction=(0.0, 1.0, 0.0))
    down_n = _ring_normals(n, direction=(0.0, -1.0, 0.0))
    parts = [
        _band_triangles(ob, ot, out_n, out_n),   # 外壁
        _band_triangles(ot, it, up_n, up_n),     # 上面（ドーナツ）
        _band_triangles(it, ib, in_n, in_n),     # 内壁
        _band_triangles(ib, ob, down_n, down_n), # 底面
    ]
    verts = np.concatenate([p[0] for p in parts]).astype(np.float32)
    norms = np.concatenate([p[1] for p in parts]).astype(np.float32)
    return np.ascontiguousarray(verts), np.ascontiguousarray(norms)


def build_peg_mesh() -> tuple[np.ndarray, np.ndarray]:
    """ペグ（円柱: 外壁+上蓋）。底面 y=0。"""
    n = PEG_SEGMENTS
    bottom = _ring(PEG_RADIUS, 0.0, n)
    top = _ring(PEG_RADIUS, PEG_HEIGHT, n)
    wall_n = _ring_normals(n)
    wall_v, wall_nn = _band_triangles(bottom, top, wall_n, wall_n)
    # 上蓋（中心と縁の三角形扇を展開）
    center = np.array([0.0, PEG_HEIGHT, 0.0])
    cap_tris = []
    for i in range(n):
        cap_tris.extend([center, top[i], top[i + 1]])
    cap_v = np.asarray(cap_tris)
    cap_n = np.tile(np.array([0.0, 1.0, 0.0]), (len(cap_v), 1))
    verts = np.concatenate([wall_v, cap_v]).astype(np.float32)
    norms = np.concatenate([wall_nn, cap_n]).astype(np.float32)
    return np.ascontiguousarray(verts), np.ascontiguousarray(norms)


def _box_mesh(x0: float, y0: float, z0: float,
              x1: float, y1: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """軸平行の箱（6面 12三角形）。"""
    faces = [
        # (4隅の順序, 法線)
        (((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)), (1, 0, 0)),
        (((x0, y0, z1), (x0, y0, z0), (x0, y1, z0), (x0, y1, z1)), (-1, 0, 0)),
        (((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)), (0, 1, 0)),
        (((x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0)), (0, -1, 0)),
        (((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)), (0, 0, 1)),
        (((x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z0)), (0, 0, -1)),
    ]
    verts = []
    norms = []
    for corners, normal in faces:
        a, b, c, d = corners
        verts.extend([a, b, c, a, c, d])
        norms.extend([normal] * 6)
    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(norms, dtype=np.float32),
    )


def build_plate_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1プレート分の板とペグ群（プレートローカル座標: ペグ位置 0..28）。"""
    ext = PLATE_PEGS - 0.5
    slab_v, slab_n = _box_mesh(-0.5, -PLATE_THICKNESS, -0.5, ext, 0.0, ext)
    peg_v, peg_n = build_peg_mesh()
    ii, jj = np.meshgrid(np.arange(PLATE_PEGS), np.arange(PLATE_PEGS), indexing="ij")
    offsets = np.stack(
        [jj.ravel(), np.zeros(PLATE_PEGS * PLATE_PEGS), ii.ravel()], axis=1
    ).astype(np.float32)
    pegs_v = (peg_v[None, :, :] + offsets[:, None, :]).reshape(-1, 3).astype(np.float32)
    pegs_n = np.tile(peg_n, (len(offsets), 1)).astype(np.float32)
    return (
        np.ascontiguousarray(slab_v),
        np.ascontiguousarray(slab_n),
        np.ascontiguousarray(pegs_v),
        np.ascontiguousarray(pegs_n),
    )


def _rot_z(points: np.ndarray, degrees: float) -> np.ndarray:
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    out = points.copy()
    out[:, 0] = points[:, 0] * c - points[:, 1] * s
    out[:, 1] = points[:, 0] * s + points[:, 1] * c
    return out


def build_tweezer_mesh() -> tuple[np.ndarray, np.ndarray]:
    """V字型ピンセット（先端が原点、上方向に開く）。"""
    arm_v, arm_n = _box_mesh(-0.04, 0.05, -0.04, 0.04, 1.5, 0.04)
    left_v, left_n = _rot_z(arm_v, 9.0), _rot_z(arm_n, 9.0)
    right_v, right_n = _rot_z(arm_v, -9.0), _rot_z(arm_n, -9.0)
    verts = np.concatenate([left_v, right_v]).astype(np.float32)
    norms = np.concatenate([left_n, right_n]).astype(np.float32)
    return np.ascontiguousarray(verts), np.ascontiguousarray(norms)


_mesh_cache: dict[str, tuple[np.ndarray, ...]] = {}


def _get_meshes() -> dict[str, tuple[np.ndarray, ...]]:
    """メッシュ配列をモジュール内で1回だけ生成して使い回す。"""
    if not _mesh_cache:
        _mesh_cache["bead"] = build_bead_mesh()
        _mesh_cache["plate"] = build_plate_mesh()
        _mesh_cache["tweezer"] = build_tweezer_mesh()
    return _mesh_cache


class BeadsSimulator3DWindow(tk.Toplevel):
    """立体設計用ウィンドウ（左パネル + 3D盤面）。"""

    def __init__(self, master: tk.Misc, on_close: Optional[Callable[[], None]] = None) -> None:
        super().__init__(master)
        self._window_state_path = Path(__file__).resolve().parent / "simulator_3d_state.json"
        self._last_geometry: Optional[tuple[int, int, int, int]] = None
        self._last_normal_geometry: Optional[tuple[int, int, int, int]] = None
        self._last_window_state: str = "normal"
        self._ghost_alpha = GHOST_ALPHA
        self._translucent_below = True
        self._inherit_color = True
        self._restrict_to_image = True
        self._brush_size = 1
        self._dome_profile = "hemisphere"
        self._dome_scale = 1.0
        self._panel_width: Optional[int] = None
        self.title("立体ビーズ設計（試作）")
        if not self._load_window_state():
            self.geometry("980x640")
        self._on_close_callback = on_close
        self._palette: Optional[BeadPalette] = None
        self._color_rows: list[dict] = []
        self._row_widgets: dict[int, dict] = {}
        self._selected_palette_index = -1
        self._show_all_colors = False
        self._syncing_layer = False

        self._paned = ttk.PanedWindow(self, orient="horizontal")
        self._paned.pack(fill="both", expand=True)
        self._build_panel()
        self._gl_frame = BeadsSimulatorGL(
            self._paned,
            on_board_changed=self._on_board_changed,
            on_ghost_toggled=self._on_ghost_toggled_from_key,
            on_layer_changed=self._on_layer_changed_from_key,
            on_alpha_changed=self._on_alpha_changed_from_key,
            on_brush_changed=self._on_brush_changed_from_key,
        )
        self._paned.add(self._panel, weight=0)
        self._paned.add(self._gl_frame, weight=1)
        self._gl_frame.animate = 0  # 操作時のみ再描画する
        self._gl_frame.set_ghost_alpha(self._ghost_alpha)
        self._gl_frame.set_translucent_below(self._translucent_below)
        self._gl_frame.set_inherit_color(self._inherit_color)
        self._gl_frame.set_restrict_to_image(self._restrict_to_image)
        self._gl_frame.set_brush_size(self._brush_size)
        self._gl_frame.focus_set()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Configure>", self._on_window_configure)
        # サッシ位置はウィンドウ表示後でないと反映されない
        self.after(80, self._restore_panel_width)

    # --- 左パネル構築 ---

    def _build_panel(self) -> None:
        panel = ttk.Frame(self._paned, padding=(8, 8, 4, 8))
        self._panel = panel
        self._summary_var = tk.StringVar(value="ビーズ: 0 個 / 層数: 0")
        ttk.Label(panel, textvariable=self._summary_var).pack(anchor="w")

        # 作業層（内部は0始まり、表示は1始まり）
        layer_row = ttk.Frame(panel)
        layer_row.pack(fill="x", pady=(6, 0))
        ttk.Label(layer_row, text="作業層", font=("", 9)).pack(side="left")
        self._layer_var = tk.IntVar(value=1)
        self._layer_spin = ttk.Spinbox(
            layer_row,
            from_=1,
            to=MAX_LAYERS,
            width=5,
            textvariable=self._layer_var,
            command=self._on_layer_spin,
        )
        self._layer_spin.pack(side="left", padx=(6, 0))
        self._layer_spin.bind("<Return>", lambda _e: self._on_layer_spin())
        self._layer_spin.bind("<FocusOut>", lambda _e: self._on_layer_spin())
        ttk.Label(layer_row, text="層目 (W/S)", font=("", 8), foreground="#555").pack(
            side="left", padx=(4, 0)
        )

        brush_row = ttk.Frame(panel)
        brush_row.pack(fill="x", pady=(4, 0))
        ttk.Label(brush_row, text="ブラシ", font=("", 9)).pack(side="left")
        self._brush_var = tk.IntVar(value=self._brush_size)
        self._brush_spin = ttk.Spinbox(
            brush_row,
            from_=1,
            to=MAX_BRUSH,
            width=5,
            textvariable=self._brush_var,
            command=self._on_brush_spin,
        )
        self._brush_spin.pack(side="left", padx=(6, 0))
        self._brush_spin.bind("<Return>", lambda _e: self._on_brush_spin())
        self._brush_spin.bind("<FocusOut>", lambda _e: self._on_brush_spin())
        ttk.Label(brush_row, text="角 (1〜9キー)", font=("", 8), foreground="#555").pack(
            side="left", padx=(4, 0)
        )

        # 表示上限（断面表示）
        top_row = ttk.Frame(panel)
        top_row.pack(fill="x", pady=(4, 0))
        ttk.Label(top_row, text="表示上限", font=("", 9)).pack(side="left")
        self._visible_top_scale = ttk.Scale(
            top_row,
            from_=1,
            to=MAX_LAYERS,
            orient="horizontal",
            command=self._on_visible_top_change,
        )
        self._visible_top_scale.pack(side="left", fill="x", expand=True, padx=(6, 2))
        self._follow_top_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            panel,
            text="表示上限を作業層に追従",
            variable=self._follow_top_var,
            command=self._on_follow_top_check,
        ).pack(anchor="w")

        self._translucent_var = tk.BooleanVar(value=self._translucent_below)
        ttk.Checkbutton(
            panel,
            text="下層を半透明にする",
            variable=self._translucent_var,
            command=self._on_translucent_check,
        ).pack(anchor="w", pady=(4, 0))
        self._ghost_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            panel,
            text="1層目のガイド表示 (G)",
            variable=self._ghost_var,
            command=self._on_ghost_check,
        ).pack(anchor="w")
        alpha_row = ttk.Frame(panel)
        alpha_row.pack(fill="x", pady=(2, 0))
        ttk.Label(alpha_row, text="半透明の濃さ", font=("", 9)).pack(side="left")
        self._ghost_alpha_var = tk.DoubleVar(value=self._ghost_alpha)
        ttk.Scale(
            alpha_row,
            from_=0.05,
            to=1.0,
            orient="horizontal",
            variable=self._ghost_alpha_var,
            command=self._on_ghost_alpha_change,
        ).pack(side="left", fill="x", expand=True, padx=(6, 2))

        undo_row = ttk.Frame(panel)
        undo_row.pack(fill="x", pady=(6, 0))
        self._undo_button = ttk.Button(
            undo_row, text="元に戻す (Ctrl+Z)", command=self._on_undo, state="disabled"
        )
        self._undo_button.pack(side="left", fill="x", expand=True)
        self._redo_button = ttk.Button(
            undo_row, text="やり直す (Ctrl+X)", command=self._on_redo, state="disabled"
        )
        self._redo_button.pack(side="left", fill="x", expand=True, padx=(4, 0))

        ttk.Button(
            panel, text="背景色以外を1層上げる", command=self._on_raise_non_background
        ).pack(anchor="w", pady=(6, 0), fill="x")
        self._background_var = tk.StringVar(value="背景色: -")
        ttk.Label(
            panel,
            textvariable=self._background_var,
            font=("", 8),
            foreground="#555",
            wraplength=200,
            justify="left",
        ).pack(anchor="w")

        dome = ttk.LabelFrame(panel, text="自動ドーム", padding=(6, 2, 6, 4))
        dome.pack(fill="x", pady=(6, 0))
        curve_row = ttk.Frame(dome)
        curve_row.pack(fill="x")
        ttk.Label(curve_row, text="カーブ", font=("", 9)).pack(side="left")
        self._dome_profile_var = tk.StringVar(value=_DOME_PROFILE_LABELS[self._dome_profile])
        ttk.Combobox(
            curve_row,
            textvariable=self._dome_profile_var,
            values=list(_DOME_PROFILE_LABELS.values()),
            state="readonly",
            width=8,
        ).pack(side="left", padx=(6, 0))
        scale_row = ttk.Frame(dome)
        scale_row.pack(fill="x", pady=(2, 0))
        ttk.Label(scale_row, text="高さ倍率", font=("", 9)).pack(side="left")
        self._dome_scale_var = tk.DoubleVar(value=self._dome_scale)
        ttk.Scale(
            scale_row,
            from_=DOME_SCALE_MIN,
            to=DOME_SCALE_MAX,
            orient="horizontal",
            variable=self._dome_scale_var,
            command=self._on_dome_scale_change,
        ).pack(side="left", fill="x", expand=True, padx=(6, 4))
        self._dome_scale_label = ttk.Label(scale_row, text=f"{self._dome_scale:.1f}", font=("", 8))
        self._dome_scale_label.pack(side="left")
        ttk.Button(dome, text="ドーム状に盛り上げる", command=self._on_apply_dome).pack(
            fill="x", pady=(4, 0)
        )

        ttk.Button(
            panel, text="1層目を変換結果で敷き詰める", command=self._on_fill_layer0
        ).pack(anchor="w", pady=(6, 2), fill="x")
        ttk.Button(panel, text="盤面をリセット", command=self._on_reset).pack(
            anchor="w", pady=(0, 4), fill="x"
        )

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=4)
        self._inherit_var = tk.BooleanVar(value=self._inherit_color)
        ttk.Checkbutton(
            panel,
            text="真下と同じ色で置く",
            variable=self._inherit_var,
            command=self._on_inherit_check,
        ).pack(anchor="w")
        ttk.Label(
            panel,
            text="（真下が空なら1層目の色。下の一覧は引き継げない場合の色）",
            font=("", 8),
            foreground="#555",
            wraplength=200,
            justify="left",
        ).pack(anchor="w")
        self._restrict_var = tk.BooleanVar(value=self._restrict_to_image)
        ttk.Checkbutton(
            panel,
            text="元画像の範囲内だけに置く",
            variable=self._restrict_var,
            command=self._on_restrict_check,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            panel,
            text="（オフにすると元画像の外へはみ出して置ける）",
            font=("", 8),
            foreground="#555",
            wraplength=200,
            justify="left",
        ).pack(anchor="w")
        ttk.Label(panel, text="使用色（クリックで選択）").pack(anchor="w", pady=(4, 0))
        self._show_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            panel,
            text="全パレット色を表示",
            variable=self._show_all_var,
            command=self._on_show_all_check,
        ).pack(anchor="w")

        list_holder = ttk.Frame(panel)
        list_holder.pack(fill="both", expand=True, pady=(2, 4))
        self._list_canvas = tk.Canvas(
            list_holder, width=200, highlightthickness=0, bg="#f5f5f5"
        )
        scroll = ttk.Scrollbar(list_holder, orient="vertical", command=self._list_canvas.yview)
        self._list_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._list_canvas.pack(side="left", fill="both", expand=True)
        self._list_inner = tk.Frame(self._list_canvas, bg="#f5f5f5")
        self._list_window = self._list_canvas.create_window(
            (0, 0), window=self._list_inner, anchor="nw"
        )
        self._list_inner.bind(
            "<Configure>",
            lambda _e: self._list_canvas.configure(scrollregion=self._list_canvas.bbox("all")),
        )
        # パネル幅の変更に合わせて行の幅を追従させる
        self._list_canvas.bind(
            "<Configure>",
            lambda e: self._list_canvas.itemconfigure(self._list_window, width=e.width),
        )
        self._bind_list_scroll(self._list_canvas)
        self._bind_list_scroll(self._list_inner)

        ttk.Label(
            panel,
            text="左ドラッグ: 配置 / 右ドラッグ: 削除\n"
                 "Ctrl+Z: 元に戻す / Ctrl+X: やり直す\n"
                 "1〜9: ブラシの一辺（N×N）\n"
                 "W / S: 作業層を上げる / 下げる\n"
                 "A / D: 半透明を薄く / 濃く\n"
                 "中ドラッグ or Shift+左: 回転\n"
                 "Ctrl+左ドラッグ: 移動 / ホイール: 拡縮",
            font=("", 8),
            foreground="#555",
        ).pack(anchor="w", side="bottom")

    def _bind_list_scroll(self, widget: tk.Misc) -> None:
        widget.bind("<MouseWheel>", self._on_list_scroll)

    def _on_list_scroll(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            self._list_canvas.yview_scroll(int(-delta / 120), "units")

    # --- 目標設定 ---

    def set_target(self, image: np.ndarray, palette: BeadPalette) -> None:
        """変換後画像から1層目のグリッドを作る。同一画像なら作品を維持する。"""
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            return
        rgb = np.asarray(image[:, :, :3], dtype=np.uint8)
        idx = _palette_rgb_to_index(rgb, palette).astype(np.int16)
        h, w = idx.shape
        plates_x = max(1, math.ceil(w / PLATE_PEGS))
        plates_y = max(1, math.ceil(h / PLATE_PEGS))
        target = np.full((plates_y * PLATE_PEGS, plates_x * PLATE_PEGS), -1, dtype=np.int16)
        target[:h, :w] = idx
        current = self._gl_frame.target_grid
        if current is not None and np.array_equal(current, target):
            # 画像が同じでもパレット定義は差し替わっている場合があるので更新する
            self._palette = palette
            return
        # 目標を差し替えると積み上げた作品は失われる
        if self._gl_frame.has_beads_above_layer0():
            if not messagebox.askyesno(
                "作品の破棄",
                "2層目以上に置いたビーズがあります。\n"
                "新しい画像に差し替えると失われますが、よろしいですか？",
                parent=self,
            ):
                return
        self._palette = palette
        self._gl_frame.set_target_grid(target, palette.rgb_uint8)
        self._gl_frame.fill_layer0()  # 1層目は完成した状態から始める
        self._gl_frame.reset_history()  # 敷き詰めた状態を履歴の起点にする
        self._build_color_rows()
        self._on_board_changed()

    def _build_color_rows(self) -> None:
        """色一覧を作り直す。既定は1層目で使う色、チェック時は全パレット色。"""
        palette = self._palette
        if palette is None:
            return
        target = self._gl_frame.target_grid
        target_counts: dict[int, int] = {}
        if target is not None:
            used = target[target >= 0]
            if used.size:
                uniq, counts = np.unique(used, return_counts=True)
                target_counts = {int(k): int(v) for k, v in zip(uniq, counts)}

        if self._show_all_colors:
            # 全表示のときはCSVの並び順のほうが目当ての色を探しやすい
            indices = list(range(len(palette)))
        else:
            placed = self._gl_frame.color_counts()
            candidates = set(target_counts) | set(placed)
            indices = sorted(candidates, key=lambda pi: (-target_counts.get(pi, 0), pi))

        self._color_rows = []
        for pi in indices:
            color = palette.colors[pi]
            self._color_rows.append(
                {
                    "palette_index": pi,
                    "color_id": color.color_id,
                    "name": color.name,
                    "rgb": tuple(int(v) for v in palette.rgb_uint8[pi]),
                }
            )
        for child in self._list_inner.winfo_children():
            child.destroy()
        self._row_widgets = {}
        for row in self._color_rows:
            self._make_color_row(row)
        # 選択色は可能なかぎり維持する
        keep = self._selected_palette_index
        if keep >= 0 and keep in self._row_widgets:
            self._select_color(keep)
        elif self._color_rows:
            self._select_color(self._color_rows[0]["palette_index"])
        else:
            self._selected_palette_index = -1
            self._gl_frame.set_selected_color(-1)

    def _make_color_row(self, row: dict) -> None:
        pi = row["palette_index"]
        r, g, b = row["rgb"]
        frame = tk.Frame(self._list_inner, bg="#f5f5f5", padx=2, pady=1)
        frame.pack(fill="x")
        swatch = tk.Label(
            frame, width=2, bg=f"#{r:02x}{g:02x}{b:02x}", relief="solid", bd=1
        )
        swatch.pack(side="left", padx=(2, 4))
        # 使用数はパネルが狭くても隠れないよう先に右側へ確保する
        count = tk.Label(frame, text="使用 0", bg="#f5f5f5", font=("", 9))
        count.pack(side="right", padx=(0, 2))
        name = tk.Label(
            frame,
            text=f"{row['color_id']} {row['name']}",
            bg="#f5f5f5",
            anchor="w",
            font=("", 9),
        )
        name.pack(side="left", fill="x", expand=True)
        widgets = {"frame": frame, "name": name, "count": count}
        self._row_widgets[pi] = widgets
        for widget in (frame, swatch, name, count):
            widget.bind("<Button-1>", lambda _e, p=pi: self._select_color(p))
            self._bind_list_scroll(widget)

    def _select_color(self, palette_index: int) -> None:
        self._selected_palette_index = palette_index
        self._gl_frame.set_selected_color(palette_index)
        for pi, widgets in self._row_widgets.items():
            bg = "#cce4ff" if pi == palette_index else "#f5f5f5"
            widgets["frame"].configure(bg=bg)
            widgets["name"].configure(bg=bg)
            widgets["count"].configure(bg=bg)

    # --- 集計表示・層コントロール ---

    def _on_board_changed(self) -> None:
        """配置/削除/リセット後に集計表示と層コントロールを更新する。"""
        counts = self._gl_frame.color_counts()
        total = sum(counts.values())
        top = self._gl_frame.top_layer()
        self._summary_var.set(f"ビーズ: {total} 個 / 層数: {top + 1}")
        for pi, widgets in self._row_widgets.items():
            widgets["count"].configure(text=f"使用 {counts.get(pi, 0)}")
        self._background_var.set(f"背景色: {self._describe_color(self._gl_frame.background_color())}")
        self._undo_button.configure(state="normal" if self._gl_frame.can_undo else "disabled")
        self._redo_button.configure(state="normal" if self._gl_frame.can_redo else "disabled")
        self._sync_layer_controls()

    def _describe_color(self, palette_index: int) -> str:
        if palette_index < 0 or self._palette is None:
            return "-"
        color = self._palette.colors[palette_index]
        return f"{color.color_id} {color.name}"

    def _on_undo(self) -> None:
        self._gl_frame.undo()

    def _on_redo(self) -> None:
        self._gl_frame.redo()

    def _on_dome_scale_change(self, _value: str) -> None:
        self._dome_scale = float(self._dome_scale_var.get())
        self._dome_scale_label.configure(text=f"{self._dome_scale:.1f}")

    def _on_apply_dome(self) -> None:
        self._dome_profile = _DOME_LABEL_TO_PROFILE.get(
            self._dome_profile_var.get(), "hemisphere"
        )
        try:
            plan = self._gl_frame.plan_dome(self._dome_profile, self._dome_scale)
        except RuntimeError as exc:
            # cv2 未導入。cv2_utils のメッセージをそのまま案内する
            messagebox.showerror("自動ドーム", str(exc), parent=self)
            return
        if plan is None or plan.islands == 0:
            messagebox.showinfo(
                "自動ドーム",
                f"背景色（{self._describe_color(self._gl_frame.background_color())}）"
                "以外の領域がありません。",
                parent=self,
            )
            return
        if not messagebox.askyesno(
            "自動ドーム",
            f"{plan.islands} 個の領域をドーム状に盛り上げます。\n"
            f"最高 {plan.top_layer + 1} 層 / ビーズ {plan.total_beads} 個になります。\n"
            "対象の列は指定の高さちょうどに整えられます。よろしいですか？",
            parent=self,
        ):
            return
        self._gl_frame.apply_dome(plan)
        self._summary_var.set(
            f"ドーム: {plan.islands} 領域 / 最高 {plan.top_layer + 1} 層 / {plan.total_beads} 個"
        )

    def _on_raise_non_background(self) -> None:
        raised, background = self._gl_frame.raise_non_background()
        if background < 0:
            messagebox.showinfo(
                "背景色以外を1層上げる", "1層目にビーズがありません。", parent=self
            )
            return
        if raised == 0:
            messagebox.showinfo(
                "背景色以外を1層上げる",
                f"背景色（{self._describe_color(background)}）以外の列がありません。",
                parent=self,
            )
            return
        self._summary_var.set(
            f"{raised} 列を1層上げました（背景色: {self._describe_color(background)}）"
        )

    def _sync_layer_controls(self) -> None:
        """作業層スピンボックスと表示上限スライダーを盤面の状態に合わせる。"""
        active = self._gl_frame.active_layer
        highest = max(self._gl_frame.top_layer(), active)
        self._syncing_layer = True
        try:
            if int(self._layer_var.get() or 0) != active + 1:
                self._layer_var.set(active + 1)
            self._visible_top_scale.configure(to=max(1, highest + 1))
            self._visible_top_scale.set(self._gl_frame.visible_top + 1)
        except Exception:
            pass
        finally:
            self._syncing_layer = False

    def _set_active_layer(self, layer: int) -> None:
        layer = max(0, min(MAX_LAYERS - 1, layer))
        self._gl_frame.set_active_layer(layer)
        if self._follow_top_var.get():
            self._gl_frame.set_visible_top(layer)
        self._sync_layer_controls()

    def _on_layer_spin(self) -> None:
        if self._syncing_layer:
            return
        try:
            value = int(self._layer_var.get())
        except (tk.TclError, ValueError):
            self._sync_layer_controls()
            return
        self._set_active_layer(value - 1)

    def _on_layer_changed_from_key(self, layer: int) -> None:
        """GL側のPgUp/PgDnで作業層が変わったときの同期。"""
        if self._follow_top_var.get():
            self._gl_frame.set_visible_top(layer)
        self._sync_layer_controls()

    def _on_visible_top_change(self, value: str) -> None:
        if self._syncing_layer:
            return
        try:
            top = int(round(float(value))) - 1
        except (TypeError, ValueError):
            return
        # 手動で動かしたら追従は解除する（作業層より下は隠さない）
        self._follow_top_var.set(False)
        self._gl_frame.set_visible_top(top)

    def _on_follow_top_check(self) -> None:
        if self._follow_top_var.get():
            self._gl_frame.set_visible_top(self._gl_frame.active_layer)
            self._sync_layer_controls()

    def _on_translucent_check(self) -> None:
        self._translucent_below = bool(self._translucent_var.get())
        self._gl_frame.set_translucent_below(self._translucent_below)

    def _on_brush_spin(self) -> None:
        try:
            size = int(self._brush_var.get())
        except (tk.TclError, ValueError):
            self._brush_var.set(self._gl_frame.brush_size)
            return
        self._gl_frame.set_brush_size(size)
        self._brush_size = self._gl_frame.brush_size
        if self._brush_var.get() != self._brush_size:
            self._brush_var.set(self._brush_size)

    def _on_brush_changed_from_key(self, size: int) -> None:
        """GL側の数字キーで変わったブラシサイズをスピンボックスへ反映する。"""
        self._brush_size = size
        self._brush_var.set(size)

    def _on_inherit_check(self) -> None:
        self._inherit_color = bool(self._inherit_var.get())
        self._gl_frame.set_inherit_color(self._inherit_color)

    def _on_restrict_check(self) -> None:
        self._restrict_to_image = bool(self._restrict_var.get())
        self._gl_frame.set_restrict_to_image(self._restrict_to_image)

    def _on_alpha_changed_from_key(self, alpha: float) -> None:
        """GL側のA/Dキーで変わった濃さをスライダーへ反映する。"""
        self._ghost_alpha = alpha
        self._ghost_alpha_var.set(alpha)

    def _on_show_all_check(self) -> None:
        self._show_all_colors = bool(self._show_all_var.get())
        self._build_color_rows()
        self._on_board_changed()

    def _on_fill_layer0(self) -> None:
        if self._gl_frame.target_grid is None:
            return
        if not messagebox.askyesno(
            "1層目の敷き詰め",
            "1層目を変換結果で置き換えますか？\n（1層目に置いたビーズは上書きされます）",
            parent=self,
        ):
            return
        self._gl_frame.fill_layer0()
        self._on_board_changed()

    def _on_reset(self) -> None:
        if not messagebox.askyesno(
            "リセット", "全ての層のビーズを取り除きますか？", parent=self
        ):
            return
        self._gl_frame.reset_board()

    def _on_ghost_check(self) -> None:
        self._gl_frame.set_ghost_visible(bool(self._ghost_var.get()))

    def _on_ghost_alpha_change(self, _value: str) -> None:
        self._ghost_alpha = float(self._ghost_alpha_var.get())
        self._gl_frame.set_ghost_alpha(self._ghost_alpha)

    def _on_ghost_toggled_from_key(self, visible: bool) -> None:
        self._ghost_var.set(visible)

    def _restore_panel_width(self) -> None:
        """保存しておいた左パネル幅（サッシ位置）を復元する。"""
        if not self._panel_width:
            return
        try:
            self._paned.sashpos(0, int(self._panel_width))
        except Exception:
            pass

    # --- ウィンドウ状態の保存/復元（preview_3d.py と同パターン） ---

    def _on_close(self) -> None:
        self._save_window_state()
        if self._on_close_callback:
            self._on_close_callback()
        self.destroy()

    def _on_window_configure(self, event: tk.Event) -> None:
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
            panel_width = int(data.get("panel_width", 0))
            if panel_width > 0:
                self._panel_width = panel_width
            alpha = float(data.get("ghost_alpha", self._ghost_alpha))
            self._ghost_alpha = max(0.05, min(1.0, alpha))
            self._translucent_below = bool(
                data.get("translucent_below", self._translucent_below)
            )
            self._inherit_color = bool(data.get("inherit_color", self._inherit_color))
            self._restrict_to_image = bool(
                data.get("restrict_to_image", self._restrict_to_image)
            )
            brush = int(data.get("brush_size", self._brush_size))
            self._brush_size = max(1, min(MAX_BRUSH, brush))
            profile = str(data.get("dome_profile", self._dome_profile))
            if profile in _DOME_PROFILE_LABELS:
                self._dome_profile = profile
            dome_scale = float(data.get("dome_scale", self._dome_scale))
            self._dome_scale = max(DOME_SCALE_MIN, min(DOME_SCALE_MAX, dome_scale))
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
        width, height, x, y = geometry
        payload = {
            "width": int(width),
            "height": int(height),
            "x": int(x),
            "y": int(y),
            "state": current_state,
            "ghost_alpha": round(float(self._ghost_alpha), 3),
            "translucent_below": bool(self._translucent_below),
            "inherit_color": bool(self._inherit_color),
            "restrict_to_image": bool(self._restrict_to_image),
            "brush_size": int(self._brush_size),
            "dome_profile": str(self._dome_profile),
            "dome_scale": round(float(self._dome_scale), 2),
        }
        try:
            payload["panel_width"] = int(self._paned.sashpos(0))
        except Exception:
            pass
        try:
            self._window_state_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass


class BeadsSimulatorGL(OpenGLFrame):
    """ペグボードとビーズを描画し、配置操作を受け付けるOpenGLフレーム。"""

    def __init__(
        self,
        master: tk.Misc,
        on_board_changed: Optional[Callable[[], None]] = None,
        on_ghost_toggled: Optional[Callable[[bool], None]] = None,
        on_layer_changed: Optional[Callable[[int], None]] = None,
        on_alpha_changed: Optional[Callable[[float], None]] = None,
        on_brush_changed: Optional[Callable[[int], None]] = None,
    ) -> None:
        super().__init__(master)
        self._on_board_changed = on_board_changed
        self._on_ghost_toggled = on_ghost_toggled
        self._on_layer_changed = on_layer_changed
        self._on_alpha_changed = on_alpha_changed
        self._on_brush_changed = on_brush_changed

        self._target_grid: Optional[np.ndarray] = None  # 1層目のガイド (H,W)
        self._board: Optional[BeadBoard] = None
        self._history = BoardHistory()
        self._palette_rgb_f: Optional[np.ndarray] = None  # (N,3) float 0-1
        self._plates_x = 0
        self._plates_y = 0
        self._selected_index = -1
        self._inherit_color = True  # 真下（無ければ1層目）と同じ色で置く
        self._restrict_to_image = True  # 元画像の範囲内だけに置く
        self._ghost_visible = True
        self._ghost_alpha = GHOST_ALPHA
        self._hover_peg: Optional[tuple[int, int, int]] = None  # (layer,row,col)

        # 層の状態
        self._active_layer = 0
        self._visible_top = 0
        self._translucent_below = True

        # カメラ（yaw/pitchは度、targetはグリッド中心基準）
        self._yaw = 0.0
        self._pitch = 55.0
        self._distance = 40.0
        self._target_x = 0.0
        self._target_z = 0.0

        # ピッキング用の行列キャッシュ（直近の描画時点）
        self._mv = None
        self._proj = None
        self._viewport = None

        # ディスプレイリスト
        self._list_bead: Optional[int] = None
        self._list_slab: Optional[int] = None
        self._list_pegs: Optional[int] = None
        self._list_tweezer: Optional[int] = None
        # 配置ビーズは層×プレート単位、ガイドは1層目のみプレート単位
        self._plate_bead_lists: dict[tuple[int, int, int], int] = {}
        self._plate_ghost_lists: dict[tuple[int, int], int] = {}
        self._dirty_plates: set[tuple[int, int, int]] = set()
        self._ghost_dirty = False
        self._all_plates_dirty = False

        # 入力状態
        self._drag_mode: Optional[str] = None  # None | "orbit" | "pan"
        self._drag_last: Optional[tuple[int, int]] = None
        # ドラッグ描画（ペイント）の状態
        self._brush_size = 1
        self._stroke_mode: Optional[str] = None  # None | "place" | "erase"
        self._stroke_last: Optional[tuple[int, int]] = None
        self._stroke_changed = False
        self._keys_held: set[str] = set()
        self._redraw_pending = False
        self._redraw_idle: Optional[str] = None
        self._redraw_timer: Optional[str] = None

        self.bind("<Enter>", lambda _event: self.focus_set())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_left_press)
        self.bind("<B1-Motion>", self._on_drag_motion)
        self.bind("<ButtonRelease-1>", self._on_drag_release)
        self.bind("<ButtonPress-2>", self._on_middle_press)
        self.bind("<B2-Motion>", self._on_drag_motion)
        self.bind("<ButtonRelease-2>", self._on_drag_release)
        self.bind("<ButtonPress-3>", self._on_right_press)
        self.bind("<B3-Motion>", self._on_drag_motion)
        self.bind("<ButtonRelease-3>", self._on_drag_release)
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", self._on_mouse_wheel_linux)
        self.bind("<Button-5>", self._on_mouse_wheel_linux)
        self.bind("<KeyPress-g>", self._on_ghost_key)
        self.bind("<KeyPress-G>", self._on_ghost_key)
        self.bind("<KeyRelease-g>", self._on_toggle_key_release)
        self.bind("<KeyRelease-G>", self._on_toggle_key_release)
        # 右手はマウスに置いたまま左手で操作できるようWASDに割り当てる
        # （長押しリピートはそのまま連続操作として活かす）
        for key in ("w", "W"):
            self.bind(f"<KeyPress-{key}>", lambda _e: self._step_layer(1))
        for key in ("s", "S"):
            self.bind(f"<KeyPress-{key}>", lambda _e: self._step_layer(-1))
        for key in ("a", "A"):
            self.bind(f"<KeyPress-{key}>", lambda _e: self._step_alpha(-GHOST_ALPHA_STEP))
        for key in ("d", "D"):
            self.bind(f"<KeyPress-{key}>", lambda _e: self._step_alpha(GHOST_ALPHA_STEP))
        for size in range(1, MAX_BRUSH + 1):
            self.bind(f"<KeyPress-{size}>", lambda _e, n=size: self._set_brush_from_key(n))
        for key in ("z", "Z"):
            self.bind(f"<Control-{key}>", lambda _e: self.undo())
        for key in ("x", "X"):
            self.bind(f"<Control-{key}>", lambda _e: self.redo())

    # --- 公開API ---

    @property
    def target_grid(self) -> Optional[np.ndarray]:
        return self._target_grid

    @property
    def board(self) -> Optional[BeadBoard]:
        return self._board

    @property
    def active_layer(self) -> int:
        return self._active_layer

    @property
    def visible_top(self) -> int:
        return max(self._visible_top, self._active_layer)

    def set_target_grid(self, target: np.ndarray, palette_rgb: np.ndarray) -> None:
        """新しいガイドを設定し、盤面とカメラを初期化する。"""
        self._target_grid = target
        self._board = BeadBoard.from_target(target)
        self._palette_rgb_f = palette_rgb.astype(np.float32) / 255.0
        self._plates_y = target.shape[0] // PLATE_PEGS
        self._plates_x = target.shape[1] // PLATE_PEGS
        self._hover_peg = None
        self._active_layer = 0
        self._visible_top = 0
        self._yaw = 0.0
        self._pitch = 55.0
        self._target_x = 0.0
        self._target_z = 0.0
        self._distance = max(12.0, max(target.shape) * 1.3)
        self._history.reset(self._board.grid)
        self._all_plates_dirty = True
        self._request_redraw()

    def fill_layer0(self) -> None:
        """1層目をガイド（変換結果）で敷き詰める。"""
        if self._board is None or self._target_grid is None:
            return
        self._board.fill_layer(0, self._target_grid)
        self._commit_change()

    def reset_board(self) -> None:
        """置いたビーズを全て取り除く。"""
        if self._board is None:
            return
        self._board.clear()
        self._active_layer = 0
        self._visible_top = 0
        self._commit_change()

    def raise_non_background(self) -> tuple[int, int]:
        """背景色（1層目の最多使用色）以外の列を1段積み増す。

        戻り値は (積んだ列数, 背景色のパレットindex)。
        """
        if self._board is None:
            return (0, -1)
        background = self._board.dominant_color(0)
        if background < 0:
            return (0, -1)
        raised = self._board.raise_non_background(background)
        if raised:
            # 増やした層で続けて作業できるよう、作業層も1つ上げて連動させる
            self._step_layer(1)
            self._commit_change()
        return (raised, background)

    def _figure_mask(self) -> Optional[np.ndarray]:
        """背景色を除いた1層目の領域（ドームの対象）。"""
        if self._board is None:
            return None
        background = self._board.dominant_color(0)
        if background < 0:
            return None
        base = self._board.grid[0]
        return (base >= 0) & (base != background)

    def plan_dome(self, profile: str, scale: float) -> Optional[DomePlan]:
        """ドームの高さを計算する。盤面は変更しない（確認ダイアログ用）。"""
        mask = self._figure_mask()
        if mask is None or not mask.any():
            return None
        return dome_heights(mask, profile=profile, scale=scale)

    def apply_dome(self, plan: DomePlan) -> int:
        """計算済みのドームを盤面へ反映する。戻り値は変化したセル数。"""
        if self._board is None or plan.islands == 0:
            return 0
        changed = self._board.apply_column_heights(plan.heights)
        if not changed:
            return 0
        # 出来上がったドームの頂上へ作業層を移す（表示上限とスピンボックスも追従する）
        target = max(0, min(MAX_LAYERS - 1, plan.top_layer))
        if target != self._active_layer:
            self.set_active_layer(target)
            if self._on_layer_changed:
                self._on_layer_changed(self._active_layer)
        self._commit_change()
        return changed

    def background_color(self) -> int:
        """背景色として扱う色（1層目の最多使用色）。"""
        return self._board.dominant_color(0) if self._board is not None else -1

    # --- アンドゥ/リドゥ ---

    def reset_history(self) -> None:
        """現在の盤面を履歴の起点にする。"""
        if self._board is not None:
            self._history.reset(self._board.grid)

    @property
    def can_undo(self) -> bool:
        return self._history.can_undo

    @property
    def can_redo(self) -> bool:
        return self._history.can_redo

    def _commit_change(self, record: bool = True) -> None:
        """盤面を書き換えたあとの後始末（履歴・再描画・通知）。"""
        if record and self._board is not None:
            self._history.push(self._board.grid)
        self._all_plates_dirty = True
        self._request_redraw()
        if self._on_board_changed:
            self._on_board_changed()

    def _restore_grid(self, grid: Optional[np.ndarray]) -> bool:
        if grid is None or self._board is None:
            return False
        self._board.grid = grid
        # 作業層は動かさない。アンドゥのたびに視点が別の層へ飛ぶと
        # 「戻して置き直す」ができないため、層を復元側に合わせる
        self._board.ensure_layer(self._active_layer)
        self._hover_peg = None
        self._commit_change(record=False)
        return True

    def undo(self) -> str:
        self._restore_grid(self._history.undo())
        return "break"

    def redo(self) -> str:
        self._restore_grid(self._history.redo())
        return "break"

    def color_counts(self) -> dict[int, int]:
        return self._board.counts() if self._board is not None else {}

    def top_layer(self) -> int:
        return self._board.top_layer() if self._board is not None else -1

    def has_beads_above_layer0(self) -> bool:
        return self._board is not None and self._board.has_beads_above(0)

    def set_active_layer(self, layer: int) -> None:
        layer = max(0, min(MAX_LAYERS - 1, int(layer)))
        if layer == self._active_layer:
            return
        self._active_layer = layer
        if self._board is not None:
            self._board.ensure_layer(layer)
        self._hover_peg = None
        self._request_redraw()

    def set_visible_top(self, layer: int) -> None:
        layer = max(0, min(MAX_LAYERS - 1, int(layer)))
        if layer == self._visible_top:
            return
        self._visible_top = layer
        self._request_redraw()

    def set_translucent_below(self, enabled: bool) -> None:
        if self._translucent_below == enabled:
            return
        self._translucent_below = enabled
        self._request_redraw()

    def _step_layer(self, delta: int) -> str:
        """W/Sキーで作業層を上下する。"""
        target = max(0, min(MAX_LAYERS - 1, self._active_layer + delta))
        if target != self._active_layer:
            self.set_active_layer(target)
            if self._on_layer_changed:
                self._on_layer_changed(self._active_layer)
        return "break"

    def _step_alpha(self, delta: float) -> str:
        """A/Dキーで下層の半透明の濃さを調整する。"""
        before = self._ghost_alpha
        self.set_ghost_alpha(self._ghost_alpha + delta)
        if self._ghost_alpha != before and self._on_alpha_changed:
            self._on_alpha_changed(self._ghost_alpha)
        return "break"

    def set_selected_color(self, palette_index: int) -> None:
        self._selected_index = palette_index
        self._request_redraw()

    def set_inherit_color(self, enabled: bool) -> None:
        """真下の色を引き継いで置くかどうかを切り替える。"""
        if self._inherit_color == enabled:
            return
        self._inherit_color = enabled
        self._request_redraw()

    def set_restrict_to_image(self, enabled: bool) -> None:
        """配置エリアを元画像の範囲内に制限するかを切り替える。"""
        if self._restrict_to_image == enabled:
            return
        self._restrict_to_image = enabled
        self._request_redraw()

    def _can_place(self, row: int, col: int) -> bool:
        """その列にビーズを置いてよいか。

        制限が有効なときは、元画像の範囲内（1層目のガイドがある位置）か、
        既に1層目にビーズを置いた位置だけを許可する。削除は常に許可する
        （制限を解除して置いたビーズを片付けられるようにするため）。
        """
        if not self._restrict_to_image:
            return True
        if self._target_grid is not None and self._target_grid[row, col] >= 0:
            return True
        return self._board is not None and self._board.get(0, row, col) >= 0

    def _color_for(self, layer: int, row: int, col: int) -> int:
        """そのセルに置く色を決める。

        立体作品では1層目と同じ色を真上に積むのが基本なので、
        引き継ぎ有効時は 真下のビーズ → 1層目のガイド の順に色を拾い、
        どちらも無い場合だけ手動選択の色を使う。
        """
        if self._inherit_color:
            if self._board is not None and layer > 0:
                below = self._board.get(layer - 1, row, col)
                if below >= 0:
                    return below
            if self._target_grid is not None:
                base = int(self._target_grid[row, col])
                if base >= 0:
                    return base
        return self._selected_index

    def set_ghost_visible(self, visible: bool) -> None:
        if self._ghost_visible == visible:
            return
        self._ghost_visible = visible
        self._request_redraw()

    def set_ghost_alpha(self, alpha: float) -> None:
        """ゴーストの不透明度（描画時の定数アルファ）を変更する。"""
        self._ghost_alpha = max(0.05, min(1.0, float(alpha)))
        self._request_redraw()

    # --- 再描画制御（preview_3d.py と同パターン） ---

    def tkMap(self, event: tk.Event) -> None:
        super().tkMap(event)
        self._request_redraw()

    def tkResize(self, event: tk.Event) -> None:
        super().tkResize(event)
        self._request_redraw()

    def _request_redraw(self) -> None:
        """次の描画を予約する（複数回呼ばれても1回にまとめる）。

        after_idle はイベントキューが空になるまで走らない。修飾キーを
        押しっぱなしにするとキーのオートリピートでキューが埋まり続け、
        描画が保留されて視点操作がガクついて見える。タイマーを併用して
        描画の遅延に上限を設ける。
        """
        if self._redraw_pending:
            return
        self._redraw_pending = True
        self._redraw_idle = self._schedule(self.after_idle, self._on_redraw_idle)
        self._redraw_timer = self._schedule(
            self.after, REDRAW_INTERVAL_MS, self._on_redraw_idle
        )

    @staticmethod
    def _schedule(scheduler, *args) -> Optional[str]:
        try:
            return scheduler(*args)
        except tk.TclError:
            return None

    def _cancel_redraw_jobs(self) -> None:
        for attr in ("_redraw_idle", "_redraw_timer"):
            job = getattr(self, attr, None)
            if job is None:
                continue
            setattr(self, attr, None)
            try:
                self.after_cancel(job)
            except Exception:
                pass

    def _on_redraw_idle(self) -> None:
        if not self._redraw_pending:
            return  # idle とタイマーのうち先に来た方だけを処理する
        self._redraw_pending = False
        self._cancel_redraw_jobs()
        if not self.winfo_exists() or not self.context_created:
            return
        if not self.winfo_ismapped():
            return
        self._display()

    # --- 盤面操作 ---

    def _grid_offsets(self) -> tuple[float, float]:
        """グリッド座標→ワールド座標の中心オフセット。"""
        grid_h, grid_w = self._target_grid.shape
        return (grid_w - 1) / 2.0, (grid_h - 1) / 2.0

    def _mark_peg_dirty(self, layer: int, row: int, col: int) -> None:
        self._dirty_plates.add((layer, row // PLATE_PEGS, col // PLATE_PEGS))
        if layer == 0:
            self._ghost_dirty = True  # 1層目のガイドは配置済みセルを除いて描く

    def set_brush_size(self, size: int) -> None:
        size = max(1, min(MAX_BRUSH, int(size)))
        if size == self._brush_size:
            return
        self._brush_size = size
        self._request_redraw()

    @property
    def brush_size(self) -> int:
        return self._brush_size

    def _set_brush_from_key(self, size: int) -> str:
        """数字キーでブラシの一辺を変える。"""
        if size != self._brush_size:
            self.set_brush_size(size)
            if self._on_brush_changed:
                self._on_brush_changed(self._brush_size)
        return "break"

    def _brush_cells(self, row: int, col: int) -> list[tuple[int, int]]:
        """ブラシ範囲のセル。奇数サイズはカーソル中心、偶数は右下寄りになる。"""
        if self._target_grid is None:
            return []
        n = self._brush_size
        r0 = row - (n - 1) // 2
        c0 = col - (n - 1) // 2
        grid_h, grid_w = self._target_grid.shape
        cells = []
        for r in range(r0, r0 + n):
            if not 0 <= r < grid_h:
                continue
            for c in range(c0, c0 + n):
                if 0 <= c < grid_w:
                    cells.append((r, c))
        return cells

    def _apply_brush(self, layer: int, row: int, col: int, erase: bool) -> bool:
        """ブラシ範囲へまとめて配置/削除する。変化があれば True。"""
        if self._board is None:
            return False
        changed = False
        for r, c in self._brush_cells(row, col):
            if erase:
                ok = self._board.remove(layer, r, c)
            elif self._can_place(r, c):
                index = self._color_for(layer, r, c)
                ok = index >= 0 and self._board.place(layer, r, c, index)
            else:
                ok = False
            if ok:
                self._mark_peg_dirty(layer, r, c)
                changed = True
        return changed

    # --- ドラッグ描画（ペイント） ---

    def _begin_stroke(self, mode: str, cell: tuple[int, int, int]) -> None:
        self._stroke_mode = mode
        self._stroke_last = None
        self._stroke_changed = False
        self._stroke_to(cell)

    def _stroke_to(self, cell: tuple[int, int, int]) -> None:
        """ストロークを指定セルまで伸ばす。

        マウスを速く動かすとMotionイベントが飛び飛びになるため、
        前回のセルとの間をグリッド上で線形補間して途切れを防ぐ。
        """
        if self._stroke_mode is None:
            return
        layer, row, col = cell
        erase = self._stroke_mode == "erase"
        changed = False
        last = self._stroke_last
        if last is not None and last != (row, col):
            r0, c0 = last
            steps = max(abs(row - r0), abs(col - c0))
            for i in range(1, steps):
                rr = r0 + int(round((row - r0) * i / steps))
                cc = c0 + int(round((col - c0) * i / steps))
                changed |= self._apply_brush(layer, rr, cc, erase)
        changed |= self._apply_brush(layer, row, col, erase)
        self._stroke_last = (row, col)
        if changed:
            self._stroke_changed = True
            self._request_redraw()

    def _end_stroke(self) -> None:
        """ストローク終了。集計の更新はここで1回だけ行う。

        1粒ごとに通知すると色一覧のラベル更新がドラッグ中の負荷になる。
        """
        was_changed = self._stroke_changed
        self._stroke_mode = None
        self._stroke_last = None
        self._stroke_changed = False
        if not was_changed:
            return
        # 1ストロークをアンドゥの1単位にする
        if self._board is not None:
            self._history.push(self._board.grid)
        if self._on_board_changed:
            self._on_board_changed()

    # --- ピッキング ---

    def _unproject_to_plane(
        self, event_x: int, event_y: int, plane_y: float = 0.0
    ) -> Optional[tuple[float, float]]:
        """マウス座標から水平面 y=plane_y との交点 (x, z) を求める。"""
        if self._mv is None:
            return None
        try:
            win_y = float(self._viewport[3]) - event_y - 1.0
            p0 = gluUnProject(event_x, win_y, 0.0, self._mv, self._proj, self._viewport)
            p1 = gluUnProject(event_x, win_y, 1.0, self._mv, self._proj, self._viewport)
        except Exception:
            return None
        dy = p1[1] - p0[1]
        if abs(dy) < 1e-9:
            return None
        t = (plane_y - p0[1]) / dy
        # t<0 はカメラの後方。t>1（遠クリップより先）は地面の点としては
        # 有効なので弾かない（弾くと遠景で平行移動が止まる）
        if t < 0.0:
            return None
        return (p0[0] + t * (p1[0] - p0[0]), p0[2] + t * (p1[2] - p0[2]))

    def _pick_peg(self, event_x: int, event_y: int) -> Optional[tuple[int, int, int]]:
        """マウス座標から作業層の平面上の最寄りセル (layer, row, col) を求める。"""
        if self._target_grid is None:
            return None
        hit = self._unproject_to_plane(event_x, event_y, layer_y(self._active_layer))
        if hit is None:
            return None
        hx, hz = hit
        off_x, off_z = self._grid_offsets()
        col = int(round(hx + off_x))
        row = int(round(hz + off_z))
        grid_h, grid_w = self._target_grid.shape
        if 0 <= row < grid_h and 0 <= col < grid_w:
            return (self._active_layer, row, col)
        return None

    # --- 入力ハンドラ ---

    def _on_motion(self, event: tk.Event) -> None:
        if self._drag_mode is not None:
            return
        peg = self._pick_peg(event.x, event.y)
        if peg != self._hover_peg:
            self._hover_peg = peg
            self._request_redraw()

    def _on_leave(self, _event: tk.Event) -> None:
        if self._hover_peg is not None:
            self._hover_peg = None
            self._request_redraw()

    def _on_left_press(self, event: tk.Event) -> None:
        state = int(getattr(event, "state", 0))
        if state & 0x0001:  # Shift → 回転
            self._drag_mode = "orbit"
            self._drag_last = (event.x, event.y)
            return
        if state & 0x0004:  # Ctrl → 平行移動
            self._drag_mode = "pan"
            self._drag_last = (event.x, event.y)
            return
        peg = self._pick_peg(event.x, event.y)
        if peg is not None:
            self._hover_peg = peg
            self._begin_stroke("place", peg)

    def _on_middle_press(self, event: tk.Event) -> None:
        state = int(getattr(event, "state", 0))
        self._drag_mode = "pan" if state & 0x0001 else "orbit"
        self._drag_last = (event.x, event.y)

    def _on_right_press(self, event: tk.Event) -> None:
        peg = self._pick_peg(event.x, event.y)
        if peg is not None:
            self._hover_peg = peg
            self._begin_stroke("erase", peg)

    def _on_drag_motion(self, event: tk.Event) -> None:
        if self._stroke_mode is not None:
            # ドラッグ描画中はカーソルを追いながら塗り続ける。
            # 浮遊ビーズの追従は盤面の変化とは切り離して再描画する
            # （既に置いてある場所をなぞると盤面が変わらないため）
            peg = self._pick_peg(event.x, event.y)
            if peg != self._hover_peg:
                self._hover_peg = peg
                self._request_redraw()
            if peg is not None:
                self._stroke_to(peg)
            return
        if self._drag_mode is None or self._drag_last is None:
            return
        x0, y0 = self._drag_last
        dx = event.x - x0
        dy = event.y - y0
        self._drag_last = (event.x, event.y)
        if self._drag_mode == "orbit":
            # ウィンドウサイズ比例の感度（横幅いっぱいのドラッグで180°、
            # 上下は軽めの操作感になるよう縦幅いっぱいで180°）
            width = max(self.winfo_width(), 1)
            height = max(self.winfo_height(), 1)
            self._yaw = (self._yaw - dx * (180.0 / width)) % 360.0
            self._pitch = max(15.0, min(85.0, self._pitch + dy * (180.0 / height)))
        elif self._drag_mode == "pan":
            self._apply_pan(x0, y0, event.x, event.y)
        self._request_redraw()

    def _on_drag_release(self, _event: tk.Event) -> None:
        self._drag_mode = None
        self._drag_last = None
        if self._stroke_mode is not None:
            self._end_stroke()

    def _ground_basis(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """カメラの右方向と、地面(XZ)へ投影した前方向。"""
        yaw = math.radians(self._yaw)
        right = (math.cos(yaw), -math.sin(yaw))
        # 画面の上方向に対応する地面のベクトル（カメラから見て奥へ向かう向き）
        forward = (math.sin(yaw), math.cos(yaw))
        return right, forward

    def _screen_pan_delta(self, dx_px: float, dy_px: float) -> tuple[float, float]:
        """画面上の移動量に対応する注視点の移動量（レイが使えないときの代替）。

        1:1追従ほど正確ではないが、視点を寝かせても破綻せず滑らかに動く。
        """
        height = max(self.winfo_height(), 1)
        # 注視点の距離における画面1pxあたりの世界座標
        world_per_px = 2.0 * self._distance * math.tan(math.radians(22.5)) / height
        right, forward = self._ground_basis()
        # 縦方向は伏せ角ぶん引き伸ばされる。浅い角度で発散しないよう下限を設ける
        tilt = max(math.sin(math.radians(self._pitch)), 0.35)
        move_r = -dx_px * world_per_px
        move_f = -dy_px * world_per_px / tilt
        return (
            right[0] * move_r + forward[0] * move_f,
            right[1] * move_r + forward[1] * move_f,
        )

    def _apply_pan(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """カーソル直下の作業層上の点が付いてくるよう注視点を動かす（1:1追従）。

        視点を寝かせるとレイが作業層の面に当たらなかったり、擦れ違いざまの
        浅い角度で交点が極端に遠くなったりする。そのままだと平行移動が
        止まる／飛ぶを繰り返してガクつくため、使えない場合は画面基準の
        移動へ切り替えて連続性を保つ。
        """
        plane = layer_y(self._active_layer)
        p_prev = self._unproject_to_plane(x0, y0, plane)
        p_cur = self._unproject_to_plane(x1, y1, plane)
        fallback = self._screen_pan_delta(x1 - x0, y1 - y0)
        move = fallback
        if p_prev is not None and p_cur is not None:
            exact = (p_prev[0] - p_cur[0], p_prev[1] - p_cur[1])
            # 画面基準の見積りから極端に外れていなければ1:1追従を使う。
            # 地平線近くの擦れ違いざまでは交点が遠くへ飛ぶため、そこだけ代替に切り替える
            expected = math.hypot(*fallback)
            if math.hypot(*exact) <= max(expected * 3.0, 1e-6):
                move = exact
        self._target_x += move[0]
        self._target_z += move[1]
        if self._target_grid is not None:
            limit = max(self._target_grid.shape) / 2.0 + 10.0
            self._target_x = max(-limit, min(limit, self._target_x))
            self._target_z = max(-limit, min(limit, self._target_z))

    def _apply_zoom(self, steps: float) -> None:
        self._distance = max(4.0, min(300.0, self._distance * (0.92 ** steps)))
        self._request_redraw()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if getattr(event, "delta", 0) == 0:
            return
        self._apply_zoom(event.delta / 120.0)

    def _on_mouse_wheel_linux(self, event: tk.Event) -> None:
        if event.num == 4:
            self._apply_zoom(1.0)
        elif event.num == 5:
            self._apply_zoom(-1.0)

    def _on_ghost_key(self, event: tk.Event) -> str:
        # Gキーでゴースト表示を切り替える（長押しリピート無効）
        key = event.keysym.lower()
        if key in self._keys_held:
            return "break"
        self._keys_held.add(key)
        self._ghost_visible = not self._ghost_visible
        if self._on_ghost_toggled:
            self._on_ghost_toggled(self._ghost_visible)
        self._request_redraw()
        return "break"

    def _on_toggle_key_release(self, event: tk.Event) -> None:
        self._keys_held.discard(event.keysym.lower())

    # --- GLライフサイクル ---

    def initgl(self) -> None:
        glClearColor(0.16, 0.16, 0.18, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.35, 0.35, 0.35, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.75, 0.75, 0.75, 1.0))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
        # コンテキストが作り直された場合に備えてリスト参照を破棄する
        self._list_bead = None
        self._list_slab = None
        self._list_pegs = None
        self._list_tweezer = None
        self._plate_bead_lists = {}
        self._plate_ghost_lists = {}
        self._all_plates_dirty = True

    def _compile_mesh_list(self, verts: np.ndarray, normals: np.ndarray) -> int:
        """頂点/法線配列をディスプレイリストへ焼き込む。

        頂点ポインタ設定は glNewList の外で行う必要がある
        （glDrawArrays がコンパイル時に配列を展開して取り込む）。
        """
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, verts)
        glNormalPointer(GL_FLOAT, 0, normals)
        list_id = int(glGenLists(1))
        glNewList(list_id, GL_COMPILE)
        glDrawArrays(GL_TRIANGLES, 0, len(verts))
        glEndList()
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        return list_id

    def _ensure_geometry_lists(self) -> None:
        if self._list_bead is not None:
            return
        meshes = _get_meshes()
        bead_v, bead_n = meshes["bead"]
        slab_v, slab_n, pegs_v, pegs_n = meshes["plate"]
        tw_v, tw_n = meshes["tweezer"]
        self._list_bead = self._compile_mesh_list(bead_v, bead_n)
        self._list_slab = self._compile_mesh_list(slab_v, slab_n)
        self._list_pegs = self._compile_mesh_list(pegs_v, pegs_n)
        self._list_tweezer = self._compile_mesh_list(tw_v, tw_n)

    def _delete_plate_list(self, lists: dict, key: tuple[int, int]) -> None:
        old = lists.pop(key, None)
        if old is not None:
            try:
                glDeleteLists(old, 1)
            except Exception:
                pass

    def _rebuild_plate_lists(self, key: tuple[int, int, int]) -> None:
        """1層×1プレート分の配置ビーズのリストを作り直す。"""
        layer, pr, pc = key
        r0, c0 = pr * PLATE_PEGS, pc * PLATE_PEGS
        off_x, off_z = self._grid_offsets()
        base_y = layer_y(layer) + BEAD_LIFT_Y

        self._delete_plate_list(self._plate_bead_lists, key)
        if self._board is None or layer >= self._board.layer_count:
            return
        placed = self._board.grid[layer, r0 : r0 + PLATE_PEGS, c0 : c0 + PLATE_PEGS]
        rows, cols = np.nonzero(placed >= 0)
        if not rows.size:
            return
        list_id = int(glGenLists(1))
        glNewList(list_id, GL_COMPILE)
        for r, c in zip(rows.tolist(), cols.tolist()):
            color = self._palette_rgb_f[placed[r, c]]
            glPushMatrix()
            glTranslatef(c0 + c - off_x, base_y, r0 + r - off_z)
            glColor3f(float(color[0]), float(color[1]), float(color[2]))
            glCallList(self._list_bead)
            glPopMatrix()
        glEndList()
        self._plate_bead_lists[key] = list_id

    def _rebuild_ghost_lists(self) -> None:
        """1層目のガイド（まだ置かれていないセル）のリストを作り直す。"""
        off_x, off_z = self._grid_offsets()
        for plate in list(self._plate_ghost_lists):
            self._delete_plate_list(self._plate_ghost_lists, plate)
        if self._target_grid is None or self._board is None:
            return
        placed0 = self._board.grid[0]
        for pr in range(self._plates_y):
            for pc in range(self._plates_x):
                r0, c0 = pr * PLATE_PEGS, pc * PLATE_PEGS
                sl = (slice(r0, r0 + PLATE_PEGS), slice(c0, c0 + PLATE_PEGS))
                target = self._target_grid[sl]
                rows, cols = np.nonzero((target >= 0) & (placed0[sl] < 0))
                if not rows.size:
                    continue
                list_id = int(glGenLists(1))
                glNewList(list_id, GL_COMPILE)
                for r, c in zip(rows.tolist(), cols.tolist()):
                    color = self._palette_rgb_f[target[r, c]]
                    glPushMatrix()
                    glTranslatef(c0 + c - off_x, BEAD_LIFT_Y, r0 + r - off_z)
                    # アルファは描画時に glBlendColor の定数値で与える
                    glColor3f(float(color[0]), float(color[1]), float(color[2]))
                    glCallList(self._list_bead)
                    glPopMatrix()
                glEndList()
                self._plate_ghost_lists[(pr, pc)] = list_id

    def redraw(self) -> None:
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # 近クリップ面をカメラ距離に応じて遠ざけ、深度バッファの精度を確保する
        # （near/far比が大きいとZファイティングによるちらつきが出る）
        near = max(0.3, self._distance * 0.05)
        gluPerspective(45.0, width / height, near, 1000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # カメラ固定のヘッドライト（gluLookAt より前に設定する）
        glLightfv(GL_LIGHT0, GL_POSITION, (0.3, 0.6, 1.0, 0.0))
        pitch_rad = math.radians(self._pitch)
        yaw_rad = math.radians(self._yaw)
        # 高い層を作業中も盤面が画面外へ流れないよう注視点を層に追従させる
        target_y = layer_y(self._active_layer) * 0.5
        eye_x = self._target_x + self._distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        eye_y = target_y + self._distance * math.sin(pitch_rad)
        eye_z = self._target_z + self._distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        gluLookAt(
            eye_x, eye_y, eye_z, self._target_x, target_y, self._target_z, 0.0, 1.0, 0.0
        )

        # ピッキング用に描画時点の行列を保存する
        self._mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._proj = glGetDoublev(GL_PROJECTION_MATRIX)
        self._viewport = glGetIntegerv(GL_VIEWPORT)

        if self._target_grid is None:
            return
        self._ensure_geometry_lists()
        if self._board is None:
            return
        if self._all_plates_dirty:
            # アンドゥなどで層数が減ると、消えた層のリストがどこからも
            # dirty 指定されずに残り、層を上げ直したときに幽霊ビーズとして
            # 描かれてしまう。全再構築時は既存のリストを必ず捨てる。
            for key in list(self._plate_bead_lists):
                self._delete_plate_list(self._plate_bead_lists, key)
            self._dirty_plates = {
                (layer, pr, pc)
                for layer in range(self._board.layer_count)
                for pr in range(self._plates_y)
                for pc in range(self._plates_x)
            }
            self._all_plates_dirty = False
            self._ghost_dirty = True
        for key in self._dirty_plates:
            self._rebuild_plate_lists(key)
        self._dirty_plates = set()
        if self._ghost_dirty:
            self._rebuild_ghost_lists()
            self._ghost_dirty = False

        visible_top = self.visible_top
        active = self._active_layer
        # 半透明で描く層（下層／1層目ガイド）は不透明パスの後にまとめて描く
        opaque_keys = []
        blended_keys = []
        for key, list_id in self._plate_bead_lists.items():
            layer = key[0]
            if layer > visible_top:
                continue
            if self._translucent_below and layer < active:
                blended_keys.append(list_id)
            else:
                opaque_keys.append(list_id)

        # 不透明パス: プレート → 配置ビーズ → ピンセット
        off_x, off_z = self._grid_offsets()
        for pr in range(self._plates_y):
            for pc in range(self._plates_x):
                glPushMatrix()
                glTranslatef(pc * PLATE_PEGS - off_x, 0.0, pr * PLATE_PEGS - off_z)
                glColor3f(*_SLAB_COLOR)
                glCallList(self._list_slab)
                glColor3f(*_PEG_COLOR)
                glCallList(self._list_pegs)
                glPopMatrix()
        for list_id in opaque_keys:
            glCallList(list_id)
        self._draw_tweezer(off_x, off_z)

        # 半透明パス: 不透明度は定数アルファで与え、リスト再構築なしで変更できる。
        # 深度書き込みは有効のまま描く: 最前面の面だけが残るため
        # 重なり順による濃度の揺れ（視点移動時のちらつき）が出ない。
        draw_ghost = self._ghost_visible and visible_top >= 0 and self._plate_ghost_lists
        if blended_keys or draw_ghost:
            glEnable(GL_BLEND)
            glBlendColor(0.0, 0.0, 0.0, self._ghost_alpha)
            glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA)
            for list_id in blended_keys:
                glCallList(list_id)
            if draw_ghost:
                for list_id in self._plate_ghost_lists.values():
                    glCallList(list_id)
            glDisable(GL_BLEND)

    def _draw_tweezer(self, off_x: float, off_z: float) -> None:
        """ホバー中のセルにピンセットと持ちビーズを描く。

        ブラシが2以上のときは塗る範囲ぶんの持ちビーズを並べて、
        どこにどの色が置かれるかを置く前に確認できるようにする。
        置けない位置（範囲制限の外）は持ちビーズを出さないことで、
        制限が効いていることがそのまま見て分かるようにする。
        """
        if self._hover_peg is None:
            return
        layer, row, col = self._hover_peg
        hold_y = layer_y(layer) + BEAD_HOLD_Y
        if self._palette_rgb_f is not None:
            for r, c in self._brush_cells(row, col):
                if not self._can_place(r, c):
                    continue
                held = self._color_for(layer, r, c)
                if held < 0:
                    continue
                color = self._palette_rgb_f[held]
                glPushMatrix()
                glTranslatef(c - off_x, hold_y, r - off_z)
                glColor3f(float(color[0]), float(color[1]), float(color[2]))
                glCallList(self._list_bead)
                glPopMatrix()
        glPushMatrix()
        glTranslatef(col - off_x, hold_y + BEAD_HEIGHT * 0.9, row - off_z)
        glColor3f(*_TWEEZER_COLOR)
        glCallList(self._list_tweezer)
        glPopMatrix()
