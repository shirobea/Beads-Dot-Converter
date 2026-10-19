"""ドット絵の復元ウィンドウ。

拡大・再圧縮・ぼかしで崩れたドット絵の画像から、元のドット解像度と色を取り戻す
（converter/dot_restore.py）。結果はファイルに保存するか、メイン画面の入力画像として送れる。
"""

from __future__ import annotations

import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from converter import dot_restore as dr

# (キー, 表示名, 最小, 最大, 刻み, 表示倍率)。表示倍率 100 は % で入力する項目
ANALYSIS_FIELDS = [
    ("fg_coef", "前景判定（×σ_bg）", 0.5, 20.0, 0.5, 1),
    ("half_pitch_tol", "半分ピッチ許容", 0.0, 1.0, 0.05, 1),
    ("inner_margin", "マス内側の除外幅（×ピッチ）", 0.0, 0.45, 0.05, 1),
    ("sample_ratio", "サンプリング範囲（×ピッチ）", 0.1, 1.0, 0.05, 1),
]
FINALIZE_FIELDS = [
    ("minor_ratio", "にじみ判定の少数色（%）", 0.0, 20.0, 0.5, 100),
    ("hole_ratio", "内部の穴の最大面積（%）", 0.0, 20.0, 0.5, 100),
    ("mixed_coef", "混ざったマス判定（×σ_cell）", 1.0, 20.0, 0.5, 1),
    ("verify_coef", "怪しいマス判定（×σ_cell）", 1.0, 20.0, 0.5, 1),
]
BACKGROUND_CHOICES = {"自動": "auto", "あり": "on", "なし": "off"}
HOLE_CHOICES = {"色を残す": "keep", "周りの色で埋める": "fill"}
VIEW_CHOICES = ("元画像", "復元結果", "比較")
MAX_MARKS = 4000  # 印の描画数の上限（多すぎると Canvas が重くなる）


def _load_rgb(path: Path) -> np.ndarray:
    """画像を RGB 配列で読む。透過は絵に使われていない色の背景に合成する。"""
    with Image.open(path) as img:
        img.load()
        has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
        if has_alpha:
            rgba = np.asarray(img.convert("RGBA"))
            if (rgba[..., 3] < 255).any():
                return dr.flatten_alpha(rgba)
            return np.ascontiguousarray(rgba[..., :3])
        return np.ascontiguousarray(np.asarray(img.convert("RGB")))


class DotRestoreWindow:
    """ドット絵の復元を行う別ウィンドウ。"""

    def __init__(
        self,
        parent: tk.Tk,
        schedule: Callable[..., None],
        get_main_input: Callable[[], Optional[Path]],
        on_send: Callable[[Path], None],
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        self._parent = parent
        self._schedule = schedule
        self._get_main_input = get_main_input
        self._on_send = on_send
        self._on_close = on_close
        self._source_path: Optional[Path] = None
        self._rgb: Optional[np.ndarray] = None
        self._analysis: Optional[dr.Analysis] = None
        self._result: Optional[dr.RestoreResult] = None
        self._compare_cache: Optional[np.ndarray] = None
        self._manual_merges: list[tuple] = []
        self._job = 0
        self._busy = False
        self._merge_after: Optional[str] = None
        self._render_after: Optional[str] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._swatches: list[ImageTk.PhotoImage] = []
        self._view_geom: Optional[tuple] = None  # (x0, y0, scale) 復元結果の表示位置
        self._highlight: set[int] = set()

        window = tk.Toplevel(parent)
        window.title("ドット絵の復元")
        window.geometry("1320x820")
        window.minsize(960, 600)
        window.protocol("WM_DELETE_WINDOW", self._handle_close)
        self._window = window

        self._coef_vars: dict[str, tk.StringVar] = {}
        defaults = dr.RestoreParams()
        for key, *_rest in ANALYSIS_FIELDS + FINALIZE_FIELDS:
            factor = _rest[-1]
            self._coef_vars[key] = tk.StringVar(value=self._fmt(getattr(defaults, key) * factor))
        self._pitch_var = tk.StringVar()
        self._origin_x_var = tk.StringVar()
        self._origin_y_var = tk.StringVar()
        self._bg_var = tk.StringVar(value="自動")
        self._hole_var = tk.StringVar(value="色を残す")
        self._merge_var = tk.DoubleVar(value=defaults.merge_coef)
        self._merge_label = tk.StringVar(value=f"{defaults.merge_coef:.1f}")
        self._view_var = tk.StringVar(value="元画像")
        self._mark_changed_var = tk.BooleanVar(value=False)
        self._mark_suspicious_var = tk.BooleanVar(value=False)
        self._status_var = tk.StringVar(value="画像を開いてください。")
        self._summary_var = tk.StringVar(value="")
        self._hover_var = tk.StringVar(value="")

        self._build()

    # ------------------------------------------------------------ 画面の組み立て
    def _build(self) -> None:
        w = self._window
        w.rowconfigure(1, weight=1)
        w.columnconfigure(0, weight=1)

        top = ttk.Frame(w, padding=(8, 8, 8, 4))
        top.grid(row=0, column=0, sticky="we")
        top.columnconfigure(4, weight=1)
        ttk.Button(top, text="画像を開く…", command=self._choose_file).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(top, text="メイン画面の入力画像を使う", command=self._use_main_input).grid(
            row=0, column=1, padx=(0, 6)
        )
        self._run_button = ttk.Button(top, text="復元する", command=self._start_restore)
        self._run_button.grid(row=0, column=2, padx=(0, 12))
        self._progress = ttk.Progressbar(top, length=160, maximum=1.0)
        self._progress.grid(row=0, column=3, padx=(0, 8))
        ttk.Label(top, textvariable=self._status_var, anchor="w").grid(row=0, column=4, sticky="we")

        body = ttk.Frame(w, padding=(8, 0, 8, 0))
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        self._build_settings(body)
        self._build_preview(body)
        self._build_palette(body)

        bottom = ttk.Frame(w, padding=(8, 4, 8, 8))
        bottom.grid(row=2, column=0, sticky="we")
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self._summary_var, anchor="w", justify="left", wraplength=900).grid(
            row=0, column=0, sticky="we"
        )
        # 必要なものだけ個別に保存する（全部まとめて書き出すと使わないファイルが増えるため）
        self._save_button = ttk.Menubutton(bottom, text="保存 ▾", state="disabled")
        save_menu = tk.Menu(self._save_button, tearoff=False)
        for kind, (label, _suffix, _ext) in dr.EXPORTS.items():
            save_menu.add_command(label=label + "…", command=lambda k=kind: self._save(k))
        self._save_button["menu"] = save_menu
        self._save_button.grid(row=0, column=2, padx=(8, 0))
        self._send_button = ttk.Button(
            bottom, text="メイン画面の入力に送る", command=self._send_to_main, state="disabled"
        )
        self._send_button.grid(row=0, column=3, padx=(8, 0))

    def _build_settings(self, parent: ttk.Frame) -> None:
        side = ttk.Frame(parent)
        side.grid(row=0, column=0, sticky="ns", padx=(0, 8))

        override = ttk.LabelFrame(side, text="自動判定の上書き（空欄＝自動）", padding=6)
        override.grid(row=0, column=0, sticky="we")
        ttk.Label(override, text="ピッチ（px）").grid(row=0, column=0, sticky="w")
        ttk.Entry(override, textvariable=self._pitch_var, width=8).grid(row=0, column=1, sticky="w", pady=1)
        ttk.Label(override, text="原点 X / Y（px）").grid(row=1, column=0, sticky="w")
        origin = ttk.Frame(override)
        origin.grid(row=1, column=1, sticky="w", pady=1)
        ttk.Entry(origin, textvariable=self._origin_x_var, width=6).grid(row=0, column=0)
        ttk.Entry(origin, textvariable=self._origin_y_var, width=6).grid(row=0, column=1, padx=(4, 0))
        ttk.Label(override, text="背景").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            override, textvariable=self._bg_var, values=list(BACKGROUND_CHOICES), state="readonly", width=10
        ).grid(row=2, column=1, sticky="w", pady=1)
        ttk.Label(override, text="内部の小さな穴").grid(row=3, column=0, sticky="w")
        hole = ttk.Combobox(override, textvariable=self._hole_var, values=list(HOLE_CHOICES), state="readonly", width=14)
        hole.grid(row=3, column=1, sticky="w", pady=1)
        hole.bind("<<ComboboxSelected>>", lambda *_: self._schedule_finalize())

        merge = ttk.LabelFrame(side, text="色の統合（×ノイズ量）", padding=6)
        merge.grid(row=1, column=0, sticky="we", pady=(8, 0))
        merge.columnconfigure(0, weight=1)
        ttk.Scale(
            merge, from_=0.5, to=12.0, variable=self._merge_var, command=lambda *_: self._on_merge_slider()
        ).grid(row=0, column=0, sticky="we")
        ttk.Label(merge, textvariable=self._merge_label, width=10).grid(row=0, column=1, padx=(6, 0))
        ttk.Label(merge, text="右へ動かすほど似た色をまとめます", foreground="#666").grid(
            row=1, column=0, columnspan=2, sticky="w"
        )

        coef = ttk.LabelFrame(side, text="係数（画像から測った値に掛ける）", padding=6)
        coef.grid(row=2, column=0, sticky="we", pady=(8, 0))
        row = 0
        for fields in (ANALYSIS_FIELDS, FINALIZE_FIELDS):
            for key, label, lo, hi, step, _factor in fields:
                ttk.Label(coef, text=label).grid(row=row, column=0, sticky="w")
                spin = ttk.Spinbox(
                    coef, textvariable=self._coef_vars[key], from_=lo, to=hi, increment=step, width=7
                )
                spin.grid(row=row, column=1, sticky="w", padx=(6, 0), pady=1)
                row += 1
        ttk.Label(
            coef,
            text="上の 4 つは「復元する」で反映、\n下の 4 つと統合・穴の扱いはすぐ反映されます",
            foreground="#666",
            justify="left",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(coef, text="反映", command=self._schedule_finalize).grid(row=row + 1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(coef, text="既定に戻す", command=self._reset_coefs).grid(row=row + 1, column=1, sticky="e", pady=(4, 0))

    def _build_preview(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        bar = ttk.Frame(frame)
        bar.grid(row=0, column=0, sticky="we", pady=(0, 4))
        for i, name in enumerate(VIEW_CHOICES):
            ttk.Radiobutton(bar, text=name, value=name, variable=self._view_var, command=self._request_render).grid(
                row=0, column=i, padx=(0, 8)
            )
        ttk.Checkbutton(
            bar, text="置き換えたマスに印", variable=self._mark_changed_var, command=self._request_render
        ).grid(row=0, column=3, padx=(16, 8))
        ttk.Checkbutton(
            bar, text="怪しいマスに印", variable=self._mark_suspicious_var, command=self._request_render
        ).grid(row=0, column=4)
        ttk.Label(bar, textvariable=self._hover_var, foreground="#444").grid(row=0, column=5, padx=(16, 0))

        canvas = tk.Canvas(frame, background="#d8d8d8", highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        canvas.bind("<Configure>", lambda *_: self._request_render())
        canvas.bind("<Motion>", self._on_canvas_motion)
        canvas.bind("<Button-1>", self._on_canvas_click)
        self._window.bind("<Escape>", self._clear_selection)
        self._canvas = canvas

    def _build_palette(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="パレット", padding=6)
        frame.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        frame.rowconfigure(0, weight=1)
        style = ttk.Style(self._window)
        style.configure("DotRestore.Treeview", rowheight=22)
        tree = ttk.Treeview(
            frame,
            columns=("rgb", "count", "merge"),
            show="tree headings",
            selectmode="extended",
            style="DotRestore.Treeview",
        )
        tree.heading("#0", text="色")
        tree.heading("rgb", text="RGB")
        tree.heading("count", text="マス数")
        tree.heading("merge", text="統合候補")
        tree.column("#0", width=70, stretch=False)
        tree.column("rgb", width=110, stretch=False)
        tree.column("count", width=56, anchor="e", stretch=False)
        tree.column("merge", width=64, anchor="center", stretch=False)
        # マス数のごく少ない色（統合候補）は黄色の背景で目立たせる
        tree.tag_configure("tiny", background="#fff3b0")
        tree.grid(row=0, column=0, sticky="ns")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)
        tree.bind("<<TreeviewSelect>>", lambda *_: self._on_palette_select())
        tree.bind("<Button-1>", self._on_tree_click, add="+")
        self._tree = tree
        ttk.Label(
            frame,
            text="クリックでその色のマスを強調（もう一度クリックで解除）\n"
            "Ctrl+クリックで 2 色を選んで統合\n黄色の行はごく少ない色（→ の色への統合候補）",
            foreground="#666",
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        btns = ttk.Frame(frame)
        btns.grid(row=2, column=0, columnspan=2, sticky="we", pady=(4, 0))
        self._merge_button = ttk.Button(btns, text="選んだ 2 色を統合", command=self._merge_selected, state="disabled")
        self._merge_button.grid(row=0, column=0, sticky="w")
        self._unmerge_button = ttk.Button(
            btns, text="手動統合を取り消す", command=self._clear_manual_merges, state="disabled"
        )
        self._unmerge_button.grid(row=0, column=1, sticky="e", padx=(6, 0))
        self._merge_tiny_button = ttk.Button(
            btns, text="統合候補をすべて統合", command=self._merge_tiny, state="disabled"
        )
        self._merge_tiny_button.grid(row=1, column=0, sticky="w", pady=(4, 0))

    # ------------------------------------------------------------ 入力
    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self._window,
            title="復元する画像を選択",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif"), ("All files", "*.*")],
        )
        if path:
            self.load_path(Path(path))

    def _use_main_input(self) -> None:
        path = self._get_main_input()
        if path is None:
            messagebox.showinfo("ドット絵の復元", "メイン画面で入力画像が選ばれていません。", parent=self._window)
            return
        self.load_path(path)

    def load_path(self, path: Path) -> None:
        try:
            rgb = _load_rgb(path)
        except Exception as exc:
            messagebox.showerror("読込エラー", f"画像を開けませんでした: {exc}", parent=self._window)
            return
        self._job += 1  # 実行中の処理の結果は捨てる
        self._source_path = path
        self._rgb = rgb
        self._analysis = None
        self._result = None
        self._compare_cache = None
        self._manual_merges = []
        self._highlight = set()
        self._set_busy(False)
        self._fill_palette()
        self._update_buttons()
        self._summary_var.set("")
        self._status_var.set(f"{path.name}（{rgb.shape[1]}×{rgb.shape[0]}px）を読み込みました。「復元する」を押してください。")
        self._view_var.set("元画像")
        self._request_render()

    # ------------------------------------------------------------ パラメータ
    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:g}"

    def _parse_float(self, text: str, label: str) -> Optional[float]:
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"「{label}」は数値で入力してください。") from None

    def _collect_params(self) -> Optional[dr.RestoreParams]:
        try:
            values = {}
            for key, label, lo, hi, _step, factor in ANALYSIS_FIELDS + FINALIZE_FIELDS:
                v = self._parse_float(self._coef_vars[key].get(), label)
                if v is None:
                    v = getattr(dr.RestoreParams(), key) * factor
                values[key] = min(max(v, lo), hi) / factor
            pitch = self._parse_float(self._pitch_var.get(), "ピッチ")
            ox = self._parse_float(self._origin_x_var.get(), "原点 X")
            oy = self._parse_float(self._origin_y_var.get(), "原点 Y")
        except ValueError as exc:
            messagebox.showerror("入力エラー", str(exc), parent=self._window)
            return None
        if pitch is not None and pitch < 1.0:
            messagebox.showerror("入力エラー", "ピッチは 1 以上で入力してください。", parent=self._window)
            return None
        if (ox is None) != (oy is None):
            messagebox.showerror("入力エラー", "原点は X と Y の両方を入力してください。", parent=self._window)
            return None
        return dr.RestoreParams(
            merge_coef=float(self._merge_var.get()),
            hole_mode=HOLE_CHOICES.get(self._hole_var.get(), "keep"),
            background=BACKGROUND_CHOICES.get(self._bg_var.get(), "auto"),
            pitch=pitch,
            origin_x=ox,
            origin_y=oy,
            manual_merges=tuple(self._manual_merges),
            **values,
        )

    def _reset_coefs(self) -> None:
        defaults = dr.RestoreParams()
        for key, *_rest in ANALYSIS_FIELDS + FINALIZE_FIELDS:
            self._coef_vars[key].set(self._fmt(getattr(defaults, key) * _rest[-1]))
        self._merge_var.set(defaults.merge_coef)
        self._merge_label.set(f"{defaults.merge_coef:.1f}")
        self._schedule_finalize()

    # ------------------------------------------------------------ 実行
    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._run_button.configure(state="disabled" if busy else "normal")
        if not busy:
            self._progress.configure(value=0)

    def _start_restore(self) -> None:
        if self._rgb is None:
            messagebox.showinfo("ドット絵の復元", "先に画像を開いてください。", parent=self._window)
            return
        params = self._collect_params()
        if params is None:
            return
        # 前半の判定に関わる値が同じなら、後半（統合・補正）だけやり直す
        if self._analysis is not None and self._analysis.params.analysis_key() == params.analysis_key():
            self._start_finalize(params)
            return
        self._job += 1
        job = self._job
        self._set_busy(True)
        self._status_var.set("復元しています…")
        rgb = self._rgb
        threading.Thread(target=self._restore_worker, args=(job, rgb, params), daemon=True).start()

    def _restore_worker(self, job: int, rgb: np.ndarray, params: dr.RestoreParams) -> None:
        def progress(value: float, message: str) -> None:
            self._schedule(0, self._on_progress, job, value, message)

        try:
            analysis, result = dr.restore(rgb, params, progress)
        except Exception as exc:  # 画像ごとの想定外の失敗は画面に出す
            self._schedule(0, self._on_failed, job, exc)
            return
        self._schedule(0, self._on_done, job, analysis, result)

    def _start_finalize(self, params: dr.RestoreParams) -> None:
        if self._analysis is None:
            return
        self._job += 1
        job = self._job
        analysis = self._analysis
        self._status_var.set("色を整理し直しています…")

        def work() -> None:
            try:
                result = dr.finalize(analysis, params)
            except Exception as exc:
                self._schedule(0, self._on_failed, job, exc)
                return
            self._schedule(0, self._on_done, job, analysis, result)

        threading.Thread(target=work, daemon=True).start()

    def _schedule_finalize(self) -> None:
        if self._analysis is None or self._busy:
            return
        params = self._collect_params()
        if params is None:
            return
        # 前半の値は変えずに、後半の係数だけを差し替える
        kept = {name: getattr(self._analysis.params, name) for name in dr.ANALYSIS_PARAM_NAMES}
        self._start_finalize(replace(params, **kept))

    def _on_merge_slider(self) -> None:
        self._merge_label.set(f"{self._merge_var.get():.1f}")
        if self._merge_after is not None:
            try:
                self._window.after_cancel(self._merge_after)
            except tk.TclError:
                pass
        self._merge_after = self._window.after(200, self._schedule_finalize)

    def _on_progress(self, job: int, value: float, message: str) -> None:
        if job != self._job or not self.is_alive():
            return
        self._progress.configure(value=value)
        self._status_var.set(message)

    def _on_failed(self, job: int, exc: Exception) -> None:
        if job != self._job or not self.is_alive():
            return
        self._set_busy(False)
        self._status_var.set("復元に失敗しました。")
        messagebox.showerror("ドット絵の復元", f"復元に失敗しました: {exc}", parent=self._window)

    def _on_done(self, job: int, analysis: dr.Analysis, result: dr.RestoreResult) -> None:
        if job != self._job or not self.is_alive():
            return
        self._set_busy(False)
        self._analysis = analysis
        self._result = result
        self._compare_cache = None
        self._highlight = set()
        w, h = result.size
        colors = sum(1 for p in result.palette if not p["background"])
        bg = "背景あり" if analysis.bg_used else "背景なし"
        kind = "可逆" if analysis.lossless else "非可逆"
        interp = "補間（ぼかし）" if analysis.model == "bilinear" else "ニアレスト"
        self._status_var.set(f"完了：{w}×{h} ドット・{colors} 色（{bg}）")
        lines = [
            f"ピッチ {analysis.pitch:.3f}px　原点 ({analysis.origin[0]:.2f}, {analysis.origin[1]:.2f})　"
            f"{kind}・{interp}　統合閾値 T={result.merge_threshold:.2f}　平均誤差 {result.mean_error:.2f}　"
            f"置き換え {len(result.changes)} マス{self._change_breakdown(result)}・怪しいマス {len(result.suspicious)}",
        ]
        lines += [f"⚠ {msg}" for msg in result.warnings]
        lines += [f"・{msg}" for msg in analysis.notes]
        self._summary_var.set("\n".join(lines))
        self._fill_palette()
        self._update_buttons()
        if self._view_var.get() == "元画像":
            self._view_var.set("復元結果")
        self._request_render()

    @staticmethod
    def _change_breakdown(result: dr.RestoreResult) -> str:
        names = {"smudge": "にじみ", "mixed": "混ざったマス", "isolated": "孤立", "hole": "穴", "manual": "手動統合"}
        counts: dict[str, int] = {}
        for ch in result.changes:
            counts[ch["reason"]] = counts.get(ch["reason"], 0) + 1
        if not counts:
            return ""
        return "（" + "・".join(f"{names.get(k, k)} {v}" for k, v in counts.items()) + "）"

    # ------------------------------------------------------------ パレット
    def _fill_palette(self) -> None:
        tree = self._tree
        tree.delete(*tree.get_children())
        self._swatches = []
        if self._result is None:
            return
        for i, entry in enumerate(self._result.palette):
            r, g, b = entry["rgb"]
            swatch = ImageTk.PhotoImage(Image.new("RGB", (28, 16), (r, g, b)), master=self._window)
            self._swatches.append(swatch)
            name = "背景" if entry["background"] else str(i + 1)
            count = entry["count"] if not entry["background"] else (entry["count"] or "")
            merge_to = entry.get("merge_to")
            merge = f"→ {merge_to + 1}" if merge_to is not None else ""
            tags = ("tiny",) if merge_to is not None else ()
            tree.insert(
                "", "end", iid=str(i), text=f" {name}", image=swatch, values=(f"{r}, {g}, {b}", count, merge), tags=tags
            )

    def _selected_indices(self) -> list[int]:
        return [int(iid) for iid in self._tree.selection()]

    def _on_palette_select(self) -> None:
        sel = self._selected_indices()
        self._highlight = set(sel)
        self._update_buttons()
        if self._result is not None and self._view_var.get() != "復元結果":
            self._view_var.set("復元結果")
        self._request_render()

    def _merge_selected(self) -> None:
        sel = self._selected_indices()
        if self._result is None or len(sel) != 2:
            return
        a, b = (self._result.palette[i] for i in sel)
        # 背景を選んだときはもう一方を背景に、それ以外はマス数の少ないほうを多いほうへまとめる
        if a["background"] or b["background"]:
            src, dst = (b, a) if a["background"] else (a, b)
        else:
            src, dst = (a, b) if a["count"] <= b["count"] else (b, a)
        self._manual_merges.append((src["rgb"], dst["rgb"]))
        self._schedule_finalize()

    def _merge_tiny(self) -> None:
        """統合候補（ごく少ない色）をすべて、それぞれの候補先の色へ統合する。"""
        if self._result is None:
            return
        pal = self._result.palette
        merges = [(e["rgb"], pal[e["merge_to"]]["rgb"]) for e in pal if e.get("merge_to") is not None]
        if not merges:
            return
        self._manual_merges.extend(merges)
        self._schedule_finalize()

    def _on_tree_click(self, event) -> Optional[str]:
        """選択中の色だけをもう一度クリックしたら選択を解除する（何も選んでいない状態に戻す）。"""
        row = self._tree.identify_row(event.y)
        if not row or event.state & 0x0004:
            return None
        if self._tree.selection() == (row,):
            self._tree.selection_set(())
            return "break"
        return None

    def _clear_manual_merges(self) -> None:
        self._manual_merges = []
        self._schedule_finalize()

    def _update_buttons(self) -> None:
        ready = self._result is not None
        state = "normal" if ready else "disabled"
        for btn in (self._save_button, self._send_button):
            btn.configure(state=state)
        has_tiny = ready and any(e.get("merge_to") is not None for e in self._result.palette)
        self._merge_tiny_button.configure(state="normal" if has_tiny else "disabled")
        self._merge_button.configure(state="normal" if ready and len(self._tree.selection()) == 2 else "disabled")
        self._unmerge_button.configure(state="normal" if ready and self._manual_merges else "disabled")

    # ------------------------------------------------------------ 描画
    def _request_render(self) -> None:
        if self._render_after is not None:
            try:
                self._window.after_cancel(self._render_after)
            except tk.TclError:
                pass
        self._render_after = self._window.after(30, self._render)

    def _result_image(self) -> np.ndarray:
        """復元結果を表示用の RGB にする（透過は市松模様、強調中は他の色を薄く）。"""
        res = self._result
        assert res is not None
        img = res.pixels.astype(np.float32)
        h, w = res.labels.shape
        yy, xx = np.mgrid[0:h, 0:w]
        checker = np.where(((xx + yy) % 2 == 0)[..., None], 236.0, 206.0)
        transparent = res.alpha == 0
        img = np.where(transparent[..., None], checker, img)
        if self._highlight:
            keep = np.isin(res.labels, list(self._highlight))
            faded = img * 0.25 + 200.0 * 0.75
            img = np.where(keep[..., None], img, faded)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _render(self) -> None:
        self._render_after = None
        if not self.is_alive():
            return
        canvas = self._canvas
        canvas.delete("all")
        self._view_geom = None
        cw, ch = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        view = self._view_var.get()
        if self._rgb is None:
            canvas.create_text(cw // 2, ch // 2, text="画像を開いてください", fill="#666")
            return
        if view == "復元結果" and self._result is not None:
            arr = self._result_image()
        elif view == "比較" and self._result is not None and self._analysis is not None:
            if self._compare_cache is None:
                self._compare_cache = dr.comparison_image(self._analysis, self._result)
            arr = self._compare_cache
        else:
            arr = self._rgb
        h, w = arr.shape[:2]
        scale = min((cw - 16) / w, (ch - 16) / h)
        if scale >= 1.0:
            scale = float(int(scale))
            resample = Image.NEAREST
        else:
            resample = Image.BOX
        dw, dh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        pil = Image.fromarray(arr).resize((dw, dh), resample)
        self._photo = ImageTk.PhotoImage(pil, master=self._window)
        x0, y0 = (cw - dw) // 2, (ch - dh) // 2
        canvas.create_image(x0, y0, image=self._photo, anchor="nw")
        if view == "復元結果" and self._result is not None:
            self._view_geom = (x0, y0, dw / w)
            self._draw_marks(x0, y0, dw / w)

    def _draw_marks(self, x0: int, y0: int, s: float) -> None:
        res = self._result
        if res is None or s < 3:
            return
        marks: list[tuple[int, int, str]] = []
        if self._mark_changed_var.get():
            marks += [(c["x"], c["y"], "#ff8c00") for c in res.changes]
        if self._mark_suspicious_var.get():
            marks += [(x, y, "#e00000") for x, y in res.suspicious]
        inset = max(1.0, s * 0.12)
        for x, y, color in marks[:MAX_MARKS]:
            self._canvas.create_rectangle(
                x0 + x * s + inset,
                y0 + y * s + inset,
                x0 + (x + 1) * s - inset,
                y0 + (y + 1) * s - inset,
                outline=color,
                width=2 if s >= 8 else 1,
            )

    def _clear_selection(self, _event=None) -> None:
        self._tree.selection_set(())

    def _cell_at(self, event) -> Optional[tuple[int, int]]:
        if self._view_geom is None or self._result is None:
            return None
        x0, y0, s = self._view_geom
        x, y = int((event.x - x0) // s), int((event.y - y0) // s)
        h, w = self._result.labels.shape
        if 0 <= x < w and 0 <= y < h:
            return x, y
        return None

    def _on_canvas_motion(self, event) -> None:
        cell = self._cell_at(event)
        if cell is None:
            self._hover_var.set("")
            return
        x, y = cell
        res = self._result
        if res.alpha[y, x] == 0:
            self._hover_var.set(f"({x}, {y}) 透明")
            return
        r, g, b = (int(v) for v in res.pixels[y, x])
        self._hover_var.set(f"({x}, {y}) RGB {r}, {g}, {b}　誤差 {float(res.cell_error[y, x]):.1f}")

    def _on_canvas_click(self, event) -> None:
        cell = self._cell_at(event)
        if cell is None or self._result is None:
            return
        idx = str(int(self._result.labels[cell[1], cell[0]]))
        if event.state & 0x0004:  # Ctrl: 選択に追加
            self._tree.selection_add(idx)
        elif self._tree.selection() == (idx,):  # 選択中の色をもう一度クリックしたら解除
            self._tree.selection_set(())
            return
        else:
            self._tree.selection_set(idx)
        self._tree.see(idx)

    # ------------------------------------------------------------ 出力
    def _stem(self) -> str:
        return self._source_path.stem if self._source_path else "restored"

    def _save(self, kind: str) -> None:
        if self._result is None or self._analysis is None:
            return
        label, suffix, ext = dr.EXPORTS[kind]
        types = [("CSV", "*.csv")] if ext == ".csv" else [("PNG", "*.png")]
        path = filedialog.asksaveasfilename(
            parent=self._window,
            title=f"{label}を保存",
            initialdir=str(self._source_path.parent) if self._source_path else None,
            initialfile=f"{self._stem()}{suffix}{ext}",
            defaultextension=ext,
            filetypes=types,
        )
        if not path:
            return
        try:
            saved = dr.save_export(self._analysis, self._result, kind, path)
        except Exception as exc:
            messagebox.showerror("保存エラー", f"保存できませんでした: {exc}", parent=self._window)
            return
        self._status_var.set(f"{label}を保存しました: {saved.name}")

    def _send_to_main(self) -> None:
        if self._result is None:
            return
        folder = Path(tempfile.gettempdir()) / "Beads_Dot_Converter"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{self._stem()}_dot.png"
            Image.fromarray(dr.rgba(self._result), "RGBA").save(path)
        except Exception as exc:
            messagebox.showerror("ドット絵の復元", f"一時ファイルを作れませんでした: {exc}", parent=self._window)
            return
        self._on_send(path)
        self._status_var.set("メイン画面の入力画像に送りました（透明部分は白として扱われます）。")

    # ------------------------------------------------------------ ウィンドウ
    def is_alive(self) -> bool:
        try:
            return bool(self._window.winfo_exists())
        except tk.TclError:
            return False

    def focus(self) -> None:
        try:
            self._window.deiconify()
            self._window.lift()
            self._window.focus_force()
        except tk.TclError:
            pass

    def _handle_close(self) -> None:
        self._job += 1  # 実行中の結果は捨てる
        try:
            self._window.destroy()
        except tk.TclError:
            pass
        if self._on_close is not None:
            self._on_close()

