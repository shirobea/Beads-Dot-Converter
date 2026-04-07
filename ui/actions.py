"""ユーザー操作を司るアクション層のMixin。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Callable, Any

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
from color_spaces import rgb_to_lab
from .models import ConversionRequest
from .color_usage_window import ColorUsageWindow
from .color_usage_service import build_color_usage_rows
from .noise_filters import build_noise_filter_registry
import numpy as np
import converter

if TYPE_CHECKING:
    from .app import BeadsApp


class ActionsMixin:
    """画像選択・保存・変換開始/停止などのユーザー操作ハンドラ。"""

    def _schedule_on_ui(self: "BeadsApp", delay_ms: int, func: Callable[..., None], *args: Any) -> None:
        """閉じる途中はUIスレッドへの投げ込みを抑止する。"""
        if getattr(self, "_closing", False):
            return
        try:
            if not self.root.winfo_exists():
                return
        except Exception:
            return
        try:
            self.root.after(delay_ms, func, *args)
        except Exception:
            pass

    def select_image(self: "BeadsApp") -> None:
        path = filedialog.askopenfilename(
            title="画像を選択",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")],
        )
        if not path:
            return
        self.input_image_path = Path(path)
        try:
            image = Image.open(self.input_image_path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("読込エラー", f"画像を開けませんでした: {exc}")
            return
        self.input_original_pil = image
        self.input_filtered_pil = None
        self.input_pil = image
        self._input_using_filtered = False
        self._showing_input_overlay = False
        filters = self._get_noise_filter_registry()
        if filters and self.noise_filter_var.get() not in filters:
            self.noise_filter_var.set(next(iter(filters)))
        self.output_pil = None
        self.output_image = None
        self.color_usage = []
        self._color_usage_base_image = None
        self._all_mode_results = None
        self._output_grid_photos = []
        # 入力を変えたら前回出力のプレビューは破棄してブレンド表示の混在を防ぐ
        self.prev_output_pil = None
        self._showing_prev = False
        self._output_photo = None
        self.original_size = image.size
        self.output_canvas.configure(image="", text="変換後")
        self._set_initial_target_size(image)
        self._refresh_previews()
        matched = self._update_color_usage_from_input(image)
        if self.mode_var.get() == "全て":
            self._set_color_usage_button_state(False)
        base_msg = "入力画像を読み込みました。RGB最適化を行う場合はボタンを押してください。"
        if matched:
            base_msg = "入力画像を読み込みました。パレット内の色のみのため色使用一覧を開けます。RGB最適化を行う場合はボタンを押してください。"
        if self.mode_var.get() == "全て":
            base_msg += " 全てモードでは色使用一覧は利用できません。"
        self.rgb_log_var.set(base_msg)
        self._request_input_shading_update(immediate=True)

    def select_normal_map(self: "BeadsApp") -> None:
        """ノーマルマップを選択して状態に保存する。"""
        path = filedialog.askopenfilename(
            title="ノーマルマップを選択",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")],
        )
        if not path:
            return
        self.normal_map_path = Path(path)
        self.normal_map_label.set(self.normal_map_path.name)
        self._request_input_shading_update()

    def select_ao_map(self: "BeadsApp") -> None:
        """AOマップを選択して状態に保存する。"""
        path = filedialog.askopenfilename(
            title="AOマップを選択",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")],
        )
        if not path:
            return
        self.ao_map_path = Path(path)
        self.ao_map_label.set(self.ao_map_path.name)
        self._request_input_shading_update()

    def select_specular_map(self: "BeadsApp") -> None:
        """Specularマップを選択して状態に保存する。"""
        path = filedialog.askopenfilename(
            title="Specularマップを選択",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")],
        )
        if not path:
            return
        self.specular_map_path = Path(path)
        self.specular_map_label.set(self.specular_map_path.name)
        self._request_input_shading_update()

    def select_displacement_map(self: "BeadsApp") -> None:
        """Displacementマップを選択して状態に保存する。"""
        path = filedialog.askopenfilename(
            title="Displacementマップを選択",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")],
        )
        if not path:
            return
        self.displacement_map_path = Path(path)
        self.displacement_map_label.set(self.displacement_map_path.name)
        self._request_input_shading_update()

    def _sanitize_kernel_size(self: "BeadsApp", raw_size: object) -> int:
        """メディアン用カーネルサイズを奇数・下限付きで整える。"""
        try:
            size = int(raw_size)
        except Exception:
            size = 3
        size = max(3, size)
        if size % 2 == 0:
            size += 1
        self.noise_filter_size_var.set(size)
        return size

    def _set_noise_busy(self: "BeadsApp", busy: bool) -> None:
        """ノイズ除去処理中はUI操作を抑止する。"""
        self._noise_busy = busy
        state_token = "disabled" if busy else "normal"
        for btn_name in ("noise_apply_button", "noise_reset_button"):
            btn = getattr(self, btn_name, None)
            if btn:
                try:
                    btn.configure(state=state_token)
                except Exception:
                    pass
        if busy:
            self.status_var.set("ノイズ除去を実行中です...")

    def _set_progress_style(self: "BeadsApp", style_name: str) -> None:
        """進捗バーの表示スタイルを切り替える。"""
        bar = getattr(self, "progress_bar", None)
        if not bar:
            return
        try:
            bar.configure(style=style_name)
        except Exception:
            pass

    def _cancel_noise_progress_timer(self: "BeadsApp") -> None:
        """ノイズ除去進捗の更新タイマーを停止する。"""
        after_id = getattr(self, "_noise_progress_after_id", None)
        if not after_id:
            return
        try:
            self.root.after_cancel(after_id)
        except Exception:
            pass
        self._noise_progress_after_id = None

    def _update_noise_progress_display(self: "BeadsApp", value: float) -> None:
        """ノイズ除去用の進捗表示を更新する。"""
        clamped = max(0.0, min(1.0, value))
        percent = int(clamped * 100)
        elapsed = 0.0
        if self._noise_progress_start is not None:
            elapsed = time.perf_counter() - self._noise_progress_start
        self.progress_label.configure(text=f"ノイズ除去: {percent}% (経過 {elapsed:.1f}s)")
        self.progress_bar["value"] = percent

    def _schedule_noise_progress_tick(self: "BeadsApp") -> None:
        """ノイズ除去中に疑似的な進捗を進める。"""
        if not getattr(self, "_noise_busy", False):
            return
        self._noise_progress_value = min(0.95, self._noise_progress_value + 0.03)
        self._update_noise_progress_display(self._noise_progress_value)
        try:
            self._noise_progress_after_id = self.root.after(120, self._schedule_noise_progress_tick)
        except Exception:
            self._noise_progress_after_id = None

    def _start_noise_progress(self: "BeadsApp") -> None:
        """ノイズ除去の進捗表示を開始する。"""
        self._cancel_noise_progress_timer()
        self._noise_progress_start = time.perf_counter()
        self._noise_progress_value = 0.0
        self._set_progress_style(self._progress_style_default)
        self._update_noise_progress_display(0.0)
        self._schedule_noise_progress_tick()

    def _finish_noise_progress(self: "BeadsApp", success: bool) -> None:
        """ノイズ除去の進捗表示を終了する。"""
        self._cancel_noise_progress_timer()
        self._set_progress_style(self._progress_style_default)
        if success:
            self._noise_progress_value = 1.0
            self._update_noise_progress_display(1.0)
        else:
            self._reset_progress_display()
        self._noise_progress_start = None

    def _get_noise_filter_registry(self: "BeadsApp") -> dict[str, Callable[[Image.Image], Image.Image]]:
        """追加しやすいようフィルタ名と実装を辞書でまとめる。"""
        size = self._sanitize_kernel_size(self.noise_filter_size_var.get())
        try:
            sigma = int(self.bilateral_sigma_var.get())
        except Exception:
            sigma = 75
        try:
            return build_noise_filter_registry(size, sigma)
        except RuntimeError as exc:
            if not getattr(self, "_noise_filter_error_shown", False):
                messagebox.showerror("ノイズ除去未対応", f"ノイズ除去にはOpenCVが必要です。\n{exc}")
                setattr(self, "_noise_filter_error_shown", True)
            return {}

    def apply_noise_reduction(self: "BeadsApp") -> None:
        """入力画像にノイズ除去フィルタを適用する（別スレッドで実行）。"""
        if self.input_original_pil is None:
            messagebox.showinfo("入力画像なし", "先に入力画像を選択してください。")
            return
        if getattr(self, "_noise_busy", False):
            return
        filters = self._get_noise_filter_registry()
        if not filters:
            messagebox.showinfo("ノイズ除去未対応", "OpenCVが見つからないためノイズ除去は利用できません。")
            return
        name = self.noise_filter_var.get()
        filter_func = filters.get(name)
        if filter_func is None:
            messagebox.showerror("フィルタ未対応", f"選択したフィルタ「{name}」は未対応です。")
            return

        self._set_noise_busy(True)
        self._start_noise_progress()

        def _worker() -> None:
            try:
                # PILオブジェクトはスレッド間で共有しないようコピー
                src = self.input_original_pil.copy() if self.input_original_pil else None
                if src is None:
                    raise ValueError("入力画像が見つかりません。")
                filtered_img = filter_func(src)
            except Exception as exc:  # スレッド内で例外を捕捉してUIスレッドへ渡す
                self._schedule_on_ui(0, lambda: self._on_noise_failed(exc))
                return
            self._schedule_on_ui(0, lambda: self._on_noise_finished(name, filtered_img))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_noise_finished(self: "BeadsApp", name: str, filtered: Image.Image) -> None:
        """ノイズ除去成功時のUI反映（UIスレッドで実行）。"""
        self._finish_noise_progress(True)
        self.input_filtered_pil = filtered
        self.input_pil = filtered
        self._input_using_filtered = True
        self._showing_input_overlay = False
        self.status_var.set(f"{name}フィルタでノイズ除去しました。")
        self.rgb_log_var.set(f"{name}フィルタでノイズ除去しました。")
        self._refresh_previews()
        self._request_input_shading_update(immediate=True)
        self._set_noise_busy(False)

    def _on_noise_failed(self: "BeadsApp", exc: Exception) -> None:
        """ノイズ除去失敗時のUI処理（UIスレッドで実行）。"""
        self._finish_noise_progress(False)
        messagebox.showerror("ノイズ除去失敗", f"ノイズ除去中にエラーが発生しました:\n{exc}")
        self._set_noise_busy(False)

    def reset_noise_reduction(self: "BeadsApp") -> None:
        """ノイズ除去前の元画像に戻す。"""
        if self.input_original_pil is None:
            messagebox.showinfo("入力画像なし", "先に入力画像を選択してください。")
            return
        if getattr(self, "_noise_busy", False):
            return
        self._set_noise_busy(True)
        self.input_filtered_pil = None
        self.input_pil = self.input_original_pil
        self._input_using_filtered = False
        self._showing_input_overlay = False
        self.status_var.set("入力画像を元に戻しました。")
        self.rgb_log_var.set("ノイズ除去をリセットしました。")
        self._refresh_previews()
        self._request_input_shading_update(immediate=True)
        self._set_noise_busy(False)

    def _on_space_key(self: "BeadsApp", _event: "tk.Event") -> str:
        if self._runner.is_running:
            self.cancel_conversion()
        else:
            self.start_conversion()
        return "break"

    def _get_active_input_array(self: "BeadsApp") -> Optional[np.ndarray]:
        """現在表示中の入力画像をRGB配列にして返す。"""
        if self.input_pil is None:
            return None
        return np.asarray(self.input_pil.convert("RGB"), dtype=np.uint8)

    def _input_shading_enabled(self: "BeadsApp") -> bool:
        """ノーマル/AO/Specular/Displacementのどれかが有効か判定する。"""
        has_normal = bool(self.normal_enabled_var.get()) and self.normal_map_path is not None
        has_ao = bool(self.ao_enabled_var.get()) and self.ao_map_path is not None
        has_specular = bool(self.specular_enabled_var.get()) and self.specular_map_path is not None
        has_disp = bool(self.displacement_enabled_var.get()) and self.displacement_map_path is not None
        return has_normal or has_ao or has_specular or has_disp

    def _request_input_shading_update(self: "BeadsApp", immediate: bool = False) -> None:
        """入力プレビューの陰影更新をデバウンスして実行する。"""
        after_id = getattr(self, "_input_shading_after_id", None)
        if after_id:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        if immediate:
            self._update_input_shading_preview()
            return
        try:
            self._input_shading_after_id = self.root.after(80, self._update_input_shading_preview)
        except Exception:
            self._input_shading_after_id = None

    def _update_input_shading_preview(self: "BeadsApp") -> None:
        """現在のノーマル/AO設定を入力プレビューに反映する。"""
        self._input_shading_after_id = None
        if self.input_pil is None:
            self._input_shaded_pil = None
            return
        if not self._input_shading_enabled():
            self._input_shaded_pil = None
            self._refresh_previews()
            return
        try:
            base = np.asarray(self.input_pil.convert("RGB"), dtype=np.uint8)
            shaded = converter.apply_shading_preview(
                image_rgb=base,
                normal_map_path=str(self.normal_map_path) if self.normal_map_path else None,
                normal_enabled=bool(self.normal_enabled_var.get()),
                normal_invert_y=bool(self.normal_invert_y_var.get()),
                normal_light_dir=(
                    float(self.normal_light_x_var.get()),
                    float(self.normal_light_y_var.get()),
                    float(self.normal_light_z_var.get()),
                ),
                normal_strength=float(self.normal_strength_var.get()),
                normal_ambient=float(self.normal_ambient_var.get()),
                normal_gamma=float(self.normal_gamma_var.get()),
                ao_map_path=str(self.ao_map_path) if self.ao_map_path else None,
                ao_enabled=bool(self.ao_enabled_var.get()),
                ao_strength=float(self.ao_strength_var.get()),
                specular_map_path=str(self.specular_map_path) if self.specular_map_path else None,
                specular_enabled=bool(self.specular_enabled_var.get()),
                specular_strength=float(self.specular_strength_var.get()),
                specular_shininess=float(self.specular_shininess_var.get()),
                displacement_map_path=str(self.displacement_map_path) if self.displacement_map_path else None,
                displacement_enabled=bool(self.displacement_enabled_var.get()),
                displacement_strength=float(self.displacement_strength_var.get()),
                displacement_midpoint=float(self.displacement_midpoint_var.get()),
                displacement_invert=bool(self.displacement_invert_var.get()),
            )
            self._input_shaded_pil = Image.fromarray(shaded)
        except Exception:
            self._input_shaded_pil = None
        self._refresh_previews()

    def start_conversion(self: "BeadsApp") -> None:
        if self._runner.is_running:
            return
        if not self.input_image_path:
            messagebox.showwarning("入力ファイル未選択", "まず入力画像を選択してください。")
            return
        request = self._gather_request()
        if request is None:
            return
        input_image = self._get_active_input_array()
        if input_image is None:
            messagebox.showerror("入力エラー", "入力画像を読み込んでから変換してください。")
            return
        self._pending_settings = self._build_pending_settings(request)
        self._prepare_conversion_ui()
        started = self._runner.start(
            request=request,
            input_path=str(self.input_image_path),
            input_image=input_image,
            palette=self.palette,
            on_progress=self.update_progress,
            on_success=self._on_conversion_success,
            on_cancelled=self._on_cancelled,
            on_error=self._handle_worker_error,
        )
        if not started:
            self.status_var.set("既に変換中です。")

    def _resolve_dither_method(self: "BeadsApp") -> str:
        """コンボボックスの表示ラベルを内部 method 文字列に変換する。"""
        from converter.dither import DITHER_SPECS
        label = self.dither_var.get()
        return next((s["method"] for s in DITHER_SPECS if s["label"] == label), "none")

    def _gather_request(self: "BeadsApp") -> Optional[ConversionRequest]:
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
        except ValueError:
            messagebox.showerror("入力エラー", "幅・高さには整数を入力してください。")
            return None
        if width <= 0 or height <= 0:
            messagebox.showerror("入力エラー", "幅・高さは1以上にしてください。")
            return None
        cmc_l = float(self.cmc_l_var.get())
        cmc_c = float(self.cmc_c_var.get())
        cmc_l = max(0.5, min(3.0, cmc_l))
        cmc_c = max(0.5, min(3.0, cmc_c))
        resize_label = self.resize_method_var.get()
        resize_method = {
            "ニアレストネイバー": "nearest",
            "バイリニア": "bilinear",
            "INTER_AREA": "area",
        }.get(resize_label, "nearest")
        keep_aspect = self.lock_aspect_var.get()
        r_w = max(0.5, min(2.0, float(self.rgb_r_weight_var.get())))
        g_w = max(0.5, min(2.0, float(self.rgb_g_weight_var.get())))
        b_w = max(0.5, min(2.0, float(self.rgb_b_weight_var.get())))
        if self.normal_enabled_var.get() and not self.normal_map_path:
            messagebox.showerror("入力エラー", "ノーマルマップを選択してください。")
            return None
        if self.ao_enabled_var.get() and not self.ao_map_path:
            messagebox.showerror("入力エラー", "AOマップを選択してください。")
            return None
        if self.specular_enabled_var.get() and not self.specular_map_path:
            messagebox.showerror("入力エラー", "Specularマップを選択してください。")
            return None
        if self.displacement_enabled_var.get() and not self.displacement_map_path:
            messagebox.showerror("入力エラー", "Displacementマップを選択してください。")
            return None
        spec_strength = max(0.0, min(2.0, float(self.specular_strength_var.get())))
        spec_shininess = max(1.0, min(64.0, float(self.specular_shininess_var.get())))
        return ConversionRequest(
            width=width,
            height=height,
            mode=self.mode_var.get().replace(" (CIEDE2000)", ""),
            lab_metric=self.lab_metric_var.get(),
            cmc_l=cmc_l,
            cmc_c=cmc_c,
            keep_aspect=keep_aspect,
            resize_method=resize_method,
            rgb_weights=(r_w, g_w, b_w),
            normal_map_path=str(self.normal_map_path) if self.normal_map_path else None,
            normal_enabled=bool(self.normal_enabled_var.get()),
            normal_invert_y=bool(self.normal_invert_y_var.get()),
            normal_light_dir=(
                float(self.normal_light_x_var.get()),
                float(self.normal_light_y_var.get()),
                float(self.normal_light_z_var.get()),
            ),
            normal_strength=float(self.normal_strength_var.get()),
            normal_ambient=float(self.normal_ambient_var.get()),
            normal_gamma=float(self.normal_gamma_var.get()),
            ao_map_path=str(self.ao_map_path) if self.ao_map_path else None,
            ao_enabled=bool(self.ao_enabled_var.get()),
            ao_strength=float(self.ao_strength_var.get()),
            specular_map_path=str(self.specular_map_path) if self.specular_map_path else None,
            specular_enabled=bool(self.specular_enabled_var.get()),
            specular_strength=spec_strength,
            specular_shininess=spec_shininess,
            displacement_map_path=str(self.displacement_map_path) if self.displacement_map_path else None,
            displacement_enabled=bool(self.displacement_enabled_var.get()),
            displacement_strength=float(self.displacement_strength_var.get()),
            displacement_midpoint=float(self.displacement_midpoint_var.get()),
            displacement_invert=bool(self.displacement_invert_var.get()),
            pseudo_gradient_strength=max(0.0, min(20.0, float(self.pseudo_gradient_var.get()))),
            use_super_sampling=bool(self.super_sampling_var.get()),
            dither_method=self._resolve_dither_method(),
            dither_strength=max(0.0, min(1.0, float(self.dither_strength_var.get()))),
        )

    def _build_pending_settings(self: "BeadsApp", request: ConversionRequest) -> dict:
        if request.mode == "全て":
            cmc_l = "2.0"
            cmc_c = "1.0"
            rgb_weights = [1.0, 1.0, 1.0]
        else:
            cmc_l = f"{request.cmc_l:.1f}"
            cmc_c = f"{request.cmc_c:.1f}"
            rgb_weights = [
                round(request.rgb_weights[0], 1),
                round(request.rgb_weights[1], 1),
                round(request.rgb_weights[2], 1),
            ]
        resize_label = {
            "nearest": "ニアレストネイバー",
            "bilinear": "バイリニア",
            "area": "INTER_AREA",
        }.get(request.resize_method, request.resize_method)
        return {
            "幅": request.width,
            "高さ": request.height,
            "モード": request.mode,
            "Lab距離式": request.lab_metric,
            "CMC l": cmc_l,
            "CMC c": cmc_c,
            "リサイズ方式": resize_label,
            "RGB重み": rgb_weights,
            "ノーマル有効": bool(request.normal_enabled),
            "ノーマルY反転": bool(request.normal_invert_y),
            "ノーマル強さ": round(float(request.normal_strength), 3),
            "ノーマル環境光": round(float(request.normal_ambient), 3),
            "ノーマルガンマ": round(float(request.normal_gamma), 3),
            "ノーマル光方向": [
                round(float(request.normal_light_dir[0]), 3),
                round(float(request.normal_light_dir[1]), 3),
                round(float(request.normal_light_dir[2]), 3),
            ],
            "ノーマルマップ": request.normal_map_path,
            "AO有効": bool(request.ao_enabled),
            "AO強さ": round(float(request.ao_strength), 3),
            "AOマップ": request.ao_map_path,
            "Specular有効": bool(request.specular_enabled),
            "Specular強さ": round(float(request.specular_strength), 3),
            "Specular鋭さ": round(float(request.specular_shininess), 3),
            "Specularマップ": request.specular_map_path,
            "Displacement有効": bool(request.displacement_enabled),
            "Displacement強さ": round(float(request.displacement_strength), 3),
            "Displacement中心": round(float(request.displacement_midpoint), 3),
            "Displacement反転": bool(request.displacement_invert),
            "Displacementマップ": request.displacement_map_path,
            "グラデーション強度": round(float(request.pseudo_gradient_strength), 1),
            "ディザリング": self.dither_var.get(),
            "ディザリング強度": round(float(request.dither_strength), 2),
        }

    def _prepare_conversion_ui(self: "BeadsApp") -> None:
        self.save_button.configure(state="disabled")
        self._set_color_usage_button_state(False)
        self._start_time = time.perf_counter()
        self._set_progress_style(self._progress_style_default)
        self.update_progress(0.0)
        self.status_var.set("変換中...")
        self.convert_button.configure(text="変換中止", state="normal", command=self.cancel_conversion)

    def _set_color_usage_button_state(self: "BeadsApp", enabled: bool) -> None:
        """色使用一覧ボタンの有効/無効を切り替える。"""
        btn = getattr(self, "color_usage_button", None)
        if not btn:
            return
        state_token = "normal" if enabled else "disabled"
        try:
            btn.configure(state=state_token)
        except Exception:
            pass

    def _set_3d_preview_button_state(self: "BeadsApp", enabled: bool) -> None:
        """3Dプレビューの有効/無効を切り替える。"""
        btn = getattr(self, "preview_3d_button", None)
        if not btn:
            return
        state_token = "normal" if enabled else "disabled"
        try:
            btn.configure(state=state_token)
        except Exception:
            pass

    def _get_3d_preview_source(self: "BeadsApp") -> Optional[np.ndarray]:
        """3Dプレビューに使える画像を取得する。"""
        if self.output_image is not None:
            return self.output_image
        base = getattr(self, "_color_usage_base_image", None)
        if isinstance(base, np.ndarray):
            return base
        return None

    def _update_3d_preview_button_state(self: "BeadsApp") -> None:
        """3Dプレビューの可否を最新状態に合わせる。"""
        self._set_3d_preview_button_state(self._get_3d_preview_source() is not None)

    def _build_color_usage_rows(self: "BeadsApp", image: np.ndarray, settings: Optional[dict]) -> list[dict]:
        """変換後画像から色使用数の一覧を作る。"""
        if image is None:
            return []
        mode = ""
        if settings:
            mode = str(settings.get("モード", ""))
        if mode.lower() in {"none", "なし", "全て"}:
            return []
        _, rows = build_color_usage_rows(image, self.palette, require_in_palette=False)
        return rows

    def _get_all_mode_grid_shape(self: "BeadsApp", width: int, height: int) -> tuple[int, int]:
        """全モード表示の行数・列数を画像比率で切り替える。"""
        if width <= 0 or height <= 0:
            return (4, 2)
        if width >= height:
            return (4, 2)
        return (2, 4)

    def _compose_all_mode_image(self: "BeadsApp", results: list[dict]) -> Optional[np.ndarray]:
        """全モード結果を横長は2列×4行、縦長は2行×4列で合成する。"""
        if not results:
            return None
        images = [entry.get("image") for entry in results if isinstance(entry, dict)]
        images = [img for img in images if isinstance(img, np.ndarray)]
        if not images:
            return None
        base_h, base_w = images[0].shape[:2]
        rows, cols = self._get_all_mode_grid_shape(base_w, base_h)
        canvas = np.zeros((base_h * rows, base_w * cols, 3), dtype=np.uint8)
        for idx, img in enumerate(images[: rows * cols]):
            if img.shape[:2] != (base_h, base_w):
                # サイズが異なる場合は最小限のリサイズを行う
                img = np.asarray(
                    Image.fromarray(img).resize((base_w, base_h), Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
            row = idx // cols
            col = idx % cols
            y0 = row * base_h
            x0 = col * base_w
            canvas[y0 : y0 + base_h, x0 : x0 + base_w] = img
        return canvas

    def _analyze_palette_usage(self: "BeadsApp", image: np.ndarray) -> tuple[bool, list[dict]]:
        """入力画像がパレット内の色だけか確認し、色使用一覧を作成する。"""
        return build_color_usage_rows(image, self.palette, require_in_palette=True)

    def _update_color_usage_from_input(self: "BeadsApp", image: Image.Image) -> bool:
        """入力画像がパレット色のみなら色使用一覧を有効化する。"""
        input_array = np.asarray(image, dtype=np.uint8)
        matched, rows = self._analyze_palette_usage(input_array)
        if not matched:
            self.color_usage = []
            self._color_usage_base_image = None
            self._set_color_usage_button_state(False)
            self._refresh_color_usage_window(reset_sort=False)
            self._update_3d_preview_button_state()
            return False
        self.color_usage = rows
        self._color_usage_base_image = input_array
        self._set_color_usage_button_state(bool(rows))
        self._refresh_color_usage_window(reset_sort=True)
        self._update_3d_preview_button_state()
        return True

    def _refresh_color_usage_window(self: "BeadsApp", reset_sort: bool) -> None:
        """色使用一覧ウィンドウが開いている場合のみ更新する。"""
        window = getattr(self, "_color_usage_window", None)
        if window and window.is_alive():
            window.update_rows(self.color_usage, reset_sort=reset_sort)

    def _update_color_usage_preview(self: "BeadsApp", rgb: Optional[tuple[int, int, int]]) -> None:
        """色使用一覧のプレビューを更新する。"""
        window = getattr(self, "_color_usage_window", None)
        if window and window.is_alive():
            preview_image = self._make_color_usage_preview(rgb)
            source_image = self._get_color_usage_base_pil()
            window.set_preview_image(preview_image, source_image)

    def _make_color_usage_preview(self: "BeadsApp", rgb: Optional[tuple[int, int, int]]) -> Optional[Image.Image]:
        """選択色以外を薄くしたプレビュー画像を作る。"""
        base_image = self._color_usage_base_image if self._color_usage_base_image is not None else self.output_image
        if base_image is None or not self.color_usage:
            return None
        base = np.asarray(base_image, dtype=np.uint8)
        if rgb is None:
            return Image.fromarray(base)
        target = np.array(rgb, dtype=np.uint8)
        mask = np.all(base == target, axis=2)
        if not mask.any():
            return Image.fromarray(base)
        # 非選択色はスライダー値で白寄せ/黒寄せする
        tone = float(self.color_usage_tone_var.get())
        tone = max(-1.0, min(1.0, tone))
        base_float = base.astype(np.float32)
        if tone >= 0:
            mix = tone
            adjusted = base_float * (1.0 - mix) + 255.0 * mix
        else:
            mix = -tone
            adjusted = base_float * (1.0 - mix)
        dim = np.clip(adjusted, 0, 255).astype(np.uint8)
        result = base.copy()
        result[~mask] = dim[~mask]
        return Image.fromarray(result)

    def _get_color_usage_base_pil(self: "BeadsApp") -> Optional[Image.Image]:
        """プレビューの色拾い用に元画像をPIL化する。"""
        base_image = self._color_usage_base_image if self._color_usage_base_image is not None else self.output_image
        if base_image is None:
            return None
        return Image.fromarray(np.asarray(base_image, dtype=np.uint8))

    def _on_color_usage_window_closed(self: "BeadsApp") -> None:
        """色使用一覧ウィンドウを閉じた後の参照をクリアする。"""
        self._color_usage_window = None
        # 非選択色の明暗値を閉じるタイミングで保存する
        self._save_settings()

    def _on_color_usage_select(self: "BeadsApp", rgb: Optional[tuple[int, int, int]]) -> None:
        """色使用一覧で選択された色に合わせてプレビューを更新する。"""
        self._color_usage_selected_rgb = rgb
        self._update_color_usage_preview(rgb)

    def _on_color_usage_tone_change(self: "BeadsApp") -> None:
        """非選択色の明暗調整を表示とプレビューに反映する。"""
        value = float(self.color_usage_tone_var.get())
        clamped = max(-1.0, min(1.0, value))
        if clamped != value:
            self.color_usage_tone_var.set(clamped)
        percent = int(round(abs(clamped) * 100))
        if clamped > 0:
            label = f"白寄せ {percent}%"
        elif clamped < 0:
            label = f"黒寄せ {percent}%"
        else:
            label = "変更なし"
        self.color_usage_tone_display.set(label)
        self._update_color_usage_preview(self._color_usage_selected_rgb)

    def show_color_usage(self: "BeadsApp") -> None:
        """色使用一覧ウィンドウを開く。"""
        if getattr(self, "_all_mode_results", None):
            messagebox.showinfo("色使用一覧", "全てモードでは色使用一覧は利用できません。")
            return
        if not getattr(self, "color_usage", None):
            messagebox.showinfo("色使用一覧", "変換後の色一覧がありません。")
            return
        window = getattr(self, "_color_usage_window", None)
        if window and window.is_alive():
            window.focus()
            return
        self._color_usage_window = ColorUsageWindow(
            self.root,
            self.color_usage,
            on_close=self._on_color_usage_window_closed,
            on_select=self._on_color_usage_select,
            dim_var=self.color_usage_tone_var,
            dim_display_var=self.color_usage_tone_display,
        )
        self._update_color_usage_preview(None)

    def _check_3d_preview_available(self: "BeadsApp") -> bool:
        """3Dプレビューの依存関係があるか確認する。"""
        try:
            import OpenGL.GL  # noqa: F401
            import OpenGL.GLU  # noqa: F401
            from pyopengltk import OpenGLFrame  # noqa: F401
        except Exception as exc:
            messagebox.showerror(
                "3Dプレビュー未対応",
                "3Dプレビューを使うには以下をインストールしてください。\n"
                "pip install PyOpenGL pyopengltk\n\n"
                f"詳細: {exc}",
            )
            return False
        return True

    def _on_3d_preview_closed(self: "BeadsApp") -> None:
        """3Dプレビューの参照をクリアする。"""
        self._preview_3d_window = None

    def _update_3d_preview(self: "BeadsApp", image: np.ndarray) -> None:
        """3Dプレビューに最新の画像を渡す。"""
        window = getattr(self, "_preview_3d_window", None)
        if window is None:
            return
        try:
            if not window.winfo_exists():
                return
        except Exception:
            return
        try:
            window.set_image(image)
        except Exception:
            pass

    def open_3d_preview(self: "BeadsApp") -> None:
        """3Dプレビュー（試作）を開く。"""
        if not self._check_3d_preview_available():
            return
        source = self._get_3d_preview_source()
        if source is None:
            messagebox.showinfo("3Dプレビュー", "プレビューできる画像がありません。")
            return
        window = getattr(self, "_preview_3d_window", None)
        try:
            if window is not None and window.winfo_exists():
                self._update_3d_preview(source)
                window.focus()
                return
        except Exception:
            pass
        try:
            from .preview_3d import BeadsPreview3DWindow
        except Exception as exc:
            messagebox.showerror("3Dプレビュー未対応", f"3Dプレビューの読み込みに失敗しました:\n{exc}")
            return
        self._preview_3d_window = BeadsPreview3DWindow(self.root, on_close=self._on_3d_preview_closed)
        self._preview_3d_window.focus()
        self._update_3d_preview(source)

    def cancel_conversion(self: "BeadsApp") -> None:
        self._runner.cancel()
        self.status_var.set("中止要求を送信しました...")
        self.convert_button.configure(state="disabled", text="停止中...")
        self._start_time = None
        self._reset_progress_display()

    def _on_conversion_success(self: "BeadsApp", result: object) -> None:
        """ワーカースレッド成功時のUI側反映。"""
        if getattr(self, "_closing", False):
            return
        self.output_path = None
        self.prev_settings = self.last_settings
        self._showing_prev = False
        # 表示準備の中で細かく進捗を動かす
        def _set_ui_fraction(value: float) -> None:
            self._set_ui_progress_fraction(value)
            self.root.update_idletasks()

        def _set_ui_phase(phase: str, value: float) -> None:
            clamped = max(0.0, min(1.0, value))
            if phase == "create":
                start, end = 0.35, 0.55
            elif phase == "resize":
                start, end = 0.55, 0.85
            elif phase == "draw":
                start, end = 0.85, 1.0
            else:
                start, end = 0.35, 1.0
            self._set_ui_progress_fraction(start + (end - start) * clamped)
            self.root.update_idletasks()

        if isinstance(result, list):
            self.prev_output_pil = None
            all_results: list[dict] = []
            total = max(1, len(result))
            for idx, entry in enumerate(result):
                if not isinstance(entry, dict):
                    continue
                image = entry.get("image")
                if not isinstance(image, np.ndarray):
                    continue
                label = str(entry.get("label", ""))
                all_results.append(
                    {"label": label, "image": image, "pil": Image.fromarray(image)}
                )
                _set_ui_phase("create", (idx + 1) / total)
            self._all_mode_results = all_results if all_results else None
            self._output_grid_photos = []
            self.output_image = self._compose_all_mode_image(all_results)
            self.output_pil = Image.fromarray(self.output_image) if self.output_image is not None else None
            self._color_usage_base_image = None
            self.color_usage = []
            self._set_color_usage_button_state(False)
            self._refresh_color_usage_window(reset_sort=False)
        else:
            self.prev_output_pil = self.output_pil
            self._all_mode_results = None
            self._output_grid_photos = []
            self.output_image = result
            self._color_usage_base_image = result
            self.output_pil = Image.fromarray(result)
            self.color_usage = self._build_color_usage_rows(result, self._pending_settings)
            self._set_color_usage_button_state(bool(self.color_usage))
            self._refresh_color_usage_window(reset_sort=True)
            _set_ui_phase("create", 1.0)
        _set_ui_fraction(0.25)
        self.last_settings = self._pending_settings
        self._pending_settings = None
        self.diff_var.set(self._build_diff_overlay())
        self._save_settings()
        _set_ui_fraction(0.35)
        self._refresh_previews(progress_cb=_set_ui_phase)
        if self.output_image is not None:
            self._update_3d_preview(self.output_image)
        # 変換結果の有無で3Dプレビューのボタンを切り替える
        self._update_3d_preview_button_state()
        _set_ui_phase("draw", 1.0)
        self._set_progress_value(1.0)
        self._restore_convert_button()
        self.save_button.configure(state="normal" if self.output_image is not None else "disabled")
        self.status_var.set("変換完了（保存ボタンで任意の場所に保存できます）")
        self._showing_prev = False

    def _handle_worker_error(self: "BeadsApp", exc: Exception) -> None:
        """ワーカースレッドで例外が出た場合のUI処理。"""
        messagebox.showerror("変換失敗", f"変換中にエラーが発生しました:\n{exc}")
        self._handle_failure("変換に失敗しました")

    def _on_cancelled(self: "BeadsApp") -> None:
        self._reset_after_stop("中止しました", clear_canvas=False, preserve_output=True)

    def _handle_failure(self: "BeadsApp", status: str) -> None:
        self._reset_after_stop(status, clear_canvas=False)

    def save_image(self: "BeadsApp") -> None:
        image_to_save = self.output_image
        all_results = getattr(self, "_all_mode_results", None)
        if all_results:
            image_to_save = self._compose_all_mode_image(all_results)
        if image_to_save is None:
            self.status_var.set("出力画像がまだありません。")
            return
        initial_dir = str(self.input_image_path.parent) if self.input_image_path else str(Path.cwd())
        default_name = (
            f"{self.input_image_path.stem}_beads.png" if self.input_image_path else "output_beads.png"
        )
        path = filedialog.asksaveasfilename(
            title="出力先を選択",
            defaultextension=".png",
            initialdir=initial_dir,
            initialfile=default_name,
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            self.status_var.set("保存をキャンセルしました。")
            return
        try:
            Image.fromarray(image_to_save).save(path)
            self.output_path = Path(path)
            self.status_var.set("保存しました")
        except Exception as exc:
            messagebox.showerror("保存失敗", f"出力画像の保存に失敗しました:\n{exc}")

    def _reset_progress_display(self: "BeadsApp") -> None:
        self.progress_label.configure(text="進捗: 0% (経過 0.0s)")
        self.progress_bar["value"] = 0

    def update_progress(self: "BeadsApp", value: float) -> None:
        self._set_progress_style(self._progress_style_default)
        clamped = max(0.0, min(1.0, value))
        self._conversion_progress_last = clamped
        start, end = getattr(self, "_conversion_progress_range", (0.0, 1.0))
        mapped = start + (end - start) * clamped
        self._set_progress_value(mapped)

    def _set_progress_value(self: "BeadsApp", value: float) -> None:
        """進捗表示を実数(0.0-1.0)で更新する。"""
        clamped = max(0.0, min(1.0, value))
        percent = int(round(clamped * 100))
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
        self.progress_label.configure(text=f"進捗: {percent}% (経過 {elapsed:.1f}s)")
        self.progress_bar["value"] = percent

    def _set_ui_progress_fraction(self: "BeadsApp", fraction: float) -> None:
        """UI側の進捗を0.0-1.0で受け取り、全体の進捗へ反映する。"""
        start, end = getattr(self, "_ui_progress_range", (0.0, 1.0))
        clamped = max(0.0, min(1.0, fraction))
        mapped = start + (end - start) * clamped
        self._set_progress_value(mapped)

    def _update_ui_progress(self: "BeadsApp", step: int, total: int) -> None:
        """表示準備側の進捗を段階的に進める。"""
        if total <= 0:
            return
        start, end = getattr(self, "_ui_progress_range", (0.0, 1.0))
        fraction = max(0.0, min(1.0, step / total))
        mapped = start + (end - start) * fraction
        self._set_progress_value(mapped)

    def _reset_after_stop(
        self: "BeadsApp",
        status: str,
        clear_canvas: bool,
        preserve_output: bool = False,
    ) -> None:
        self._start_time = None
        self._conversion_progress_last = 0.0
        if not preserve_output:
            self.output_image = None
            self.output_pil = None
            self.prev_output_pil = None
            self._all_mode_results = None
            self._output_grid_photos = []
            self.output_path = None
            self.diff_var.set("")
            self.color_usage = []
            self._color_usage_base_image = None
        elif self.output_image is not None:
            self._color_usage_base_image = self.output_image
        self._pending_settings = None
        self._reset_progress_display()
        self._restore_convert_button()
        if preserve_output:
            self.save_button.configure(state="normal" if self.output_image is not None else "disabled")
        else:
            self.save_button.configure(state="disabled")
        self._update_3d_preview_button_state()
        self._set_color_usage_button_state(bool(self.color_usage and self._color_usage_base_image is not None))
        self._refresh_color_usage_window(reset_sort=False)
        if clear_canvas and not preserve_output:
            self.output_canvas.configure(image="", text="変換後")
        self.status_var.set(status)

    def _restore_convert_button(self: "BeadsApp") -> None:
        self.convert_button.configure(text="変換実行", state="normal", command=self.start_conversion)

    # --- CMC(l:c)最適化 ---
    def compute_optimal_cmc_weights(self: "BeadsApp") -> None:
        """入力画像の明度/彩度分布からCMC(l:c)の重みを推定する。"""
        if self.input_pil is None:
            messagebox.showinfo("入力画像なし", "先に入力画像を選択してください。")
            return
        # Lab空間での明度と彩度の広がりを見てバランスを決める
        try:
            lab = rgb_to_lab(np.asarray(self.input_pil.convert("RGB"), dtype=np.uint8))
        except RuntimeError as exc:
            messagebox.showerror("CMC最適化失敗", f"CMC最適化にはOpenCVが必要です。\n{exc}")
            return
        l_vals = lab[..., 0]
        chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
        l_spread = float(np.percentile(l_vals, 95) - np.percentile(l_vals, 5))
        c_spread = float(np.percentile(chroma, 95) - np.percentile(chroma, 5))
        l_score = float(l_vals.std()) + 0.5 * l_spread
        c_score = float(chroma.std()) + 0.5 * c_spread
        balance = float(np.clip((l_score - c_score) / max(l_score, c_score, 1e-3), -0.8, 0.8))
        l_w = float(np.clip(2.0 + balance * 1.2, 0.5, 3.0))
        c_w = float(np.clip(1.0 - balance * 0.8, 0.5, 3.0))
        self.cmc_l_var.set(l_w)
        self.cmc_c_var.set(c_w)
        self._on_cmc_l_change()
        self._on_cmc_c_change()
        msg = f"CMC最適化結果: l={l_w:.2f}, c={c_w:.2f}（明度/彩度分布から算出）"
        self.status_var.set(msg)
        self.rgb_log_var.set(msg)

    def reset_cmc_weights(self: "BeadsApp") -> None:
        """CMC(l:c)の重みをデフォルトに戻す。"""
        self.cmc_l_var.set(2.0)
        self.cmc_c_var.set(1.0)
        self._on_cmc_l_change()
        self._on_cmc_c_change()
        self.status_var.set("CMC重みをデフォルト(2.0 / 1.0)に戻しました。")
        self.rgb_log_var.set("CMC重みをデフォルト(2.0 / 1.0)に戻しました。")

    # --- RGB最適化/プレビュー ---
    def compute_optimal_rgb_weights(self: "BeadsApp") -> None:
        """入力画像から単純な平均値ベースでRGB重みを算出する。"""
        if self.input_pil is None:
            messagebox.showinfo("入力画像なし", "先に入力画像を選択してください。")
            return
        arr = np.asarray(self.input_pil, dtype=np.float32)
        means = arr.mean(axis=(0, 1))
        safe_means = np.clip(means, 1e-3, None)
        target = float(safe_means.mean())
        weights = target / safe_means
        weights = np.clip(weights, 0.5, 1.5)
        r_w, g_w, b_w = (float(weights[0]), float(weights[1]), float(weights[2]))
        self.rgb_r_weight_var.set(r_w)
        self.rgb_g_weight_var.set(g_w)
        self.rgb_b_weight_var.set(b_w)
        self._on_rgb_r_change()
        self._on_rgb_g_change()
        self._on_rgb_b_change()
        msg = f"最適化結果: R={r_w:.2f}, G={g_w:.2f}, B={b_w:.2f}（単純平均合わせ）"
        self.status_var.set(msg)
        self.rgb_log_var.set(msg)

    def reset_rgb_weights(self: "BeadsApp") -> None:
        """RGB重みを1.0に戻し、表示を更新する。"""
        self.rgb_r_weight_var.set(1.0)
        self.rgb_g_weight_var.set(1.0)
        self.rgb_b_weight_var.set(1.0)
        self._on_rgb_r_change()
        self._on_rgb_g_change()
        self._on_rgb_b_change()
        self.rgb_log_var.set("RGB重みを1.0にリセットしました。")
        self.status_var.set("RGB重みを1.0にリセットしました。")
