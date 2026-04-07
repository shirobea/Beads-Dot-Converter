"""変換実行のワーカー管理を担うコントローラ。"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

import converter
from palette import BeadPalette
from .models import ConversionRequest


class ConversionRunner:
    """UIとは疎結合に変換スレッドを管理する。"""

    def __init__(
        self,
        schedule_ui: Callable[[int, Callable[..., None], object], None],
        is_closing: Callable[[], bool],
    ) -> None:
        self._schedule_ui = schedule_ui
        self._is_closing = is_closing
        self._thread: Optional[threading.Thread] = None
        self._cancel_event: Optional[threading.Event] = None

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def start(
        self,
        request: ConversionRequest,
        input_path: str,
        palette: BeadPalette,
        on_progress: Callable[[float], None],
        on_success: Callable[[object], None],
        on_cancelled: Callable[[], None],
        on_error: Callable[[Exception], None],
        input_image: np.ndarray | None = None,
    ) -> bool:
        """変換スレッドを開始。すでに実行中ならFalseを返す。"""
        if self.is_running:
            return False
        self._cancel_event = threading.Event()

        def _progress_cb(value: float) -> None:
            self._schedule_ui(0, on_progress, value)

        def _worker() -> None:
            try:
                if request.mode == "全て":
                    # 全モードは固定パラメータで実行する
                    result = converter.convert_all_modes(
                        input_path=input_path,
                        input_image=input_image,
                        output_size=(request.width, request.height),
                        palette=palette,
                        keep_aspect=request.keep_aspect,
                        resize_method=request.resize_method,
                        rgb_weights=(1.0, 1.0, 1.0),
                        cmc_l=2.0,
                        cmc_c=1.0,
                        normal_map_path=request.normal_map_path,
                        normal_enabled=request.normal_enabled,
                        normal_invert_y=request.normal_invert_y,
                        normal_light_dir=request.normal_light_dir,
                        normal_strength=request.normal_strength,
                        normal_ambient=request.normal_ambient,
                        normal_gamma=request.normal_gamma,
                        ao_map_path=request.ao_map_path,
                        ao_enabled=request.ao_enabled,
                        ao_strength=request.ao_strength,
                        specular_map_path=request.specular_map_path,
                        specular_enabled=request.specular_enabled,
                        specular_strength=request.specular_strength,
                        specular_shininess=request.specular_shininess,
                        displacement_map_path=request.displacement_map_path,
                        displacement_enabled=request.displacement_enabled,
                        displacement_strength=request.displacement_strength,
                        displacement_midpoint=request.displacement_midpoint,
                        displacement_invert=request.displacement_invert,
                        pseudo_gradient_strength=request.pseudo_gradient_strength,
                        use_super_sampling=request.use_super_sampling,
                        dither_method=request.dither_method,
                        dither_strength=request.dither_strength,
                        progress_callback=_progress_cb,
                        cancel_event=self._cancel_event,
                    )
                else:
                    result = converter.convert_image(
                        input_path=input_path,
                        input_image=input_image,
                        output_size=(request.width, request.height),
                        mode=request.mode,
                        lab_metric=request.lab_metric,
                        cmc_l=request.cmc_l,
                        cmc_c=request.cmc_c,
                        palette=palette,
                        keep_aspect=request.keep_aspect,
                        resize_method=request.resize_method,
                        rgb_weights=request.rgb_weights,
                        normal_map_path=request.normal_map_path,
                        normal_enabled=request.normal_enabled,
                        normal_invert_y=request.normal_invert_y,
                        normal_light_dir=request.normal_light_dir,
                        normal_strength=request.normal_strength,
                        normal_ambient=request.normal_ambient,
                        normal_gamma=request.normal_gamma,
                        ao_map_path=request.ao_map_path,
                        ao_enabled=request.ao_enabled,
                        ao_strength=request.ao_strength,
                        specular_map_path=request.specular_map_path,
                        specular_enabled=request.specular_enabled,
                        specular_strength=request.specular_strength,
                        specular_shininess=request.specular_shininess,
                        displacement_map_path=request.displacement_map_path,
                        displacement_enabled=request.displacement_enabled,
                        displacement_strength=request.displacement_strength,
                        displacement_midpoint=request.displacement_midpoint,
                        displacement_invert=request.displacement_invert,
                        pseudo_gradient_strength=request.pseudo_gradient_strength,
                        use_super_sampling=request.use_super_sampling,
                        dither_method=request.dither_method,
                        dither_strength=request.dither_strength,
                        progress_callback=_progress_cb,
                        cancel_event=self._cancel_event,
                    )
            except converter.ConversionCancelled:
                self._schedule_ui(0, on_cancelled)
                return
            except Exception as exc:  # pragma: no cover - GUI依存
                self._schedule_ui(0, on_error, exc)
                return

            if self._is_closing():
                return
            self._schedule_ui(0, on_success, result)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        if self._cancel_event:
            self._cancel_event.set()

    def cancel_and_wait(self, timeout: float = 2.0) -> None:
        self.cancel()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        self._cancel_event = None
