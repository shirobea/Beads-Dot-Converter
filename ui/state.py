"""ウィンドウ位置・サイズの保存/復元など状態管理をまとめるMixin。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import tkinter as tk

if TYPE_CHECKING:
    from .app import BeadsApp


def _write_json_atomic(path: Path, payload: dict) -> None:
    """一時ファイル経由でJSONを書き込み、途中クラッシュによる破損を防ぐ。"""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp_path, path)


class StateMixin:
    def _on_window_configure(self: "BeadsApp", event: tk.Event) -> None:
        """ウィンドウが動いたりリサイズされたら現在位置を覚えておく。"""
        if event.widget is not self.root:
            return
        current_state = self.root.state()
        self._last_window_state = current_state
        geometry = (
            self.root.winfo_width(),
            self.root.winfo_height(),
            self.root.winfo_x(),
            self.root.winfo_y(),
        )
        self._last_geometry = geometry
        if current_state == "normal":
            # 最大化時に復元したときのために通常サイズも保持しておく
            self._last_normal_geometry = geometry

    def _load_window_state(self: "BeadsApp") -> bool:
        """前回終了時のウィンドウ配置を読み込む。"""
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
                self.root.geometry(f"{width}x{height}+{x}+{y}")
                self._last_geometry = (width, height, x, y)
                self._last_normal_geometry = (width, height, x, y)
                self._last_window_state = state
                if state == "zoomed":
                    # 起動時も最大化を維持
                    self.root.state("zoomed")
                return True
        except Exception:
            return False
        return False

    # --- 設定の永続化 ---
    def _load_settings(self: "BeadsApp") -> None:
        """前回変換時の設定差分を復元する。"""
        try:
            if not hasattr(self, "_settings_path"):
                return
            if not self._settings_path.exists():
                return
            data = json.loads(self._settings_path.read_text(encoding="utf-8"))
            self.last_settings = data.get("last_settings")
            self.prev_settings = data.get("prev_settings")
            self._saved_mode = data.get("selected_mode")
            if hasattr(self, "normal_detail_var"):
                detail_raw = data.get("normal_detail_visible")
                if detail_raw is not None:
                    try:
                        self.normal_detail_var.set(bool(detail_raw))
                    except Exception:
                        pass
            if hasattr(self, "map_detail_var"):
                map_raw = data.get("map_detail_visible")
                if map_raw is not None:
                    try:
                        self.map_detail_var.set(bool(map_raw))
                    except Exception:
                        pass
            # 色使用一覧の明暗スライダー値も復元する
            if hasattr(self, "color_usage_tone_var"):
                tone_raw = data.get("color_usage_tone")
                if tone_raw is not None:
                    try:
                        tone_val = float(tone_raw)
                        tone_val = max(-1.0, min(1.0, tone_val))
                        self.color_usage_tone_var.set(tone_val)
                    except Exception:
                        pass
        except Exception:
            # 復元に失敗しても起動は続ける
            self.last_settings = None
            self.prev_settings = None
            self._saved_mode = None

    def _save_settings(self: "BeadsApp") -> None:
        """現在の設定差分を保存する。"""
        try:
            if not hasattr(self, "_settings_path"):
                return
            selected_mode = getattr(self, "_saved_mode", None)
            if hasattr(self, "mode_var"):
                try:
                    selected_mode = self.mode_var.get()
                except Exception:
                    pass
            payload = {
                "last_settings": self.last_settings,
                "prev_settings": self.prev_settings,
                "selected_mode": selected_mode,
            }
            if hasattr(self, "normal_detail_var"):
                try:
                    payload["normal_detail_visible"] = bool(self.normal_detail_var.get())
                except Exception:
                    pass
            if hasattr(self, "map_detail_var"):
                try:
                    payload["map_detail_visible"] = bool(self.map_detail_var.get())
                except Exception:
                    pass
            # 色使用一覧の明暗スライダー値も保存する
            if hasattr(self, "color_usage_tone_var"):
                try:
                    tone_val = float(self.color_usage_tone_var.get())
                    payload["color_usage_tone"] = max(-1.0, min(1.0, tone_val))
                except Exception:
                    pass
            _write_json_atomic(self._settings_path, payload)
        except Exception:
            # 保存失敗は致命的ではないので無視
            pass

    def _remember_mode_selection(self: "BeadsApp") -> None:
        """選択中の変換モードだけを保存する。"""
        try:
            if not hasattr(self, "mode_var"):
                return
            # 変換しなくても次回のモード復元に使う
            self._saved_mode = self.mode_var.get()
            self._save_settings()
        except Exception:
            pass

    def _save_window_state(self: "BeadsApp") -> None:
        """直近に記録したウィンドウ配置をファイルへ保存する。"""
        if self._last_geometry is None:
            self._last_geometry = (
                self.root.winfo_width(),
                self.root.winfo_height(),
                self.root.winfo_x(),
                self.root.winfo_y(),
            )
        current_state = self.root.state()
        if current_state not in ("normal", "zoomed"):
            current_state = "normal"
        geometry = self._last_geometry
        if current_state == "zoomed" and getattr(self, "_last_normal_geometry", None):
            # 最大化時でも通常サイズを復元できるよう別途保存
            geometry = self._last_normal_geometry
        if geometry is None:
            geometry = (
                self.root.winfo_width(),
                self.root.winfo_height(),
                self.root.winfo_x(),
                self.root.winfo_y(),
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
            _write_json_atomic(self._window_state_path, payload)
        except Exception:
            pass
