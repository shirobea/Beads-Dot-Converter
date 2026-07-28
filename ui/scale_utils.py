"""Shared helpers for slider pointer interactions."""

from __future__ import annotations

import tkinter as tk


def calc_scale_value_from_pointer(event: tk.Event) -> float:
    """Calculate scale value from pointer position."""
    scale = event.widget
    orient = str(scale.cget("orient") or "horizontal").lower()
    if orient == "vertical":
        height = max(1, scale.winfo_height())
        fraction = 1.0 - (event.y / height)
    else:
        width = max(1, scale.winfo_width())
        fraction = event.x / width
    fraction = max(0.0, min(1.0, fraction))
    start_val = float(scale.cget("from"))
    end_val = float(scale.cget("to"))
    return start_val + (end_val - start_val) * fraction


def bind_scale_click_jump(root: tk.Misc) -> None:
    """Bind click-to-jump behavior to all scales."""

    def _is_disabled(widget: tk.Widget) -> bool:
        # ttk.Scale と tk.Scale の両方に対応する
        if hasattr(widget, "instate"):
            try:
                return bool(widget.instate(["disabled"]))
            except Exception:
                return False
        try:
            return str(widget.cget("state")) == "disabled"
        except Exception:
            return False

    def _invoke_scale_command(scale: tk.Widget, value: float) -> None:
        # コールバックがあれば実行する
        try:
            cmd = scale.cget("command")
        except Exception:
            return
        if not cmd:
            return
        try:
            scale.tk.call(cmd, value)
        except Exception:
            pass

    def _on_pointer(event: tk.Event) -> str:
        if _is_disabled(event.widget):
            return "break"
        try:
            new_val = calc_scale_value_from_pointer(event)
            event.widget.set(new_val)
        except Exception:
            return "break"
        _invoke_scale_command(event.widget, new_val)
        return "break"

    for class_name in ("TScale", "Scale"):
        try:
            # 既定のクラスバインド（トラフクリックの増分ジャンプ+オートリピート）を
            # 置き換える。add="+" で共存させると、範囲の狭いスライダーでは増分±1が
            # 最大/最小へ飽和し、ポインタ追従と交互に発火して振動する
            root.bind_class(class_name, "<Button-1>", _on_pointer)
            root.bind_class(class_name, "<B1-Motion>", _on_pointer)
        except Exception:
            pass
