"""Application entry point."""

import sys

from ui import BeadsApp
import tkinter as tk
from palette import load_default_palette
from tkinter import messagebox


def main() -> None:
    """Launch the Tkinter application."""
    root = tk.Tk()
    root.title("Beads Dot Converter")

    try:
        # 起動時は先頭のパレットを読み込み、保存済みの選択があればUI側で差し替える
        palette, palette_file = load_default_palette()
    except Exception as exc:  # pragma: no cover - GUI entry
        messagebox.showerror("パレット読込エラー", f"カラーパレットの読み込みに失敗しました。\n{exc}")
        root.destroy()
        return

    BeadsApp(root, palette, palette_file)
    root.mainloop()


if __name__ == "__main__":
    # 作品イメージは pywebview がメインスレッドを占有するため別プロセスで起動する。
    # exe 版も自分自身をこの引数付きで起動するので、tkinter は立ち上げずにここで分岐する
    if len(sys.argv) >= 3 and sys.argv[1] == "--desk-preview":
        from ui.desk_preview import run

        run(sys.argv[2])
    else:
        main()
