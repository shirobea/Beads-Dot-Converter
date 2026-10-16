"""作品イメージ（デスクプレビュー）。

変換結果のビーズ作品を実寸のPCデスクに置いた状態で three.js に描かせる。
pywebview の ``webview.start()`` はメインスレッド必須で tkinter の mainloop と
共存できないため、表示は別プロセスで行う。

親（本体）側:
    ``build_scene_data()`` でシーンJSONを作り、``DeskPreviewLauncher.show()`` が
    一時フォルダに書き出して子プロセスを起動する。開いている間に再度呼ばれたら
    JSONを書き換えるだけで、子プロセス側が検知して再読込する。
子（表示）側:
    ``main.py --desk-preview <json_path>`` から ``run()`` が呼ばれる。
    tkinter は起動しない。
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# 照明・置き方などの表示設定を次回に引き継ぐファイル
_STATE_PATH = Path(__file__).resolve().parent / "desk_preview_state.json"
# アイロン後の実測色。選択中パレットと色番号が一致すれば、こちらの値で表示する
IRONED_PALETTE_FILE = "Scan_ironed.csv"
DESK_PREVIEW_ARG = "--desk-preview"
# 子プロセスが自分でメッセージボックスを出してから終了するときの終了コード
EXIT_NO_PYWEBVIEW = 2
EXIT_WINDOW_FAILED = 3


def web_root() -> Path:
    """HTML/JS 一式の置き場所（PyInstaller 版は _MEIPASS 配下）。"""
    candidates = [Path(__file__).resolve().parent / "desk_preview_web"]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "ui" / "desk_preview_web")
    for path in candidates:
        if (path / "index.html").is_file():
            return path
    return candidates[0]


# ---------------------------------------------------------------------------
# シーンデータ生成（本体側）
# ---------------------------------------------------------------------------


def _load_ironed_rgb() -> dict[str, tuple[float, float, float]]:
    """色番号 → アイロン後の実測RGB。CSVが無ければ空。"""
    try:
        from palette import load_palette, resolve_palette_path

        ironed = load_palette(resolve_palette_path(IRONED_PALETTE_FILE))
    except Exception:
        return {}
    return {c.color_id: tuple(float(v) for v in c.rgb) for c in ironed.colors}


def build_scene_data(image: np.ndarray, palette: Any, pitch_mm: float, palette_name: str = "") -> dict:
    """変換結果の画像からシーンJSON用の辞書を作る。

    画像の各ピクセルを最も近いパレット色へ割り当て、セルにはシーン内パレットの
    index（空白は -1）を入れる。シーン内パレットは使われている色だけに絞る。
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("RGB画像が必要です。")
    height, width = arr.shape[:2]
    rgb = arr[:, :, :3].astype(np.int32)
    empty = np.zeros((height, width), dtype=bool)
    if arr.shape[2] >= 4:
        empty = arr[:, :, 3] < 128

    # 変換結果はパレット色そのものなので、ユニーク色ごとに最近傍を1回引けば足りる
    flat = rgb.reshape(-1, 3)
    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    pal_rgb = palette.rgb_array.astype(np.float32)
    dist = ((uniq[:, None, :].astype(np.float32) - pal_rgb[None, :, :]) ** 2).sum(axis=2)
    nearest = dist.argmin(axis=1)

    ironed = _load_ironed_rgb()
    used_pal = sorted(set(int(i) for i in np.unique(nearest[inverse][~empty.reshape(-1)])))
    scene_index = {pal_idx: n for n, pal_idx in enumerate(used_pal)}
    scene_palette = []
    for pal_idx in used_pal:
        color = palette.colors[pal_idx]
        src = ironed.get(color.color_id)
        rgb_val = src if src is not None else color.rgb
        scene_palette.append(
            {
                "id": color.color_id,
                "name": color.name,
                "rgb": [int(round(v)) for v in rgb_val],
                "rgb_source": "ironed" if src is not None else "palette",
                "material": "opaque",
            }
        )

    lookup = np.array([scene_index.get(int(p), -1) for p in nearest], dtype=np.int32)
    cells = lookup[inverse].reshape(height, width)
    cells[empty] = -1
    return {
        "grid": {"width": int(width), "height": int(height), "cells": cells.tolist()},
        "palette": scene_palette,
        "palette_name": palette_name,
        "bead_pitch_mm": float(pitch_mm),
        "ironing": "both_sides",
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def _child_command(json_path: Path) -> list[str]:
    """別プロセス起動コマンド。exe 版は自分自身を再起動する。"""
    if getattr(sys, "frozen", False):
        return [sys.executable, DESK_PREVIEW_ARG, str(json_path)]
    main_py = Path(__file__).resolve().parent.parent / "main.py"
    return [sys.executable, str(main_py), DESK_PREVIEW_ARG, str(json_path)]


class DeskPreviewLauncher:
    """作品イメージの子プロセスを1つだけ管理する。"""

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._json_path = Path(tempfile.gettempdir()) / f"beads_desk_preview_{os.getpid()}.json"
        # パイプだと読み手がいないと詰まるので、子の stderr はファイルに流す
        self._log_path = self._json_path.with_suffix(".log")
        # 本体を閉じたら作品イメージも閉じる
        atexit.register(self.close)

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def show(self, scene: dict) -> None:
        """シーンを書き出し、未起動なら子プロセスを起動する。"""
        _write_json_atomic(self._json_path, scene)
        if self.is_running():
            # 子プロセスがJSONの更新を検知して再読込・前面化する
            return
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with open(self._log_path, "wb") as log:
            self._process = subprocess.Popen(
                _child_command(self._json_path),
                creationflags=flags,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
            )

    def update(self, scene: dict) -> None:
        """開いている場合だけ内容を差し替える。"""
        if self.is_running():
            _write_json_atomic(self._json_path, scene)

    def wait_startup_error(self, timeout: float = 4.0) -> Optional[str]:
        """起動直後に落ちた場合のエラー文を返す（生きていれば None）。"""
        proc = self._process
        if proc is None:
            return None
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if proc.returncode in (0, EXIT_NO_PYWEBVIEW, EXIT_WINDOW_FAILED):
            # 正常終了、または子プロセスが既にエラーを表示済み
            return None
        try:
            err = self._log_path.read_bytes().decode("utf-8", "replace")
        except Exception:
            err = ""
        return err.strip()[-1500:] or f"終了コード {proc.returncode}"

    def close(self) -> None:
        if self.is_running():
            try:
                self._process.terminate()
            except Exception:
                pass
        for path in (self._json_path, self._log_path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 表示プロセス（子側）
# ---------------------------------------------------------------------------


def _load_state() -> dict:
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(data: dict) -> None:
    try:
        _write_json_atomic(_STATE_PATH, data)
    except Exception:
        pass


def _list_assets(sub: str, suffixes: tuple[str, ...]) -> list[str]:
    folder = web_root() / "assets" / sub
    if not folder.is_dir():
        return []
    # JS 側は拡張子を除いた名前で引き、ファイル名でロードする
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in suffixes)


class _DeskApi:
    """JS から ``window.pywebview.api.*`` で呼ばれる窓口。"""

    def __init__(self, json_path: Path) -> None:
        self._json_path = json_path
        self._window = None
        self._state = _load_state()

    def attach(self, window: Any) -> None:
        self._window = window

    def get_scene(self) -> dict:
        data = json.loads(self._json_path.read_text(encoding="utf-8"))
        data["models"] = _list_assets("models", (".glb", ".gltf"))
        data["hdri"] = _list_assets("hdri", (".hdr",))
        return data

    def get_settings(self) -> dict:
        return dict(self._state.get("settings", {}))

    def save_settings(self, settings: dict) -> None:
        self._state["settings"] = settings
        _save_state(self._state)

    def save_screenshot(self, data_url: str) -> str:
        """PNG の dataURL を保存ダイアログで選んだ場所へ書く。"""
        import webview

        if self._window is None or "," not in data_url:
            return ""
        name = time.strftime("beads_desk_%Y%m%d_%H%M%S.png")
        result = self._window.create_file_dialog(
            webview.FileDialog.SAVE, save_filename=name, file_types=("PNG 画像 (*.png)",)
        )
        if not result:
            return ""
        target = result if isinstance(result, str) else result[0]
        if not target.lower().endswith(".png"):
            target += ".png"
        Path(target).write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
        return target


def _remember_geometry(api: _DeskApi, window: Any) -> None:
    try:
        api._state["window"] = {
            "width": int(window.width),
            "height": int(window.height),
            "x": int(window.x),
            "y": int(window.y),
        }
        _save_state(api._state)
    except Exception:
        pass


def _watch_scene(api: _DeskApi, window: Any, json_path: Path, stop: threading.Event) -> None:
    """本体がJSONを書き換えたら JS に再読込させて前面に出す。"""
    try:
        last = json_path.stat().st_mtime_ns
    except OSError:
        last = 0
    while not stop.wait(0.4):
        try:
            mtime = json_path.stat().st_mtime_ns
        except OSError:
            continue
        if mtime == last:
            continue
        last = mtime
        try:
            window.evaluate_js("window.deskPreview && window.deskPreview.reload()")
            window.restore()
            # 他プロセスのウィンドウは素直に前面へ来ないので、最前面を一瞬だけ立てる
            window.on_top = True
            window.on_top = False
        except Exception:
            pass


def _show_fatal(message: str) -> None:
    # exe 版（console=False）では sys.stderr が None になる
    if sys.stderr is not None:
        sys.stderr.write(message + "\n")
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, "作品イメージ", 0x10)
        except Exception:
            pass


def run(json_path: str) -> None:
    """子プロセスのエントリーポイント。"""
    path = Path(json_path)
    try:
        import webview
    except Exception as exc:
        _show_fatal(
            "作品イメージを表示するには pywebview が必要です。\n"
            f"pip install pywebview\n\n詳細: {exc}"
        )
        sys.exit(EXIT_NO_PYWEBVIEW)

    api = _DeskApi(path)
    geom = api._state.get("window", {})
    window = webview.create_window(
        "作品イメージ",
        url=str(web_root() / "index.html"),
        js_api=api,
        width=int(geom.get("width", 1280)),
        height=int(geom.get("height", 800)),
        x=geom.get("x"),
        y=geom.get("y"),
        min_size=(640, 420),
        background_color="#1d1f22",
    )
    api.attach(window)
    stop = threading.Event()
    window.events.closing += lambda: _remember_geometry(api, window)
    window.events.closed += stop.set

    def _on_start() -> None:
        threading.Thread(target=_watch_scene, args=(api, window, path, stop), daemon=True).start()

    try:
        # ES Modules は file:// では読めないため、内蔵 HTTP サーバー経由で配信する
        webview.start(_on_start, http_server=True, private_mode=True)
    except Exception as exc:
        _show_fatal(
            "作品イメージのウィンドウを開けませんでした。\n"
            "Microsoft Edge WebView2 ランタイムがインストールされているか確認してください。\n\n"
            f"詳細: {exc}"
        )
        sys.exit(EXIT_WINDOW_FAILED)
    finally:
        stop.set()
