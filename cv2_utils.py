"""OpenCV (cv2) の依存解決ユーティリティ。"""

from __future__ import annotations


def require_cv2():
    """cv2 をインポートして返す。未インストールの場合は RuntimeError を送出する。"""
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV (cv2) が必要です。pip install opencv-python") from exc
    return cv2
