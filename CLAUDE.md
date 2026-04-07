# Beads_Dot_Converter 開発ガイド

## プロジェクト概要
Python + Tkinter ベースのアイロンビーズ用画像変換ツール。
画像をビーズパレット色にマッピングして出力するアプリ。

## フォルダ構造
```
Beads_Dot_Converter/
├── main.py                    # エントリーポイント
├── color_spaces.py            # 色空間変換・距離計算
├── palette.py                 # ビーズパレットCSV管理
├── normalize_settings.py      # settings.json の移行ユーティリティ
├── ColorPallet.csv            # ビーズカラーデータ（14列以上、日英ヘッダ対応）
├── Beads_Dot_Converter.spec   # PyInstaller ビルド設定
├── create_shortcut.ps1        # デスクトップショートカット作成スクリプト
├── run.vbs                    # pythonw ランチャースクリプト
├── CHANGELOG.md
├── VERSION                    # 現在のバージョン
├── converter/                 # 変換パイプライン
│   ├── __init__.py
│   ├── pipeline.py            # 変換フロー本体
│   ├── quantize.py            # パレットへの色マッピング
│   ├── dither.py              # ディザリング（Floyd-Steinberg / Atkinson / Bayer）
│   └── io_utils.py            # 画像 I/O ユーティリティ
└── ui/                        # GUI コンポーネント
    ├── __init__.py            # BeadsApp をエクスポート
    ├── app.py                 # メインウィンドウクラス（BeadsApp）
    ├── layout.py              # ウィジェット構築（LayoutMixin）
    ├── actions.py             # ユーザー操作ハンドラ（ActionsMixin）
    ├── state.py               # 設定永続化（StateMixin）
    ├── preview.py             # プレビュー描画（PreviewMixin）
    ├── preview_3d.py          # 3D ビーズプレビュー OpenGL
    ├── controller.py          # 変換スレッド管理（ConversionRunner）
    ├── models.py              # 変換パラメータモデル（ConversionRequest）
    ├── noise_filters.py       # ノイズ除去フィルタ
    ├── color_usage_window.py  # 色使用量ウィンドウ
    ├── color_usage_list.py    # 色リスト表示
    ├── color_usage_preview.py # 色使用量プレビュー
    ├── color_usage_service.py # 色使用量サービス層
    ├── scale_utils.py         # スライダークリックジャンプ
    ├── settings.json          # アプリ設定（永続化）
    └── preview_3d_state.json  # 3D プレビューウィンドウ状態
```

## 重要ルール
- UI からの操作は必ず `_schedule_on_ui()` 経由でスレッド安全に
- `ConversionRequest` は frozen dataclass、変更不可
- 指示された内容に関係のないフォルダやファイルの参照を禁止し、無駄な調査を減らし、常にトークン節約に努めること
- 修正や変更点があるたびに VERSION ファイル内のバージョン番号を適切な番号へ更新すること
