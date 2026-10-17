# 作品イメージの素材と出典

## 同梱しているもの

| 素材 | 出典 | ライセンス |
|---|---|---|
| `vendor/three/`（three.js r170 本体と addons） | https://github.com/mrdoob/three.js | MIT（`vendor/three/LICENSE`） |
| デスク・マット・小物（キーボード、マウス、モニター、スマートフォン、マグカップ、鉛筆） | `props.js` でコードから生成 | 本プロジェクトのコード |
| `hdri/lebombo_2k.hdr`（昼の窓際・昼白色の環境光） | https://polyhaven.com/a/lebombo （Greg Zaal） | CC0 |
| `hdri/wooden_lounge_2k.hdr`（電球色の環境光） | https://polyhaven.com/a/wooden_lounge （Greg Zaal） | CC0 |
| 色確認用の環境光 | three.js の `RoomEnvironment`（`vendor/three/`） | MIT |
| HDRI が読めないときの簡易環境光 | `lighting.js` でコードから生成 | 本プロジェクトのコード |

## 差し替え用の素材を置く場所

ファイルを置くだけで、次に作品イメージを開いたときから使われる（コード変更不要）。
置いた素材は必ず下の表に出典とライセンスを追記すること。CC0（Poly Haven など）を推奨。

### `models/` … 小物の glTF（`.glb` / `.gltf`）

ファイル名（拡張子を除く）で差し替える小物が決まる：
`keyboard` / `mouse` / `monitor` / `phone` / `mug` / `pencil`

読み込み時にバウンディングボックスの最大辺を実寸（`props.js` の `PROP_SIZES`）に合わせて
等倍スケールし、底面中心を原点にそろえる。モデルは Y 軸が上、手前が +Z の向きで用意する。
サイズを抑えるため Draco 圧縮を検討する。

### `hdri/` … 環境光の HDRI（`.hdr`、2k 以下）

各プリセットの既定の HDRI は `lighting.js` の `PRESETS` に書いてある（上の表の2つ）。
ファイル名（拡張子を除く）が照明プリセットのキーと一致するファイルを置くと、既定より優先される：
`window`（昼の窓際） / `neutral`（昼白色） / `warm`（電球色）。`check`（色確認用）は常に `RoomEnvironment`。

- 「窓が1つある明るい部屋」のように、明るい光源がはっきり1か所にあるものを選ぶ
- 主光源（影を落とす DirectionalLight）は、HDRI の最も明るい方向に自動で合わせる（作品から見て左奥に来るよう HDRI を回す）
- HDRI 自体の色かぶりは読み込み時に取り除き、プリセットの色温度を掛け直す

## 追加した素材

| ファイル | 出典 URL | 作者 | ライセンス |
|---|---|---|---|
| `hdri/lebombo_2k.hdr` | https://polyhaven.com/a/lebombo | Greg Zaal | CC0 |
| `hdri/wooden_lounge_2k.hdr` | https://polyhaven.com/a/wooden_lounge | Greg Zaal | CC0 |
