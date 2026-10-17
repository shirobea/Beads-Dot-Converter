# 作品イメージの素材と出典

## 同梱しているもの

| 素材 | 出典 | ライセンス |
|---|---|---|
| `vendor/three/`（three.js r170 本体と addons） | https://github.com/mrdoob/three.js | MIT（`vendor/three/LICENSE`） |
| `models/mouse.glb`（マウス） | 「Computer mouse」by zhe_kan https://sketchfab.com/3d-models/computer-mouse-6e7940d9e2144efeae3468a906f27e07 | CC BY 4.0（作者名とライセンスの表示が必要。作品イメージのパネルにも表示する） |
| `models/keyboard.glb`（テンキーなしキーボード） | 「Keyboard」by thethieme https://sketchfab.com/3d-models/keyboard-cc7e7ed38d9a469ea499893295bc767a | CC BY 4.0（作者名とライセンスの表示が必要。作品イメージのパネルにも表示する） |
| `models/pencil.glb`（鉛筆） | 「Pencil」by fredbear1211 https://sketchfab.com/3d-models/pencil-768e1a7e70ca4badb5cec0705c5a1b7c | CC BY 4.0（作者名とライセンスの表示が必要。作品イメージのパネルにも表示する） |
| `models/monitor.glb`（モニター） | 「LG moniter」by Mohamed Ahmed https://sketchfab.com/3d-models/lg-moniter-f899bb43701f4e95b1e44c4bf21ed29c （INNOCN 25M2S の幅・奥行に合わせて台座を変形） | CC BY 4.0（作者名とライセンスの表示が必要。作品イメージのパネルにも表示する） |
| `models/mug.glb`（マグカップ） | 「enamel cup」by schteppn https://sketchfab.com/3d-models/enamel-cup-32d5033eb7e7435696369ee300ad16e7 | CC BY 4.0（作者名とライセンスの表示が必要。作品イメージのパネルにも表示する） |
| `models/phone.glb`（スマートフォン） | 「Realistic Smartphone 3D Model」by LukeModels75 https://sketchfab.com/3d-models/realistic-smartphone-3d-model-77e5794dde144965b5bd4aeab9cb50e8 | CC BY 4.0（作者名とライセンスの表示が必要。作品イメージのパネルにも表示する） |
| デスクの脚・マット | `props.js` でコードから生成 | 本プロジェクトのコード |
| `hdri/lebombo_2k.hdr`（昼の窓際・昼白色の環境光） | https://polyhaven.com/a/lebombo （Greg Zaal） | CC0 |
| `hdri/wooden_lounge_2k.hdr`（電球色の環境光） | https://polyhaven.com/a/wooden_lounge （Greg Zaal） | CC0 |
| 色確認用の環境光 | three.js の `RoomEnvironment`（`vendor/three/`） | MIT |
| `textures/laminate_floor_02_*_1k.jpg`（部屋の床） | https://polyhaven.com/a/laminate_floor_02 （Dario Barresi, Charlotte Baglioni） | CC0 |
| `textures/white_plaster_02_*_1k.jpg`（壁・天井の凹凸とツヤ） | https://polyhaven.com/a/white_plaster_02 （Rob Tuytel） | CC0 |
| `textures/oak_veneer_01_*_2k.jpg`（机の天板・ドア） | https://polyhaven.com/a/oak_veneer_01 （Jenelle van Heerden） | CC0 |
| `textures/terlenka_*_1k.jpg`（カーテン） | https://polyhaven.com/a/terlenka （colormass, Rico Cilliers） | CC0 |
| 部屋のその他（窓の外の景色・巾木など） | `room.js` でコードから生成 | 本プロジェクトのコード |
| HDRI が読めないときの簡易環境光 | `lighting.js` でコードから生成 | 本プロジェクトのコード |

## 差し替え用の素材を置く場所

ファイルを置くだけで、次に作品イメージを開いたときから使われる（コード変更不要）。
置いた素材は必ず下の表に出典とライセンスを追記すること。CC0（Poly Haven など）を推奨。

### `models/` … 小物の glTF（`.glb` / `.gltf`）

ファイル名（拡張子を除く）で差し替える小物が決まる：
`keyboard` / `mouse` / `monitor` / `phone` / `mug` / `pencil`

読み込み時にバウンディングボックスの最大辺を実寸（`props.js` の `PROP_SIZES`）に合わせて
等倍スケールし、底面中心を原点にそろえる。モデルは Y 軸が上、使う人の側が +Z の向きで用意する。
向きが違うときは、同じ名前の `<名前>.json` に `{ "rotateY": 180 }` のように回転（度）を書く。
`rotateX` / `rotateZ` も使え、X → Z → Y の順にワールドの軸まわりに回す（寝かせる・裏返す → 水平に回す）。
実寸が `PROP_SIZES` と違うとき（テンキーなしキーボードなど）は `"size": { "w": 355, "d": 132, "h": 33 }`（mm）を書くと、
モデルの縮尺と机の上の配置の両方にその寸法が使われる。
縦横比が実物と違うときは `"deform"` で部品ごとに動かせる（`props.js` の `deformModel()`）。
`"raise": { "above": 60, "by": 74 }` は高さ 60mm より上を 74mm 持ち上げ（モニターの首を伸ばす）、
`"scaleBase": { "below": 40, "z": 1.26 }` は高さ 40mm より下を奥行方向に 1.26 倍にする（台座を広げる）。
glTF の `asset.extras` に作者・ライセンスがあれば、作品イメージのパネルに出典として表示する。
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
| `models/mouse.glb` | https://sketchfab.com/3d-models/computer-mouse-6e7940d9e2144efeae3468a906f27e07 | zhe_kan | CC BY 4.0 |
| `models/keyboard.glb` | https://sketchfab.com/3d-models/keyboard-cc7e7ed38d9a469ea499893295bc767a | thethieme | CC BY 4.0 |
| `models/pencil.glb` | https://sketchfab.com/3d-models/pencil-768e1a7e70ca4badb5cec0705c5a1b7c | fredbear1211 | CC BY 4.0 |
| `models/monitor.glb` | https://sketchfab.com/3d-models/lg-moniter-f899bb43701f4e95b1e44c4bf21ed29c | Mohamed Ahmed | CC BY 4.0 |
| `models/mug.glb` | https://sketchfab.com/3d-models/enamel-cup-32d5033eb7e7435696369ee300ad16e7 | schteppn | CC BY 4.0 |
| `models/phone.glb` | https://sketchfab.com/3d-models/realistic-smartphone-3d-model-77e5794dde144965b5bd4aeab9cb50e8 | LukeModels75 | CC BY 4.0 |
| `textures/laminate_floor_02_{diff,nor_gl,rough}_1k.jpg` | https://polyhaven.com/a/laminate_floor_02 | Dario Barresi, Charlotte Baglioni | CC0 |
| `textures/white_plaster_02_{nor_gl,rough}_1k.jpg` | https://polyhaven.com/a/white_plaster_02 | Rob Tuytel | CC0 |
| `textures/oak_veneer_01_{diff,nor_gl,rough}_2k.jpg` | https://polyhaven.com/a/oak_veneer_01 | Jenelle van Heerden | CC0 |
| `textures/terlenka_{diff,nor_gl,rough}_1k.jpg` | https://polyhaven.com/a/terlenka | colormass, Rico Cilliers | CC0 |
