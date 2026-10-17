// 照明プリセット：HDRI の環境光（映り込み・陰の明るさ）と、影を落とす主光源（窓の光）を組で切り替える。
// 主光源は HDRI の最も明るい方向（窓）に置き、影の向きと映り込みの窓の位置をそろえる。
import * as THREE from 'three';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// key: 主光源の強度、env: environmentIntensity、exposure: 露出（パネルの露出スライダーはこれに掛け算）
export const PRESETS = {
  window: { label: '昼の窓際', hdri: 'lebombo_2k.hdr', kelvin: 6000, key: 3.5, env: 0.5, exposure: 1.0 },
  neutral: { label: '昼白色', hdri: 'lebombo_2k.hdr', kelvin: 5000, key: 2.5, env: 0.6, exposure: 1.0 },
  warm: { label: '電球色', hdri: 'wooden_lounge_2k.hdr', kelvin: 2700, key: 2.0, env: 0.4, exposure: 1.1 },
  // 無彩色の RoomEnvironment と白い主光源。天板・マットも無彩色にして照り返しの色かぶりをなくす
  check: { label: '色確認用', room: true, kelvin: 6500, key: 2.0, env: 0.8, exposure: 1.0, neutralSurfaces: true },
};

export const DEFAULT_PRESET = 'neutral';

// 窓（主光源）を作品から見て左奥に置く。HDRI はこの方位に窓が来るよう Y 軸回りに回す。
// 手前からの視点で天板やスマホに窓が映り込み、影は手前右に落ちて見やすい。
// 真後ろ寄りにすると、窓が低いときにモニターの長い影が作品に掛かるので左に寄せている
const KEY_AZIMUTH = Math.atan2(-0.53, -0.85);
const MIN_ELEVATION = THREE.MathUtils.degToRad(15);
const CLAMPED_ELEVATION = THREE.MathUtils.degToRad(25);
const KEY_DISTANCE = 1500;

// 色温度 → sRGB（Tanner Helland の近似式）
function kelvinRaw(kelvin) {
  const t = kelvin / 100;
  let r, g, b;
  if (t <= 66) {
    r = 255;
    g = 99.4708025861 * Math.log(t) - 161.1195681661;
  } else {
    r = 329.698727446 * Math.pow(t - 60, -0.1332047592);
    g = 288.1221695283 * Math.pow(t - 60, -0.0755148492);
  }
  if (t >= 66) b = 255;
  else if (t <= 19) b = 0;
  else b = 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  return [r, g, b].map((v) => Math.min(255, Math.max(0, v)) / 255);
}

const D65 = kelvinRaw(6500);

// 表示の白（D65 ≒ 6500K）を基準に正規化し、6500K でちょうど (1, 1, 1) になるようにする
export function kelvinToColor(kelvin) {
  const raw = kelvinRaw(kelvin);
  const c = raw.map((v, i) => v / D65[i]);
  const m = Math.max(...c);
  return new THREE.Color().setRGB(c[0] / m, c[1] / m, c[2] / m, THREE.SRGBColorSpace);
}

// 目の色順応の度合い（0 = 光の色をそのまま表示、1 = 完全に白へ補正）。
// 人は照明の色にかなり順応するので、5000K の白い紙を見ても肌色には感じない（CIECAM02 の
// 標準的な観察条件で 0.9 前後）。照明ごとの違いは残したいので、それより少し弱めにする（von Kries 型）
const ADAPTATION = 0.8;

// 主光源と環境光の両方に掛ける光の色（リニア）。色温度はここで1回だけ掛ける
export function lightTint(kelvin) {
  const c = kelvinToColor(kelvin);
  const adapt = (v) => v / (1 - ADAPTATION + ADAPTATION * v);
  return new THREE.Color(adapt(c.r), adapt(c.g), adapt(c.b));
}

// HDRI 自体の色かぶりを取る：上向きの面（天板・ビーズ）に当たる環境光の色が無彩色になるよう
// 各チャンネルを合わせ（明るさは保つ）、そのうえでプリセットの光の色を掛ける。
// これで白いビーズに当たる環境光はちょうどプリセットの色になり、HDRI の色味と主光源の色温度が
// 二重にかからない。基準は上半球の照度（cos 重み）。単純平均だと暗い壁や床に、明るさ重みだと
// 窓だけに引っ張られて、実際に机を照らす光の色とずれる
function neutralizedCopy(hdr, tint) {
  const { width, height } = hdr.image;
  const src = hdr.image.data;
  const ch = src.length / (width * height);
  const sum = [0, 0, 0];
  for (let y = 0; y < height / 2; y++) {
    const elevation = (0.5 - (y + 0.5) / height) * Math.PI;
    // 照度の cos 重み × 正距円筒の画素の面積（仰角の cos）
    const w = Math.sin(elevation) * Math.cos(elevation);
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * ch;
      sum[0] += src[i] * w; sum[1] += src[i + 1] * w; sum[2] += src[i + 2] * w;
    }
  }
  const lum = 0.2126 * sum[0] + 0.7152 * sum[1] + 0.0722 * sum[2];
  const k = [lum / sum[0] * tint.r, lum / sum[1] * tint.g, lum / sum[2] * tint.b];
  const data = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += ch) {
    data[i] = src[i] * k[0];
    data[i + 1] = src[i + 1] * k[1];
    data[i + 2] = src[i + 2] * k[2];
    if (ch === 4) data[i + 3] = 1;
  }
  const tex = new THREE.DataTexture(data, hdr.image.width, hdr.image.height, THREE.RGBAFormat, THREE.FloatType);
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.flipY = hdr.flipY;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

// ---- 環境光の解析 -----------------------------------------------------------

// 正距円筒の HDR から最も明るい方向を求める（1/8 に縮小して走査）。
// three.js の equirect と同じ対応：u → 方位 atan2(z, x)、v → 仰角（データの先頭行が真上）
function brightestDirection(hdr) {
  const { data, width, height } = hdr.image;
  const step = 8;
  const ch = data.length / (width * height);
  let best = -1, bu = 0, bv = 0;
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      let lum = 0;
      for (let dy = 0; dy < step; dy++) {
        for (let dx = 0; dx < step; dx++) {
          const i = ((y + dy) * width + (x + dx)) * ch;
          lum += 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
        }
      }
      if (lum > best) { best = lum; bu = (x + step / 2) / width; bv = (y + step / 2) / height; }
    }
  }
  const azimuth = (bu - 0.5) * Math.PI * 2;
  const elevation = (0.5 - bv) * Math.PI;
  return { azimuth, elevation };
}

// シーンで作る環境光（RoomEnvironment など）は、最も明るい発光面の方向を使う
function brightestSceneLight(envScene) {
  let best = -1;
  const dir = new THREE.Vector3(0, 1, 0);
  envScene.updateMatrixWorld(true);
  envScene.traverse((o) => {
    if (!o.isMesh || !o.material?.color) return;
    const c = o.material.color;
    const lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    if (lum > best) { best = lum; o.getWorldPosition(dir); }
  });
  const n = dir.clone().normalize();
  return { azimuth: Math.atan2(n.z, n.x), elevation: Math.asin(THREE.MathUtils.clamp(n.y, -1, 1)) };
}

function directionFrom(azimuth, elevation) {
  const ce = Math.cos(elevation);
  return new THREE.Vector3(Math.cos(azimuth) * ce, Math.sin(elevation), Math.sin(azimuth) * ce);
}

// ---- HDRI が読めないときの簡易ルーム -----------------------------------------

const FALLBACK_ROOM = {
  wall: 0.42,
  window: { pos: [-4.9, 1.2, 0.6], rot: [0, Math.PI / 2, 0], size: [4.2, 2.6], power: 9 },
};

function basic(level) {
  const m = new THREE.MeshBasicMaterial();
  m.color.setScalar(level);
  return m;
}

function buildFallbackRoom() {
  const scene = new THREE.Scene();
  const wall = FALLBACK_ROOM.wall;
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(10, 10, 10), Object.assign(basic(wall), { side: THREE.BackSide })));
  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), basic(wall * 1.2));
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.y = 4.94;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), basic(wall * 0.28));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -4.94;
  scene.add(ceiling, floor);
  const w = FALLBACK_ROOM.window;
  const pane = new THREE.Mesh(new THREE.PlaneGeometry(...w.size), Object.assign(basic(w.power), { side: THREE.DoubleSide }));
  pane.position.set(...w.pos);
  pane.rotation.set(...w.rot);
  // 窓の桟：明るい面に暗い線が入ると、ツヤのある物に「窓が映っている」ように見える
  for (const [bw, bh, x, y] of [[0.07, w.size[1], 0, 0], [w.size[0], 0.07, 0, w.size[1] * 0.15]]) {
    const bar = new THREE.Mesh(new THREE.PlaneGeometry(bw, bh), Object.assign(basic(wall * 0.35), { side: THREE.DoubleSide }));
    bar.position.set(x, y, 0.02);
    pane.add(bar);
  }
  scene.add(pane);
  const shelf = new THREE.Mesh(new THREE.BoxGeometry(3.2, 4.2, 0.6), basic(wall * 0.18));
  shelf.position.set(2.4, -2.8, -4.65);
  scene.add(shelf);
  return scene;
}

// ---- 照明本体 ---------------------------------------------------------------

export class Lighting {
  constructor(renderer, scene, hdriFiles = []) {
    this.renderer = renderer;
    this.scene = scene;
    this.pmrem = new THREE.PMREMGenerator(renderer);
    // assets/hdri/<プリセット名>.hdr があれば、プリセット既定の HDRI より優先する
    this.files = new Set(hdriFiles);
    this.cache = new Map();
    this.hdrFiles = new Map(); // ファイル名 → 読み込み済み HDR（プリセット間で使い回す）
    this.key = new THREE.DirectionalLight(0xffffff, 3);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(4096, 4096);
    this.key.shadow.bias = -0.0002;
    this.key.shadow.normalBias = 0.5;
    scene.add(this.key, this.key.target);
    this.focus = new THREE.Vector3();
    this.bounds = new THREE.Box3(new THREE.Vector3(-700, -30, -350), new THREE.Vector3(700, 450, 350));
    this.direction = new THREE.Vector3(-0.6, 0.6, 0.4).normalize();
    this.presetKey = null;
    this.helper = null;
  }

  // 影の範囲：デスク全体（小物の高さ込み）がシャドウカメラに収まるように合わせる
  setFocus(center, bounds) {
    this.focus.copy(center);
    this.bounds.copy(bounds);
    this._placeKey();
  }

  setShadowMapSize(size) {
    if (this.key.shadow.mapSize.x === size) return;
    this.key.shadow.mapSize.set(size, size);
    this.key.shadow.map?.dispose();
    this.key.shadow.map = null;
    this.key.shadow.needsUpdate = true;
  }

  // 開発用：シャドウカメラの範囲を表示する
  showShadowCamera(on) {
    if (on && !this.helper) {
      this.helper = new THREE.CameraHelper(this.key.shadow.camera);
      this.scene.add(this.helper);
    } else if (!on && this.helper) {
      this.scene.remove(this.helper);
      this.helper.dispose();
      this.helper = null;
    }
    this.helper?.update();
  }

  _placeKey() {
    const key = this.key;
    key.position.copy(this.focus).addScaledVector(this.direction, KEY_DISTANCE);
    key.target.position.copy(this.focus);
    key.updateMatrixWorld();
    key.target.updateMatrixWorld();
    // 光の向きから見たデスクの 8 隅を囲む最小の直方体を影カメラにする
    const cam = key.shadow.camera;
    cam.position.copy(key.position);
    cam.lookAt(key.target.position);
    cam.updateMatrixWorld();
    const inv = cam.matrixWorldInverse;
    const b = this.bounds;
    const box = new THREE.Box3();
    for (const x of [b.min.x, b.max.x]) for (const y of [b.min.y, b.max.y]) for (const z of [b.min.z, b.max.z]) {
      box.expandByPoint(new THREE.Vector3(x, y, z).applyMatrix4(inv));
    }
    const margin = 20;
    cam.left = box.min.x - margin; cam.right = box.max.x + margin;
    cam.bottom = box.min.y - margin; cam.top = box.max.y + margin;
    cam.near = Math.max(1, -box.max.z - margin); cam.far = -box.min.z + margin;
    cam.updateProjectionMatrix();
    key.shadow.needsUpdate = true;
    this.helper?.update();
  }

  // プリセットを適用し、プリセット定義を返す（露出や無彩色化は呼び出し側で反映）
  async apply(presetKey) {
    const id = PRESETS[presetKey] ? presetKey : DEFAULT_PRESET;
    const preset = PRESETS[id];
    this.presetKey = id;
    const env = await this._envFor(id, preset);
    const scene = this.scene;
    scene.environment = env.texture;
    scene.environmentIntensity = preset.env;
    scene.background = env.texture;
    scene.backgroundBlurriness = 0.35;
    scene.backgroundIntensity = 0.8;
    scene.environmentRotation.set(0, env.rotation, 0);
    scene.backgroundRotation.set(0, env.rotation, 0);
    this.key.color.copy(lightTint(preset.kelvin));
    this.key.intensity = preset.key;
    this.direction.copy(env.direction);
    this._placeKey();
    return preset;
  }

  async _envFor(id, preset) {
    if (this.cache.has(id)) return this.cache.get(id);
    let result = null;
    if (!preset.room) {
      const file = this.files.has(`${id}.hdr`) ? `${id}.hdr` : preset.hdri;
      try {
        if (!this.hdrFiles.has(file)) {
          this.hdrFiles.set(file, new RGBELoader().setDataType(THREE.FloatType).loadAsync(`assets/hdri/${file}`));
        }
        const hdr = await this.hdrFiles.get(file);
        const tinted = neutralizedCopy(hdr, lightTint(preset.kelvin));
        result = this._fromLight(this.pmrem.fromEquirectangular(tinted).texture, brightestDirection(hdr));
        tinted.dispose();
      } catch (err) {
        console.warn('HDRI の読み込みに失敗したため簡易環境光を使います', err);
      }
    }
    if (!result) {
      const room = preset.room ? new RoomEnvironment() : buildFallbackRoom();
      const light = brightestSceneLight(room);
      result = this._fromLight(this.pmrem.fromScene(room, 0.04).texture, light);
      room.traverse((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
    }
    this.cache.set(id, result);
    return result;
  }

  // 最も明るい方向が KEY_AZIMUTH に来るよう環境を回し、主光源の向きを決める。
  // 仰角が低すぎると影が長くなりすぎるので 15° 未満は 25° に起こす
  _fromLight(texture, { azimuth, elevation }) {
    const vertical = Math.abs(elevation) > THREE.MathUtils.degToRad(80);
    // 真上の光は方位に意味がないので回さない
    const rotation = vertical ? 0 : azimuth - KEY_AZIMUTH;
    const dir = directionFrom(azimuth, elevation).applyEuler(new THREE.Euler(0, rotation, 0));
    let elev = Math.asin(THREE.MathUtils.clamp(dir.y, -1, 1));
    if (elev < MIN_ELEVATION) elev = CLAMPED_ELEVATION;
    const az = Math.atan2(dir.z, dir.x);
    return { texture, rotation, direction: directionFrom(az, elev), source: { azimuth, elevation } };
  }
}
