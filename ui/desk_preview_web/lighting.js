// 照明プリセット：環境光（PMREM）と太陽光（影を落とす DirectionalLight）をまとめて切り替える。
// 環境光は HDRI があればそれを使い、無ければ色温度で染めた簡易ルームから生成する。
import * as THREE from 'three';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';

export const PRESETS = {
  window: {
    label: '昼の窓際', kelvin: 6000, envIntensity: 1.0,
    // 左手前上の窓から差す自然光。影はやや硬め
    sun: { dir: [-0.75, 1.0, 0.45], intensity: 2.6, radius: 3 },
    room: { wall: 0.42, panels: [{ pos: [-4.9, 1.2, 0.6], rot: [0, Math.PI / 2, 0], size: [4.2, 2.6], power: 9, window: true }] },
  },
  neutral: {
    label: '昼白色', kelvin: 5000, envIntensity: 1.0,
    sun: { dir: [0.25, 1.0, 0.35], intensity: 1.7, radius: 6 },
    room: {
      wall: 0.45, panels: [
        { pos: [-1.6, 4.9, 0], rot: [Math.PI / 2, 0, 0], size: [1.2, 3.2], power: 7 },
        { pos: [1.6, 4.9, 0], rot: [Math.PI / 2, 0, 0], size: [1.2, 3.2], power: 7 },
      ],
    },
  },
  warm: {
    label: '電球色', kelvin: 2700, envIntensity: 0.75,
    sun: { dir: [0.55, 1.0, 0.2], intensity: 1.3, radius: 5 },
    room: {
      wall: 0.22, panels: [
        { pos: [1.0, 4.9, -0.5], rot: [Math.PI / 2, 0, 0], size: [1.4, 1.4], power: 9 },
      ],
    },
  },
  check: {
    label: '色確認用', kelvin: 6500, envIntensity: 1.0,
    // ライトブース相当：周囲から均一に照らし、影はほぼ出さない
    sun: { dir: [0.0, 1.0, 0.15], intensity: 0.55, radius: 12 },
    room: { wall: 1.0, furnished: false, panels: [{ pos: [0, 4.9, 0], rot: [Math.PI / 2, 0, 0], size: [6, 6], power: 1.6 }] },
  },
};

export const DEFAULT_PRESET = 'neutral';

// 色温度 → sRGB（Tanner Helland の近似式）。明るさは最大成分を1にそろえる
export function kelvinToColor(kelvin) {
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
  const c = [r, g, b].map((v) => Math.min(255, Math.max(0, v)) / 255);
  const m = Math.max(...c);
  return new THREE.Color().setRGB(c[0] / m, c[1] / m, c[2] / m, THREE.SRGBColorSpace);
}

function buildRoomScene(preset, tint) {
  const scene = new THREE.Scene();
  const wallMat = new THREE.MeshBasicMaterial({ side: THREE.BackSide });
  wallMat.color.copy(tint).multiplyScalar(preset.room.wall);
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(10, 10, 10), wallMat));
  // 床は少し暗く、色ムラの照り返しを抑える
  const floorMat = new THREE.MeshBasicMaterial();
  floorMat.color.copy(tint).multiplyScalar(preset.room.wall * 0.55);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -4.95;
  scene.add(floor);
  for (const p of preset.room.panels) {
    const mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    mat.color.copy(tint).multiplyScalar(p.power);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(p.size[0], p.size[1]), mat);
    mesh.position.set(...p.pos);
    mesh.rotation.set(...p.rot);
    scene.add(mesh);
    if (p.window) addMullions(scene, mesh, p.size, tint, preset.room.wall);
  }
  if (preset.room.furnished !== false) furnish(scene, tint, preset.room.wall);
  return scene;
}

function basic(tint, level) {
  const m = new THREE.MeshBasicMaterial();
  m.color.copy(tint).multiplyScalar(level);
  return m;
}

// 窓の桟：明るい面の中に暗い線が入ると、ツヤのある物に「窓が映っている」ように見える
function addMullions(scene, panel, size, tint, wall) {
  const mat = basic(tint, wall * 0.35);
  const [w, h] = size;
  const bars = [
    [0.07, h, 0, 0], [0.07, h, -w / 4, 0], [0.07, h, w / 4, 0], [w, 0.07, 0, h * 0.15],
  ];
  for (const [bw, bh, x, y] of bars) {
    const bar = new THREE.Mesh(new THREE.PlaneGeometry(bw, bh), mat);
    bar.position.set(x, y, 0.02);
    panel.add(bar);
    const back = bar.clone();
    back.position.z = -0.02;
    panel.add(back);
  }
}

// 部屋の家具・天井・床の濃淡。映り込みに変化を付けるための大まかな形だけ
function furnish(scene, tint, wall) {
  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), basic(tint, wall * 1.2));
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.y = 4.94;
  scene.add(ceiling);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), basic(tint, wall * 0.28));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -4.94;
  scene.add(floor);
  // 奥の壁の本棚（暗い枠に明るさの違う背表紙）
  const shelf = new THREE.Mesh(new THREE.BoxGeometry(3.2, 4.2, 0.6), basic(tint, wall * 0.18));
  shelf.position.set(2.4, -2.8, -4.65);
  scene.add(shelf);
  for (let row = 0; row < 3; row++) {
    for (let i = 0; i < 9; i++) {
      const level = wall * (0.25 + ((i * 7 + row * 3) % 5) * 0.12);
      const book = new THREE.Mesh(new THREE.PlaneGeometry(0.28, 0.9 + ((i + row) % 3) * 0.12), basic(tint, level));
      book.position.set(1.1 + i * 0.32, -4.25 + row * 1.3, -4.34);
      scene.add(book);
    }
  }
  // 右の壁のドアと、奥の壁の額
  const door = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 2.3), basic(tint, wall * 0.42));
  door.rotation.y = -Math.PI / 2;
  door.position.set(4.93, -2.4, 2.5);
  scene.add(door);
  const frame = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.0), basic(tint, wall * 0.15));
  frame.position.set(-1.8, 0.8, -4.93);
  scene.add(frame);
  const picture = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 0.8), basic(tint, wall * 0.7));
  picture.position.set(-1.8, 0.8, -4.92);
  scene.add(picture);
}

export class Lighting {
  constructor(renderer, scene, hdriFiles = []) {
    this.renderer = renderer;
    this.scene = scene;
    this.pmrem = new THREE.PMREMGenerator(renderer);
    this.hdri = new Map(hdriFiles.map((f) => [f.replace(/\.[^.]+$/, ''), f]));
    this.cache = new Map();
    this.sun = new THREE.DirectionalLight(0xffffff, 1);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(4096, 4096);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.6;
    scene.add(this.sun, this.sun.target);
    this.focus = new THREE.Vector3();
    this.extent = 800;
    this.key = null;
  }

  // 影カメラを作品＋デスクの範囲に合わせる。範囲が狭いほど影が細かくなる
  fitShadow(center, extent) {
    this.focus.copy(center);
    this.extent = extent;
    this._placeSun();
  }

  _placeSun() {
    const preset = PRESETS[this.key] || PRESETS[DEFAULT_PRESET];
    const dir = new THREE.Vector3(...preset.sun.dir).normalize();
    this.sun.position.copy(this.focus).addScaledVector(dir, this.extent * 2.5);
    this.sun.target.position.copy(this.focus);
    const cam = this.sun.shadow.camera;
    const half = this.extent * 0.75;
    cam.left = -half; cam.right = half; cam.top = half; cam.bottom = -half;
    cam.near = 10; cam.far = this.extent * 6;
    cam.updateProjectionMatrix();
    this.sun.shadow.needsUpdate = true;
  }

  async apply(key) {
    const preset = PRESETS[key] || PRESETS[DEFAULT_PRESET];
    this.key = PRESETS[key] ? key : DEFAULT_PRESET;
    const tint = kelvinToColor(preset.kelvin);
    this.sun.color.copy(tint);
    this.sun.intensity = preset.sun.intensity;
    this.sun.shadow.radius = preset.sun.radius;
    this._placeSun();
    this.scene.environment = await this._envFor(this.key, preset, tint);
    this.scene.environmentIntensity = preset.envIntensity;
  }

  async _envFor(key, preset, tint) {
    if (this.cache.has(key)) return this.cache.get(key);
    let tex = null;
    const file = this.hdri.get(key);
    if (file) {
      try {
        const hdr = await new RGBELoader().loadAsync(`assets/hdri/${file}`);
        hdr.mapping = THREE.EquirectangularReflectionMapping;
        tex = this.pmrem.fromEquirectangular(hdr).texture;
        hdr.dispose();
      } catch (err) {
        console.warn('HDRI の読み込みに失敗したため簡易環境光を使います', err);
      }
    }
    if (!tex) {
      const room = buildRoomScene(preset, tint);
      tex = this.pmrem.fromScene(room, 0.03).texture;
      room.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
    }
    this.cache.set(key, tex);
    return tex;
  }
}
