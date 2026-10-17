// 作品イメージ：実寸のデスクに作品を置いて描画する。単位は 1 = 1mm。
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { buildArtwork, IRONING } from './beads.js';
import { Lighting, PRESETS, DEFAULT_PRESET } from './lighting.js';
import { buildDesk, buildMat, buildProps, modelCredits, PROP_SIZES } from './props.js';
import { buildRoom, loadRoomTextures } from './room.js';

const DEFAULTS = {
  lighting: DEFAULT_PRESET,
  exposure: 1.0,
  ironing: 'both_sides',
  placement: 'flat',
  view: 'oblique',
  pitch: 'app',
  dims: true,
  ao: true,
  props: true,
  paper: false,
};
const STAND_TILT = THREE.MathUtils.degToRad(12);

const $ = (id) => document.getElementById(id);
const api = () => window.pywebview?.api;

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
// 色相のずれが少ないトーンマップ（色確認向き）
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
$('view').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1d1f22);
const camera = new THREE.PerspectiveCamera(38, 1, 5, 20000);
// 寸法線は AO の法線パスに混ざらないよう別シーンにして最後に重ねる
const overlay = new THREE.Scene();

// 接地・すき間の陰影（GTAO）。単位が mm なので半径も mm で指定する
const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));
composer.addPass(new RenderPass(scene, camera));
const gtao = new GTAOPass(scene, camera, 1, 1);
gtao.output = GTAOPass.OUTPUT.Default;
// 影の主役はシャドウマップ。AO は小物の下・キーや粒のすき間に薄く効く程度（半径 6mm）
gtao.blendIntensity = 0.5;
gtao.updateGtaoMaterial({ radius: 6, distanceExponent: 1, thickness: 2, scale: 1.0, samples: 16, distanceFallOff: 1, screenSpaceRadius: false });
gtao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, rings: 2, samples: 16 });
composer.addPass(gtao);
composer.addPass(new OutputPass());

function renderFrame() {
  if (settings.ao) composer.render();
  else renderer.render(scene, camera);
  renderer.autoClear = false;
  renderer.render(overlay, camera);
  renderer.autoClear = true;
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.minDistance = 25;
controls.maxDistance = 6000;
controls.maxPolarAngle = THREE.MathUtils.degToRad(88);
controls.screenSpacePanning = true;

let lighting = null;
let props = null;
let data = null;
let settings = { ...DEFAULTS };
let world = null; // デスク・作品など、シーン再構築のたびに作り直す部分
let layout = null;
let dirty = true;

// ---- データ受け渡し ---------------------------------------------------------

function demoScene() {
  // pywebview 外（ブラウザ単体）で開いたとき用の見本データ。?demo=100 で大きさを変えられる（負荷確認用）
  const n = Math.max(5, Math.min(400, parseInt(new URLSearchParams(location.search).get('demo') || '29', 10) || 29));
  const palette = [
    { id: '#001', name: 'ホワイト', rgb: [243, 250, 246] },
    { id: '#005', name: 'レッド', rgb: [196, 38, 52] },
    { id: '#006', name: 'ピンク', rgb: [236, 128, 160] },
    { id: '#018', name: 'ブラック', rgb: [38, 38, 42] },
  ];
  const cells = [];
  for (let r = 0; r < n; r++) {
    const row = [];
    for (let c = 0; c < n; c++) {
      const x = (c - n / 2 + 0.5) / (n * 0.31), y = -(r - n / 2 - 0.5) / (n * 0.31);
      const heart = Math.pow(x * x + y * y - 1, 3) - x * x * y * y * y;
      const edge = heart < 0 && heart > -0.06;
      row.push(heart < 0 ? (edge ? 3 : (x < -0.2 && y > 0.3 ? 2 : 1)) : 0);
    }
    cells.push(row);
  }
  // ?models=mouse.glb のように、置いてある小物モデルを指定して確認できる（ブラウザ単体ではフォルダを列挙できないため）
  const models = (new URLSearchParams(location.search).get('models') || '').split(',').filter(Boolean);
  return { grid: { width: n, height: n, cells }, palette, bead_pitch_mm: 2.6, ironing: 'both_sides', models, hdri: [], demo: true };
}

async function waitForApi(timeout = 1500) {
  if (api()) return api();
  return new Promise((resolve) => {
    const done = () => resolve(api() || null);
    window.addEventListener('pywebviewready', done, { once: true });
    setTimeout(done, timeout);
  });
}

async function fetchScene() {
  const a = api();
  return a ? await a.get_scene() : demoScene();
}

let saveTimer = 0;
function saveSettings() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => api()?.save_settings(settings), 400);
}

// ---- レイアウト ------------------------------------------------------------

function pitchMm() {
  return settings.pitch === 'app' ? data.bead_pitch_mm : parseFloat(settings.pitch);
}

function rectOverlap(a, b, margin) {
  return Math.abs(a.x - b.x) * 2 < a.w + b.w + margin * 2 && Math.abs(a.z - b.z) * 2 < a.d + b.d + margin * 2;
}

// 作品の大きさからデスク寸法と小物の位置を決める。
// 小物は候補位置を順に試し、作品や他の小物と重なる／デスクからはみ出す候補は捨てる。
function computeLayout(artW, artD, artH) {
  const flat = settings.placement === 'flat';
  const footD = flat ? artD : 130;
  // デスクは幅 90cm × 奥行 60cm。作品が大きくてマットが収まらないときだけ広げる
  const matW = Math.max(600, artW + 80);
  const matD = Math.max(330, footD + 80);
  const W = Math.max(900, matW + 80);
  const D = Math.max(600, matD + 260);
  const matZ = D / 2 - 15 - matD / 2;
  const art = { x: 0, z: flat ? matZ : D / 2 - 30 - 40 - footD / 2, w: artW + 10, d: footD };
  const placed = { art };
  const fits = (r) => Math.abs(r.x) + r.w / 2 <= W / 2 - 15 && Math.abs(r.z) + r.d / 2 <= D / 2 - 10
    && !Object.values(placed).some((o) => rectOverlap(r, o, 14));
  const aL = art.x - art.w / 2, aR = art.x + art.w / 2;
  // 白の基準にする A6 の紙。作品の右横 20mm に平置きし、小物はこれも避けて置く。
  // 奥の端を作品にそろえ、奥のキーボードを押し出さないようにする
  let paper = null;
  if (settings.paper) {
    const back = art.z - (flat ? artD : footD) / 2;
    paper = { x: art.x + artW / 2 + 20 + PAPER.w / 2, z: Math.max(back + PAPER.d / 2, art.z), w: PAPER.w, d: PAPER.d };
    // 作品との間は 20mm と決めているので、ここではデスクからはみ出さないかだけを見る
    const onDesk = Math.abs(paper.x) + paper.w / 2 <= W / 2 - 15 && Math.abs(paper.z) + paper.d / 2 <= D / 2 - 10;
    if (onDesk) placed.paper = paper;
    else paper = null;
  }
  const candidates = {
    monitor: () => [[0, -D / 2 + 110]],
    // モニターのすぐ手前（奥行 60cm の机でも作品を置く余地を残す）
    keyboard: () => [[0, (placed.monitor ? -D / 2 + 225 : -D / 2 + 20) + 68], [aR + 250, D / 2 - 100], [aL - 250, D / 2 - 100]],
    mouse: () => {
      const kb = placed.keyboard;
      const list = kb ? [[kb.x + kb.w / 2 + 70, kb.z], [kb.x - kb.w / 2 - 70, kb.z]] : [];
      return list.concat([[aR + 80, D / 2 - 110], [aL - 80, D / 2 - 110]]);
    },
    mug: () => [[W / 2 - 75, -D / 2 + 110], [-W / 2 + 75, -D / 2 + 110], [W / 2 - 75, 0], [-W / 2 + 75, 0]],
    // 作品の横に並べる。奥のキーボードに当たるときは手前の縁へ寄せる
    phone: () => [[aL - 75, art.z], [aL - 75, D / 2 - 100], [aR + 75, D / 2 - 100], [-W / 2 + 120, D / 2 - 120]],
    pencil: () => [[aR + 30, art.z], [aR + 30, D / 2 - 112], [aL - 30, D / 2 - 112], [W / 2 - 60, D / 2 - 130]],
  };
  const props = {};
  for (const name of ['monitor', 'keyboard', 'mouse', 'mug', 'phone', 'pencil']) {
    const size = PROP_SIZES[name];
    for (const [x, z] of candidates[name]()) {
      const r = { x, z, w: size.w, d: size.d };
      if (fits(r)) { placed[name] = r; props[name] = r; break; }
    }
  }
  return { W, D, matW, matD, matZ, art, artW, artD, artH, flat, props, paper };
}

// 白の基準紙：A6（105 × 148 × 0.1mm）、sRGB (243, 243, 240)
const PAPER = { w: 105, d: 148, h: 0.1 };
function buildPaper() {
  const m = new THREE.Mesh(
    new THREE.BoxGeometry(PAPER.w, PAPER.h, PAPER.d),
    new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(243 / 255, 243 / 255, 240 / 255, THREE.SRGBColorSpace), roughness: 0.9 }),
  );
  m.receiveShadow = true;
  return m;
}

// 色確認用プリセットでは天板とマットを無彩色のグレー（sRGB 118 ≒ 反射率 18%）にして照り返しの色かぶりをなくす
const NEUTRAL_GRAY = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(118 / 255, 118 / 255, 118 / 255, THREE.SRGBColorSpace), roughness: 0.8 });
function applySurfaceMode() {
  if (!world) return;
  const neutral = !!activePreset?.neutralSurfaces;
  for (const mesh of world.surfaces) {
    mesh.userData.original ??= mesh.material;
    mesh.material = neutral ? NEUTRAL_GRAY : mesh.userData.original;
  }
  dirty = true;
}

let activePreset = null;
function applyExposure() {
  renderer.toneMappingExposure = (activePreset?.exposure ?? 1) * settings.exposure;
  dirty = true;
}

// 置いた感じを出すための小さな回転（重なり判定の余白に収まる程度）
const PROP_YAW = { mouse: -0.12, phone: 0.1, pencil: -0.07, mug: 0.7 };
function propSurface(L, r) {
  return Math.abs(r.z - L.matZ) < L.matD / 2 && Math.abs(r.x) < L.matW / 2 ? 2 : 0;
}

// ---- 寸法線 ----------------------------------------------------------------

// ラベルはズームしても読める大きさのまま（画面に対して一定サイズ）
function labelSprite(text) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const font = 'bold 44px "Segoe UI", "Yu Gothic UI", sans-serif';
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 36;
  canvas.width = w; canvas.height = 64;
  ctx.font = font;
  ctx.fillStyle = 'rgba(20,22,26,0.82)';
  ctx.beginPath();
  ctx.roundRect(0, 0, w, 64, 14);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, w / 2, 34);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, toneMapped: false, sizeAttenuation: false }));
  const h = 0.02;
  sprite.scale.set(h * (w / 64), h, 1);
  sprite.renderOrder = 10;
  return sprite;
}

function dimensionLine(a, b, text, labelSize, offsetDir) {
  const g = new THREE.Group();
  const lineMat = new THREE.LineBasicMaterial({ color: 0xffd25a, depthTest: false, toneMapped: false });
  const tick = offsetDir.clone().multiplyScalar(labelSize * 0.35);
  const pts = [a.clone().add(tick), a, a, b, b, b.clone().add(tick)];
  const line = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lineMat);
  line.renderOrder = 9;
  g.add(line);
  const label = labelSprite(text);
  label.position.copy(a).add(b).multiplyScalar(0.5).addScaledVector(offsetDir, labelSize * 0.6);
  g.add(label);
  return g;
}

// 縦横で単位がばらつかないよう、作品の長辺で cm / mm を決める
let unitCm = false;
const fmt = (mm) => (unitCm ? `${(mm / 10).toFixed(1)} cm` : `${mm.toFixed(1)} mm`);

function buildDimensions(L, artGroup) {
  const g = new THREE.Group();
  const { artW, artD, artH } = L;
  const size = THREE.MathUtils.clamp(Math.max(artW, artD) * 0.09, 9, 60);
  const gap = size * 0.6;
  if (L.flat) {
    const y = artH + 1;
    const front = artD / 2 + gap;
    const right = artW / 2 + gap;
    g.add(dimensionLine(new THREE.Vector3(-artW / 2, y, front), new THREE.Vector3(artW / 2, y, front), fmt(artW), size, new THREE.Vector3(0, 0, 1)));
    g.add(dimensionLine(new THREE.Vector3(right, y, -artD / 2), new THREE.Vector3(right, y, artD / 2), fmt(artD), size, new THREE.Vector3(1, 0, 0)));
    g.position.copy(artGroup.position);
  } else {
    // 立てた作品は作品と同じ傾きの平面上に寸法を描く
    const front = artH + 2;
    g.add(dimensionLine(new THREE.Vector3(-artW / 2, front, artD / 2 + gap), new THREE.Vector3(artW / 2, front, artD / 2 + gap), fmt(artW), size, new THREE.Vector3(0, 0, 1)));
    g.add(dimensionLine(new THREE.Vector3(artW / 2 + gap, front, -artD / 2), new THREE.Vector3(artW / 2 + gap, front, artD / 2), fmt(artD), size, new THREE.Vector3(1, 0, 0)));
    g.position.copy(artGroup.position);
    g.quaternion.copy(artGroup.quaternion);
  }
  return g;
}

// ---- スタンド ---------------------------------------------------------------

// 作品の背面に沿う背板＋手前の受け。作品下端の背面は台の上 (y=4, z=22) に来る
function buildStand(artW, artD) {
  const g = new THREE.Group();
  const acrylic = new THREE.MeshPhysicalMaterial({
    color: 0xffffff, roughness: 0.05, metalness: 0, transparent: true, opacity: 0.28, clearcoat: 1,
  });
  const w = Math.max(60, artW * 0.55);
  const base = new THREE.Mesh(new THREE.BoxGeometry(w, 4, 90), acrylic);
  base.position.set(0, 2, -15);
  const lip = new THREE.Mesh(new THREE.BoxGeometry(w, 9, 3), acrylic);
  lip.position.set(0, 4 + 4.5, 26);
  const hb = Math.min(artD * 0.7, 110);
  const back = new THREE.Mesh(new THREE.BoxGeometry(w * 0.5, hb, 3), acrylic);
  back.position.set(0, 4 + Math.cos(STAND_TILT) * hb / 2, 22 - Math.sin(STAND_TILT) * hb / 2 - 2);
  back.rotation.x = -STAND_TILT;
  for (const m of [base, lip, back]) { m.castShadow = true; m.receiveShadow = true; g.add(m); }
  return g;
}

// ---- シーン構築 ------------------------------------------------------------

const TEXTURE_SLOTS = ['map', 'bumpMap', 'normalMap', 'roughnessMap', 'emissiveMap', 'aoMap'];

function disposeWorld() {
  if (!world) return;
  scene.remove(world.root);
  overlay.remove(world.dims);
  world.dims.traverse((o) => { o.geometry?.dispose?.(); o.material?.map?.dispose?.(); o.material?.dispose?.(); });
  // 共有のグレー材質を破棄しないよう、元の材質に戻してから片付ける
  for (const mesh of world.surfaces) if (mesh.userData.original) mesh.material = mesh.userData.original;
  // 小物は使い回すので、破棄の前に外しておく
  if (props) Object.values(props).forEach((p) => p.parent?.remove(p));
  world.artGroup.userData.dispose?.();
  world.root.traverse((o) => {
    if (o.isInstancedMesh && o.parent === world.artGroup) return;
    o.geometry?.dispose?.();
    if (o.material) {
      for (const slot of TEXTURE_SLOTS) {
        const t = o.material[slot];
        if (t && !t.userData.shared) t.dispose();
      }
      o.material.dispose?.();
    }
  });
  world = null;
}

function rebuild({ resetCamera = false } = {}) {
  if (!data) return;
  const prev = layout;
  disposeWorld();
  const pitch = pitchMm();
  const artGroup = buildArtwork(data, settings.ironing, pitch);
  const { widthMm: artW, depthMm: artD, heightMm: artH, count } = artGroup.userData;
  unitCm = Math.max(artW, artD) >= 100;
  const L = computeLayout(artW, artD, artH);
  layout = L;

  const root = new THREE.Group();
  const desk = buildDesk(L.W, L.D);
  root.add(desk);
  const matMesh = buildMat(L.matW, L.matD);
  matMesh.position.z = L.matZ;
  root.add(matMesh);
  const surface = 2; // マットの厚み
  if (L.paper) {
    const paper = buildPaper();
    paper.position.set(L.paper.x, propSurface(L, L.paper) + PAPER.h / 2, L.paper.z);
    root.add(paper);
  }

  if (L.flat) {
    artGroup.position.set(L.art.x, surface, L.art.z);
  } else {
    // 画像の上側（-Z）が上を向くよう X 軸回りに起こし、少し後ろへ傾ける
    const stand = buildStand(artW, artD);
    stand.position.set(L.art.x, surface, L.art.z);
    root.add(stand);
    artGroup.rotation.x = Math.PI / 2 - STAND_TILT;
    const lift = surface + 4 + Math.cos(STAND_TILT) * artD / 2;
    artGroup.position.set(L.art.x, lift, L.art.z + 22 - Math.sin(STAND_TILT) * artD / 2);
  }
  root.add(artGroup);

  if (settings.props && props) {
    for (const [name, r] of Object.entries(L.props)) {
      const p = props[name];
      p.position.set(r.x, name === 'monitor' ? 0 : propSurface(L, r), r.z);
      p.rotation.y = PROP_YAW[name] || 0;
      root.add(p);
    }
  }
  const dims = buildDimensions(L, artGroup);
  dims.visible = settings.dims;
  overlay.add(dims);
  scene.add(root);
  world = { root, artGroup, dims, surfaces: [desk.userData.top, matMesh], room: null };
  applySurfaceMode();
  refreshRoom();

  updateInfo(count, artW, artD, pitch);
  const sizeChanged = !prev || prev.artW !== L.artW || prev.artD !== L.artD || prev.flat !== L.flat;
  if (resetCamera || sizeChanged) applyView(settings.view);
  dirty = true;
}

// 部屋（窓の位置と天井照明は照明プリセットで変わるので、プリセットを切り替えるたびに作り直す）。
// 主光源は作品の中心を狙い、影カメラは部屋全体を囲む
function refreshRoom() {
  if (!world || !layout) return;
  const L = layout;
  if (world.room) {
    world.root.remove(world.room.group);
    world.room.group.traverse((o) => {
      o.geometry?.dispose?.();
      if (o.material) {
        for (const slot of TEXTURE_SLOTS) {
          const t = o.material[slot];
          if (t && !t.userData.shared) t.dispose();
        }
        o.material.dispose?.();
      }
    });
  }
  const room = buildRoom({
    desk: { W: L.W, D: L.D },
    art: { x: L.art.x, z: L.art.z },
    light: { direction: lighting.direction.clone(), inside: !!activePreset?.inside, color: lighting.key.color.clone() },
  });
  world.room = room;
  world.root.add(room.group);
  lighting.setFocus(new THREE.Vector3(L.art.x, 0, L.art.z), room.bounds);
  dirty = true;
}

function artCenter() {
  const L = layout;
  const y = L.flat ? L.artH : 6 + (L.artD / 2) * Math.cos(STAND_TILT);
  return new THREE.Vector3(L.art.x, y, L.art.z);
}

function applyView(view) {
  if (!layout) return;
  const L = layout;
  const target = artCenter();
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const span = Math.max(L.artW, L.flat ? L.artD : L.artD * 0.8);
  if (view === 'top') {
    // 作品と周りの小物が収まる範囲を真上から
    const regionW = L.artW + 700, regionD = Math.max(L.artD + 260, 600);
    const fitH = Math.max(regionD, regionW / camera.aspect) / 2 / Math.tan(fov / 2);
    target.set(L.art.x, 0, Math.min(L.art.z, L.D / 2 - regionD / 2));
    camera.position.set(target.x, fitH * 1.05, target.z + 0.01);
  } else if (view === 'seated') {
    // 椅子に座った目線：天板から約43cm上、手前の縁から約33cm
    camera.position.set(L.art.x, 430, L.D / 2 + 330);
    target.y = L.flat ? 0 : target.y;
    target.z -= L.flat ? 40 : 0;
  } else {
    const dist = (span + 520) / 2 / Math.tan(fov / 2);
    const dir = new THREE.Vector3(0.42, 0.78, 0.9).normalize();
    camera.position.copy(target).addScaledVector(dir, dist);
  }
  controls.target.copy(target);
  controls.update();
  dirty = true;
}

// ---- UI --------------------------------------------------------------------

function updateInfo(count, artW, artD, pitch) {
  const { width, height } = data.grid;
  $('size-main').textContent = `${fmt(artW)} × ${fmt(artD)}`;
  $('size-sub').textContent = `${width} × ${height} 粒（${count.toLocaleString()} 粒・${data.palette.length} 色）／ ピッチ ${pitch} mm`;
  const ironed = data.palette.filter((p) => p.rgb_source === 'ironed').length;
  $('color-src').textContent = data.demo
    ? '見本データを表示中（本体から開くと変換結果が表示されます）'
    : ironed === data.palette.length
      ? '色：アイロン後の実測値（Scan_ironed）'
      : ironed > 0
        ? `色：${ironed}/${data.palette.length} 色をアイロン後の実測値で表示`
        : `色：パレット「${data.palette_name || '選択中'}」の値`;
  const shown = Object.keys(layout.props);
  const hidden = Object.keys(PROP_SIZES).filter((k) => !shown.includes(k));
  $('props-note').textContent = settings.props && hidden.length
    ? `置き場所がないため非表示：${hidden.map((k) => PROP_SIZES[k].label).join('・')}` : '';
  // 外部の 3D モデルの出典表示（CC BY は作者名とライセンスの表示が条件）
  $('model-credit').textContent = modelCredits
    .map((c) => `${c.prop}のモデル：「${c.title}」${c.author.replace(/\s*\(https?:[^)]*\)/, '')}（${c.license.replace(/\s*\(https?:[^)]*\)/, '')}）`)
    .join(' ／ ');
}

function syncControls() {
  for (const el of document.querySelectorAll('[data-setting]')) {
    const key = el.dataset.setting;
    const val = el.dataset.value;
    el.classList.toggle('active', String(settings[key]) === val);
  }
  $('exposure').value = settings.exposure;
  $('exposure-val').textContent = Number(settings.exposure).toFixed(2);
  $('dims').checked = settings.dims;
  $('ao').checked = settings.ao;
  $('props').checked = settings.props;
  $('paper').checked = settings.paper;
  $('pitch').value = settings.pitch;
  const appOpt = $('pitch').querySelector('option[value="app"]');
  if (data) appOpt.textContent = `本体の設定（${data.bead_pitch_mm} mm）`;
}

async function setSetting(key, value) {
  settings[key] = value;
  syncControls();
  saveSettings();
  if (key === 'lighting') {
    activePreset = await lighting.apply(value);
    applyExposure();
    applySurfaceMode();
    refreshRoom();
  } else if (key === 'exposure') {
    applyExposure();
  } else if (key === 'view') {
    applyView(value);
  } else if (key === 'dims') {
    if (world) world.dims.visible = value;
  } else if (key === 'ao') {
    // 描画方法だけの切り替えなので作り直さない
  } else {
    rebuild();
  }
  dirty = true;
}

function buildPanel() {
  const groups = {
    placement: { flat: '平置き', stand: 'スタンド' },
    ironing: Object.fromEntries(Object.entries(IRONING).map(([k, v]) => [k, v.label])),
    lighting: Object.fromEntries(Object.entries(PRESETS).map(([k, v]) => [k, v.label])),
    view: { top: '真上', seated: '座った目線', oblique: '斜め' },
  };
  for (const [key, options] of Object.entries(groups)) {
    const box = $(`opt-${key}`);
    for (const [val, label] of Object.entries(options)) {
      const b = document.createElement('button');
      b.textContent = label;
      b.dataset.setting = key;
      b.dataset.value = val;
      b.addEventListener('click', () => {
        // 視点は同じボタンを押し直しても初期位置に戻す
        if (key === 'view' || settings[key] !== val) setSetting(key, val);
      });
      box.appendChild(b);
    }
  }
  $('exposure').addEventListener('input', (e) => setSetting('exposure', parseFloat(e.target.value)));
  $('dims').addEventListener('change', (e) => setSetting('dims', e.target.checked));
  $('ao').addEventListener('change', (e) => setSetting('ao', e.target.checked));
  $('props').addEventListener('change', (e) => setSetting('props', e.target.checked));
  $('paper').addEventListener('change', (e) => setSetting('paper', e.target.checked));
  $('pitch').addEventListener('change', (e) => setSetting('pitch', e.target.value));
  $('shot').addEventListener('click', screenshot);
  $('toggle-panel').addEventListener('click', () => document.body.classList.toggle('panel-hidden'));
}

async function screenshot() {
  renderFrame();
  const url = renderer.domElement.toDataURL('image/png');
  const a = api();
  if (a) {
    const saved = await a.save_screenshot(url);
    if (saved) toast(`保存しました：${saved}`);
  } else {
    const link = document.createElement('a');
    link.href = url;
    link.download = 'beads_desk.png';
    link.click();
  }
}

let toastTimer = 0;
function toast(text) {
  const el = $('toast');
  el.textContent = text;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3500);
}

// ---- 描画ループ（変化があったときだけ描く） --------------------------------

function resize() {
  const el = $('view');
  const w = el.clientWidth, h = el.clientHeight;
  renderer.setSize(w, h, false);
  composer.setPixelRatio(renderer.getPixelRatio());
  composer.setSize(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  dirty = true;
}

const frameTimes = [];
function loop(now) {
  requestAnimationFrame(loop);
  const moved = controls.update();
  if (!moved && !dirty) return;
  dirty = false;
  renderFrame();
  frameTimes.push(now);
  while (frameTimes.length && now - frameTimes[0] > 1000) frameTimes.shift();
  if (frameTimes.length > 5) $('fps').textContent = `${frameTimes.length} fps`;
}

// ---- 起動・再読込 -----------------------------------------------------------

async function reload() {
  try {
    data = await fetchScene();
    syncControls();
    rebuild();
  } catch (err) {
    showError(err);
  }
}

function showError(err) {
  console.error(err);
  $('loading').classList.add('show');
  $('loading').textContent = `読み込みに失敗しました：${err?.message || err}`;
}

async function start() {
  buildPanel();
  window.addEventListener('resize', resize);
  new ResizeObserver(resize).observe($('view'));
  controls.addEventListener('change', () => { dirty = true; });
  resize();
  try {
    await waitForApi();
    data = await fetchScene();
    const saved = api() ? await api().get_settings() : {};
    settings = { ...DEFAULTS, ...saved };
    if (!saved || !saved.ironing) settings.ironing = data.ironing || DEFAULTS.ironing;
    lighting = new Lighting(renderer, scene, data.hdri || []);
    activePreset = await lighting.apply(settings.lighting);
    applyExposure();
    // 開発用：?shadowcam=1 でシャドウカメラの範囲を表示する
    if (new URLSearchParams(location.search).has('shadowcam')) lighting.showShadowCamera(true);
    [props] = await Promise.all([buildProps(data.models || []), loadRoomTextures()]);
    syncControls();
    rebuild({ resetCamera: true });
    $('loading').classList.remove('show');
    requestAnimationFrame(loop);
  } catch (err) {
    showError(err);
  }
}

// 負荷確認用：カメラを回しながら n フレーム描いて1フレームの平均時間(ms)を返す
function bench(n = 60) {
  const gl = renderer.getContext();
  const px = new Uint8Array(4);
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.01);
    camera.lookAt(controls.target);
    renderFrame();
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); // GPU の完了を待つ
  }
  dirty = true;
  return (performance.now() - t0) / n;
}

// 確認用：小物（または作品 'art'）を斜め上から映す
function focus(name, dist = 260, dir = [0.35, 0.55, 0.75]) {
  const r = name === 'art' ? layout.art : layout.props[name];
  if (!r) return false;
  const target = new THREE.Vector3(r.x, name === 'monitor' ? 250 : 15, r.z);
  controls.target.copy(target);
  camera.position.copy(target).add(new THREE.Vector3(...dir).normalize().multiplyScalar(dist));
  controls.update();
  dirty = true;
  return true;
}

// 確認用：画面上の矩形の平均色（sRGB 0〜255）。描画直後に読むので同じタスク内で描き直す
function sampleRect(x, y, w = 8, h = 8) {
  renderFrame();
  const gl = renderer.getContext();
  const px = new Uint8Array(w * h * 4);
  const H = renderer.domElement.height;
  gl.readPixels(x, H - y - h, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const sum = [0, 0, 0];
  for (let i = 0; i < w * h; i++) for (let c = 0; c < 3; c++) sum[c] += px[i * 4 + c];
  return sum.map((v) => Math.round(v / (w * h)));
}

// 確認用：ワールド座標を描画バッファ上のピクセル位置へ
function toScreen(x, y, z) {
  camera.updateMatrixWorld();
  const v = new THREE.Vector3(x, y, z).project(camera);
  const c = renderer.domElement;
  return [Math.round((v.x * 0.5 + 0.5) * c.width), Math.round((-v.y * 0.5 + 0.5) * c.height)];
}

window.deskPreview = {
  reload, bench, focus, sampleRect, toScreen, setSetting,
  debug: { THREE, scene, renderer, camera, controls, get lighting() { return lighting; }, get world() { return world; }, get layout() { return layout; }, settings: () => settings, render: () => { dirty = true; } },
};
start();
