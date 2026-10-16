// デスク・小物。単位は 1 = 1mm。各小物は原点が底面中心、+Z が手前。
// assets/models/<名前>.glb があればそちらを読み込み、実寸に合わせて差し替える。
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// 小物の実寸（w: 幅X, d: 奥行Z, h: 高さY）
export const PROP_SIZES = {
  keyboard: { label: 'キーボード', w: 440, d: 135, h: 30 },
  mouse: { label: 'マウス', w: 63, d: 120, h: 40 },
  monitor: { label: '24インチモニター', w: 540, d: 200, h: 430 },
  phone: { label: 'スマートフォン', w: 72, d: 150, h: 8 },
  mug: { label: 'マグカップ', w: 115, d: 80, h: 95 },
  pencil: { label: '鉛筆', w: 8, d: 175, h: 7 },
};

const mat = (color, roughness = 0.5, extra = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness, metalness: 0, ...extra });

function shadowed(obj) {
  obj.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return obj;
}

function canvasTexture(w, h, draw, srgb = true) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  draw(canvas.getContext('2d'), w, h);
  const tex = new THREE.CanvasTexture(canvas);
  if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

// ---- デスク・マット ---------------------------------------------------------

function woodTexture() {
  return canvasTexture(1024, 512, (ctx, w, h) => {
    ctx.fillStyle = '#b48a5e';
    ctx.fillRect(0, 0, w, h);
    let seed = 3;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    // 木目：横方向にゆらぐ細線を重ねる
    for (let i = 0; i < 260; i++) {
      const y0 = rnd() * h;
      const amp = 2 + rnd() * 6;
      const freq = 0.002 + rnd() * 0.01;
      const phase = rnd() * 10;
      const dark = rnd() < 0.5;
      ctx.strokeStyle = dark ? `rgba(90,58,30,${0.05 + rnd() * 0.12})` : `rgba(222,186,140,${0.05 + rnd() * 0.1})`;
      ctx.lineWidth = 0.6 + rnd() * 2.2;
      ctx.beginPath();
      for (let x = 0; x <= w; x += 8) {
        const y = y0 + Math.sin(x * freq + phase) * amp + Math.sin(x * freq * 3.1) * amp * 0.3;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  });
}

function matTexture(wMm, dMm) {
  const pxPerMm = Math.min(4, 2048 / Math.max(wMm, dMm));
  const W = Math.round(wMm * pxPerMm), H = Math.round(dMm * pxPerMm);
  return canvasTexture(W, H, (ctx) => {
    ctx.fillStyle = '#2f5f4f';
    ctx.fillRect(0, 0, W, H);
    // 方眼：10mm ごとに細線、50mm ごとに太線（原点はマット中心）
    const cx = W / 2, cy = H / 2;
    for (const [step, alpha, lw] of [[10, 0.22, 1], [50, 0.5, 1.6]]) {
      ctx.strokeStyle = `rgba(220,240,230,${alpha})`;
      ctx.lineWidth = lw;
      ctx.beginPath();
      for (let mm = 0; mm <= wMm / 2; mm += step) {
        for (const s of [1, -1]) {
          const x = cx + s * mm * pxPerMm;
          ctx.moveTo(x, 0); ctx.lineTo(x, H);
        }
      }
      for (let mm = 0; mm <= dMm / 2; mm += step) {
        for (const s of [1, -1]) {
          const y = cy + s * mm * pxPerMm;
          ctx.moveTo(0, y); ctx.lineTo(W, y);
        }
      }
      ctx.stroke();
    }
    // 縁の目盛り数字（cm）
    ctx.fillStyle = 'rgba(230,245,238,0.75)';
    ctx.font = `${Math.max(10, 5 * pxPerMm)}px sans-serif`;
    ctx.textAlign = 'center';
    for (let mm = -Math.floor(wMm / 100) * 50; mm <= wMm / 2 - 20; mm += 50) {
      if (mm === 0) continue;
      ctx.fillText(`${mm / 10}`, cx + mm * pxPerMm, H - 3 * pxPerMm);
    }
  });
}

export function buildDesk(W, D) {
  const group = new THREE.Group();
  const thickness = 25;
  const wood = woodTexture();
  wood.repeat.set(W / 1200, D / 600);
  wood.wrapS = wood.wrapT = THREE.RepeatWrapping;
  const top = new THREE.Mesh(
    new RoundedBoxGeometry(W, thickness, D, 3, 3),
    mat(0xffffff, 0.55, { map: wood }),
  );
  top.position.y = -thickness / 2;
  top.receiveShadow = true;
  group.add(top);
  // 脚（パネル脚）と床・壁。視点を下げたときに空中に浮いて見えないように
  const legMat = mat(0x3a3b3e, 0.6);
  for (const s of [-1, 1]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(30, 695, D - 80), legMat);
    leg.position.set(s * (W / 2 - 60), -thickness - 695 / 2, 0);
    leg.receiveShadow = true;
    group.add(leg);
  }
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(8000, 8000), mat(0x8a8178, 0.85));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -720;
  floor.receiveShadow = true;
  group.add(floor);
  const wall = new THREE.Mesh(new THREE.PlaneGeometry(8000, 3000), mat(0xe8e4dc, 0.9));
  wall.position.set(0, -720 + 1500, -D / 2 - 80);
  wall.receiveShadow = true;
  group.add(wall);
  return group;
}

export function buildMat(w, d) {
  const m = new THREE.Mesh(
    new RoundedBoxGeometry(w, 2, d, 2, 1),
    mat(0xffffff, 0.8, { map: matTexture(w, d) }),
  );
  m.position.y = 1;
  m.receiveShadow = true;
  return m;
}

// ---- 小物（簡易モデル） -----------------------------------------------------

function keyboard() {
  const g = new THREE.Group();
  const { w, d } = PROP_SIZES.keyboard;
  // 奥が高いくさび形を、少し傾けた箱で近似
  const body = new THREE.Mesh(new RoundedBoxGeometry(w, 20, d, 3, 5), mat(0x2b2d31, 0.55));
  body.position.y = 12;
  body.rotation.x = 0.06;
  g.add(body);
  const keyGeo = new RoundedBoxGeometry(15.5, 8, 15.5, 2, 2.5);
  const keyMat = mat(0x3c3f45, 0.6);
  const keys = [];
  const pitch = 19;
  // メイン部（15列×5段）＋スペース行、テンキー、カーソル部をざっくり配置
  const addKey = (x, z, kw = 1) => keys.push([x, z, kw]);
  const left = -w / 2 + 14;
  for (let r = 0; r < 5; r++) for (let c = 0; c < 15; c++) addKey(left + c * pitch, -d / 2 + 22 + r * pitch);
  addKey(left + 4 * pitch, -d / 2 + 22 + 5 * pitch - 2, 6.2);
  for (let c of [0, 1, 2, 11, 12, 13, 14]) addKey(left + c * pitch, -d / 2 + 22 + 5 * pitch - 2);
  const nav = left + 15.6 * pitch;
  for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) addKey(nav + c * pitch, -d / 2 + 22 + r * pitch);
  addKey(nav + pitch, -d / 2 + 22 + 4 * pitch); for (let c = 0; c < 3; c++) addKey(nav + c * pitch, -d / 2 + 22 + 5 * pitch - 2);
  const pad = nav + 3.6 * pitch;
  for (let r = 0; r < 6; r++) for (let c = 0; c < 4; c++) if (pad + c * pitch < w / 2 - 8) addKey(pad + c * pitch, -d / 2 + 22 + r * pitch - (r === 5 ? 2 : 0));
  const inst = new THREE.InstancedMesh(keyGeo, keyMat, keys.length);
  const m4 = new THREE.Matrix4();
  keys.forEach(([x, z, kw], i) => {
    const y = 23 + (z + d / 2) * 0.06;
    m4.compose(new THREE.Vector3(x + (kw - 1) * pitch / 2, y, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(0.06, 0, 0)),
      new THREE.Vector3(kw === 1 ? 1 : (kw * pitch - 3.5) / 15.5, 1, 1));
    inst.setMatrixAt(i, m4);
  });
  g.add(inst);
  return g;
}

function mouse() {
  const g = new THREE.Group();
  const { w, d, h } = PROP_SIZES.mouse;
  // 楕円体の上半分（下半分は天板に埋まって見えない）
  const shell = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24, 0, Math.PI * 2, 0, Math.PI / 2), mat(0xd9dadc, 0.38));
  shell.scale.set(w / 2, h, d / 2);
  g.add(shell);
  const split = new THREE.Mesh(new THREE.BoxGeometry(0.8, 4, d * 0.32), mat(0x55585e, 0.5));
  split.position.set(0, h - 3.2, -d * 0.24);
  split.scale.y = 1;
  g.add(split);
  const wheel = new THREE.Mesh(new THREE.CylinderGeometry(4.5, 4.5, 5, 20), mat(0x4a4d52, 0.6));
  wheel.rotation.z = Math.PI / 2;
  wheel.position.set(0, h - 3.5, -d * 0.18);
  g.add(wheel);
  return g;
}

function screenTexture() {
  return canvasTexture(1024, 576, (ctx, w, h) => {
    const grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, '#1f3b63');
    grad.addColorStop(0.55, '#3a6ea5');
    grad.addColorStop(1, '#8fb8de');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
    // タスクバーとウィンドウの影で「画面らしさ」を出す
    ctx.fillStyle = 'rgba(20,24,32,0.85)';
    ctx.fillRect(0, h - 28, w, 28);
    ctx.fillStyle = 'rgba(245,247,250,0.93)';
    ctx.fillRect(w * 0.18, h * 0.14, w * 0.5, h * 0.58);
    ctx.fillStyle = 'rgba(60,110,190,0.9)';
    ctx.fillRect(w * 0.18, h * 0.14, w * 0.5, 22);
  });
}

function monitor() {
  const g = new THREE.Group();
  const dark = mat(0x1d1e21, 0.45);
  const base = new THREE.Mesh(new RoundedBoxGeometry(230, 12, 180, 3, 5), mat(0x2a2b2f, 0.4));
  base.position.y = 6;
  g.add(base);
  const neck = new THREE.Mesh(new RoundedBoxGeometry(46, 260, 24, 3, 6), mat(0x2a2b2f, 0.4));
  neck.position.set(0, 140, -40);
  neck.rotation.x = -0.08;
  g.add(neck);
  const panelW = 540, panelH = 320;
  const panel = new THREE.Mesh(new RoundedBoxGeometry(panelW, panelH, 22, 3, 4), dark);
  const panelY = 110 + panelH / 2;
  panel.position.set(0, panelY, -10);
  panel.rotation.x = -0.06;
  g.add(panel);
  // 画面（24インチ 16:9 ＝ 約 531×299mm）。自発光は控えめにして周囲の照明を邪魔しない
  const tex = screenTexture();
  const screen = new THREE.Mesh(
    new THREE.PlaneGeometry(527, 297),
    new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.15, emissive: 0xffffff, emissiveMap: tex, emissiveIntensity: 0.55 }),
  );
  screen.position.set(0, 0, 11.2);
  panel.add(screen);
  return g;
}

function phone() {
  const g = new THREE.Group();
  const { w, d, h } = PROP_SIZES.phone;
  const body = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, 4, 7), mat(0x1b1c1f, 0.25));
  body.position.y = h / 2;
  g.add(body);
  const glass = new THREE.Mesh(
    new THREE.PlaneGeometry(w - 5, d - 5),
    new THREE.MeshStandardMaterial({ color: 0x050608, roughness: 0.06, metalness: 0.0 }),
  );
  glass.rotation.x = -Math.PI / 2;
  glass.position.y = h + 0.05;
  g.add(glass);
  return g;
}

function mug() {
  const g = new THREE.Group();
  const r = 40, h = 95, t = 4;
  // 外壁・縁・内壁・底を1本の断面で回転
  const prof = [
    [0, 0], [r - 3, 0], [r, 3], [r, h - 1.5], [r - t / 2, h], [r - t, h - 1.5], [r - t, 8], [0, 8],
  ].map(([x, y]) => new THREE.Vector2(x, y));
  const ceramic = mat(0xf1efe9, 0.28);
  const body = new THREE.Mesh(new THREE.LatheGeometry(prof, 48), ceramic);
  g.add(body);
  const coffee = new THREE.Mesh(new THREE.CircleGeometry(r - t, 40), mat(0x3b2416, 0.12));
  coffee.rotation.x = -Math.PI / 2;
  coffee.position.y = h - 18;
  g.add(coffee);
  const handle = new THREE.Mesh(new THREE.TorusGeometry(24, 5.5, 14, 30, Math.PI * 1.15), ceramic);
  handle.rotation.z = -Math.PI * 0.575;
  handle.position.set(r + 2, h / 2, 0);
  g.add(handle);
  return g;
}

function pencil() {
  const g = new THREE.Group();
  const len = 175, r = 3.6;
  const bodyLen = len - 30;
  // 六角軸を平面で接地させる
  const body = new THREE.Mesh(new THREE.CylinderGeometry(r, r, bodyLen, 6), mat(0xe8b923, 0.4));
  body.rotation.x = Math.PI / 2;
  body.rotation.y = Math.PI / 6;
  body.position.set(0, r * 0.866, 0);
  const pivot = new THREE.Group();
  pivot.add(body);
  const wood = new THREE.Mesh(new THREE.ConeGeometry(r, 18, 6, 1, true), mat(0xdcb98a, 0.7));
  wood.rotation.x = -Math.PI / 2;
  wood.position.set(0, r * 0.866, -bodyLen / 2 - 9);
  pivot.add(wood);
  const lead = new THREE.Mesh(new THREE.ConeGeometry(r * 0.32, 5.6, 12), mat(0x2b2b2b, 0.35));
  lead.rotation.x = -Math.PI / 2;
  lead.position.set(0, r * 0.866, -bodyLen / 2 - 18 - 0.5);
  pivot.add(lead);
  const ferrule = new THREE.Mesh(new THREE.CylinderGeometry(r * 1.02, r * 1.02, 6, 20), mat(0xb9b9b9, 0.3, { metalness: 0.8 }));
  ferrule.rotation.x = Math.PI / 2;
  ferrule.position.set(0, r * 0.866, bodyLen / 2 + 3);
  pivot.add(ferrule);
  const eraser = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.95, r * 0.95, 6, 20), mat(0xe58c96, 0.8));
  eraser.rotation.x = Math.PI / 2;
  eraser.position.set(0, r * 0.866, bodyLen / 2 + 9);
  pivot.add(eraser);
  pivot.position.z = 3;
  g.add(pivot);
  return g;
}

const BUILDERS = { keyboard, mouse, monitor, phone, mug, pencil };

// glTF をバウンディングボックス基準で実寸（mm）へ合わせる
function normalizeModel(root, size) {
  const box = new THREE.Box3().setFromObject(root);
  const dim = box.getSize(new THREE.Vector3());
  const scale = Math.max(size.w, size.d, size.h) / Math.max(dim.x, dim.y, dim.z, 1e-6);
  root.scale.multiplyScalar(scale);
  box.setFromObject(root);
  const center = box.getCenter(new THREE.Vector3());
  root.position.x -= center.x;
  root.position.z -= center.z;
  root.position.y -= box.min.y;
  const g = new THREE.Group();
  g.add(root);
  return g;
}

export async function buildProps(modelFiles = []) {
  const files = new Map(modelFiles.map((f) => [f.replace(/\.[^.]+$/, ''), f]));
  const loader = new GLTFLoader();
  const props = {};
  for (const [name, build] of Object.entries(BUILDERS)) {
    let obj = null;
    if (files.has(name)) {
      try {
        const gltf = await loader.loadAsync(`assets/models/${files.get(name)}`);
        obj = normalizeModel(gltf.scene, PROP_SIZES[name]);
      } catch (err) {
        console.warn(`${name} のモデル読み込みに失敗したため簡易モデルを使います`, err);
      }
    }
    props[name] = shadowed(obj || build());
  }
  return props;
}
