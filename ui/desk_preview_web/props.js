// デスク・小物。単位は 1 = 1mm。各小物は原点が底面中心、+Z が手前。
// assets/models/<名前>.glb があればそちらを読み込み、実寸に合わせて差し替える。
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { canvasTexture, grainSurface, matSurface, woodSurface } from './textures.js';

// 小物の実寸（w: 幅X, d: 奥行Z, h: 高さY）
export const PROP_SIZES = {
  keyboard: { label: 'キーボード', w: 440, d: 135, h: 30 },
  mouse: { label: 'マウス', w: 63, d: 120, h: 40 },
  monitor: { label: '24インチモニター', w: 540, d: 200, h: 430 },
  phone: { label: 'スマートフォン', w: 72, d: 150, h: 8 },
  mug: { label: 'マグカップ', w: 115, d: 80, h: 95 },
  pencil: { label: '鉛筆', w: 8, d: 175, h: 7 },
};

// 物理ベースの材質。grain を渡すと細かな凹凸（シボ）と粗さムラを付ける。
// 粗さマップの平均は (1 - amount) なので、roughness が実効値になるよう割り戻す
function phys(color, roughness, extra = {}, grain = null, Material = THREE.MeshPhysicalMaterial) {
  const params = { color, roughness, metalness: 0, ...extra };
  if (grain) {
    const amount = grain.amount ?? 0.15;
    params.roughness = roughness / (1 - amount);
    const g = grainSurface(grain.cells, grain.seed || 3, amount);
    // 大きな部品ほど繰り返して、凹凸の細かさを実寸でそろえる
    const rep = grain.repeat || [1, 1];
    const tiled = (t) => {
      if (rep[0] === 1 && rep[1] === 1) return t;
      const c = t.clone();
      c.repeat.set(rep[0], rep[1]);
      c.userData.shared = false;
      return c;
    };
    params.normalMap = tiled(g.normal);
    params.normalScale = new THREE.Vector2(grain.normal ?? 0.3, grain.normal ?? 0.3);
    params.roughnessMap = tiled(g.roughness);
  }
  return new Material(params);
}

// クリア層などを使わない材質は MeshStandardMaterial にする
const std = (color, roughness, extra = {}, grain = null) => phys(color, roughness, extra, grain, THREE.MeshStandardMaterial);

// 樹脂（シボ加工のマット）、塗装、金属など、繰り返し使う材質
const M = {
  matteBlack: (rep) => phys(0x1f2023, 0.62, {}, { cells: 64, normal: 0.25, repeat: rep }),
  // キーボード本体：半ツヤの樹脂
  satinBlack: (rep) => std(0x26272b, 0.45, {}, { cells: 64, normal: 0.04, amount: 0.06, repeat: rep }),
  // モニタースタンド：金属調の塗装
  gunmetal: (rep) => std(0x4a4c52, 0.4, { metalness: 0.6 }, { cells: 64, normal: 0.08, amount: 0.12, repeat: rep }),
  // スマホの側面：アルミ
  aluminum: (rep) => std(0x9a9da3, 0.3, { metalness: 0.8 }, { cells: 96, normal: 0.05, amount: 0.12, repeat: rep }),
};

function shadowed(obj) {
  obj.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return obj;
}

// ---- デスク・マット・部屋 ---------------------------------------------------

export function buildDesk(W, D) {
  const group = new THREE.Group();
  const thickness = 25;
  const wood = woodSurface();
  // ウレタン塗装の天板：木目の凹凸と導管、クリア層に窓が映り込む。
  // 木目の粗さマップの平均は約 0.55 なので、実効 0.6 になるよう割り戻す
  const top = new THREE.Mesh(
    new RoundedBoxGeometry(W, thickness, D, 3, 3),
    phys(0xffffff, 0.6 / 0.55, {
      map: wood.map, roughnessMap: wood.roughness, bumpMap: wood.bump, bumpScale: 1.2,
      clearcoat: 0.4, clearcoatRoughness: 0.25,
    }),
  );
  top.position.y = -thickness / 2;
  top.receiveShadow = true;
  group.add(top);
  // 脚（パネル脚）。視点を下げたときに天板が浮いて見えないように
  const legMat = std(0x3a3b3e, 0.4, { metalness: 0.6 });
  for (const s of [-1, 1]) {
    const leg = new THREE.Mesh(new RoundedBoxGeometry(30, 695, D - 80, 2, 3), legMat);
    leg.position.set(s * (W / 2 - 60), -thickness - 695 / 2, 0);
    leg.receiveShadow = true;
    group.add(leg);
  }
  // 床と壁は置かない。背景は HDRI（ぼかし）で見せる
  group.userData.top = top;
  return group;
}

export function buildMat(w, d) {
  const surf = matSurface(w, d);
  const m = new THREE.Mesh(
    new RoundedBoxGeometry(w, 2, d, 2, 1),
    std(0xffffff, 0.8, { map: surf.map, bumpMap: surf.bump, bumpScale: 0.6 }),
  );
  m.position.y = 1;
  m.receiveShadow = true;
  return m;
}

// ---- キーボード -------------------------------------------------------------

// 日本語配列フルサイズの概略。[ラベル, 幅(u), 高さ(u)]、null は隙間。1u = 19mm
const KEY_ROWS = [
  [['Esc'], [null, 1], ['F1'], ['F2'], ['F3'], ['F4'], [null, 0.5], ['F5'], ['F6'], ['F7'], ['F8'], [null, 0.5], ['F9'], ['F10'], ['F11'], ['F12']],
  [['半/全'], ['1'], ['2'], ['3'], ['4'], ['5'], ['6'], ['7'], ['8'], ['9'], ['0'], ['-'], ['^'], ['¥'], ['BS']],
  [['Tab', 1.5], ['Q'], ['W'], ['E'], ['R'], ['T'], ['Y'], ['U'], ['I'], ['O'], ['P'], ['@'], ['['], ['Enter', 1.5, 2]],
  [['Caps', 1.75], ['A'], ['S'], ['D'], ['F'], ['G'], ['H'], ['J'], ['K'], ['L'], [';'], [':'], [']']],
  [['Shift', 2.25], ['Z'], ['X'], ['C'], ['V'], ['B'], ['N'], ['M'], [','], ['.'], ['/'], ['\\'], ['Shift', 1.75]],
  [['Ctrl', 1.25], ['Win', 1.25], ['Alt', 1.25], ['無変換', 1.25], ['', 4.75], ['変換', 1.25], ['かな', 1.25], ['Alt', 1.25], ['Ctrl', 1.25]],
];
const NAV_ROWS = {
  0: [['PrtSc'], ['ScrLk'], ['Pause']],
  1: [['Ins'], ['Home'], ['PgUp']],
  2: [['Del'], ['End'], ['PgDn']],
  4: [[null, 1], ['↑'], [null, 1]],
  5: [['←'], ['↓'], ['→']],
};
const PAD_ROWS = {
  1: [['Num'], ['/'], ['*'], ['-']],
  2: [['7'], ['8'], ['9'], ['+', 1, 2]],
  3: [['4'], ['5'], ['6']],
  4: [['1'], ['2'], ['3'], ['Enter', 1, 2]],
  5: [['0', 2], ['.']],
};

function keyboardLayout() {
  const u = 19;
  const keys = [];
  const place = (rows, x0, y0) => {
    for (const [r, row] of Object.entries(rows)) {
      let x = x0;
      for (const [label, w = 1, h = 1] of row) {
        if (label !== null) keys.push({ label, x: x + (w * u) / 2, z: y0 + Number(r) * u + (h * u) / 2 + (Number(r) > 0 ? 4 : 0), w, h });
        x += w * u;
      }
    }
  };
  place(Object.fromEntries(KEY_ROWS.entries()), 0, 0);
  place(NAV_ROWS, 15.25 * u, 0);
  place(PAD_ROWS, 18.5 * u, 0);
  return keys;
}

function keyboard() {
  const g = new THREE.Group();
  const { w, d } = PROP_SIZES.keyboard;
  const tilt = 0.06;
  // 奥が高いくさび形を、少し傾けた箱で近似
  const body = new THREE.Mesh(new RoundedBoxGeometry(w, 20, d, 3, 5), M.satinBlack([6, 2]));
  body.position.y = 12;
  body.rotation.x = tilt;
  g.add(body);

  // キートップ：上へ行くほど細る形にし、全キーを1つのジオメトリにまとめる。
  // UV は真上からの投影（キーボード全体で1枚の印字テクスチャ）
  const keys = keyboardLayout();
  const u = 19;
  const ox = -w / 2 + 12, oz = -d / 2 + 9;
  const parts = [];
  for (const k of keys) {
    const kw = k.w * u - 3.5, kd = k.h * u - 3.5;
    const geo = new RoundedBoxGeometry(kw, 8, kd, 2, 1.6);
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const t = (p.getY(i) + 4) / 8; // 0=底, 1=天面
      const s = 1 - 0.16 * t;
      p.setX(i, p.getX(i) * (1 - (1 - s) * 15.5 / kw));
      p.setZ(i, p.getZ(i) * (1 - (1 - s) * 15.5 / kd));
    }
    geo.translate(ox + k.x, 0, oz + k.z);
    const uv = geo.attributes.uv;
    for (let i = 0; i < p.count; i++) uv.setXY(i, (p.getX(i) + w / 2) / w, 1 - (p.getZ(i) + d / 2) / d);
    geo.computeVertexNormals();
    parts.push(geo);
  }
  const keyGeo = mergeGeometries(parts);
  parts.forEach((p) => p.dispose());
  const legend = canvasTexture(2048, Math.round((2048 * d) / w), (ctx, cw, ch) => {
    ctx.fillStyle = '#34373c';
    ctx.fillRect(0, 0, cw, ch);
    const sx = cw / w, sz = ch / d;
    ctx.fillStyle = '#d4d7dc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const k of keys) {
      if (!k.label) continue;
      const big = k.label.length === 1;
      ctx.font = `${big ? 600 : 500} ${(big ? 5.2 : 3.1) * sx}px "Segoe UI", "Yu Gothic UI", sans-serif`;
      // 印字はキートップのやや左上（実物の刻印位置）
      const px = (ox + k.x + w / 2 - (big ? 2.5 : 0)) * sx;
      const pz = (oz + k.z + d / 2 - (big ? 2.5 : 0)) * sz;
      ctx.fillText(k.label, px, pz);
    }
  });
  // シボは縦横の実寸比に合わせて繰り返す（印字テクスチャは全体で1枚のまま）
  const capMat = std(0xffffff, 0.45, { map: legend }, { cells: 128, normal: 0.1, amount: 0.12, repeat: [w / d, 1] });
  const caps = new THREE.Mesh(keyGeo, capMat);
  caps.position.y = 23;
  caps.rotation.x = tilt;
  g.add(caps);
  return g;
}

// ---- マウス -----------------------------------------------------------------

// 前（-Z）ほど背を低くして、手のひらが乗る後ろ寄りに山があるマウスらしい形にする
const mouseLift = (zn) => 1 - 0.22 * (zn < 0 ? Math.min(1, -zn) ** 2 * (3 - 2 * Math.min(1, -zn)) : 0);

function mouseShellGeometry() {
  const geo = new THREE.SphereGeometry(1, 64, 32, 0, Math.PI * 2, 0, Math.PI / 2);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) p.setY(i, p.getY(i) * mouseLift(p.getZ(i)));
  geo.computeVertexNormals();
  return geo;
}

// 上面の印刷・溝：左右ボタンの分割線とホイールの穴。
// 半球の UV は u が周方向（u = 0.75 が前方の中心線）、v が 1 = 頂点 / 0 = 裾
function mouseShellTexture() {
  return canvasTexture(1024, 512, (ctx, w, h) => {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(40,42,46,0.85)';
    ctx.fillRect(w * 0.75 - 1.5, h * 0.02, 3, h * 0.72);
    ctx.fillStyle = '#1c1d20';
    ctx.beginPath();
    ctx.roundRect(w * (0.75 - 0.046), h * 0.19, w * 0.092, h * 0.16, 12);
    ctx.fill();
  });
}

function mouse() {
  const g = new THREE.Group();
  const { w, d, h } = PROP_SIZES.mouse;
  // 楕円体の上半分（下半分は天板に埋まって見えない）。上面はツヤのある樹脂、側面は濃色のグリップ
  const hemi = mouseShellGeometry();
  const shell = new THREE.Mesh(hemi, std(0xdfe0e2, 0.35, { map: mouseShellTexture() }, { cells: 32, normal: 0.04, amount: 0.08 }));
  const rx = w / 2 - 1.2, rz = d / 2 - 1;
  shell.scale.set(rx, h, rz);
  shell.position.y = 0.6;
  g.add(shell);
  const skirt = new THREE.Mesh(hemi, phys(0x2b2c30, 0.7, {}, { cells: 64, normal: 0.5, amount: 0.2 }));
  skirt.scale.set(w / 2, h * 0.32, d / 2);
  g.add(skirt);
  // ホイールは上面の穴から 2mm ほど顔を出す高さに沈める
  const wz = -d * 0.2;
  const zn = wz / rz;
  const surfaceY = 0.6 + h * Math.sqrt(1 - zn * zn) * mouseLift(zn);
  const wheelGrain = grainSurface(4, 9, 0.3);
  const wheelMat = phys(0x3a3c40, 0.75, { normalMap: wheelGrain.normal, normalScale: new THREE.Vector2(0.8, 0.8) });
  const wheel = new THREE.Mesh(new THREE.CylinderGeometry(5, 5, 5.5, 28), wheelMat);
  wheel.rotation.z = Math.PI / 2;
  wheel.position.set(0, surfaceY - 3, wz);
  g.add(wheel);
  return g;
}

// ---- モニター ---------------------------------------------------------------

function screenTexture() {
  return canvasTexture(1600, 900, (ctx, w, h) => {
    // 壁紙：やわらかいグラデーションと光のにじみ
    const grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, '#16283f');
    grad.addColorStop(0.5, '#2f5c8a');
    grad.addColorStop(1, '#8cb6d8');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
    for (const [x, y, r, c] of [[0.75, 0.3, 0.45, 'rgba(255,214,170,0.35)'], [0.2, 0.8, 0.5, 'rgba(120,190,255,0.25)']]) {
      const rg = ctx.createRadialGradient(x * w, y * h, 0, x * w, y * h, r * w);
      rg.addColorStop(0, c);
      rg.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = rg;
      ctx.fillRect(0, 0, w, h);
    }
    // ウィンドウ（本アプリ風：左に設定パネル、右にドット絵）
    const wx = w * 0.16, wy = h * 0.12, ww = w * 0.56, wh = h * 0.66;
    ctx.shadowColor = 'rgba(0,0,0,0.35)';
    ctx.shadowBlur = 30;
    ctx.fillStyle = '#f4f5f7';
    ctx.fillRect(wx, wy, ww, wh);
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#e1e4e9';
    ctx.fillRect(wx, wy, ww, 26);
    ctx.fillStyle = '#c9ced6';
    for (let i = 0; i < 9; i++) ctx.fillRect(wx + 18, wy + 50 + i * 30, ww * 0.26, 10);
    const cell = 12, gx = wx + ww * 0.4, gy = wy + 50;
    const cols = ['#e9e3d6', '#d63b3b', '#f08aa5', '#3b2f2f'];
    for (let r = 0; r < 26; r++) {
      for (let c = 0; c < 26; c++) {
        const x = (c - 12.5) / 8, y = -(r - 13) / 8;
        const v = Math.pow(x * x + y * y - 1, 3) - x * x * y * y * y;
        ctx.fillStyle = v < 0 ? (v > -0.06 ? cols[3] : cols[1]) : cols[0];
        ctx.fillRect(gx + c * cell, gy + r * cell, cell - 1, cell - 1);
      }
    }
    ctx.fillStyle = 'rgba(18,22,30,0.88)';
    ctx.fillRect(0, h - 40, w, 40);
  });
}

function monitor() {
  const g = new THREE.Group();
  const base = new THREE.Mesh(new RoundedBoxGeometry(230, 12, 180, 3, 5), M.gunmetal([3, 3]));
  base.position.y = 6;
  g.add(base);
  const neck = new THREE.Mesh(new RoundedBoxGeometry(46, 260, 24, 3, 6), M.gunmetal());
  neck.position.set(0, 140, -40);
  neck.rotation.x = -0.08;
  g.add(neck);
  const panelW = 540, panelH = 320;
  const panel = new THREE.Mesh(new RoundedBoxGeometry(panelW, panelH, 22, 3, 4), M.matteBlack([8, 5]));
  panel.position.set(0, 110 + panelH / 2, -10);
  panel.rotation.x = -0.06;
  g.add(panel);
  // 画面（24インチ 16:9 ＝ 約 531×299mm）。ノングレアの表面に周囲がぼんやり映る
  const screen = new THREE.Mesh(
    new THREE.PlaneGeometry(527, 297),
    phys(0x050505, 0.38, { emissive: 0xffffff, emissiveMap: screenTexture(), emissiveIntensity: 0.5, clearcoat: 0.25, clearcoatRoughness: 0.55 }),
  );
  screen.position.set(0, 4, 11.2);
  panel.add(screen);
  const led = new THREE.Mesh(new THREE.CircleGeometry(1.2, 12), new THREE.MeshBasicMaterial({ color: 0x66ccff }));
  led.position.set(panelW / 2 - 18, -panelH / 2 + 5, 11.3);
  panel.add(led);
  return g;
}

// ---- スマートフォン ---------------------------------------------------------

// ロック画面（暗めの壁紙と時計）。机に置いた画面が点いている状態
function lockScreenTexture() {
  return canvasTexture(360, 750, (ctx, w, h) => {
    const grad = ctx.createLinearGradient(0, 0, w * 0.4, h);
    grad.addColorStop(0, '#1d2b4a');
    grad.addColorStop(0.55, '#4b3a6b');
    grad.addColorStop(1, '#c0705a');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
    const glow = ctx.createRadialGradient(w * 0.7, h * 0.75, 0, w * 0.7, h * 0.75, w * 0.8);
    glow.addColorStop(0, 'rgba(255,200,150,0.45)');
    glow.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.textAlign = 'center';
    ctx.font = '300 92px "Segoe UI", sans-serif';
    ctx.fillText('12:34', w / 2, h * 0.24);
    ctx.font = '500 22px "Segoe UI", "Yu Gothic UI", sans-serif';
    ctx.fillText('10月17日 土曜日', w / 2, h * 0.12);
    ctx.fillStyle = 'rgba(255,255,255,0.18)';
    ctx.beginPath();
    ctx.roundRect(w * 0.08, h * 0.62, w * 0.84, 70, 18);
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.fillRect(w * 0.35, h - 18, w * 0.3, 5);
  });
}

function phone() {
  const g = new THREE.Group();
  const { w, d, h } = PROP_SIZES.phone;
  // アルミの側面フレーム＋ガラスの前面（消灯中で周囲が映り込む）
  const frame = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, 6, 7.5), M.aluminum());
  frame.position.y = h / 2;
  g.add(frame);
  const glass = new THREE.Mesh(
    new RoundedBoxGeometry(w - 1.2, 0.6, d - 1.2, 4, 7),
    phys(0x020203, 0.05, { clearcoat: 1, clearcoatRoughness: 0.03, ior: 1.52 }),
  );
  glass.position.y = h + 0.05;
  g.add(glass);
  // 表示面（ガラスの表面と同じツヤ）。発光は暗めにして、窓や部屋の映り込みが見えるようにする
  const display = new THREE.Mesh(
    new THREE.PlaneGeometry(w - 4, d - 4),
    phys(0x000000, 0.05, { emissive: 0xffffff, emissiveMap: lockScreenTexture(), emissiveIntensity: 0.12, clearcoat: 1, clearcoatRoughness: 0.03 }),
  );
  display.rotation.x = -Math.PI / 2;
  display.position.y = h + 0.38;
  g.add(display);
  const speaker = new THREE.Mesh(new RoundedBoxGeometry(10, 0.5, 1.2, 2, 0.5), phys(0x111214, 0.8));
  speaker.position.set(0, h + 0.3, -d / 2 + 4);
  g.add(speaker);
  return g;
}

// ---- マグカップ -------------------------------------------------------------

function mug() {
  const g = new THREE.Group();
  const r = 40, h = 95, t = 4;
  // 外壁・縁・内壁・底を1本の断面で回転。釉薬のかかった陶器（下地は半ツヤ＋透明な釉の層）
  const prof = [
    [0, 0.5], [r - 6, 0.5], [r - 4, 0], [r - 1.5, 1.2], [r, 4], [r, h - 1.5], [r - t / 2, h], [r - t, h - 1.5], [r - t, 9], [r - t - 3, 7], [0, 7],
  ].map(([x, y]) => new THREE.Vector2(x, y));
  const glaze = phys(0xf3f0ea, 0.42, { clearcoat: 1, clearcoatRoughness: 0.05 }, { cells: 16, normal: 0.06, amount: 0.05 });
  const body = new THREE.Mesh(new THREE.LatheGeometry(prof, 64), glaze);
  g.add(body);
  // コーヒー：縁にうっすら泡（クレマ）の輪
  const crema = canvasTexture(256, 256, (ctx, w2) => {
    const rg = ctx.createRadialGradient(w2 / 2, w2 / 2, w2 * 0.2, w2 / 2, w2 / 2, w2 / 2);
    rg.addColorStop(0, '#2a1409');
    rg.addColorStop(0.82, '#3a1d0e');
    rg.addColorStop(1, '#7a4a28');
    ctx.fillStyle = rg;
    ctx.fillRect(0, 0, w2, w2);
  });
  const coffee = new THREE.Mesh(new THREE.CircleGeometry(r - t, 48), phys(0xffffff, 0.05, { map: crema, clearcoat: 1, clearcoatRoughness: 0.02 }));
  coffee.rotation.x = -Math.PI / 2;
  coffee.position.y = h - 18;
  g.add(coffee);
  const handle = new THREE.Mesh(new THREE.TorusGeometry(24, 5.5, 20, 40, Math.PI * 1.15), glaze);
  handle.rotation.z = -Math.PI * 0.575;
  handle.position.set(r + 2, h / 2, 0);
  g.add(handle);
  return g;
}

// ---- 鉛筆 -------------------------------------------------------------------

function pencilPrint() {
  // 六角軸の UV：u が周方向（1面 = 1/6）、v が長さ方向。真上を向く面は u = 1/3〜1/2
  return canvasTexture(384, 2048, (ctx, w, h) => {
    ctx.fillStyle = '#e9b721';
    ctx.fillRect(0, 0, w, h);
    // 稜線ごとにわずかに明るさを変えて面の切り替わりを出す
    for (let f = 0; f < 6; f++) {
      ctx.fillStyle = f % 2 ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)';
      ctx.fillRect((f * w) / 6, 0, w / 6, h);
    }
    ctx.save();
    ctx.translate((w * 5) / 12, h * 0.2);
    ctx.rotate(Math.PI / 2);
    ctx.fillStyle = '#1b1b1b';
    ctx.font = 'bold 38px "Segoe UI", sans-serif';
    ctx.textBaseline = 'middle';
    ctx.fillText('HB', 0, 0);
    ctx.font = '600 30px "Segoe UI", sans-serif';
    ctx.fillStyle = '#2a2a2a';
    ctx.fillText('DRAWING PENCIL  ·  JAPAN', 110, 0);
    ctx.restore();
  });
}

function pencil() {
  const g = new THREE.Group();
  const len = 175, r = 3.6;
  const bodyLen = len - 30;
  const ay = r * 0.866;
  // 六角軸を平面で接地させる。塗装は半ツヤのラッカー、面はフラットシェーディング
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(r, r, bodyLen, 6),
    std(0xffffff, 0.35, { map: pencilPrint(), flatShading: true }),
  );
  body.rotation.x = Math.PI / 2;
  body.rotation.y = Math.PI / 6;
  body.position.set(0, ay, 0);
  const pivot = new THREE.Group();
  pivot.add(body);
  const woodGrain = canvasTexture(128, 256, (ctx, w, h) => {
    ctx.fillStyle = '#dcb98a';
    ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 40; i++) {
      ctx.strokeStyle = `rgba(150,105,60,${0.1 + Math.random() * 0.15})`;
      ctx.beginPath();
      const x = Math.random() * w;
      ctx.moveTo(x, 0);
      ctx.lineTo(x + (Math.random() - 0.5) * 8, h);
      ctx.stroke();
    }
  });
  const wood = new THREE.Mesh(new THREE.ConeGeometry(r, 18, 6, 1, true), phys(0xffffff, 0.8, { map: woodGrain, flatShading: true }));
  wood.rotation.x = -Math.PI / 2;
  wood.rotation.y = Math.PI / 6;
  wood.position.set(0, ay, -bodyLen / 2 - 9);
  pivot.add(wood);
  const lead = new THREE.Mesh(new THREE.ConeGeometry(r * 0.32, 5.6, 16), phys(0x2c2c2e, 0.42, { metalness: 0.55 }));
  lead.rotation.x = -Math.PI / 2;
  lead.position.set(0, ay, -bodyLen / 2 - 18 - 0.5);
  pivot.add(lead);
  // 口金：細かい溝のある金属
  const ridges = canvasTexture(16, 64, (ctx, w, h) => {
    for (let y = 0; y < h; y++) {
      const v = 128 + Math.sin((y / h) * Math.PI * 2 * 6) * 100;
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(0, y, w, 1);
    }
  }, false);
  const ferrule = new THREE.Mesh(
    new THREE.CylinderGeometry(r * 1.03, r * 1.03, 6, 32),
    phys(0xc9c6bd, 0.28, { metalness: 1, bumpMap: ridges, bumpScale: 2 }),
  );
  ferrule.rotation.x = Math.PI / 2;
  ferrule.position.set(0, ay, bodyLen / 2 + 3);
  pivot.add(ferrule);
  const eraser = new THREE.Mesh(new THREE.CapsuleGeometry(r * 0.95, 4, 6, 24), phys(0xe08995, 0.85, {}, { cells: 64, normal: 0.4, amount: 0.1 }));
  eraser.rotation.x = Math.PI / 2;
  eraser.position.set(0, ay, bodyLen / 2 + 7.5);
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
