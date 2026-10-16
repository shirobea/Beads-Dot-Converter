// ビーズ形状と作品メッシュ。全ビーズを1つの InstancedMesh にまとめ、色は instanceColor で持つ。
// 形状は 5mm ピッチ基準で定義し、実際のピッチに合わせて相似拡大縮小する。
import * as THREE from 'three';

export const IRONING = {
  none: { label: 'なし', roughness: 0.32, normalScale: 0.05 },
  one_side: { label: '片面', roughness: 0.45, normalScale: 0.18 },
  both_sides: { label: '両面', roughness: 0.52, normalScale: 0.28 },
};

// 断面（r, y）: 下内→下外→外壁を上へ→上外→上内→内壁を下へ。
// この向きだと LatheGeometry の法線が外向き・穴の内側向きになる。
const SECTIONS = {
  // アイロン前：外径約5mm・高さ約5mm・穴径約2.5mmの円筒チューブ
  none: { rInB: 1.3, rOutB: 2.4, rInT: 1.3, rOutT: 2.4, h: 4.8, cB: 0.65, cT: 0.65, bulge: 0.04 },
  // 片面：上面が平らに融着して広がり、穴が上側だけ縮む
  one_side: { rInB: 1.3, rOutB: 2.4, rInT: 0.85, rOutT: 2.62, h: 4.3, cB: 0.6, cT: 0.22, bulge: 0.08 },
  // 両面：高さ約3.8mm、穴が小さく、隣のビーズと重なって連結する
  both_sides: { rInB: 0.78, rOutB: 2.63, rInT: 0.72, rOutT: 2.63, h: 3.8, cB: 0.28, cT: 0.24, bulge: 0.14 },
};

function arc(points, cx, cy, radius, from, to, steps) {
  for (let i = 0; i <= steps; i++) {
    const a = from + ((to - from) * i) / steps;
    points.push(new THREE.Vector2(cx + Math.cos(a) * radius, cy + Math.sin(a) * radius));
  }
}

function sectionPoints(s, steps) {
  const pts = [];
  const half = Math.PI / 2;
  // 下面（内→外）
  arc(pts, s.rInB + s.cB, s.cB, s.cB, Math.PI, Math.PI + half, steps);
  arc(pts, s.rOutB - s.cB, s.cB, s.cB, -half, 0, steps);
  // 外壁の中ほどを少し膨らませて溶けた丸みを出す
  pts.push(new THREE.Vector2((s.rOutB + s.rOutT) / 2 + s.bulge, s.h / 2));
  // 上面（外→内）
  arc(pts, s.rOutT - s.cT, s.h - s.cT, s.cT, 0, half, steps);
  arc(pts, s.rInT + s.cT, s.h - s.cT, s.cT, half, Math.PI, steps);
  // 内壁を下へ（始点に戻って閉じる）
  pts.push(new THREE.Vector2(s.rInB, s.cB));
  return pts;
}

const geometryCache = new Map();

export function beadGeometry(ironing, pitch, segments, arcSteps) {
  const key = `${ironing}:${pitch}:${segments}:${arcSteps}`;
  if (geometryCache.has(key)) return geometryCache.get(key);
  const s = SECTIONS[ironing] || SECTIONS.both_sides;
  const geo = new THREE.LatheGeometry(sectionPoints(s, arcSteps), segments);
  geo.scale(pitch / 5, pitch / 5, pitch / 5);
  geometryCache.set(key, geo);
  return geo;
}

export function beadHeight(ironing, pitch) {
  return (SECTIONS[ironing] || SECTIONS.both_sides).h * (pitch / 5);
}

// 溶けムラ用の法線マップ（値ノイズ→高さ→法線）
let meltNormal = null;
function meltNormalMap() {
  if (meltNormal) return meltNormal;
  const n = 128;
  const height = new Float32Array(n * n);
  const rand = mulberry32(7);
  for (const [cell, amp] of [[16, 1.0], [32, 0.5], [64, 0.25]]) {
    const g = n / cell;
    const grid = Array.from({ length: (g + 1) * (g + 1) }, () => rand());
    const at = (x, y) => grid[(y % g) * (g + 1) + (x % g)];
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const gx = x / cell, gy = y / cell;
        const x0 = Math.floor(gx), y0 = Math.floor(gy);
        const fx = smooth(gx - x0), fy = smooth(gy - y0);
        const v = lerp(lerp(at(x0, y0), at(x0 + 1, y0), fx), lerp(at(x0, y0 + 1), at(x0 + 1, y0 + 1), fx), fy);
        height[y * n + x] += v * amp;
      }
    }
  }
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = n;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(n, n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const h = (xx, yy) => height[((yy + n) % n) * n + ((xx + n) % n)];
      const dx = (h(x + 1, y) - h(x - 1, y)) * 2.0;
      const dy = (h(x, y + 1) - h(x, y - 1)) * 2.0;
      const len = Math.hypot(dx, dy, 1);
      const i = (y * n + x) * 4;
      img.data[i] = (-dx / len * 0.5 + 0.5) * 255;
      img.data[i + 1] = (-dy / len * 0.5 + 0.5) * 255;
      img.data[i + 2] = (1 / len * 0.5 + 0.5) * 255;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  meltNormal = new THREE.CanvasTexture(canvas);
  meltNormal.wrapS = meltNormal.wrapT = THREE.RepeatWrapping;
  meltNormal.repeat.set(3, 2);
  return meltNormal;
}

function smooth(t) { return t * t * (3 - 2 * t); }
function lerp(a, b, t) { return a + (b - a) * t; }

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 粒数が多いほど断面・周方向の分割を減らして描画負荷を抑える
function detailFor(count) {
  if (count <= 3000) return { segments: 24, arcSteps: 3 };
  if (count <= 12000) return { segments: 14, arcSteps: 2 };
  if (count <= 30000) return { segments: 10, arcSteps: 2 };
  return { segments: 8, arcSteps: 1 };
}

/**
 * 作品を組み立てる。返す Group は原点が作品中心・底面、+Z が画像の下側。
 * 列 → X、行 → Z、ビーズの穴は Y 方向。
 */
export function buildArtwork(scene, ironing, pitch) {
  const { width, height, cells } = scene.grid;
  const colors = scene.palette.map((p) =>
    new THREE.Color().setRGB(p.rgb[0] / 255, p.rgb[1] / 255, p.rgb[2] / 255, THREE.SRGBColorSpace));
  let count = 0;
  for (const row of cells) for (const v of row) if (v >= 0) count++;

  const detail = detailFor(count);
  const geometry = beadGeometry(ironing, pitch, detail.segments, detail.arcSteps);
  const spec = IRONING[ironing] || IRONING.both_sides;
  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: spec.roughness,
    metalness: 0,
    normalMap: meltNormalMap(),
    normalScale: new THREE.Vector2(spec.normalScale, spec.normalScale),
  });
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(count, 1));
  mesh.count = count;
  mesh.castShadow = count <= 30000;
  mesh.receiveShadow = true;

  const rand = mulberry32(width * 7919 + height);
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const e = new THREE.Euler();
  const pos = new THREE.Vector3();
  const scl = new THREE.Vector3();
  const ox = (width - 1) / 2, oz = (height - 1) / 2;
  const tilt = ironing === 'none' ? 0.035 : 0.0;
  const squash = ironing === 'none' ? 0.0 : 0.035;
  let i = 0;
  for (let r = 0; r < height; r++) {
    const row = cells[r];
    for (let c = 0; c < width; c++) {
      const v = row[c];
      if (v < 0) continue;
      // 1粒ずつわずかに傾き・潰れ具合を変えて、手作業の不揃いさを出す
      e.set((rand() - 0.5) * tilt, rand() * Math.PI * 2, (rand() - 0.5) * tilt);
      q.setFromEuler(e);
      const sx = 1 + (rand() - 0.5) * squash;
      scl.set(sx, 1 + (rand() - 0.5) * squash, sx);
      pos.set((c - ox) * pitch, 0, (r - oz) * pitch);
      m.compose(pos, q, scl);
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, colors[v] || colors[0]);
      i++;
    }
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();

  const group = new THREE.Group();
  group.add(mesh);
  group.userData = {
    count,
    widthMm: width * pitch,
    depthMm: height * pitch,
    heightMm: beadHeight(ironing, pitch),
    dispose() { material.dispose(); mesh.dispose(); },
  };
  return group;
}
