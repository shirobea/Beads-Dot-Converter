// ビーズ形状と作品メッシュ。全ビーズを1つの InstancedMesh にまとめ、色は instanceColor で持つ。
// 形状は 5mm ピッチ基準で定義し、実際のピッチに合わせて相似拡大縮小する。
import * as THREE from 'three';
import { mulberry32, plasticSurface, clamp01, dataTexture } from './textures.js';

// roughness は最終的な粗さの目安（粗さマップの平均 0.75 で割って material に渡す）。
// アイロン前は成形したままの半ツヤ、溶かすほどツヤが鈍り表面のムラが増える
export const IRONING = {
  none: { label: 'なし', roughness: 0.35, normalScale: 0.035 },
  one_side: { label: '片面', roughness: 0.45, normalScale: 0.16 },
  both_sides: { label: '両面', roughness: 0.45, normalScale: 0.26 },
};

// 断面（r, y）: 下内→下外→外壁を上へ→上外→上内→内壁を下へ。
// この向きだと LatheGeometry の法線が外向き・穴の内側向きになる。
const SECTIONS = {
  // アイロン前：天面が平らな円筒チューブ。外径＝ピッチで隣と接し、穴径は外径の約4割。
  // 角は成形品のわずかな面取り程度にとどめる（大きく丸めるとドーナツ形に見える）
  none: { rInB: 1.0, rOutB: 2.5, rInT: 1.0, rOutT: 2.5, h: 4.8, cB: 0.18, cT: 0.18, bulge: 0 },
  // 片面：下側はアイロン前のまま、上面が平らに融着して広がり、穴が上側だけ縮む
  one_side: { rInB: 1.0, rOutB: 2.5, rInT: 0.7, rOutT: 2.62, h: 4.3, cB: 0.18, cT: 0.22, bulge: 0.06 },
  // 両面：高さ約3.8mm、穴が小さく、隣のビーズと重なって連結する
  both_sides: { rInB: 0.78, rOutB: 2.63, rInT: 0.72, rOutT: 2.63, h: 3.8, cB: 0.28, cT: 0.24, bulge: 0.14 },
};

const WALL_STEPS = 6;

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
  // 外壁：遮蔽を高さ方向に変えられるよう分割する。溶けたものは中ほどが膨らむ
  for (let k = 1; k < WALL_STEPS; k++) {
    const t = k / WALL_STEPS;
    const y = s.cB + (s.h - s.cT - s.cB) * t;
    pts.push(new THREE.Vector2(s.rOutB + (s.rOutT - s.rOutB) * t + s.bulge * Math.sin(Math.PI * t), y));
  }
  // 上面（外→内）
  arc(pts, s.rOutT - s.cT, s.h - s.cT, s.cT, 0, half, steps);
  arc(pts, s.rInT + s.cT, s.h - s.cT, s.cT, half, Math.PI, steps);
  // 内壁を下へ（始点に戻って閉じる）
  for (let k = 1; k < WALL_STEPS; k++) {
    const t = k / WALL_STEPS;
    pts.push(new THREE.Vector2(s.rInT + (s.rInB - s.rInT) * t, s.h - s.cT - (s.h - s.cT - s.cB) * t));
  }
  pts.push(new THREE.Vector2(s.rInB, s.cB));
  return pts;
}

const smoothstep = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };

// 断面の各点の遮蔽度（1 = 開けている）。
// 光は穴の上からしか入らない（下はマットでふさがる）ので、穴の奥ほど暗い。
// 外壁は隣のビーズと接していて、下へ行くほど隙間の奥になる。
function sectionOcclusion(s, pts) {
  const rMid = (Math.min(s.rInB, s.rInT) + Math.max(s.rOutB, s.rOutT)) / 2;
  const hole = Math.min(s.rInB, s.rInT);
  return pts.map((p) => {
    const top = p.y >= s.h - s.cT - 1e-4;
    const bottom = p.y <= s.cB + 1e-4 && !top;
    if (p.x < rMid) {
      if (top) return 0.97;
      if (bottom) return 0.3;
      return 0.95 - 0.78 * smoothstep(0, hole * 3.2, s.h - p.y);
    }
    if (top) return 1.0;
    if (bottom) return 0.35;
    return 0.42 + 0.56 * Math.pow(p.y / s.h, 1.4);
  });
}

// LatheGeometry の v は断面の点番号に比例するので、点ごとの値を縦1列のテクスチャにする
function occlusionTexture(values) {
  const h = 256;
  const data = new Uint8Array(h * 4 * 4);
  const n = values.length - 1;
  for (let y = 0; y < h; y++) {
    const f = (y / (h - 1)) * n;
    const i0 = Math.floor(f), i1 = Math.min(n, i0 + 1);
    const v = values[i0] + (values[i1] - values[i0]) * (f - i0);
    for (let x = 0; x < 4; x++) {
      const k = (y * 4 + x) * 4;
      data[k] = data[k + 1] = data[k + 2] = clamp01(v) * 255;
      data[k + 3] = 255;
    }
  }
  const tex = dataTexture(data, 4, h, false);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearFilter;
  return tex;
}

const geometryCache = new Map();

export function beadGeometry(ironing, pitch, segments, arcSteps) {
  const key = `${ironing}:${pitch}:${segments}:${arcSteps}`;
  if (geometryCache.has(key)) return geometryCache.get(key);
  const s = SECTIONS[ironing] || SECTIONS.both_sides;
  const pts = sectionPoints(s, arcSteps);
  const geo = new THREE.LatheGeometry(pts, segments);
  geo.scale(pitch / 5, pitch / 5, pitch / 5);
  geo.userData.aoMap = occlusionTexture(sectionOcclusion(s, pts));
  geo.userData.aoMap.userData.shared = true;
  geometryCache.set(key, geo);
  return geo;
}

export function beadHeight(ironing, pitch) {
  return (SECTIONS[ironing] || SECTIONS.both_sides).h * (pitch / 5);
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
  const surface = plasticSurface();
  surface.normal.repeat.set(3, 2);
  surface.roughness.repeat.set(3, 2);
  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    metalness: 0,
    roughness: Math.min(1, spec.roughness / 0.75),
    roughnessMap: surface.roughness,
    normalMap: surface.normal,
    normalScale: new THREE.Vector2(spec.normalScale, spec.normalScale),
    aoMap: geometry.userData.aoMap,
    aoMapIntensity: 1,
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
  const tint = new THREE.Color();
  const ox = (width - 1) / 2, oz = (height - 1) / 2;
  // アイロン前は隣と接して並ぶので形を崩さない。溶かしたものだけ潰れ具合を不揃いにする
  const squash = ironing === 'none' ? 0.0 : 0.035;
  let i = 0;
  for (let r = 0; r < height; r++) {
    const row = cells[r];
    for (let c = 0; c < width; c++) {
      const v = row[c];
      if (v < 0) continue;
      e.set(0, rand() * Math.PI * 2, 0);
      q.setFromEuler(e);
      const sx = 1 + (rand() - 0.5) * squash;
      scl.set(sx, 1 + (rand() - 0.5) * squash, sx);
      pos.set((c - ox) * pitch, 0, (r - oz) * pitch);
      m.compose(pos, q, scl);
      mesh.setMatrixAt(i, m);
      // 同じ色番号でも製造ロットや光の当たり方で粒ごとにわずかに明るさが違う
      tint.copy(colors[v] || colors[0]).multiplyScalar(1 + (rand() - 0.5) * 0.07);
      mesh.setColorAt(i, tint);
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
