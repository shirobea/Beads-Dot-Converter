// 手続き的に作るテクスチャ（外部画像を使わずに質感を出すためのもの）。
// 色テクスチャは sRGB、法線・粗さ・高さはリニアとして扱う。
import * as THREE from 'three';

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

const smooth = (t) => t * t * (3 - 2 * t);
const lerp = (a, b, t) => a + (b - a) * t;
export const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

// 座標から決まる値ノイズ（0〜1）。period を渡すとその周期で繰り返す
function hash2(x, y, seed) {
  let h = Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(seed, 1442695041);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

export function valueNoise(x, y, seed = 0, period = 0) {
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const fx = smooth(x - x0), fy = smooth(y - y0);
  const w = (v) => (period ? ((v % period) + period) % period : v);
  const a = hash2(w(x0), w(y0), seed), b = hash2(w(x0 + 1), w(y0), seed);
  const c = hash2(w(x0), w(y0 + 1), seed), d = hash2(w(x0 + 1), w(y0 + 1), seed);
  return lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
}

export function fbm(x, y, seed = 0, octaves = 4, period = 0) {
  let sum = 0, amp = 0.5, norm = 0, f = 1;
  for (let i = 0; i < octaves; i++) {
    sum += valueNoise(x * f, y * f, seed + i * 17, period ? period * f : 0) * amp;
    norm += amp;
    amp *= 0.5;
    f *= 2;
  }
  return sum / norm;
}

// 高さ配列 → 法線マップ（端はラップ）
export function normalTextureFromHeight(height, w, h, strength = 2) {
  const data = new Uint8Array(w * h * 4);
  const at = (x, y) => height[((y + h) % h) * w + ((x + w) % w)];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = (at(x + 1, y) - at(x - 1, y)) * strength;
      const dy = (at(x, y + 1) - at(x, y - 1)) * strength;
      const len = Math.hypot(dx, dy, 1);
      const i = (y * w + x) * 4;
      data[i] = (-dx / len * 0.5 + 0.5) * 255;
      data[i + 1] = (dy / len * 0.5 + 0.5) * 255;
      data[i + 2] = (1 / len * 0.5 + 0.5) * 255;
      data[i + 3] = 255;
    }
  }
  return dataTexture(data, w, h, false);
}

// 0〜1 の配列 → グレースケールのテクスチャ（粗さ・バンプ用）
export function grayTexture(values, w, h) {
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const v = clamp01(values[i]) * 255;
    data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  }
  return dataTexture(data, w, h, false);
}

export function dataTexture(data, w, h, srgb) {
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 8;
  if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

export function canvasTexture(w, h, draw, srgb = true) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  draw(canvas.getContext('2d'), w, h);
  const tex = new THREE.CanvasTexture(canvas);
  if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

// 周期的なノイズの高さ場（タイル可能）
export function tileableHeight(n, cells, seed, octaves = 3) {
  const out = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      out[y * n + x] = fbm((x / n) * cells, (y / n) * cells, seed, octaves, cells);
    }
  }
  return out;
}

// ---- 素材ごとのテクスチャ ---------------------------------------------------

const cache = new Map();
// 使い回すテクスチャには shared を立て、シーン再構築時に破棄されないようにする
function cached(key, make) {
  if (!cache.has(key)) {
    const v = make();
    for (const t of v?.isTexture ? [v] : Object.values(v)) if (t?.isTexture) t.userData.shared = true;
    cache.set(key, v);
  }
  return cache.get(key);
}

// 樹脂の細かなムラ（成形肌・溶けムラ用）。normal と roughness の組
export function plasticSurface(seed = 7) {
  return cached(`plastic:${seed}`, () => {
    const n = 256;
    const height = tileableHeight(n, 8, seed, 4);
    const rough = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) rough[i] = 0.75 + (height[i] - 0.5) * 0.9;
    return { normal: normalTextureFromHeight(height, n, n, 3), roughness: grayTexture(rough, n, n) };
  });
}

// 天板の木目（ウォールナット寄りのオーク）。map / roughness / bump をまとめて返す
export function woodSurface(w = 2048, h = 1024, seed = 11, tone = 'oak') {
  return cached(`wood:${w}:${h}:${seed}:${tone}`, () => {
    const palette = tone === 'floor'
      ? [[120, 84, 56], [86, 58, 38]]
      : [[198, 160, 118], [150, 108, 70]];
    const color = new Uint8Array(w * h * 4);
    const rough = new Float32Array(w * h);
    const bump = new Float32Array(w * h);
    const plank = tone === 'floor' ? h / 6 : 0;
    const rand = mulberry32(seed);
    const pores = [];
    for (let i = 0; i < (w * h) / 2500; i++) pores.push([rand() * w, rand() * h, 30 + rand() * 90]);
    for (let y = 0; y < h; y++) {
      const row = plank ? Math.floor(y / plank) : 0;
      for (let x = 0; x < w; x++) {
        const nx = x / w, ny = y / h;
        // 年輪：縦方向の位置をノイズでゆがめた縞
        const warp = fbm(nx * 3 + row * 5.3, ny * 9, seed, 3) * 7;
        const ring = Math.sin((ny * 38 + warp + row * 1.7) * Math.PI);
        const grain = Math.pow(Math.abs(ring), 0.8);
        const fine = valueNoise(nx * 30, ny * 700, seed + 3);
        let t = clamp01(0.62 * grain + 0.38 * fine);
        const blotch = fbm(nx * 4 + row, ny * 3, seed + 9, 2) - 0.5;
        t = clamp01(t + blotch * 0.25);
        const i = y * w + x;
        let edge = 1;
        if (plank) {
          const py = y % plank;
          if (py < 2 || py > plank - 2) edge = 0.55;
        }
        for (let c = 0; c < 3; c++) {
          color[i * 4 + c] = lerp(palette[1][c], palette[0][c], t) * edge;
        }
        color[i * 4 + 3] = 255;
        rough[i] = 0.42 + (1 - t) * 0.18 + fine * 0.08;
        bump[i] = 0.5 + (t - 0.5) * 0.35 + (fine - 0.5) * 0.25 - (1 - edge) * 0.5;
      }
    }
    // 導管（細長い溝）を木目方向に散らす
    for (const [px, py, len] of pores) {
      for (let k = 0; k < len; k++) {
        const x = Math.floor(px + k) % w, y = Math.floor(py);
        const i = y * w + x;
        for (let c = 0; c < 3; c++) color[i * 4 + c] *= 0.9;
        bump[i] -= 0.15;
        rough[i] += 0.08;
      }
    }
    const map = dataTexture(color, w, h, true);
    return { map, roughness: grayTexture(rough, w, h), bump: grayTexture(bump, w, h) };
  });
}

// グレースケールのノイズ画像（canvas）。cells はノイズの粗さ
function noiseCanvas(n, seed, cells, octaves) {
  return cached(`noiseCanvas:${n}:${seed}:${cells}:${octaves}`, () => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = n;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(n, n);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const v = octaves > 1 ? fbm(x / n * cells, y / n * cells, seed, octaves, cells) : hash2(x, y, seed);
        const i = (y * n + x) * 4;
        img.data[i] = img.data[i + 1] = img.data[i + 2] = v * 255;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return canvas;
  });
}

// カッティングマット：方眼・目盛り・細かな切り傷。w, d は mm
export function matSurface(wMm, dMm) {
  const pxPerMm = Math.min(4, 2048 / Math.max(wMm, dMm));
  const W = Math.round(wMm * pxPerMm), H = Math.round(dMm * pxPerMm);
  const rand = mulberry32(5);
  const scratches = [];
  for (let i = 0; i < 70; i++) {
    const x = rand() * W, y = rand() * H, a = rand() * Math.PI, l = (8 + rand() * 60) * pxPerMm;
    scratches.push([x, y, x + Math.cos(a) * l, y + Math.sin(a) * l]);
  }
  const map = canvasTexture(W, H, (ctx) => {
    ctx.fillStyle = '#2d5c4c';
    ctx.fillRect(0, 0, W, H);
    // うっすらした色ムラ（小さなノイズ画像を引き伸ばして重ねる）＋細かなざらつき
    ctx.globalCompositeOperation = 'soft-light';
    ctx.globalAlpha = 0.35;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(noiseCanvas(48, 21, 6, 3), 0, 0, W, H);
    ctx.globalAlpha = 0.12;
    ctx.fillStyle = ctx.createPattern(noiseCanvas(128, 4, 128, 1), 'repeat');
    ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    const cx = W / 2, cy = H / 2;
    for (const [step, alpha, lw] of [[10, 0.22, 1], [50, 0.5, 1.6]]) {
      ctx.strokeStyle = `rgba(220,240,230,${alpha})`;
      ctx.lineWidth = lw;
      ctx.beginPath();
      for (let mm = 0; mm <= wMm / 2; mm += step) {
        for (const s of [1, -1]) { const x = cx + s * mm * pxPerMm; ctx.moveTo(x, 0); ctx.lineTo(x, H); }
      }
      for (let mm = 0; mm <= dMm / 2; mm += step) {
        for (const s of [1, -1]) { const y = cy + s * mm * pxPerMm; ctx.moveTo(0, y); ctx.lineTo(W, y); }
      }
      ctx.stroke();
    }
    ctx.fillStyle = 'rgba(230,245,238,0.75)';
    ctx.font = `${Math.max(10, 5 * pxPerMm)}px sans-serif`;
    ctx.textAlign = 'center';
    for (let mm = -Math.floor(wMm / 100) * 50; mm <= wMm / 2 - 20; mm += 50) {
      if (mm === 0) continue;
      ctx.fillText(`${mm / 10}`, cx + mm * pxPerMm, H - 3 * pxPerMm);
    }
    // 使い込んだ切り傷（明るい細線）
    ctx.strokeStyle = 'rgba(200,225,215,0.13)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (const [x0, y0, x1, y1] of scratches) { ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); }
    ctx.stroke();
  });
  const bump = canvasTexture(W, H, (ctx) => {
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = '#3a3a3a';
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    for (const [x0, y0, x1, y1] of scratches) { ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); }
    ctx.stroke();
  }, false);
  return { map, bump };
}

// 細かい凹凸のタイル（樹脂のシボ・塗装面など）。scale は 1 タイルあたりのセル数
export function grainSurface(cells = 32, seed = 3, amount = 0.15) {
  return cached(`grain:${cells}:${seed}:${amount}`, () => {
    const n = 256;
    const height = tileableHeight(n, cells, seed, 2);
    const rough = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) rough[i] = 1 - amount + (height[i] - 0.5) * amount * 2;
    return { normal: normalTextureFromHeight(height, n, n, 2.5), roughness: grayTexture(rough, n, n) };
  });
}
