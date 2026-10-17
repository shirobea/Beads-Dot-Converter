// デスクを置く部屋：床・壁・天井・巾木・腰窓とカーテン・ドア・天井照明・スイッチ・コンセント。
// 単位は 1 = 1mm、天板の上面が y = 0。壁と天井は内側にだけ面を持つので、
// 部屋の外へカメラを出すと手前の壁が透けて中が見える（断面図のような見え方）。
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { canvasTexture, grainSurface, woodSurface } from './textures.js';

export const ROOM = {
  floorY: -720, // 天板の高さ 72cm
  height: 2400, // 天井高 2.4m
  depth: 3400,
  minWidth: 3000,
  wallGap: 40, // 机の奥の縁と壁のすき間
  // 腰窓：天板の 10cm 上から、床から約 1.97m まで
  window: { w: 1100, bottom: 100, top: 1250, reveal: 120 },
};

const CEIL_Y = ROOM.floorY + ROOM.height;

// 写真から作った PBR テクスチャ（Poly Haven, CC0）。size は 1 枚が実物で何 m 分か
const TEXTURE_SETS = {
  floor: { name: 'laminate_floor_02', maps: ['diff', 'nor_gl', 'rough'], size: 1.7 },
  plaster: { name: 'white_plaster_02', maps: ['nor_gl', 'rough'], size: 1.0 },
  door: { name: 'oak_veneer_01', maps: ['diff', 'nor_gl', 'rough'], size: 1.83 },
  fabric: { name: 'terlenka', maps: ['diff', 'nor_gl', 'rough'], size: 0.266 },
};
let photoTextures = null;

// 起動時に1回だけ読み込む。読めなかったセットは手続き生成のテクスチャで代用する
export async function loadRoomTextures() {
  if (photoTextures) return photoTextures;
  const loader = new THREE.TextureLoader();
  const result = {};
  await Promise.all(Object.entries(TEXTURE_SETS).map(async ([key, set]) => {
    try {
      const maps = {};
      for (const m of set.maps) {
        const t = await loader.loadAsync(`assets/textures/${set.name}_${m}_1k.jpg`);
        t.colorSpace = m === 'diff' ? THREE.SRGBColorSpace : THREE.NoColorSpace;
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.anisotropy = 8;
        t.userData.shared = true;
        maps[m] = t;
      }
      result[key] = maps;
    } catch (err) {
      console.warn(`${set.name} を読み込めなかったため手続き生成のテクスチャを使います`, err);
    }
  }));
  photoTextures = result;
  return result;
}

// 共有テクスチャを、面の UV（1 = 1m）に対して実寸で繰り返すよう複製する。GPU 上の画像は共有される
function tiled(texture, repeatX, repeatY = repeatX) {
  const c = texture.clone();
  c.repeat.set(repeatX, repeatY);
  c.userData.shared = false;
  return c;
}

// 面の大きさに合わせて UV を伸ばし、どの面でもテクスチャの細かさが実寸でそろうようにする（1 = 1m）
function plane(w, h) {
  const geo = new THREE.PlaneGeometry(w, h);
  const uv = geo.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * w / 1000, uv.getY(i) * h / 1000);
  return geo;
}

function mesh(geo, mat, { cast = false, receive = true } = {}) {
  const m = new THREE.Mesh(geo, mat);
  m.castShadow = cast;
  m.receiveShadow = receive;
  return m;
}

function materials() {
  const P = photoTextures || {};
  const wallColor = new THREE.Color().setRGB(245 / 255, 243 / 255, 238 / 255, THREE.SRGBColorSpace);
  // 壁と天井：白い塗り壁。写真の凹凸とツヤのムラだけ使い、色は汚れのない白にする。
  // 壁は外から光を遮るので影を両面で描く
  let wall, ceiling;
  if (P.plaster) {
    const n = 1 / TEXTURE_SETS.plaster.size;
    wall = new THREE.MeshStandardMaterial({
      color: wallColor, roughness: 1, roughnessMap: tiled(P.plaster.rough, n),
      normalMap: tiled(P.plaster.nor_gl, n), normalScale: new THREE.Vector2(0.6, 0.6), shadowSide: THREE.DoubleSide,
    });
    // 天井はクロスの細かな凹凸に見えるよう、同じ写真を 2 倍細かく使う
    ceiling = new THREE.MeshStandardMaterial({
      color: 0xf8f8f6, roughness: 1, roughnessMap: tiled(P.plaster.rough, 2 * n),
      normalMap: tiled(P.plaster.nor_gl, 2 * n), normalScale: new THREE.Vector2(0.4, 0.4), shadowSide: THREE.DoubleSide,
    });
  } else {
    const paint = grainSurface(24, 5, 0.06);
    wall = new THREE.MeshStandardMaterial({
      color: wallColor, roughness: 0.92, normalMap: tiled(paint.normal, 4), normalScale: new THREE.Vector2(0.12, 0.12),
      shadowSide: THREE.DoubleSide,
    });
    ceiling = new THREE.MeshStandardMaterial({ color: 0xf8f8f6, roughness: 0.95, shadowSide: THREE.DoubleSide });
  }
  // 床：フローリング（板は奥行方向に流す）
  let floor;
  if (P.floor) {
    const n = 1 / TEXTURE_SETS.floor.size;
    const along = (t) => { const c = tiled(t, n); c.rotation = Math.PI / 2; return c; };
    floor = new THREE.MeshStandardMaterial({
      color: 0xffffff, roughness: 1, map: along(P.floor.diff), roughnessMap: along(P.floor.rough), normalMap: along(P.floor.nor_gl),
    });
  } else {
    // 手続き生成の板目：1 枚に 6 枚の板（UV 1 = 1m なので 0.9m で 1 周）
    const wood = woodSurface(1024, 1024, 31, 'floor');
    const n = 1 / 0.9;
    floor = new THREE.MeshPhysicalMaterial({
      color: 0xffffff, roughness: 1, map: tiled(wood.map, n), roughnessMap: tiled(wood.roughness, n),
      bumpMap: tiled(wood.bump, n), bumpScale: 0.8, clearcoat: 0.3, clearcoatRoughness: 0.35,
    });
  }
  // ドア：オークの突板。ドア板（幅 0.78m × 高さ 2.0m）の面に実寸で貼る（木目は縦）
  let door;
  if (P.door) {
    const sx = 0.78 / TEXTURE_SETS.door.size, sy = 2.0 / TEXTURE_SETS.door.size;
    door = new THREE.MeshStandardMaterial({
      color: 0xffffff, roughness: 1, map: tiled(P.door.diff, sx, sy), roughnessMap: tiled(P.door.rough, sx, sy),
      normalMap: tiled(P.door.nor_gl, sx, sy), normalScale: new THREE.Vector2(0.6, 0.6),
    });
  } else {
    door = new THREE.MeshStandardMaterial({ color: 0xc4a582, roughness: 0.55 });
  }
  // カーテン：細かい織りの布（UV 1 = 1m）
  let fabric;
  if (P.fabric) {
    const n = 1 / TEXTURE_SETS.fabric.size;
    fabric = new THREE.MeshStandardMaterial({
      color: 0xffffff, roughness: 1, map: tiled(P.fabric.diff, n), roughnessMap: tiled(P.fabric.rough, n),
      normalMap: tiled(P.fabric.nor_gl, n), side: THREE.DoubleSide,
    });
  } else {
    fabric = new THREE.MeshStandardMaterial({ color: 0xdcd3c4, roughness: 0.95, side: THREE.DoubleSide });
  }
  const trim = new THREE.MeshStandardMaterial({ color: 0xf2f0ea, roughness: 0.5 });
  const frame = new THREE.MeshStandardMaterial({ color: 0xd9dadc, roughness: 0.35, metalness: 0.6 });
  const metal = new THREE.MeshStandardMaterial({ color: 0xb8b8bc, roughness: 0.25, metalness: 1 });
  return { wall, ceiling, floor, trim, frame, fabric, door, metal };
}

// 窓の外の景色（空と遠くの街並み）。発光させ、昼は白飛び気味、夜は暗い紺にする。
// 室内に露出を合わせた写真のように、ピントの外れたぼけた景色にする
function outsideTexture() {
  return canvasTexture(1024, 768, (ctx, w, h) => {
    ctx.filter = 'blur(10px)';
    const sky = ctx.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, '#9cc4ea');
    sky.addColorStop(0.65, '#dceaf6');
    sky.addColorStop(1, '#eef2f2');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);
    let seed = 9;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    // 遠景のビルと木（低いコントラストでぼんやり）
    for (let i = 0; i < 26; i++) {
      const bw = 30 + rnd() * 90, bh = 60 + rnd() * 260, x = rnd() * w;
      ctx.fillStyle = `rgba(${150 + rnd() * 40},${160 + rnd() * 30},${175 + rnd() * 30},0.55)`;
      ctx.fillRect(x, h * 0.72 - bh, bw, bh + h * 0.3);
    }
    for (let i = 0; i < 40; i++) {
      ctx.fillStyle = `rgba(${90 + rnd() * 40},${130 + rnd() * 40},${90 + rnd() * 30},0.5)`;
      ctx.beginPath();
      ctx.arc(rnd() * w, h * 0.8 + rnd() * 60, 30 + rnd() * 50, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

// ひだのあるカーテン（平面を波打たせる）。UV は 1 = 1m
function curtain(w, h, mat) {
  const geo = new THREE.PlaneGeometry(w, h, 32, 1);
  const uv = geo.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * w / 1000, uv.getY(i) * h / 1000);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) p.setZ(i, Math.sin((p.getX(i) / w) * Math.PI * 9) * 22);
  geo.computeVertexNormals();
  return mesh(geo, mat, { cast: true });
}

/**
 * 部屋を作る。
 * desk: { W, D } 机の寸法、art: 作品の中心（x, z）、
 * light: { direction, inside, color }。inside が false なら主光源は窓から差す日光で、
 * 窓をその延長上に置き、壁とカーテンで光を遮る。true なら天井照明が光源なので、
 * 照明器具をその方向に置いて点灯させ、壁は光を遮らない。
 */
export function buildRoom({ desk, art, light }) {
  const M = materials();
  const g = new THREE.Group();
  const roomW = Math.max(ROOM.minWidth, desk.W + 1400);
  const backZ = -desk.D / 2 - ROOM.wallGap;
  const frontZ = backZ + ROOM.depth;
  const left = -roomW / 2, right = roomW / 2;
  const midZ = (backZ + frontZ) / 2;
  const fy = ROOM.floorY;
  const wallH = ROOM.height;
  const casters = [];
  const blocking = !light.inside;

  // ---- 床・天井 ----
  const floor = mesh(plane(roomW, ROOM.depth), M.floor);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, fy, midZ);
  g.add(floor);
  const ceiling = mesh(plane(roomW, ROOM.depth), M.ceiling);
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.set(0, CEIL_Y, midZ);
  g.add(ceiling);

  // ---- 奥の壁（窓の開口を空けて 4 枚で組む） ----
  // 窓は作品の中心から主光源へ伸ばした線が壁に当たる位置（左右だけ合わせ、高さは腰窓で固定）
  const win = ROOM.window;
  let wx = left + 900;
  if (light.direction && light.direction.z < -0.05) {
    const t = (art.z - backZ) / -light.direction.z;
    wx = art.x + light.direction.x * t;
  }
  wx = THREE.MathUtils.clamp(wx, left + win.w / 2 + 250, right - win.w / 2 - 250);
  const wx0 = wx - win.w / 2, wx1 = wx + win.w / 2;
  const wy0 = win.bottom, wy1 = win.top;
  const backPart = (x0, x1, y0, y1) => {
    if (x1 - x0 < 1 || y1 - y0 < 1) return;
    const m = mesh(plane(x1 - x0, y1 - y0), M.wall, { cast: blocking });
    m.position.set((x0 + x1) / 2, (y0 + y1) / 2, backZ);
    g.add(m);
    casters.push(m);
  };
  backPart(left, wx0, fy, CEIL_Y);
  backPart(wx1, right, fy, CEIL_Y);
  backPart(wx0, wx1, fy, wy0);
  backPart(wx0, wx1, wy1, CEIL_Y);
  // 開口の側面（壁の厚み）
  const rd = win.reveal;
  const reveal = (w, h, pos, rot) => {
    const m = mesh(plane(w, h), M.wall, { cast: blocking });
    m.position.set(...pos);
    m.rotation.set(...rot);
    g.add(m);
    casters.push(m);
  };
  reveal(rd, wy1 - wy0, [wx0, (wy0 + wy1) / 2, backZ - rd / 2], [0, Math.PI / 2, 0]);
  reveal(rd, wy1 - wy0, [wx1, (wy0 + wy1) / 2, backZ - rd / 2], [0, -Math.PI / 2, 0]);
  reveal(win.w, rd, [wx, wy0, backZ - rd / 2], [-Math.PI / 2, 0, 0]);
  reveal(win.w, rd, [wx, wy1, backZ - rd / 2], [Math.PI / 2, 0, 0]);
  // 窓台（白い化粧板）
  const sill = mesh(new RoundedBoxGeometry(win.w + 60, 18, rd + 30, 2, 3), M.trim, { cast: blocking });
  sill.position.set(wx, wy0 - 9, backZ - rd / 2 + 15);
  g.add(sill);
  // サッシ（引き違い窓：外枠＋中央の召し合わせ）
  const sz = backZ - rd + 30;
  const bar = (w, h, x, y) => {
    const m = mesh(new THREE.BoxGeometry(w, h, 40), M.frame, { cast: blocking });
    m.position.set(x, y, sz);
    g.add(m);
  };
  bar(win.w, 45, wx, wy0 + 22);
  bar(win.w, 45, wx, wy1 - 22);
  bar(45, wy1 - wy0, wx0 + 22, (wy0 + wy1) / 2);
  bar(45, wy1 - wy0, wx1 - 22, (wy0 + wy1) / 2);
  bar(50, wy1 - wy0, wx, (wy0 + wy1) / 2);
  // 窓の外の景色。影は落とさず受けもしない
  const outsideMat = new THREE.MeshBasicMaterial({ map: outsideTexture() });
  const outside = new THREE.Mesh(new THREE.PlaneGeometry(win.w * 3, (wy1 - wy0) * 2.4), outsideMat);
  outside.position.set(wx, (wy0 + wy1) / 2 + 150, backZ - rd - 600);
  g.add(outside);
  // カーテン（左右に寄せて開けた状態）とカーテンレール。
  // 昼は窓の光が薄い布を透けて明るく見えるので、布の色で少しだけ発光させる（夜は透けない）
  if (!light.inside && M.fabric.map) {
    M.fabric.emissive.setRGB(1, 1, 1);
    M.fabric.emissiveMap = M.fabric.map;
    M.fabric.emissiveIntensity = 0.35;
  }
  const cw = 330, ch = wy1 - wy0 + 330;
  for (const [x, s] of [[wx0 - cw / 2 + 90, 1], [wx1 + cw / 2 - 90, -1]]) {
    const c = curtain(cw, ch, M.fabric);
    c.castShadow = blocking;
    c.position.set(x, wy1 + 150 - ch / 2, backZ + 55);
    c.scale.x = s;
    g.add(c);
    casters.push(c);
  }
  const rail = mesh(new RoundedBoxGeometry(win.w + 2 * cw, 30, 40, 2, 6), M.trim, { cast: blocking });
  rail.position.set(wx, wy1 + 165, backZ + 40);
  g.add(rail);

  // ---- 左右・手前の壁 ----
  const side = (w, x, z, rotY) => {
    const m = mesh(plane(w, wallH), M.wall, { cast: blocking });
    m.position.set(x, fy + wallH / 2, z);
    m.rotation.y = rotY;
    g.add(m);
    casters.push(m);
  };
  side(ROOM.depth, left, midZ, Math.PI / 2);
  side(ROOM.depth, right, midZ, -Math.PI / 2);
  side(roomW, 0, frontZ, Math.PI);

  // ---- 巾木 ----
  const skirt = (len, x, z, rotY) => {
    const m = mesh(new THREE.BoxGeometry(len, 60, 12), M.trim);
    m.position.set(x, fy + 30, z);
    m.rotation.y = rotY;
    g.add(m);
  };
  skirt(roomW, 0, backZ + 6, 0);
  skirt(roomW, 0, frontZ - 6, 0);
  skirt(ROOM.depth, left + 6, midZ, Math.PI / 2);
  skirt(ROOM.depth, right - 6, midZ, Math.PI / 2);

  // ---- ドア（右の壁の手前寄り）とスイッチ ----
  const doorZ = frontZ - 900;
  const panel = mesh(new RoundedBoxGeometry(36, 2000, 780, 2, 4), M.door, { cast: true });
  panel.position.set(right - 18, fy + 1000, doorZ);
  g.add(panel);
  // 枠（左右の縦枠と上枠）。[高さ, 奥行方向の幅, 中心の高さ, 中心の z]
  for (const [h, w, y, z] of [[2060, 60, fy + 1030, doorZ - 420], [2060, 60, fy + 1030, doorZ + 420], [60, 900, fy + 2030, doorZ]]) {
    const casing = mesh(new THREE.BoxGeometry(14, h, w), M.trim);
    casing.position.set(right - 7, y, z);
    g.add(casing);
  }
  const lever = mesh(new THREE.CylinderGeometry(9, 9, 120, 16), M.metal, { cast: true });
  lever.rotation.x = Math.PI / 2;
  lever.position.set(right - 70, fy + 950, doorZ - 300);
  g.add(lever);
  const rose = mesh(new THREE.CylinderGeometry(26, 26, 14, 24), M.metal);
  rose.rotation.z = Math.PI / 2;
  rose.position.set(right - 40, fy + 950, doorZ - 330);
  g.add(rose);
  const sw = mesh(new RoundedBoxGeometry(8, 120, 72, 1, 2), M.trim);
  sw.position.set(right - 4, fy + 1200, doorZ - 560);
  g.add(sw);
  // コンセント（机の奥の壁、床から 25cm）
  const outlet = mesh(new RoundedBoxGeometry(72, 120, 8, 1, 2), M.trim);
  outlet.position.set(320, fy + 250, backZ + 4);
  g.add(outlet);

  // ---- 天井照明（丸いシーリングライト） ----
  // 室内灯が光源のプリセットでは、作品から主光源へ伸ばした線が天井に当たる位置に置いて点灯させる
  let lx = 0, lz = midZ - 600;
  if (light.inside && light.direction && light.direction.y > 0.1) {
    const t = (CEIL_Y - 0) / light.direction.y;
    lx = art.x + light.direction.x * t;
    lz = art.z + light.direction.z * t;
  }
  lx = THREE.MathUtils.clamp(lx, left + 400, right - 400);
  lz = THREE.MathUtils.clamp(lz, backZ + 400, frontZ - 400);
  const body = mesh(new THREE.CylinderGeometry(270, 290, 70, 64), M.trim);
  body.position.set(lx, CEIL_Y - 35, lz);
  g.add(body);
  const lampMat = new THREE.MeshStandardMaterial({ color: 0xf6f6f4, roughness: 0.6, emissive: 0xffffff, emissiveIntensity: 0 });
  if (light.inside) {
    lampMat.emissive.copy(light.color || new THREE.Color(1, 1, 1));
    lampMat.emissiveIntensity = 2.2;
  }
  const diffuser = new THREE.Mesh(new THREE.CircleGeometry(255, 64), lampMat);
  diffuser.rotation.x = Math.PI / 2;
  diffuser.position.set(lx, CEIL_Y - 71, lz);
  g.add(diffuser);

  // 窓の外：昼は白飛び気味の明るさ、夜（室内灯のプリセット）は暗い紺
  if (light.inside) outsideMat.color.setRGB(0.012, 0.016, 0.03);
  else outsideMat.color.setScalar(3.2);

  const bounds = new THREE.Box3(new THREE.Vector3(left, fy, backZ - rd), new THREE.Vector3(right, CEIL_Y, frontZ));
  return { group: g, bounds, window: { x0: wx0, x1: wx1, y0: wy0, y1: wy1, z: backZ }, casters };
}
