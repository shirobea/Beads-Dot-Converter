"""簡易リグレッションテスト。

images/ にある画像すべてを cases.json の各設定（profiles）で変換し、基準（baseline/）と
今回（results/）を数値と比較画像で並べる。ケース名は「画像名__設定名」。
合否は判定せず、基準からの変化率が大きいケースに印を付ける。

    python tests/regression/run_regression.py --baseline   # 現在の結果を基準として保存
    python tests/regression/run_regression.py              # 今回の結果を作り、基準と比較
    python tests/regression/run_regression.py --case rgb   # 名前に rgb を含むケースだけ
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv2_utils import require_cv2  # noqa: E402
from color_spaces import rgb_to_lab  # noqa: E402
from converter import convert_image, ShadingConfig, DitherConfig, PostFilterConfig, BalanceConfig  # noqa: E402
from converter.io_utils import _load_image_rgb  # noqa: E402
from palette import load_palette, resolve_palette_path  # noqa: E402

CASES_PATH = HERE / "cases.json"
IMAGES_DIR = HERE / "images"
BASELINE_DIR = HERE / "baseline"
RESULTS_DIR = HERE / "results"
SEED = 12345
# images/ から拾う拡張子（cv2.imread で読めるもの）
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
# 比較画像の 1 パネルの長辺（ニアレストネイバーで拡大する）
PANEL_LONG_SIDE = 320

METRIC_KEYS = ["colors", "islands", "mean_dL", "mean_dC", "mean_dE"]
METRIC_LABELS = {
    "colors": "使用色数",
    "islands": "アイランド数",
    "mean_dL": "平均L差",
    "mean_dC": "平均C差",
    "mean_dE": "平均ΔE2000",
}


def _load_module_from_file(name: str, path: Path):
    """ui/__init__.py（Tk・OpenGL を読み込む）を通さずに ui 配下の単体モジュールを読む。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclass が sys.modules から自分のモジュールを引くため、実行前に登録する
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ui_models = _load_module_from_file("_regression_ui_models", ROOT / "ui" / "models.py")
ConversionRequest = _ui_models.ConversionRequest


def _fix_seed() -> None:
    """乱数を使う処理が入っても毎回同じ結果になるよう、各ケースの前にシードを固定する。"""
    random.seed(SEED)
    np.random.seed(SEED)
    require_cv2().setRNGSeed(SEED)


# ---------------------------------------------------------------- ケース定義

def _build_request(settings: dict) -> ConversionRequest:
    """cases.json の設定（ConversionRequest と同じ項目名）からリクエストを組み立てる。"""
    data = dict(settings)
    data["rgb_weights"] = tuple(data.get("rgb_weights", (1.0, 1.0, 1.0)))
    shading = dict(data.get("shading", {}))
    if "normal_light_dir" in shading:
        shading["normal_light_dir"] = tuple(shading["normal_light_dir"])
    data["shading"] = ShadingConfig(**shading)
    data["dither"] = DitherConfig(**data.get("dither", {}))
    data["post_filter"] = PostFilterConfig(**data.get("post_filter", {}))
    data["balance"] = BalanceConfig(**data.get("balance", {}))
    request = ConversionRequest(**data)
    if request.mode == "全て":
        raise ValueError("リグレッションテストでは mode「全て」は使えません。モードごとにケースを分けてください。")
    return request


def list_images() -> dict[str, str]:
    """images/ の画像を {ケース名に使う名前: ファイル名} で返す。拡張子違いの同名は拡張子を付けて区別する。"""
    files = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ) if IMAGES_DIR.exists() else []
    stems = [p.stem for p in files]
    return {
        (p.stem if stems.count(p.stem) == 1 else f"{p.stem}_{p.suffix[1:].lower()}"): p.name
        for p in files
    }


def load_cases(name_filter: str | None) -> list[dict]:
    """images/ の全画像 × cases.json の全設定をケースにする。"""
    raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    profiles = []
    for entry in raw["profiles"]:
        settings = _deep_merge(defaults.get("settings", {}), entry.get("settings", {}))
        profiles.append({
            "name": entry["name"],
            "palette": entry.get("palette", defaults.get("palette")),
            "noise_filter": entry.get("noise_filter", defaults.get("noise_filter")),
            "request": _build_request(settings),
        })
    cases = []
    for image_key, filename in list_images().items():
        for profile in profiles:
            name = f"{image_key}__{profile['name']}"
            if name_filter and name_filter not in name:
                continue
            cases.append({**profile, "name": name, "image": filename})
    return cases


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------- 変換

def _apply_noise_filter(image_rgb: np.ndarray, spec: dict | None) -> np.ndarray:
    """UI の「ノイズ除去」と同じフィルタを変換前にかける（name/size/sigma）。"""
    if not spec:
        return image_rgb
    from PIL import Image

    noise = _load_module_from_file("_regression_noise_filters", ROOT / "ui" / "noise_filters.py")
    registry = noise.build_noise_filter_registry(int(spec.get("size", 3)), int(spec.get("sigma", 75)))
    filtered = registry[spec["name"]](Image.fromarray(image_rgb))
    return np.asarray(filtered.convert("RGB"), dtype=np.uint8)


def run_case(case: dict, palettes: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """1 ケースを変換し、(入力の縮小画像, 出力, 秒数) を返す。"""
    req: ConversionRequest = case["request"]
    palette_name = case["palette"]
    if palette_name not in palettes:
        palettes[palette_name] = load_palette(resolve_palette_path(palette_name))
    image_rgb = _load_image_rgb(str(IMAGES_DIR / case["image"]))
    image_rgb = _apply_noise_filter(image_rgb, case["noise_filter"])

    _fix_seed()
    start = time.perf_counter()
    output = convert_image(
        input_path=None,
        input_image=image_rgb,
        output_size=(req.width, req.height),
        mode=req.mode,
        palette=palettes[palette_name],
        keep_aspect=req.keep_aspect,
        resize_method=req.resize_method,
        lab_metric=req.lab_metric,
        cmc_l=req.cmc_l,
        cmc_c=req.cmc_c,
        rgb_weights=req.rgb_weights,
        use_super_sampling=req.use_super_sampling,
        shading=req.shading,
        dither=req.dither,
        post_filter=req.post_filter,
        balance=req.balance,
    )
    elapsed = time.perf_counter() - start

    # 比較用の入力は、変換時の設定に関係なく面積平均で出力と同じサイズへ縮小する
    cv2 = require_cv2()
    h, w = output.shape[:2]
    reference = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
    return reference, np.ascontiguousarray(output[:, :, :3], dtype=np.uint8), elapsed


# ---------------------------------------------------------------- 計測

def _ciede2000_pairwise(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """同じ形の Lab 配列どうしの画素ごとの CIEDE2000。"""
    l1, a1, b1 = (lab1[..., i].astype(np.float64) for i in range(3))
    l2, a2, b2 = (lab2[..., i].astype(np.float64) for i in range(3))
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    avg_c7 = ((c1 + c2) * 0.5) ** 7
    g = 0.5 * (1 - np.sqrt(avg_c7 / (avg_c7 + 25.0 ** 7)))
    a1p = (1 + g) * a1
    a2p = (1 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dl = l2 - l1
    dc = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0.0, dh)
    dhh = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2)

    avg_l = (l1 + l2) * 0.5
    avg_cp = (c1p + c2p) * 0.5
    hsum = h1p + h2p
    avg_hp = np.where(
        c1p * c2p == 0,
        hsum,
        np.where(np.abs(h1p - h2p) <= 180, hsum / 2,
                 np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)),
    )
    t = (1 - 0.17 * np.cos(np.radians(avg_hp - 30)) + 0.24 * np.cos(np.radians(2 * avg_hp))
         + 0.32 * np.cos(np.radians(3 * avg_hp + 6)) - 0.20 * np.cos(np.radians(4 * avg_hp - 63)))
    d_theta = 30 * np.exp(-(((avg_hp - 275) / 25) ** 2))
    avg_cp7 = avg_cp ** 7
    rc = 2 * np.sqrt(avg_cp7 / (avg_cp7 + 25.0 ** 7))
    sl = 1 + (0.015 * (avg_l - 50) ** 2) / np.sqrt(20 + (avg_l - 50) ** 2)
    sc = 1 + 0.045 * avg_cp
    sh = 1 + 0.015 * avg_cp * t
    rt = -np.sin(np.radians(2 * d_theta)) * rc
    return np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dhh / sh) ** 2 + rt * (dc / sc) * (dhh / sh))


def count_islands(image: np.ndarray) -> int:
    """上下左右に同じ色が 1 つもない画素の数。"""
    packed = (image[..., 0].astype(np.int32) << 16) | (image[..., 1].astype(np.int32) << 8) | image[..., 2]
    padded = np.pad(packed, 1, constant_values=-1)
    center = padded[1:-1, 1:-1]
    same = (
        (padded[:-2, 1:-1] == center)
        | (padded[2:, 1:-1] == center)
        | (padded[1:-1, :-2] == center)
        | (padded[1:-1, 2:] == center)
    )
    return int((~same).sum())


def compute_metrics(reference: np.ndarray, output: np.ndarray) -> dict:
    lab_ref = rgb_to_lab(reference)
    lab_out = rgb_to_lab(output)
    chroma_ref = np.hypot(lab_ref[..., 1], lab_ref[..., 2])
    chroma_out = np.hypot(lab_out[..., 1], lab_out[..., 2])
    return {
        "colors": int(len(np.unique(output.reshape(-1, 3), axis=0))),
        "islands": count_islands(output),
        # 出力 − 入力。負なら出力が暗い／くすんでいる
        "mean_dL": round(float(lab_out[..., 0].mean() - lab_ref[..., 0].mean()), 3),
        "mean_dC": round(float(chroma_out.mean() - chroma_ref.mean()), 3),
        "mean_dE": round(float(_ciede2000_pairwise(lab_ref, lab_out).mean()), 3),
    }


def change_mask(baseline: np.ndarray | None, current: np.ndarray) -> np.ndarray | None:
    """基準と色が異なる画素の真偽マスク。サイズが違えば None。"""
    if baseline is None or baseline.shape != current.shape:
        return None
    return np.any(baseline != current, axis=2)


# ---------------------------------------------------------------- 画像入出力

def _write_png(path: Path, rgb: np.ndarray) -> None:
    cv2 = require_cv2()
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"PNG を書き出せませんでした: {path}")
    path.write_bytes(buf.tobytes())


def _read_png(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return _load_image_rgb(str(path))


def _upscale(rgb: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def build_comparison(reference: np.ndarray, baseline: np.ndarray | None,
                     current: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """入力（縮小）／基準／今回／変化マスク を横に並べた画像。"""
    from PIL import Image, ImageDraw

    h, w = current.shape[:2]
    scale = max(1, PANEL_LONG_SIDE // max(h, w))
    blank = np.full_like(current, 255)
    if mask is None:
        mask_rgb = np.full_like(current, 200)  # 基準が無い／サイズ違いは灰色
    else:
        # 変化した画素を赤、それ以外は今回の結果を薄くして位置が分かるようにする
        mask_rgb = (current.astype(np.float32) * 0.25 + 191).astype(np.uint8)
        mask_rgb[mask] = (230, 0, 0)
    base_panel = baseline if baseline is not None and baseline.shape == current.shape else blank

    panels = [
        ("入力（縮小）", reference),
        ("基準", base_panel),
        ("今回", current),
        ("変化した画素", mask_rgb),
    ]
    gap, label_h = 8, 22
    pw, ph = w * scale, h * scale
    canvas = Image.new("RGB", (len(panels) * pw + (len(panels) + 1) * gap, ph + label_h + gap * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font = _label_font()
    for i, (label, img) in enumerate(panels):
        x = gap + i * (pw + gap)
        canvas.paste(Image.fromarray(_upscale(img, scale)), (x, label_h + gap))
        draw.text((x, 4), label, fill="black", font=font)
    return np.asarray(canvas)


def _label_font():
    from PIL import ImageFont

    for name in ("meiryo.ttc", "YuGothM.ttc", "msgothic.ttc"):
        try:
            return ImageFont.truetype(name, 14)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------- 実行

def _case_record(case: dict) -> dict:
    """基準に残す入力と設定。JSON を一度通して tuple/list の違いを消し、比較できる形にする。"""
    record = {
        "image": case["image"],
        "palette": case["palette"],
        "noise_filter": case["noise_filter"],
        "settings": asdict(case["request"]),
    }
    return json.loads(json.dumps(record, ensure_ascii=False))


def save_baseline(cases: list[dict], replace_all: bool) -> None:
    BASELINE_DIR.mkdir(exist_ok=True)
    metrics_path = BASELINE_DIR / "metrics.json"
    if replace_all:
        # 全ケースを作り直すときは、消した画像や設定の古い基準を残さない
        stored = {}
        for old in BASELINE_DIR.glob("*.png"):
            old.unlink()
    else:
        stored = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    palettes: dict = {}
    for case in cases:
        reference, output, elapsed = run_case(case, palettes)
        _write_png(BASELINE_DIR / f"{case['name']}.png", output)
        metrics = compute_metrics(reference, output)
        stored[case["name"]] = {"metrics": metrics, **_case_record(case)}
        print(f"[基準] {case['name']}: {_format_metrics(metrics)} ({elapsed:.2f}s)")
    metrics_path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n基準を保存しました: {BASELINE_DIR}")


def compare(cases: list[dict], mark_threshold: float) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    # 前回の結果が混ざらないよう、画像は毎回入れ替える
    for old in RESULTS_DIR.glob("*.png"):
        old.unlink()
    metrics_path = BASELINE_DIR / "metrics.json"
    stored = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    palettes: dict = {}
    rows = []
    for case in cases:
        name = case["name"]
        reference, output, elapsed = run_case(case, palettes)
        baseline = _read_png(BASELINE_DIR / f"{name}.png")
        mask = change_mask(baseline, output)
        current = compute_metrics(reference, output)
        base_entry = stored.get(name)
        base_metrics = base_entry["metrics"] if base_entry else None

        if baseline is None:
            change_text, note = "", "基準なし"
        elif mask is None:
            change_text, note = "100.00", "サイズ変更"
        else:
            change = float(mask.mean() * 100)
            change_text = f"{change:.2f}"
            note = "★ 変化大" if change >= mark_threshold else ""
        if base_entry and _case_record(case) != {k: base_entry.get(k) for k in _case_record(case)}:
            note = (note + " / 設定が基準と異なる").lstrip(" /")

        _write_png(RESULTS_DIR / f"{name}.png", output)
        _write_png(RESULTS_DIR / f"{name}_compare.png", build_comparison(reference, baseline, output, mask))
        rows.append({"name": name, "base": base_metrics, "current": current,
                     "change": change_text, "note": note, "elapsed": elapsed})
        print(f"[比較] {name}: 変化率 {change_text or '-'}% {note}")

    _write_reports(rows, mark_threshold)
    print(f"\n結果: {RESULTS_DIR / 'report.md'}")


def _format_metrics(m: dict) -> str:
    return ", ".join(f"{METRIC_LABELS[k]}={m[k]}" for k in METRIC_KEYS)


def _diff(base, cur):
    if base is None:
        return ""
    d = cur - base
    return f"{d:+d}" if isinstance(d, int) else f"{d:+.3f}"


def _write_reports(rows: list[dict], mark_threshold: float) -> None:
    header = ["ケース", "変化率(%)", "印"]
    for key in METRIC_KEYS:
        label = METRIC_LABELS[key]
        header += [f"{label}(基準)", f"{label}(今回)", f"{label}(差)"]
    header.append("変換秒数")

    table = []
    for row in rows:
        line = [row["name"], row["change"], row["note"]]
        for key in METRIC_KEYS:
            base = row["base"][key] if row["base"] else None
            cur = row["current"][key]
            line += ["" if base is None else base, cur, _diff(base, cur)]
        line.append(f"{row['elapsed']:.2f}")
        table.append(line)

    # Excel で開いても文字化けしないよう BOM 付き UTF-8
    with (RESULTS_DIR / "report.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(table)

    md = [
        "# リグレッションテスト結果",
        "",
        f"- 実行日時: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 印（★）: 基準と色が異なる画素が {mark_threshold}% 以上",
        "- 平均L差・平均C差は「出力 − 入力（同サイズに縮小）」。負なら暗い／くすんでいる",
        "",
        "| ケース | 変化率(%) | 印 | " + " | ".join(METRIC_LABELS[k] + "（基準→今回）" for k in METRIC_KEYS) + " |",
        "|---|---:|---|" + "---:|" * len(METRIC_KEYS),
    ]
    for row in rows:
        cells = [row["name"], row["change"] or "-", row["note"]]
        for key in METRIC_KEYS:
            cur = row["current"][key]
            if row["base"]:
                base = row["base"][key]
                cells.append(f"{base} → {cur} ({_diff(base, cur)})")
            else:
                cells.append(str(cur))
        md.append("| " + " | ".join(str(c) for c in cells) + " |")
    md += ["", "比較画像: `<ケース名>_compare.png`（入力（縮小）／基準／今回／変化した画素を赤）", ""]
    (RESULTS_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="ビーズ変換の簡易リグレッションテスト")
    parser.add_argument("--baseline", action="store_true", help="現在の結果を基準として保存する")
    parser.add_argument("--case", help="名前にこの文字列を含むケースだけ実行する")
    parser.add_argument("--mark", type=float, default=1.0,
                        help="基準と色が異なる画素がこの%%以上なら印を付ける（既定 1.0）")
    args = parser.parse_args()

    cases = load_cases(args.case)
    if not cases:
        print(f"該当するケースがありません（画像の置き場所: {IMAGES_DIR}）。")
        return 1
    if args.baseline:
        save_baseline(cases, replace_all=args.case is None)
    else:
        compare(cases, args.mark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
