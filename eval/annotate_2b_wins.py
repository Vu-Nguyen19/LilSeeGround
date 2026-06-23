"""
Generate annotated images for 2B_wins cases into seeground_annotated/.
Reads examples.csv, remaps image paths to seeground_images_2b/, and
overlays query + prediction metadata onto each rendered scene image.

Usage:
    python3 eval/annotate_2b_wins.py
"""
import os
import csv
import sys
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "examples.csv")
IMG_BASE  = os.path.join(BASE_DIR, "seeground_images_2b")
OUT_DIR   = os.path.join(BASE_DIR, "seeground_annotated")
N         = 30  # max images to produce


def remap_image_path(csv_img_path):
    # csv paths look like: outputs/scanrefer_qwen3vl_4b/projection_img/<scene>/<idx>/rendered.png
    # local images live at:  seeground_images_2b/<scene>/<idx>/rendered.png
    parts = csv_img_path.replace("\\", "/").split("/")
    # last 3 parts are always <scene>/<idx>/rendered.png
    scene, idx, fname = parts[-3], parts[-2], parts[-1]
    return os.path.join(IMG_BASE, scene, idx, fname)


def load_fonts():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return (
                    ImageFont.truetype(path, 16),
                    ImageFont.truetype(path, 13),
                )
            except Exception:
                pass
    default = ImageFont.load_default()
    return default, default


def annotate_image(img_path, gt_id, pred_4b, iou_4b, pred_2b, iou_2b, query, out_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    banner_h = 120
    canvas = Image.new("RGB", (w, h + banner_h), (30, 30, 30))
    canvas.paste(img, (0, banner_h))

    draw = ImageDraw.Draw(canvas)
    font, font_sm = load_fonts()

    query_short = query if len(query) < 100 else query[:97] + "..."
    draw.text((10, 8),  f"Query: {query_short}",          fill=(220, 220, 220), font=font_sm)
    draw.text((10, 40), f"GT Object ID: {gt_id}",         fill=(255, 255, 255), font=font)

    color_4b = (80, 200, 80) if iou_4b >= 0.25 else (220, 80, 80)
    tick_4b  = "✓" if iou_4b >= 0.25 else "✗"
    draw.text((10, 70), f"Qwen3-VL 4B: {tick_4b} predicted ID {pred_4b}  (IoU={iou_4b:.3f})",
              fill=color_4b, font=font)

    color_2b = (80, 200, 80) if iou_2b >= 0.25 else (220, 80, 80)
    tick_2b  = "✓" if iou_2b >= 0.25 else "✗"
    draw.text((10, 95), f"Qwen3-VL 2B: {tick_2b} predicted ID {pred_2b}  (IoU={iou_2b:.3f})",
              fill=color_2b, font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def main():
    with open(CSV_PATH, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["outcome"] == "2B_wins"]

    os.makedirs(OUT_DIR, exist_ok=True)
    produced = 0

    for i, row in enumerate(rows):
        if produced >= N:
            break

        img_path = remap_image_path(row["image"])
        if not os.path.exists(img_path):
            print(f"  Missing: {img_path}")
            continue

        out_path = os.path.join(
            OUT_DIR,
            f"{produced:02d}_{row['scene_id']}_q{row['query_idx']}.png",
        )
        annotate_image(
            img_path,
            gt_id   = row["gt_id"],
            pred_4b = row["pred_id_4b"],
            iou_4b  = float(row["iou_4b"]),
            pred_2b = row["pred_id_2b"],
            iou_2b  = float(row["iou_2b"]),
            query   = row["query"],
            out_path= out_path,
        )
        print(f"  [{produced+1}/{N}] {out_path}")
        produced += 1

    print(f"\nDone. {produced} images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
