"""
Annotate rendered scene images with GT and model prediction labels.
Usage: python3 eval/annotate_examples.py --csv outputs/comparison/examples.csv
                                          --out_dir outputs/comparison/annotated/
"""
import os
import csv
import argparse
from PIL import Image, ImageDraw, ImageFont


def annotate_image(img_path, gt_id, pred_4b, iou_4b, pred_2b, iou_2b, query, out_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    banner_h = 120
    new_img = Image.new('RGB', (w, h + banner_h), (30, 30, 30))
    new_img.paste(img, (0, banner_h))

    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font = ImageFont.load_default()
        font_sm = font

    # Query text (truncated)
    query_short = query if len(query) < 100 else query[:97] + '...'
    draw.text((10, 8), f"Query: {query_short}", fill=(220, 220, 220), font=font_sm)

    # GT label
    draw.text((10, 40), f"GT Object ID: {gt_id}", fill=(255, 255, 255), font=font)

    # 4B prediction
    color_4b = (80, 200, 80) if iou_4b >= 0.25 else (220, 80, 80)
    tick_4b = '✓' if iou_4b >= 0.25 else '✗'
    draw.text((10, 70), f"Qwen3-VL 4B: {tick_4b} predicted ID {pred_4b}  (IoU={iou_4b:.3f})", fill=color_4b, font=font)

    # 2B prediction
    color_2b = (80, 200, 80) if iou_2b >= 0.25 else (220, 80, 80)
    tick_2b = '✓' if iou_2b >= 0.25 else '✗'
    draw.text((10, 95), f"Qwen3-VL 2B: {tick_2b} predicted ID {pred_2b}  (IoU={iou_2b:.3f})", fill=color_2b, font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new_img.save(out_path)


def main(csv_path, out_dir, outcome_filter=None, n=30):
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    if outcome_filter:
        rows = [r for r in rows if r['outcome'] == outcome_filter]

    print(f'Annotating {min(n, len(rows))} images...')
    for i, row in enumerate(rows[:n]):
        img_path = row['image']
        if not os.path.exists(img_path):
            print(f'  Missing image: {img_path}')
            continue

        out_path = os.path.join(out_dir, row['outcome'],
                                f"{i:02d}_{row['scene_id']}_q{row['query_idx']}.png")
        annotate_image(
            img_path,
            gt_id=row['gt_id'],
            pred_4b=row['pred_id_4b'],
            iou_4b=float(row['iou_4b']),
            pred_2b=row['pred_id_2b'],
            iou_2b=float(row['iou_2b']),
            query=row['query'],
            out_path=out_path,
        )

    print(f'Done. Images saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out_dir', default='outputs/comparison/annotated')
    parser.add_argument('--outcome', default=None, help='Filter: 4B_wins or 2B_wins')
    parser.add_argument('--n', type=int, default=30)
    args = parser.parse_args()

    main(args.csv, args.out_dir, args.outcome, args.n)
