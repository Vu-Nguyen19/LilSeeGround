"""
Find interesting comparison examples and print them with image paths.
Usage: python3 eval/find_examples.py --pred_dir_a outputs/scanrefer_qwen3vl_4b/pred/
                                      --pred_dir_b outputs/scanrefer_qwen3vl_2b/pred/
                                      --img_dir outputs/scanrefer_qwen3vl_4b/projection_img/
"""
import os
import argparse
import csv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.utils import load_json, calc_iou


def find_examples(pred_dir_a, pred_dir_b, img_dir, out_csv, iou_thresh=0.25):
    files = [f for f in os.listdir(pred_dir_a) if f.endswith('.json')]
    rows = []

    for fname in sorted(files):
        path_a = os.path.join(pred_dir_a, fname)
        path_b = os.path.join(pred_dir_b, fname)
        if not os.path.exists(path_b):
            continue

        preds_a = load_json(path_a)
        preds_b = load_json(path_b)

        for idx, (pa, pb) in enumerate(zip(preds_a, preds_b)):
            scene_id = fname.replace('.json', '')
            iou_a = round(calc_iou(pa['gt_bbox'], pa['pred_bbox']), 3)
            iou_b = round(calc_iou(pb['gt_bbox'], pb['pred_bbox']), 3)
            correct_a = iou_a >= iou_thresh
            correct_b = iou_b >= iou_thresh

            if correct_a == correct_b:
                continue  # skip cases where both agree

            outcome = '4B_wins' if correct_a else '2B_wins'
            img_path = os.path.join(img_dir, scene_id, str(idx), 'rendered.png')

            rows.append({
                'outcome': outcome,
                'scene_id': scene_id,
                'query_idx': idx,
                'gt_id': pa.get('gt_id'),
                'pred_id_4b': pa.get('predicted_id'),
                'iou_4b': iou_a,
                'pred_id_2b': pb.get('predicted_id'),
                'iou_2b': iou_b,
                'query': pa.get('query', ''),
                'image': img_path,
            })

    rows.sort(key=lambda x: x['outcome'])

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n_4b = sum(1 for r in rows if r['outcome'] == '4B_wins')
    n_2b = sum(1 for r in rows if r['outcome'] == '2B_wins')
    print(f'4B wins: {n_4b}, 2B wins: {n_2b}')
    print(f'CSV saved to {out_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir_a', required=True)
    parser.add_argument('--pred_dir_b', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--out_csv', default='outputs/comparison/examples.csv')
    parser.add_argument('--iou_thresh', type=float, default=0.25)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    find_examples(args.pred_dir_a, args.pred_dir_b, args.img_dir, args.out_csv, args.iou_thresh)
