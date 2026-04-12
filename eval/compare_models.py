"""
Compare predictions from two model runs and find interesting differing cases.
Usage: python3 eval/compare_models.py --pred_dir_a outputs/scanrefer_qwen3vl_4b/pred/
                                       --pred_dir_b outputs/scanrefer_qwen3vl_2b/pred/
                                       --img_dir_a outputs/scanrefer_qwen3vl_4b/projection_img/
                                       --img_dir_b outputs/scanrefer_qwen3vl_2b/projection_img/
                                       --out_dir outputs/comparison/
"""
import json
import os
import shutil
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.utils import load_json, calc_iou


def compare(pred_dir_a, pred_dir_b, img_dir_a, img_dir_b, out_dir, iou_thresh=0.25):
    files = [f for f in os.listdir(pred_dir_a) if f.endswith('.json')]

    a_wins = []  # A correct, B wrong
    b_wins = []  # B correct, A wrong
    both_correct = []
    both_wrong = []

    for fname in sorted(files):
        path_a = os.path.join(pred_dir_a, fname)
        path_b = os.path.join(pred_dir_b, fname)
        if not os.path.exists(path_b):
            continue

        preds_a = load_json(path_a)
        preds_b = load_json(path_b)

        for idx, (pa, pb) in enumerate(zip(preds_a, preds_b)):
            scene_id = fname.replace('.json', '')
            iou_a = calc_iou(pa['gt_bbox'], pa['pred_bbox'])
            iou_b = calc_iou(pb['gt_bbox'], pb['pred_bbox'])
            correct_a = iou_a >= iou_thresh
            correct_b = iou_b >= iou_thresh

            entry = {
                'scene_id': scene_id,
                'query_idx': idx,
                'query': pa.get('query', ''),
                'iou_a': round(iou_a, 3),
                'iou_b': round(iou_b, 3),
                'gt_id': pa.get('gt_id'),
                'pred_id_a': pa.get('predicted_id'),
                'pred_id_b': pb.get('predicted_id'),
            }

            if correct_a and not correct_b:
                a_wins.append(entry)
            elif correct_b and not correct_a:
                b_wins.append(entry)
            elif correct_a and correct_b:
                both_correct.append(entry)
            else:
                both_wrong.append(entry)

    print(f'Model A wins (A correct, B wrong): {len(a_wins)}')
    print(f'Model B wins (B correct, A wrong): {len(b_wins)}')
    print(f'Both correct: {len(both_correct)}')
    print(f'Both wrong:   {len(both_wrong)}')

    # Copy images for top differing cases
    def copy_images(cases, label, n=20):
        out = os.path.join(out_dir, label)
        os.makedirs(out, exist_ok=True)
        for i, c in enumerate(cases[:n]):
            sid, qidx = c['scene_id'], c['query_idx']
            img_a = os.path.join(img_dir_a, sid, str(qidx), 'rendered.png')
            img_b = os.path.join(img_dir_b, sid, str(qidx), 'rendered.png')
            prefix = f"{i:02d}_{sid}_q{qidx}_iouA{c['iou_a']}_iouB{c['iou_b']}"
            if os.path.exists(img_a):
                shutil.copy(img_a, os.path.join(out, prefix + '_modelA.png'))
            if os.path.exists(img_b):
                shutil.copy(img_b, os.path.join(out, prefix + '_modelB.png'))
            # Save query text
            with open(os.path.join(out, prefix + '_query.txt'), 'w') as f:
                f.write(c['query'])

    copy_images(a_wins, 'A_wins_4B_correct_2B_wrong')
    copy_images(b_wins, 'B_wins_2B_correct_4B_wrong')
    copy_images(both_wrong, 'both_wrong')

    print(f'\nImages saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir_a', required=True)
    parser.add_argument('--pred_dir_b', required=True)
    parser.add_argument('--img_dir_a', required=True)
    parser.add_argument('--img_dir_b', required=True)
    parser.add_argument('--out_dir', default='outputs/comparison')
    parser.add_argument('--iou_thresh', type=float, default=0.25)
    args = parser.parse_args()

    compare(args.pred_dir_a, args.pred_dir_b, args.img_dir_a, args.img_dir_b, args.out_dir, args.iou_thresh)
