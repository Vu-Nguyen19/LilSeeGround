"""
Compare predictions from two model runs and generate an HTML report.
Usage: python3 eval/compare_models.py --pred_dir_a outputs/scanrefer_qwen3vl_4b/pred/
                                       --pred_dir_b outputs/scanrefer_qwen3vl_2b/pred/
                                       --img_dir_a outputs/scanrefer_qwen3vl_4b/projection_img/
                                       --img_dir_b outputs/scanrefer_qwen3vl_2b/projection_img/
                                       --label_a "Qwen3-VL 4B" --label_b "Qwen3-VL 2B"
                                       --out_dir outputs/comparison/
"""
import base64
import os
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.utils import load_json, calc_iou


def img_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def compare(pred_dir_a, pred_dir_b, img_dir_a, img_dir_b, out_dir, label_a, label_b, iou_thresh=0.25, n=30):
    files = [f for f in os.listdir(pred_dir_a) if f.endswith('.json')]

    a_wins = []
    b_wins = []
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
                'img_a': os.path.join(img_dir_a, scene_id, str(idx), 'rendered.png'),
                'img_b': os.path.join(img_dir_b, scene_id, str(idx), 'rendered.png'),
            }

            if correct_a and not correct_b:
                a_wins.append(entry)
            elif correct_b and not correct_a:
                b_wins.append(entry)
            elif correct_a and correct_b:
                both_correct.append(entry)
            else:
                both_wrong.append(entry)

    print(f'{label_a} wins (correct, {label_b} wrong): {len(a_wins)}')
    print(f'{label_b} wins (correct, {label_a} wrong): {len(b_wins)}')
    print(f'Both correct: {len(both_correct)}')
    print(f'Both wrong:   {len(both_wrong)}')

    os.makedirs(out_dir, exist_ok=True)

    def render_section(cases, title, n):
        if not cases:
            return f'<h2>{title}</h2><p>No cases.</p>'
        rows = f'<h2>{title} (showing {min(n, len(cases))} of {len(cases)})</h2>'
        for c in cases[:n]:
            img_a_b64 = img_to_base64(c['img_a'])
            img_tag = f'<img src="data:image/png;base64,{img_a_b64}" style="max-width:700px;width:100%">' if img_a_b64 else '<i>Image not found</i>'
            a_color = 'green' if c['iou_a'] >= iou_thresh else 'red'
            b_color = 'green' if c['iou_b'] >= iou_thresh else 'red'
            rows += f'''
            <div style="border:1px solid #ccc;margin:16px 0;padding:16px;border-radius:8px">
                <p><b>Scene:</b> {c["scene_id"]} &nbsp;|&nbsp; <b>Query #{c["query_idx"]}</b></p>
                <p><b>Query:</b> {c["query"]}</p>
                <p><b>GT Object ID:</b> {c["gt_id"]}</p>
                <table style="width:100%;margin-bottom:8px">
                    <tr>
                        <td style="width:50%;padding:4px"><b>{label_a}</b> predicted ID: <b>{c["pred_id_a"]}</b> &nbsp; IoU: <b style="color:{a_color}">{c["iou_a"]}</b></td>
                        <td style="width:50%;padding:4px"><b>{label_b}</b> predicted ID: <b>{c["pred_id_b"]}</b> &nbsp; IoU: <b style="color:{b_color}">{c["iou_b"]}</b></td>
                    </tr>
                </table>
                {img_tag}
            </div>'''
        return rows

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Model Comparison: {label_a} vs {label_b}</title>
<style>body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}</style>
</head>
<body>
<h1>Model Comparison: {label_a} vs {label_b}</h1>
<p>IoU threshold: {iou_thresh} &nbsp;|&nbsp; {label_a} wins: {len(a_wins)} &nbsp;|&nbsp; {label_b} wins: {len(b_wins)} &nbsp;|&nbsp; Both correct: {len(both_correct)} &nbsp;|&nbsp; Both wrong: {len(both_wrong)}</p>
{render_section(a_wins, f"✅ {label_a} correct, ❌ {label_b} wrong", n)}
{render_section(b_wins, f"❌ {label_a} wrong, ✅ {label_b} correct", n)}
{render_section(both_wrong, f"❌ Both wrong", n)}
</body></html>'''

    out_file = os.path.join(out_dir, 'comparison.html')
    with open(out_file, 'w') as f:
        f.write(html)
    print(f'\nHTML report saved to {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir_a', required=True)
    parser.add_argument('--pred_dir_b', required=True)
    parser.add_argument('--img_dir_a', required=True)
    parser.add_argument('--img_dir_b', required=True)
    parser.add_argument('--label_a', default='Model A')
    parser.add_argument('--label_b', default='Model B')
    parser.add_argument('--out_dir', default='outputs/comparison')
    parser.add_argument('--iou_thresh', type=float, default=0.25)
    parser.add_argument('--n', type=int, default=30, help='Max cases per section')
    args = parser.parse_args()

    compare(args.pred_dir_a, args.pred_dir_b, args.img_dir_a, args.img_dir_b,
            args.out_dir, args.label_a, args.label_b, args.iou_thresh, args.n)
