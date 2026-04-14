"""
Analyze token usage and performance metrics from prediction JSON files.
Usage: python3 eval/analyze_tokens.py --pred_dir outputs/scanrefer_qwen3vl_4b/pred/ --model_name "Qwen3-VL 4B" --runtime_min 50.97
"""
import os
import json
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.utils import load_json, calc_iou


def analyze(pred_dir, model_name, runtime_min, iou_thresh=0.25):
    files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]

    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    n_queries = 0
    correct_25 = 0
    correct_50 = 0
    unique_correct_25 = 0
    unique_total = 0

    for fname in sorted(files):
        preds = load_json(os.path.join(pred_dir, fname))
        for p in preds:
            n_queries += 1
            iou = calc_iou(p['gt_bbox'], p['pred_bbox'])
            if iou >= 0.25:
                correct_25 += 1
            if iou >= 0.5:
                correct_50 += 1
            if p.get('unique'):
                unique_total += 1
                if iou >= 0.25:
                    unique_correct_25 += 1

            if 'token_usage' in p:
                total_prompt += p['token_usage'].get('prompt_tokens', 0)
                total_completion += p['token_usage'].get('completion_tokens', 0)
                total_tokens += p['token_usage'].get('total_tokens', 0)

    has_tokens = total_tokens > 0
    runtime_sec = runtime_min * 60

    print(f'\n{"="*50}')
    print(f'  Model: {model_name}')
    print(f'{"="*50}')
    print(f'\n--- Accuracy ---')
    print(f'  Acc@25:        {correct_25/n_queries:.2%}  ({correct_25}/{n_queries})')
    print(f'  Acc@50:        {correct_50/n_queries:.2%}  ({correct_50}/{n_queries})')
    if unique_total > 0:
        print(f'  Unique@25:     {unique_correct_25/unique_total:.2%}  ({unique_correct_25}/{unique_total})')
    else:
        print(f'  Unique@25:     N/A (no unique queries in sample)')

    print(f'\n--- Runtime ---')
    print(f'  Total time:    {runtime_min:.1f} min')
    print(f'  Scenes:        {len(files)}')
    print(f'  Queries:       {n_queries}')
    print(f'  Time/scene:    {runtime_sec/len(files):.1f}s')
    print(f'  Time/query:    {runtime_sec/n_queries:.1f}s')

    if has_tokens:
        print(f'\n--- Token Usage ---')
        print(f'  Prompt tokens:      {total_prompt:,}  (avg {total_prompt//n_queries}/query)')
        print(f'  Completion tokens:  {total_completion:,}  (avg {total_completion//n_queries}/query)')
        print(f'  Total tokens:       {total_tokens:,}  (avg {total_tokens//n_queries}/query)')
        print(f'  Tokens/sec:         {total_tokens/runtime_sec:.1f}')
    else:
        print(f'\n--- Token Usage ---')
        print(f'  Not recorded (re-run with updated inference script to capture)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--runtime_min', type=float, required=True, help='Total inference runtime in minutes')
    args = parser.parse_args()
    analyze(args.pred_dir, args.model_name, args.runtime_min)
