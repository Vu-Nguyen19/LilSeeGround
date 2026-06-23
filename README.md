# Towards Edge Device Perception with Vision-Language Models for 3D Visual Grounding

**Manan Arora, Vu Nguyen, Sarang Rajeev**  
University of Michigan — {aromanan, nvtnghia, sarangr}@umich.edu

## Overview

This project investigates whether the heavy 72B-parameter backbone in [SeeGround](https://arxiv.org/abs/2412.04383) (CVPR 2025) can be replaced with smaller, edge-deployable Vision-Language Models without sacrificing grounding accuracy.

We substitute Qwen2-VL-72B with **Qwen3-VL at the 2B and 4B scales** as drop-in replacements, keeping all other SeeGround pipeline components fixed. The 4B model fits in ~8 GB of GPU memory; the 2B model fits in ~4 GB — both compatible with edge platforms like the Jetson AGX Orin.

## Results on Nr3D

| Backbone | Params | VRAM | Unique@25 | Unique@50 | Multiple@25 | Multiple@50 | Acc@25 | Acc@50 |
|---|---|---|---|---|---|---|---|---|
| Qwen2-VL-7B† | 7B | 14 GB | — | — | — | — | 33.3 | — |
| InternVL2-8B† | 8B | 16 GB | — | — | — | — | 34.3 | — |
| InternVL2-26B† | 26B | 52 GB | — | — | — | — | 38.0 | — |
| Qwen2-VL-72B† | 72B | 144 GB | 75.7 | 68.9 | 34.0 | 30.0 | 44.1 | 39.4 |
| **Qwen3-VL-2B (ours)** | 2B | 4 GB | 73.03 | **68.54** | 32.07 | 29.22 | 42.41 | 39.15 |
| **Qwen3-VL-4B (ours)** | 4B | 8 GB | 74.16 | 67.98 | **38.14** | **34.35** | **47.23** | **42.84** |

†Results from the original SeeGround paper. All models evaluated at IoU thresholds of 0.25 and 0.50.

**Qwen3-VL-4B surpasses the 72B baseline on every metric except Unique@25** (+3.13 points overall at Acc@25, +3.44 at Acc@50). The 2B model matches the 72B baseline at Acc@50 at 36× fewer parameters.

## Key Findings

- **Scale is not the bottleneck.** Qwen3-VL-4B, 18× smaller than the original backbone, outperforms Qwen2-VL-72B overall — pointing to Qwen3's Interleaved-MRoPE architecture as a key factor.
- **Biggest gains on Multiple queries.** Qwen3-VL-4B improves +4.14 points at Multiple@25 over 72B, the exact split requiring spatial relational reasoning where we expected the sharpest degradation.
- **2B is viable for constrained hardware.** At ~4 GB and no quantization, Qwen3-VL-2B fits entirely within Jetson AGX Orin unified memory while remaining within 1.7 points of the 72B baseline overall.

## Setup

### Environment

```bash
docker pull qwenllm/qwenvl
```

### Model Weights

Download Qwen3-VL weights from [Hugging Face](https://huggingface.co/Qwen):
- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`

### Dataset

Download the Nr3D dataset from the [ReferIt3D repo](https://github.com/referit3d/referit3d) and place it at:
```
data/Nr3D/Nr3D.jsonl
```

Download the preprocessed Vil3dref data from [vil3dref](https://github.com/cshizhe/vil3dref):
```
referit3d/
├── annotations/
│   ├── meta_data/
│   └── ...
└── scan_data/
    ├── instance_id_to_name/
    └── pcd_with_global_alignment/
```

### Object Lookup Table

Build from Mask3D instance predictions:
```bash
python prepare_data/object_lookup_table_nr3d.py
```

Or download the [preprocessed OLT](https://github.com/user-attachments/files/18056532/seeground_object_lookup_table.zip) from the original SeeGround repo.

## Inference

### Deploy VLM

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3-VL-4B-Instruct \
    --served-model-name Qwen3-VL-4B-Instruct \
    --tensor_parallel_size=1
```

### Generate Query Data

```bash
python parse_query/generate_query_data_nr3d.py
```

### Run Inference

```bash
python inference/inference_nr3d.py
```

### Evaluate

```bash
python eval/eval_nr3d.py
```

## Repository Structure

```
├── prepare_data/       # OLT construction scripts
├── parse_query/        # Query parsing (anchor/target extraction)
├── inference/          # Grounding inference scripts
├── eval/               # Evaluation scripts
└── paper.pdf           # Full project report
```

## Citation

This work builds directly on SeeGround:
```
@inproceedings{li2025seeground,
  title     = {SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding},
  author    = {Rong Li and Shijie Li and Lingdong Kong and Xulei Yang and Junwei Liang},
  booktitle = {CVPR},
  year      = {2025},
}
```
