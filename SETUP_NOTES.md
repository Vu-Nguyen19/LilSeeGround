# SeeGround Setup & Troubleshooting Notes

This document summarizes the steps taken to set up the SeeGround environment and resolve common issues encountered during the initial configuration.

## 1. Environment Setup

A custom Conda environment was created to replace the incomplete Docker image.

### Essential Files:
- **`environment.yml`**: Contains all core dependencies (PyTorch, PyTorch3D, Open3D, etc.). Note that `vllm` was removed from this environment to keep it lightweight, as the model is served from a separate `vLLM` environment.
- **`setup_full_env.sh`**: Automates the creation of the `seeground` conda environment.

### Installation Steps:
1. Run `./setup_full_env.sh` to create the environment.
2. Activate with `conda activate seeground`.
3. Install missing utilities if needed: `pip install jsonlines nltk`.

## 2. PointNeXt Extensions

The PointNeXt backbone requires several C++ extensions to be compiled manually.

- **`setup_pointnext.sh`**: This script handles the entire build process:
  1. Initializes git submodules.
  2. Installs `torch-scatter` (specifically version `2.1.0+cu118` to match the environment).
  3. Compiles `pointnet2_batch`, `subsampling`, and `pointops`.

Run this script **after** activating the `seeground` environment.

## 3. Data Requirements & Preprocessing

SeeGround does **not** run directly on raw ScanNet `.ply` files. It requires preprocessed `.pth` files from the `vil3dref` pipeline.

### Required Structure:
The data should be organized as follows:
```
data/ScanRefer/ScanRefer_filtered_val.json/scans/
├── pcd_with_global_alignment/
│   └── scene0000_00.pth  <-- Globally aligned point cloud & labels
└── instance_id_to_name/
    └── scene0000_00.json <-- Mapping of IDs to object names
```

### Generating the Object Lookup Table:
When running `prepare_data/object_lookup_table_scanrefer.py`:
- Use `--scan_dir` to point to the folder containing the `pcd_with_global_alignment` and `instance_id_to_name` subdirectories.
- If you don't have Mask3D predictions (`.npz` files), comment out the `scanrefer_pred` call at the bottom of the script.

## 4. Query Parsing (vLLM Integration)

To use your existing vLLM server (e.g., running `Qwen3-32B` in an `agile` environment), you must override the hardcoded defaults in the parsing scripts.

### Command Example for `parse_query/generate_query_data_scanrefer.py`:
**Warning:** Ensure `--anno_file` points to an actual `.json` file. If your ScanRefer folder is named `ScanRefer_filtered_val.json`, you need to look inside it for the actual JSON file (e.g., `ScanRefer_filtered_val.json/ScanRefer_filtered_val.json`).

```bash
export LD_LIBRARY_PATH=/nfs/turbo/coe-mavens/nvtnghia/conda/envs/seeground/lib:$LD_LIBRARY_PATH
python parse_query/generate_query_data_scanrefer.py 
    --scan_id_file /home/nvtnghia/project/SeeGround/data/scannet/scene_list.txt \s
    --anno_file data/ScanRefer/ScanRefer_filtered_val.json 
    --openai_api_base http://localhost:8000/v1 
    --model_name Qwen/Qwen3-VL-8B-Instruct 
    --prompt_file prompts/parsing_query.txt 
    --scan_data /home/nvtnghia/project/SeeGround/data/ScanRefer/referit3d/scan_data 
    --label_map_file /home/nvtnghia/project/SeeGround/data/ScanRefer/referit3d/annotations/meta_data/scannetv2-labels.combined.tsv --save_dir data/scanrefer/query
```

### Options Breakdown:
- **`--save_dir`**: Specifies where to save the generated JSON query files (e.g., `data/scanrefer/query`). You can change this to any folder you have write access to.
- **`--model_name`**: Must match the `id` served by your vLLM server (e.g., `Qwen/Qwen3-32B`).
- **`--scan_data`**: Points to the `vil3dref` preprocessed data folder.

## 5. Inference (Visual Grounding)

The `inference/inference_scanrefer.py` script performs the final 3D grounding. It uses the VLM to pick an object ID from a rendered 2D image and maps it back to 3D.

### Command Example:
**Note:** Use absolute paths for `--gt_bbox_dir` and `--pred_bbox_dir` to ensure the script finds the lookup tables, especially if they are inside `seeground_object_lookup_table`.

```bash
python inference/inference_scanrefer.py \
    --language_annotation_dir data/scanrefer/query/ \
    --gt_bbox_dir data/seeground_object_lookup_table/scanrefer/gt \
    --pred_bbox_dir data/seeground_object_lookup_table/scanrefer/pred \
    --pcd_dir
      /home/nvtnghia/project/SeeGround/data/ScanRefer/referit3d/scan_data/pcd_with_global_alignment/ \
    --output_dir outputs/scanrefer_results \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --openai_api_base http://gl1517.arc-ts.umich.edu:8000/v1
```

### Why two bbox paths?
- **`--gt_bbox_dir`**: Loads Ground Truth boxes for **evaluation** (calculates IoU and Accuracy).
- **`--pred_bbox_dir`**: Loads the **candidate objects** (e.g., from Mask3D) that the model chooses from.
- **Output**: Generates JSON files in `output_dir/pred/` containing the `predicted_id` and the corresponding `pred_bbox` (3D coordinates).

## 6. Troubleshooting Log

### `ImportError: ... libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`
This happens when the system loads an older C++ library.
**Fix:** Set the library path to prioritize your environment's library:
```bash
export LD_LIBRARY_PATH=/nfs/turbo/coe-mavens/nvtnghia/conda/envs/seeground/lib:$LD_LIBRARY_PATH
```

### Conflicts with User-Local Packages
If you see `/home/user/.local/lib/...` in your traceback, it means your local packages are overriding the environment.
**Fix:** Run commands with `PYTHONNOUSERSITE=1`:
```bash
PYTHONNOUSERSITE=1 python ...
```

### `TypeError: join() got an unexpected keyword argument 'weights_only'`
Fixed by moving `weights_only=False` from `os.path.join` to `torch.load` in `parse_query/generate_query_data_scanrefer.py`.
