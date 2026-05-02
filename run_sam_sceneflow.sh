#!/bin/bash
set -euo pipefail

export CUDA_HOME=/usr
export TORCH_CUDA_ARCH_LIST="8.6"

SAM_DATA_ROOT="$(pwd)/sceneflow_pipeline_output/25e5c600-36fe-3245-9cc0-40ef91620c22/2026-04-07-02-39-02-FlowPairs/icp_flow_pairs"
ICP_FLOW_DIR="$(pwd)/../ICP-Flow"

echo "SAM data root: $SAM_DATA_ROOT"
echo "ICP-Flow dir:  $ICP_FLOW_DIR"

if [ ! -d "$SAM_DATA_ROOT" ]; then
    echo "ERROR: SAM data not found: $SAM_DATA_ROOT"
    exit 1
fi

cd "$ICP_FLOW_DIR"
echo "Working directory: $(pwd)"

python main.py --dataset sam --root "$SAM_DATA_ROOT" --num_frames 2 --if_gpu --gpu_idx 0 --if_save --num_workers 1 --batch_size 1 --thres_iou 0.0 --thres_box 0.0 --thres_rot 0.3 --thres_error 0.3 --thres_dist 0.3

echo "Completed ICP Run"
