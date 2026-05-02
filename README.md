# README

This repository contains the code used for the dissertation project
`Self-Supervised Vision-Based Clustering Analysis for Scene Flow Estimation`.

The code here is not a complete standalone project. It is intended to be applied on top of two external repositories:

- `sam3` / SAM 3 for image segmentation
- `ICP-Flow` for downstream clustering, registration and flow estimation

The files in this repository provide the custom code used in the dissertation:

- `lidarpipeline.py`
  Multiview SAM-LiDAR association pipeline. This performs segmentation, LiDAR projection, cross-camera merging, temporal tracking and export of ICP-Flow-compatible `.npz` pairs.
- `dataset_sam.py`
  Custom ICP-Flow dataloader for reading the exported SAM-guided `.npz` files.
- `main.py`
  Modified ICP-Flow entry point with support for `--dataset sam`.
- `run_sam_sceneflow.sh`
  Example shell script for running ICP-Flow on exported SAM-guided pairs.
- `evalgroundtruth.py`
  Ground-truth evaluation script for comparing predicted flow with OpenSceneFlow / Argoverse 2 HDF5 data.
- `sceneflowsam-environment.yaml`
  Conda environment used during development.

## Prerequisites

Before starting, you should have:

- Linux or WSL2
- Conda installed
- a working NVIDIA driver and CUDA-capable GPU
- enough disk space for Argoverse 2 sensor data
- both external repositories cloned locally

The code in this repository was developed with:

- Python 3.10
- WSL2-style paths
- GPU execution enabled for SAM 3 and ICP-Flow

Practical hardware notes:

- A CUDA-capable NVIDIA GPU is strongly recommended.
- Argoverse 2 can require very large storage depending on how much data you install.
- The dissertation experiments used a single Argoverse 2 log for the main workflow, but processing larger subsets will increase storage and runtime significantly.

## Workflow summary

The overall workflow is:

1. Use `lidarpipeline.py` in the SAM 3 workspace to segment RGB images, associate LiDAR points and export ICP-Flow-compatible `.npz` pairs.
2. Use the modified `main.py` and `dataset_sam.py` in ICP-Flow to run downstream flow inference on those exported pairs.
3. Optionally use `evalgroundtruth.py` to compare the resulting flow against OpenSceneFlow / Argoverse 2 ground truth.

## 1. External repositories required

You will need both repositories locally:

1. Clone the SAM 3 repository.
2. Clone the ICP-Flow repository.

Example:

```bash
git clone https://github.com/facebookresearch/sam3
git clone https://github.com/yanconglin/ICP-Flow
```

This code assumes:

- the SAM 3 repo is used as the preprocessing workspace
- the ICP-Flow repo is used for the downstream flow inference stage

## 2. Set up ICP-Flow first

Before copying any dissertation files into ICP-Flow, go through the setup steps given in the official ICP-Flow repository.

That usually includes:

- creating its expected environment
- building any custom CUDA or C++ extensions
- verifying that the original ICP-Flow code runs before modification

Important:

This project required local CUDA compatibility fixes in the ICP-Flow histogram CUDA extension in order to work with the local PyTorch and CUDA toolchain used during development.

The files that required manual fixes were:

- `ICP-Flow/hist_cuda/cpp/hist.cpp`
- `ICP-Flow/hist_cuda/cpp/hist_cuda.cu`

The exact changes may vary depending on:

- your PyTorch version
- your CUDA toolkit version
- your compiler toolchain
- your GPU architecture

Treat this as a compatibility step rather than a guaranteed copy-paste patch. If ICP-Flow fails during extension compilation, these are the first files to inspect.

Typical compatibility fixes may include:

- replacing deprecated PyTorch extension macros such as `AT_CHECK` with `TORCH_CHECK`
- updating older tensor access or type-check code to match newer PyTorch C++ APIs
- resolving include or namespace changes introduced by newer CUDA / PyTorch builds

The exact fix will depend on your local toolchain, but those are the kinds of issues that occurred in this project.

## 3. Create the conda environment

This project used the environment file:

- `CodeFiles/sceneflowsam-environment.yaml`

Create the environment with:

```bash
conda env create -f sceneflowsam-environment.yaml
conda activate sam3sceneflow
```

Notes:

- The environment was developed under Windows Sub-System for Linux (WSL2) pathing.
- You may need to adjust the final `prefix:` entry or remove it before creation.
- You should confirm that your local CUDA driver, PyTorch and compiler versions are compatible.

## 4. Install SAM 3 model weights

Install the SAM 3 weights separately from Hugging Face:

https://huggingface.co/facebook/sam3

The dissertation code expects SAM 3 to be available through the installed package and model weights. Follow the official SAM 3 instructions for:

- authentication if required
- downloading the correct checkpoint
- placing the checkpoint where the SAM 3 code can load it

If SAM 3 cannot find its weights, `lidarpipeline.py` will not run.

## 5. Install Argoverse 2 data

The preprocessing stage expects an Argoverse 2 sensor log to be available locally.

In `lidarpipeline.py`, the default dataset root is:

```python
DATA_ROOT = Path("data/argoverse2")
```

and the default log ID is:

```python
LOG_ID = "25e5c600-36fe-3245-9cc0-40ef91620c22"
```

So you should place the required Argoverse 2 data under the SAM repo in a structure compatible with:

```text
<SAM_REPO>/data/argoverse2/
```

If you want to use a different log or data root, edit `lidarpipeline.py`.

For evaluation mode, the script also expects:

- `demo/val/index_eval.pkl`

and `evalgroundtruth.py` expects:

- an OpenSceneFlow / Argoverse 2 `.h5` ground-truth file
   - Can be located here: https://github.com/KTH-RPL/OpenSceneFlow

The evaluation script does not download this automatically. 

## 6. Copy the dissertation files into the correct repositories

### Files to place in the SAM 3 repository

Copy:

- `lidarpipeline.py`

Recommended location:

- the SAM repo root, or another working directory from which relative paths such as `data/argoverse2`, `demo/val` and `sceneflow_pipeline_output` make sense

### Files to place in the ICP-Flow repository

Copy:

- `main.py`
- `dataset_sam.py`

Recommended location:

- `main.py` directly into the ICP-Flow repo which will replace the original
- `dataset_sam.py` will exist alongside the other dataset modules

This custom `main.py` adds support for `--dataset sam` and imports `Dataset_sam`.

### Optional evaluation file

Copy:

- `evalgroundtruth.py`

Recommended location:

- wherever you want to run the ground-truth evaluation from, provided its relative paths are updated correctly

## Hard-coded paths to review

Several paths are hard-coded in the dissertation scripts and should be reviewed before running anything.

The most important ones are:

- `lidarpipeline.py`
  `DATA_ROOT`, `LOG_ID`, `EVAL_INDEX_PATH`, output directory settings
- `run_sam_sceneflow.sh`
  `SAM_DATA_ROOT`, `ICP_FLOW_DIR`
- `evalgroundtruth.py`
  `H5_PATH`, `EVAL_INDEX_PATH`, `FLOW_OUTPUT_DIR`, `LOG_ID`
- `main.py`
  runtime dataset paths are mainly supplied by `--root`, but defaults should still be reviewed

## 7. Run the SAM-LiDAR preprocessing stage

Run `lidarpipeline.py` inside the SAM 3 environment.

This script:

1. loads Argoverse 2 images and LiDAR
2. runs SAM 3 over multiple prompts
3. projects LiDAR points into image space
4. associates points to SAM masks
5. merges object groups across cameras
6. tracks groups across two timestamps
7. exports synthetic ICP-Flow `.npz` input pairs

### Main `lidarpipeline.py` configuration parameters

The most important parameters are defined near the top of the file.

#### Dataset and log selection

```python
DATA_ROOT = Path("data/argoverse2")
LOG_ID = "25e5c600-36fe-3245-9cc0-40ef91620c22"
```

Use these to choose the Argoverse 2 data location and the log to process.

#### Mode

```python
MODE = "single"
```

Available modes described in the file:

- `"single"`
  Process one timestamp pair defined by `CAM_TIMESTAMP_NS`
- `"full_log"`
  Process the whole log at a stride
- `"eval"`
  Process timestamps listed in an evaluation index

#### Single timestamp selection

```python
CAM_TIMESTAMP_NS = 315966110460174000
TEMPORAL_GAP = 1
```

- `CAM_TIMESTAMP_NS` selects the first timestamp in `single` mode
- `TEMPORAL_GAP` controls how far ahead the paired timestamp is chosen

#### Full log stride

```python
FULL_LOG_STRIDE = 3
```

In `full_log` mode this reduces processing load by skipping sweeps.

#### Evaluation index

```python
EVAL_INDEX_PATH = Path("demo/val/index_eval.pkl")
```

Used in `eval` mode for timestamp selection.

#### SAM prompts and confidence

```python
CONFIDENCE_THRESHOLD = 0.5
TEXT_PROMPTS = ["bus", "truck", "car", "person"]
```

These determine which object prompts are used and how mask filtering is handled.

In practice, `run_sam()` also contains additional hard-coded filters such as:

- score threshold
- minimum pixel count
- overlap filtering

If recall is too low, these are key settings to inspect.

#### Temporal tracking thresholds

```python
MAX_CENTROID_DIST = 2.0
MIN_POINT_OVERLAP = 50
MIN_POINTS_FOR_EXPORT = 200
```

These strongly affect:

- whether objects are matched across time
- whether matched groups are exported at all
- final dynamic coverage

#### Output paths

```python
OUTPUT_ROOT = Path("sceneflow_pipeline_output") / LOG_ID
ICP_FLOW_OUTPUT = OUTPUT_ROOT / str_current_datetime / "icp_flow_pairs"
```

The exported `.npz` files for ICP-Flow are written here.

### Expected output from `lidarpipeline.py`

After a successful run, you should see a new output directory under:

```text
sceneflow_pipeline_output/<LOG_ID>/<timestamp>-FlowPairs/icp_flow_pairs/
```

That directory should contain:

- one or more exported `.npz` timestamp-pair files
- usually a `metadata.json` file if metadata export is enabled

If the script runs successfully but exports no usable `.npz` pairs, the first places to inspect are:

- SAM prompt and mask filtering settings
- temporal matching thresholds
- minimum point-count thresholds for export

## 8. Run ICP-Flow on the exported SAM pairs

After `lidarpipeline.py` exports `.npz` files, run ICP-Flow using the modified `main.py`.

An example is provided in:

- `run_sam_sceneflow.sh`

That script currently expects:

- exported SAM data inside `sceneflow_pipeline_output/.../icp_flow_pairs`
- the ICP-Flow repo to sit next to the current working directory

Example command from the script:

```bash
python main.py \
  --dataset sam \
  --root "$SAM_DATA_ROOT" \
  --num_frames 2 \
  --if_gpu \
  --gpu_idx 0 \
  --if_save \
  --num_workers 1 \
  --batch_size 1 \
  --thres_iou 0.0 \
  --thres_box 0.0 \
  --thres_rot 0.3 \
  --thres_error 0.3 \
  --thres_dist 0.3
```

### Important `main.py` / ICP-Flow parameters

The modified `main.py` exposes many ICP-Flow parameters. The most important for the SAM-guided setup are:

#### Dataset selection

```bash
--dataset sam
--root <path_to_exported_npz_pairs>
--num_frames 2
```

These are required for the SAM-guided workflow.

#### GPU settings

```bash
--if_gpu
--gpu_idx 0
```

Use these when running with CUDA.

#### Save outputs

```bash
--if_save
```

This writes `_flow.npz` files alongside the exported input pairs.

### Expected output from ICP-Flow

After a successful run with `--if_save`, ICP-Flow should write:

- one `_flow.npz` file for each exported input `.npz` pair

These are normally written into the same directory as the exported SAM-guided pairs.

If the run completes but `_flow.npz` files are missing, first check:

- that `--if_save` was enabled
- that `--root` points to the exported SAM pair directory
- that the modified `main.py` with `--dataset sam` support is actually the one being executed

#### ICP filtering thresholds

```bash
--thres_iou
--thres_box
--thres_rot
--thres_error
--thres_dist
```

These are especially important for SAM-guided objects because they determine which matches are accepted or rejected.

The dissertation experiments used:

```bash
--thres_iou 0.0
--thres_box 0.0
--thres_rot 0.3
--thres_error 0.3
--thres_dist 0.3
```

#### Clustering-related arguments

The script also exposes arguments such as:

```bash
--num_clusters
--min_cluster_size
--epsilon
--if_hdbscan
```

For the SAM-guided setup, object labels are already reconstructed by `dataset_sam.py`, so these are less central than they are in the native ICP-Flow baseline.

## 9. Ground-truth evaluation

Use `evalgroundtruth.py` to compare predicted flow against OpenSceneFlow / Argoverse 2 ground truth.

This script expects paths such as:

```python
H5_PATH = Path("demo/val/<log_id>.h5")
EVAL_INDEX_PATH = Path("demo/val/index_eval.pkl")
FLOW_OUTPUT_DIR = Path("sceneflow_pipeline_output/.../icp_flow_pairs")
```

You will likely need to edit:

- `H5_PATH`
- `EVAL_INDEX_PATH`
- `FLOW_OUTPUT_DIR`
- `LOG_ID`

to match your own setup.

### Metrics produced

The evaluator reports:

- `EPE3D`
- `ACC3DS`
- `ACC3DR`
- `Outlier`

and also computes diagnostics such as:

- coverage
- per-object purity
- category-wise results

## 10. Suggested setup order

For a fresh machine, the recommended order is:

1. Clone SAM 3 and ICP-Flow.
2. Follow the official ICP-Flow setup instructions and verify baseline installation.
3. Fix CUDA compatibility issues in:
   `ICP-Flow/hist_cuda/cpp/hist.cpp`
   and
   `ICP-Flow/hist_cuda/cpp/hist_cuda.cu`
   if needed.
4. Create the conda environment from `sceneflowsam-environment.yaml`.
5. Install SAM 3 weights from Hugging Face.
6. Place Argoverse 2 data under the SAM repo data root.
7. Copy `lidarpipeline.py` into the SAM repo.
8. Copy `main.py` and `dataset_sam.py` into the ICP-Flow repo.
9. Run `lidarpipeline.py` to export ICP-Flow input pairs.
10. Run `main.py` in ICP-Flow, or use `run_sam_sceneflow.sh`.
11. Run `evalgroundtruth.py` if ground-truth evaluation is required.

## 11. Practical notes

- Several paths are hard-coded and should be reviewed before running.
- The main hard-coded paths are listed earlier in section `6.5 Hard-coded paths to review`.
- The code was developed for a specific Argoverse 2 log and evaluation subset, so it is not plug-and-play for arbitrary datasets without editing paths and parameters.
- Runtime is dominated by SAM inference and repeated multiview processing.
- Coverage is sensitive to `TEXT_PROMPTS`, mask filtering, centroid matching thresholds, overlap thresholds and export minimum point counts.
- If ICP-Flow runs but produces weak results, the most important places to tune are the filtering and matching thresholds in `lidarpipeline.py`, followed by the ICP thresholds passed to `main.py`.

## 12. Minimal file placement summary

### Into SAM 3 repo

- `lidarpipeline.py`

### Into ICP-Flow repo

- `main.py`
- `dataset_sam.py`

### Run as supporting utilities

- `run_sam_sceneflow.sh`
- `evalgroundtruth.py`
- `sceneflowsam-environment.yaml`
