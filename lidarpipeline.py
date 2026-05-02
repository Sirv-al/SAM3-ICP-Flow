"""
Multiview LiDAR-SAM Association Pipeline

features:
Ego-frame merging across cameras 
Temporal tracking across consecutive sweeps
NPZ export
MODE: 'single', 'full_log', or 'eval'
"""

import json
import pickle
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from datetime import datetime
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
import av2.utils.io as io_utils

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor



# Config

DATA_ROOT = Path("data/argoverse2")
LOG_ID = "25e5c600-36fe-3245-9cc0-40ef91620c22"

CAMERAS = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
]

current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
str_current_datetime = str(current_datetime + "-FlowPairs")


OUTPUT_ROOT = Path("sceneflow_pipeline_output") / LOG_ID
ICP_FLOW_OUTPUT = OUTPUT_ROOT / str_current_datetime / "icp_flow_pairs"

DEVICE = "cuda"
CONFIDENCE_THRESHOLD = 0.5 
#TEXT_PROMPT = "car, person" # can be car, person
TEXT_PROMPTS = ["bus", "truck", "car", "person"] 

# Mode config
# "single"   : process one timestamp pair defined by CAM_TIMESTAMP_NS
# "full_log" : process all sweep pairs across the entire log at FULL_LOG_STRIDE
# "eval"     : index_eval.pkl file from ground truth data, use this for ground truth comparison
MODE = "single"


# will test on 315966105299927216 (car) and 315966110899927208 (truck)
# Used only in single mode: the timestamp to process
CAM_TIMESTAMP_NS = 315966110460174000

# Temporal gap between t0 and t1, influences what pair is picked after t0
TEMPORAL_GAP = 1

# In "full_log" mode, decides what Nth log is processed to influence processing time
FULL_LOG_STRIDE = 3

# Used for eval against ground truth, eval mode path to the index_eval.pkl
EVAL_INDEX_PATH = Path("demo/val/index_eval.pkl")


MAX_RANGE_M = 80.0
DOWNSAMPLE = 1 #visualisation

# Temporal tracking thresholds
MAX_CENTROID_DIST = 2.0   # metres: max distance between centroids for matching
MIN_POINT_OVERLAP = 50    # minimum overlapping points to confirm match
MIN_POINTS_FOR_EXPORT = 200  # minimum points needed to expor



# SAM segmentation

def run_sam(image, processor):
    #Run SAM once per prompt and merge results into a single instance map
    
    H, W = image.size[1], image.size[0]
    instance_map = -1 * np.ones((H, W), dtype=np.int16)

    instance_id = 0
    ##total_masks = 0

    for prompt in TEXT_PROMPTS:
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)

        inference_state = processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )

        masks = inference_state.get("masks", [])
        scores = inference_state.get("scores", [])

        min_pixels = 300
       
        accepted_prompt = 0
        rejected_score = 0
        rejected_size = 0
        rejected_overlap = 0

        print(f"'{prompt}' -> {len(masks)} masks")

        for m, s in zip(masks, scores):

            # manual filtering
            if s < 0.4:
                rejected_score += 1
                continue

            m2d = np.squeeze(m.cpu().numpy()) > 0

            if m2d.sum() < min_pixels:
                rejected_size += 1
                continue

            free = (instance_map == -1)

            write_mask = m2d & free

            if write_mask.sum() < 50:
                rejected_overlap += 1
                continue

            instance_map[write_mask] = instance_id
            instance_id += 1
            accepted_prompt += 1

        print(f" '{prompt}' accepted={accepted_prompt}"
          f"rejected: score={rejected_score} size={rejected_size} "
          f"overlap={rejected_overlap}")

    print(f"total merged instances = {instance_id}")
    

    if instance_id == 0:
        return None, None

    return instance_map, None  # masks not needed downstream



# Visualization

def draw_overlay_with_mask(
    img_path,
    uv,
    points_cam,
    valid,
    instance_map,
    gid_map,
    merged_id_map,
    object_colours,
    out_path
):
    # Draw SAM masks and Lidar points coloured by merged object ID
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype(np.float32)

    H, W = instance_map.shape

    uv_valid = uv[valid]
    x = np.round(uv_valid[:,0]).astype(int)
    y = np.round(uv_valid[:,1]).astype(int)

    x = np.clip(x, 0, W-1)
    y = np.clip(y, 0, H-1)

    instance_ids = instance_map[y, x]

    # Draw masks
    alpha = 0.18
    for iid in np.unique(instance_map):
        if iid == -1:
            continue
        gid = gid_map.get(iid)
        if gid is None or gid not in merged_id_map:
            continue
        merged_id = merged_id_map[gid]
        colour = object_colours[merged_id]
        mask = instance_map == iid
        img_np[mask] = (1-alpha)*img_np[mask] + alpha*colour

    # Draw lidar points
    for xx, yy, iid in zip(x[::DOWNSAMPLE], y[::DOWNSAMPLE], instance_ids[::DOWNSAMPLE]):
        if iid == -1:
            colour = np.array([255, 255, 255])
        else:
            gid = gid_map.get(iid)
            if gid is not None and gid in merged_id_map:
                merged_id = merged_id_map[gid]
                colour = object_colours[merged_id]
            else:
                colour = np.array([255, 255, 255])
        img_np[
            max(0, yy-1):min(H, yy+2),
            max(0, xx-1):min(W, xx+2)
        ] = colour

    img = Image.fromarray(img_np.astype(np.uint8))
    img.save(out_path)



# Association 

def associate_points(uv, points_cam, points_ego, valid_mask, instance_map):
    # Associate Lidar points to SAM instances in ego frame
    uv_valid = uv[valid_mask]
    pts_ego_valid = points_ego[valid_mask]

    x = np.round(uv_valid[:,0]).astype(int)
    y = np.round(uv_valid[:,1]).astype(int)

    instance_ids = instance_map[y, x]

    results = {}

    for iid in np.unique(instance_ids):
        if iid == -1:
            continue

        mask = instance_ids == iid
        pts = pts_ego_valid[mask]

        min_xyz = pts.min(axis=0)
        max_xyz = pts.max(axis=0)

        results[int(iid)] = {
            "points": pts,
            "num_points": int(mask.sum()),
            "centroid_3d": np.mean(pts, axis=0),
            "bbox": [min_xyz.tolist(), max_xyz.tolist()],
        }
    
    return results



# Spatial clustering (cross-camera)

def clusters_overlap(ptsA, ptsB, threshold=0.8, min_matches=15):
    
    # Check if two 3D point clusters overlap using kdtree
    
    if len(ptsA) == 0 or len(ptsB) == 0:
        return False
    tree = cKDTree(ptsB)
    distances, _ = tree.query(ptsA, k=1, workers=-1)
    return int(np.sum(distances < threshold)) >= min_matches


def merge_clusters_spatial(instances):
    
    # Merge point clouds across different cameras 
    # Only merge if from different cameras at the same time.

    # Camera global ID format: cam_idx * 1000 + iid
    
    merged = []
    used = set()
    ids = list(instances.keys())

    for i in range(len(ids)):
        if ids[i] in used:
            continue

        cam_i = ids[i] // 1000
        clusterA = instances[ids[i]]["points"]
        group = [ids[i]]
        used.add(ids[i])

        for j in range(i+1, len(ids)):
            if ids[j] in used:
                continue

            cam_j = ids[j] // 1000

            # only merge if from different cameras
            if cam_i == cam_j:
                continue

            clusterB = instances[ids[j]]["points"]

            if clusters_overlap(clusterA, clusterB):
                group.append(ids[j])
                used.add(ids[j])

        merged.append(group)

    # # debugging logging 
    # n_multi = sum(1 for g in merged if len(g) > 1)
    # n_solo = sum(1 for g in merged if len(g) == 1)
    # large_merges = [g for g in merged if len(g) > 3]
    # print(f"  spatial {len(ids)} clusters -> {len(merged)} objects "
    #     f"({n_multi} cross-cam merges, {n_solo} single-cam)")
    # for g in large_merges:
    #     print(f"  spatial warning for large merge: {len(g)} clusters -> "
    #         f"likely chimera")

    return merged



# Temporal tracking 

def track_objects_temporal(objects_t0, objects_t1):
    
    # temporal tracking using centroid matching, mutual nearest and point cloud overlap:

    # Returns a list of (merged_id_t0, merged_id_t1)

    # logging
    reject_used = 0
    reject_mutual = 0
    reject_small = 0
    reject_overlap = 0
    accepted = 0

    if len(objects_t0) == 0 or len(objects_t1) == 0:
        return []

    ids_t0 = list(objects_t0.keys())
    ids_t1 = list(objects_t1.keys())

    centroids_t0 = np.array([objects_t0[i]["centroid"] for i in ids_t0])
    centroids_t1 = np.array([objects_t1[i]["centroid"] for i in ids_t1])

    # # debugging
    # sizes_t0 = [objects_t0[i]["num_points"] for i in ids_t0]
    # sizes_t1 = [objects_t1[i]["num_points"] for i in ids_t1]
    # print(f"temporal tracking: Size dist t0: min={min(sizes_t0)} "
    #     f"mean={np.mean(sizes_t0):.0f} max={max(sizes_t0)}")
    # print(f"temporal tracking: Size dist t1: min={min(sizes_t1)} "
    #     f"mean={np.mean(sizes_t1):.0f} max={max(sizes_t1)}")

    # compute pairwise distances
    dist_matrix = cdist(centroids_t0, centroids_t1, metric='euclidean')

    # Build all candidate pairs (distance, i, j)
    candidates = []
    for i in range(len(ids_t0)):
        for j in range(len(ids_t1)):
            d = dist_matrix[i, j]
            if d <= MAX_CENTROID_DIST:
                candidates.append((d, i, j))

    #     ## debugging Logging
    # print(f"temporal tracking: t0 objs: {len(ids_t0)}, t1 objs: {len(ids_t1)}")
    # print(f"temporal tracking: Candidates within dist ({MAX_CENTROID_DIST}m): {len(candidates)}")

    if len(candidates) > 0:
        dists = [c[0] for c in candidates]
        print(f"temporal tracking: Candidate dist range: min={min(dists):.2f}, "
            f"max={max(dists):.2f}, mean={np.mean(dists):.2f}")

    # Sort globally by distance (smallest first)
    candidates.sort(key=lambda x: x[0])

    matches = []
    used_t0 = set()
    used_t1 = set()

    for dist, i, j in candidates:
        id_t0 = ids_t0[i]
        id_t1 = ids_t1[j]

        
        if id_t0 in used_t0 or id_t1 in used_t1:
            reject_used += 1
            continue

        # mutual nearest check
        nearest_t1 = np.argmin(dist_matrix[i])
        nearest_t0 = np.argmin(dist_matrix[:, j])

        if nearest_t1 != j or nearest_t0 != i:
            reject_mutual += 1
            continue

        pts_t0 = objects_t0[id_t0]["points"]
        pts_t1 = objects_t1[id_t1]["points"]

        # reject objects under point threshold
        if pts_t0.shape[0] < MIN_POINTS_FOR_EXPORT or pts_t1.shape[0] < MIN_POINTS_FOR_EXPORT:
            reject_small += 1
            continue

        #  reject points under overlap threshold
        if not clusters_overlap(
            pts_t0,
            pts_t1,
            threshold=1.0,          # was 1.5
            min_matches=MIN_POINT_OVERLAP
        ):
            reject_overlap += 1
            continue

        # accept match
        matches.append((id_t0, id_t1))
        used_t0.add(id_t0)
        used_t1.add(id_t1)
        accepted += 1

        

        print(f"obj {id_t0} (t0) -> obj {id_t1} (t1) "
              f"dist={dist:.2f}m"
              f"pts=({pts_t0.shape[0]}, {pts_t1.shape[0]})")
        

    print("Temporal tracking summary:")
    print(f"  Candidates checked: {len(candidates)}")
    print(f"  Accepted: {accepted}")
    print(f"  Rejected (already used): {reject_used}")
    print(f"  Rejected (mutual nearest): {reject_mutual}")
    print(f"  Rejected (too small): {reject_small}")
    print(f"  Rejected (overlap fail): {reject_overlap}")

    
    return matches

# NPZ export 

def export_tracked_pairs_npz(objects_t0, objects_t1, matches, output_dir, ts0, ts1):
    
    # Export all tracked object pairs as a single numpy array zip per timestamp pair.

    # Objects are arranged on a 1D grid with 10m spacing across the x-axis after centring
    # this is to avoid the hdbscan epsilon paramter and faulty clustering

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts0_short = str(ts0)[:15]
    ts1_short = str(ts1)[:15]

    # Filter matches to only those meeting minimum point threshold
    valid_matches = [
        (id_t0, id_t1) for id_t0, id_t1 in matches
        if objects_t0[id_t0]["points"].shape[0] >= MIN_POINTS_FOR_EXPORT
        and objects_t1[id_t1]["points"].shape[0] >= MIN_POINTS_FOR_EXPORT
    ]

    if len(valid_matches) == 0:
        print("No valid matches to export")
        return 0

    GRID_SPACING = 10.0  # in metres, hbd epsilon is 0.25m 

    pc1_parts = []
    pc2_parts = []
    grid_offsets = []
    real_centroids_t0 = []
    real_centroids_t1 = []
    object_point_counts_1 = []
    object_point_counts_2 = []

    for idx, (id_t0, id_t1) in enumerate(valid_matches):
        pts_t0 = objects_t0[id_t0]["points"].copy()
        pts_t1 = objects_t1[id_t1]["points"].copy()

        centroid_t0 = pts_t0.mean(axis=0)
        centroid_t1 = pts_t1.mean(axis=0)

        # grid position for this object
        offset = np.array([idx * GRID_SPACING, 0.0, 0.0], dtype=np.float32)

        # center both clouds on t1 centroid, then apply grid
        pts_t0_grid = (pts_t0 - centroid_t1 + offset).astype(np.float32)
        pts_t1_grid = (pts_t1 - centroid_t1 + offset).astype(np.float32)

        pc1_parts.append(pts_t0_grid)
        pc2_parts.append(pts_t1_grid)
        grid_offsets.append(offset)
        real_centroids_t0.append(centroid_t0.astype(np.float32))
        real_centroids_t1.append(centroid_t1.astype(np.float32))
        object_point_counts_1.append(pts_t0_grid.shape[0])
        object_point_counts_2.append(pts_t1_grid.shape[0])

        translation_magnitude = np.linalg.norm(centroid_t1 - centroid_t0)
        print(f"Exporting obj {idx:03d} "
              f"pts=({pts_t0.shape[0]}, {pts_t1.shape[0]})"
              f"motion_magnitude={translation_magnitude:.3f}m "
              f"grid_offset=[{offset[0]:.1f}, 0, 0]")

    pc1 = np.vstack(pc1_parts)
    pc2 = np.vstack(pc2_parts)

    npz_path = output_dir / f"{ts0_short}_{ts1_short}.npz"
    np.savez(
        npz_path,
        pc1=pc1,
        pc2=pc2,
        object_point_counts_1=np.array(object_point_counts_1, dtype=np.int32),
        object_point_counts_2=np.array(object_point_counts_2, dtype=np.int32),
        grid_offsets=np.array(grid_offsets, dtype=np.float32),
        real_centroids_t0=np.array(real_centroids_t0, dtype=np.float32),
        real_centroids_t1=np.array(real_centroids_t1, dtype=np.float32),
        ts0=np.int64(ts0),
        ts1=np.int64(ts1),
    )

    print(f"Wrote {len(valid_matches)} objects into NPZ at {npz_path.name}")
    print(f"pc1={pc1.shape}, pc2={pc2.shape}")

    return len(valid_matches)


# Process single timestamp

def process_timestamp(loader, processor, lidar_paths, center_index, sam_cache=None):

    # Process all cameras for a single timestamp
    # if sam_cache dict is provided, SAM inference is skipped for camera images already processed at this timestamp. 
    # useful when temporal gap is small enough that cache can be used
   
    if sam_cache is None:
        sam_cache = {}

    path = lidar_paths[center_index]
    lidar_ts = int(path.stem)
    pts = io_utils.read_lidar_sweep(path, attrib_spec="xyz")
    r = np.linalg.norm(pts[:, :2], axis=1)
    pts = pts[r < MAX_RANGE_M]
    lidar_sweeps = [(lidar_ts, pts)]

    all_results = {}
    overlay_infos = []

    for cam_idx, cam in enumerate(CAMERAS):
        for lidar_ts, lidar_pts_ego in lidar_sweeps:

            # Function gets closest timestamps to lidar sweep
            cam_img_fpaths = loader.get_ordered_log_cam_fpaths(LOG_ID, cam)
            if len(cam_img_fpaths) == 0:
                continue
            cam_img_fpath = min(cam_img_fpaths, key=lambda p: abs(int(p.stem) - lidar_ts))
            cam_ts = int(cam_img_fpath.stem)

            cache_key = (cam, cam_ts)
            if cache_key in sam_cache:
                instance_map, masks = sam_cache[cache_key]
            else:
                image = Image.open(cam_img_fpath).convert("RGB")
                instance_map, masks = run_sam(image, processor)
                sam_cache[cache_key] = (instance_map, masks)

            if instance_map is None:
                continue

            # project points into this camera
            uv, points_cam, valid = loader.project_ego_to_img_motion_compensated(
                points_lidar_time=lidar_pts_ego,
                cam_name=cam,
                cam_timestamp_ns=cam_ts,
                lidar_timestamp_ns=lidar_ts,
                log_id=LOG_ID,
            )

            if uv is None:
                continue

            results = associate_points(uv, points_cam, lidar_pts_ego, valid, instance_map)

            gid_map = {}
            for iid in results.keys():
                gid_map[iid] = cam_idx * 1000 + iid

            overlay_infos.append({
                "img_path": cam_img_fpath,
                "uv": uv,
                "points_cam": points_cam,
                "valid": valid,
                "instance_map": instance_map,
                "gid_map": gid_map,
                "out_path": OUTPUT_ROOT / cam / f"{cam_ts}_lidar_{lidar_ts}.png"
            })
            overlay_infos[-1]["out_path"].parent.mkdir(parents=True, exist_ok=True)

            for iid, data in results.items():
                gid = cam_idx * 1000 + iid

                if gid not in all_results:
                    all_results[gid] = {
                        "points": data["points"],
                        "centroids": [data["centroid_3d"]],
                        "num_points": data["num_points"],
                        "bbox_min": np.array(data["bbox"][0]),
                        "bbox_max": np.array(data["bbox"][1]),
                    }
                else:
                    all_results[gid]["points"] = np.vstack([
                        all_results[gid]["points"],
                        data["points"]
                    ])
                    all_results[gid]["centroids"].append(data["centroid_3d"])
                    all_results[gid]["num_points"] += data["num_points"]
                    all_results[gid]["bbox_min"] = np.minimum(
                        all_results[gid]["bbox_min"], np.array(data["bbox"][0])
                    )
                    all_results[gid]["bbox_max"] = np.maximum(
                        all_results[gid]["bbox_max"], np.array(data["bbox"][1])
                    )

    merged = merge_clusters_spatial(all_results)

    merged_id_map = {}
    for merged_id, group in enumerate(merged):
        for gid in group:
            merged_id_map[gid] = merged_id

    merged_objects = {}
    for merged_id, group in enumerate(merged):
        merged_points = []
        all_centroids = []
        for gid in group:
            if gid in all_results:
                merged_points.append(all_results[gid]["points"])
                all_centroids.extend(all_results[gid]["centroids"])

        if merged_points:
            merged_points = np.vstack(merged_points)
            all_centroids = np.array(all_centroids)
            merged_objects[merged_id] = {
                "points": merged_points,
                "centroid": all_centroids.mean(axis=0),
                "num_points": merged_points.shape[0],
            }

    point_counts = [obj["num_points"] for obj in merged_objects.values()]


    if point_counts:
        print(f"Processing {len(merged_objects)} merged objects"
            f"pts: min={min(point_counts)}"
            f"mean={np.mean(point_counts):.0f}"
            f"max={max(point_counts)}")
    else:
        print("No objects detected for processing")


    return merged_objects, merged_id_map, overlay_infos


# Process one timestamp pair

def process_pair(loader, processor, lidar_paths, index_t0, render_overlays=True):
    
    # Process a single t0/t1 pair and export
    # Returns number of pairs exported

    index_t1 = index_t0 + TEMPORAL_GAP

    if index_t1 >= len(lidar_paths):
        return 0

    ts0 = int(lidar_paths[index_t0].stem)
    ts1 = int(lidar_paths[index_t1].stem)
    print(f"\n Pair: t0={ts0} t1={ts1} ")

    # Shared SAM cache across t0 and t1
    sam_cache = {}

    objects_t0, merged_id_map_t0, overlay_infos_t0 = process_timestamp(
        loader, processor, lidar_paths, index_t0, sam_cache=sam_cache
    )
    objects_t1, merged_id_map_t1, overlay_infos_t1 = process_timestamp(
        loader, processor, lidar_paths, index_t1, sam_cache=sam_cache
    )

    print(f" Objects: t0={len(objects_t0)}, t1={len(objects_t1)}")

    matches = track_objects_temporal(objects_t0, objects_t1)
    print(f" Tracked {len(matches)} objects")

    # Build colours
    global_colours = {}
    for id_t0 in objects_t0.keys():
        rng = np.random.RandomState(id_t0)
        global_colours[('t0', id_t0)] = np.array([
            rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255)
        ])
    for id_t1 in objects_t1.keys():
        matched_id_t0 = next((a for a, b in matches if b == id_t1), None)
        if matched_id_t0 is not None:
            global_colours[('t1', id_t1)] = global_colours[('t0', matched_id_t0)]
        else:
            rng = np.random.RandomState(id_t1 + 1000)
            global_colours[('t1', id_t1)] = np.array([
                rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255)
            ])

    # Render overlay
    if render_overlays:
        object_colours_t0 = {i: global_colours[('t0', i)] for i in objects_t0.keys()}
        object_colours_t1 = {i: global_colours[('t1', i)] for i in objects_t1.keys()}

        for overlay_info in overlay_infos_t0:
            draw_overlay_with_mask(
                overlay_info["img_path"], overlay_info["uv"], overlay_info["points_cam"],
                overlay_info["valid"], overlay_info["instance_map"], overlay_info["gid_map"],
                merged_id_map_t0, object_colours_t0, overlay_info["out_path"]
            )
        for overlay_info in overlay_infos_t1:
            draw_overlay_with_mask(
                overlay_info["img_path"], overlay_info["uv"], overlay_info["points_cam"],
                overlay_info["valid"], overlay_info["instance_map"], overlay_info["gid_map"],
                merged_id_map_t1, object_colours_t1, overlay_info["out_path"]
            )

    # Export pairs
    num_exported = export_tracked_pairs_npz(
        objects_t0, objects_t1, matches, ICP_FLOW_OUTPUT, ts0=ts0, ts1=ts1
    )
    print(f" Exported: {num_exported} NPZ pairs")
    return num_exported



# Main pipeline

def main():
    loader = AV2SensorDataLoader(DATA_ROOT, DATA_ROOT)

    model = build_sam3_image_model(device=DEVICE)
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)

    lidar_paths = loader.get_ordered_log_lidar_fpaths(LOG_ID)

    
    ICP_FLOW_OUTPUT.mkdir(parents=True, exist_ok=True)

    if MODE == "single":
        print(f"\n Mode: single, Log: {LOG_ID} ")

        lidar_index = min(
            range(len(lidar_paths)),
            key=lambda i: abs(int(lidar_paths[i].stem) - CAM_TIMESTAMP_NS)
        )

        total_exported = process_pair(
            loader, processor, lidar_paths,
            index_t0=lidar_index,
            render_overlays=True
        )

        metadata = {
            "log_id": LOG_ID,
            "mode": "single",
            "center_timestamp": CAM_TIMESTAMP_NS,
            "temporal_gap": TEMPORAL_GAP,
            "exported_pairs": total_exported,
        }

    elif MODE == "full_log":
        print(f"\n Mode: full_log, Log: {LOG_ID}  Stride: {FULL_LOG_STRIDE} ")

        # Build list of t0 indices at stride intervals
        # Stop early enough that t1 = t0 + TEMPORAL_GAP is not out of bounds
        t0_indices = list(range(0, len(lidar_paths) - TEMPORAL_GAP, FULL_LOG_STRIDE))
        print(f"Processing {len(t0_indices)} timestamp pairs")

        total_exported = 0
        pair_summaries = []

        for step, index_t0 in enumerate(t0_indices):
            print(f"\n[{step+1}/{len(t0_indices)}] index_t0={index_t0}")
            num_exported = process_pair(
                loader, processor, lidar_paths,
                index_t0=index_t0,
                render_overlays=False,  # skip overlays in full
            )
            pair_summaries.append({
                "index_t0": index_t0,
                "index_t1": index_t0 + TEMPORAL_GAP,
                "t0_timestamp": int(lidar_paths[index_t0].stem),
                "t1_timestamp": int(lidar_paths[index_t0 + TEMPORAL_GAP].stem),
                "exported_pairs": num_exported,
            })
            total_exported += num_exported

        print(f"\n full log processing complete: {total_exported} total pairs exported ")

        metadata = {
            "log_id": LOG_ID,
            "mode": "full_log",
            "temporal_gap": TEMPORAL_GAP,
            "stride": FULL_LOG_STRIDE,
            "total_timestamp_pairs": len(t0_indices),
            "total_exported_pairs": total_exported,
            "pairs": pair_summaries,
        }

    elif MODE == "eval":
        print(f" Mode: eval, Log: {LOG_ID}")
        print(f"Eval index: {EVAL_INDEX_PATH}")

        # Load eval index and filter for this log
        with open(EVAL_INDEX_PATH, 'rb') as f:
            eval_index = pickle.load(f)

        eval_entries = [e for e in eval_index if e[0] == LOG_ID]
        if len(eval_entries) == 0:
            raise ValueError(
                f"No eval entries found for log {LOG_ID} in {EVAL_INDEX_PATH}.\n"
                f"Set LOG_ID to one of: {list(set(e[0] for e in eval_index))}"
            )

        print(f"Found {len(eval_entries)} eval timestamps for this log")

        # Build timestamp -> lidar path index for fast lookup
        ts_to_index = {int(p.stem): i for i, p in enumerate(lidar_paths)}

        total_exported = 0
        pair_summaries = []
        skipped = 0

        for step, (log_id_entry, ts_str) in enumerate(eval_entries):
            ts = int(ts_str)
            print(f"\n[{step+1}/{len(eval_entries)}] eval timestamp: {ts}")

            if ts not in ts_to_index:
                print(f" caution!: timestamp {ts} not found in lidar paths, skipping")
                skipped += 1
                continue

            index_t0 = ts_to_index[ts]
            index_t1 = index_t0 + TEMPORAL_GAP

            if index_t1 >= len(lidar_paths):
                print(f" caution!: t1 index {index_t1} out of bounds, skipping")
                skipped += 1
                continue

            num_exported = process_pair(
                loader, processor, lidar_paths,
                index_t0=index_t0,
                render_overlays=False,
            )

            ts1 = int(lidar_paths[index_t1].stem)
            pair_summaries.append({
                "eval_timestamp": ts,
                "index_t0": index_t0,
                "index_t1": index_t1,
                "t0_timestamp": ts,
                "t1_timestamp": ts1,
                "exported_pairs": num_exported,
            })
            total_exported += num_exported

        print(f"\n eval complete: {total_exported} pairs exported "
              f"({skipped} timestamps skipped)")

        metadata = {
            "log_id": LOG_ID,
            "mode": "eval",
            "eval_index": str(EVAL_INDEX_PATH),
            "temporal_gap": TEMPORAL_GAP,
            "eval_timestamps": len(eval_entries),
            "skipped": skipped,
            "total_exported_pairs": total_exported,
            "pairs": pair_summaries,
        }

    else:
        raise ValueError(f"Unknown Mode: '{MODE}'. Use 'single', 'full_log', or 'eval'.")

    with open(ICP_FLOW_OUTPUT / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPipeline complete.")
    print(f"Output: {ICP_FLOW_OUTPUT}")


if __name__ == "__main__":
    main()