"""
Evaluation Script

Compares scene flow estimates from the SAM pipeline against
OpenSceneFlow H5 ground truth for the Argoverse2 timestamps.

Metrics:
  EPE3D: mean endpoint error in metres 
  ACC3DS: % points with error < 0.05m or < 5% of GT magnitude 
  ACC3DR: % points with error < 0.10m or < 10% of GT magnitude 
  Outlier: % points with error > 0.30m or > 30% of GT magnitude 

"""

import h5py
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


# Config

H5_PATH = Path("demo/val/25e5c600-36fe-3245-9cc0-40ef91620c22.h5")
EVAL_INDEX_PATH = Path("demo/val/index_eval.pkl")
LOG_ID = "25e5c600-36fe-3245-9cc0-40ef91620c22"

FLOW_OUTPUT_DIR = Path(
    "sceneflow_pipeline_output/25e5c600-36fe-3245-9cc0-40ef91620c22/2026-04-07-02-39-02-FlowPairs/icp_flow_pairs"
)


from av2.datasets.sensor.constants import AnnotationCategories
CATEGORY_TO_INDEX = {
    **{"NONE": 0},
    **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)},
}
INDEX_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_INDEX.items()}

EVAL_CATEGORIES = {
    "REGULAR_VEHICLE": CATEGORY_TO_INDEX.get("REGULAR_VEHICLE", 19),
    "PEDESTRIAN":      CATEGORY_TO_INDEX.get("PEDESTRIAN", 17),
    "LARGE_VEHICLE":   CATEGORY_TO_INDEX.get("LARGE_VEHICLE", 11),
    "BOX_TRUCK":       CATEGORY_TO_INDEX.get("BOX_TRUCK", 6),
    "BICYCLIST":       CATEGORY_TO_INDEX.get("BICYCLIST", 4),
}


# ego to sensor transform

_qw, _qx, _qy, _qz = 0.999991, 0.0, 0.0, -0.004246
_tx, _ty, _tz = 1.35018, 0.0, 1.64042
_R_ego_sensor = Rotation.from_quat([_qx, _qy, _qz, _qw]).as_matrix()
_t_ego_sensor = np.array([_tx, _ty, _tz])
EGO_TO_SENSOR_R = _R_ego_sensor.T
EGO_TO_SENSOR_T = -_R_ego_sensor.T @ _t_ego_sensor


def ego_to_sensor(pts_ego: np.ndarray) -> np.ndarray:
    return (EGO_TO_SENSOR_R @ pts_ego.T).T + EGO_TO_SENSOR_T



# Result metrics

def compute_metrics(flow_pred, flow_gt):
    if len(flow_pred) == 0:
        return {"EPE3D": float("nan"), "ACC3DS": float("nan"),
                "ACC3DR": float("nan"), "Outlier": float("nan"), "n_points": 0}

    error = np.linalg.norm(flow_pred - flow_gt, axis=1)
    gt_mag = np.linalg.norm(flow_gt, axis=1)

    epe3d = error.mean()
    acc3ds = np.mean((error < 0.05) | (error / (gt_mag + 1e-6) < 0.05))
    acc3dr = np.mean((error < 0.10) | (error / (gt_mag + 1e-6) < 0.10))
    outlier = np.mean((error > 0.30) & (error / (gt_mag + 1e-6) > 0.30))

    return {
        "EPE3D":   float(epe3d),
        "ACC3DS":  float(acc3ds),
        "ACC3DR":  float(acc3dr),
        "Outlier": float(outlier),
        "n_points": int(len(error)),
    }



# flow reconstruction and diagnostics

def reconstruct_full_scene_flow(eval_ts, lidar_pts, flow_gt_comp, instance_ids,
                               flow_cat, dufo_label, npz_files_for_ts):

    N = len(lidar_pts)
    flow_pred = np.zeros((N, 3), dtype=np.float32)
    coverage_mask = np.zeros(N, dtype=bool)

    #track assignment confidence 
    flow_confidence = np.zeros(N, dtype=np.float32)

    per_object_diag = []
    missing_flow_count = 0
    found_flow_count = 0

    if len(npz_files_for_ts) == 0:
        return flow_pred, coverage_mask, {
            "per_object": [],
            "missing_flow_files": 0,
            "found_flow_files": 0,
        }

    scene_tree = cKDTree(lidar_pts)

    for npz_path in npz_files_for_ts:

        flow_npz_path = npz_path.parent / (npz_path.stem + "_flow.npz")

        if not flow_npz_path.exists():
            missing_flow_count += 1
            print(f"missing flow file: {flow_npz_path.name}")
            continue

        found_flow_count += 1
        input_data = np.load(npz_path)
        flow_data = np.load(flow_npz_path)

        pc1 = input_data["pc1"]
        pc2 = input_data["pc2"]
        counts_1 = input_data["object_point_counts_1"]
        grid_offsets = input_data["grid_offsets"]
        real_centroids_t1 = input_data["real_centroids_t1"]

        n_pc2 = len(pc2)
        full_flow = flow_data["scene_flow"]
        obj_flow_all = full_flow[n_pc2 : n_pc2 + len(pc1)]

        if len(obj_flow_all) == 0:
            continue

        idx1 = 0
        for obj_idx, n1 in enumerate(counts_1):
            n1 = int(n1)

            pts_t0_grid = pc1[idx1 : idx1 + n1]
            obj_flow = obj_flow_all[idx1 : idx1 + n1]
            idx1 += n1

            offset = grid_offsets[obj_idx]
            centroid_t1 = real_centroids_t1[obj_idx]

            # Recover real coordinates
            pts_t0_real = pts_t0_grid - offset + centroid_t1

            #
            flow_mean = np.mean(obj_flow, axis=0)
            flow_magnitude = np.linalg.norm(flow_mean)

            flow_residuals = obj_flow - flow_mean
            disp = np.linalg.norm(flow_residuals, axis=1)
            flow_std = np.sqrt(np.mean(np.sum(flow_residuals**2, axis=1)))

            
            if flow_magnitude < 0.05:
                print(f" obj: {obj_idx:03d} negligible motion ({flow_magnitude:.3f}m)")
                continue

            
            threshold = max(0.2, 2.5 * np.median(disp))
            inliers = disp < threshold

            if inliers.sum() < 20:
                print(f" obj: {obj_idx:03d} no consistent motion ({inliers.sum()} inliers)")
                continue

            pts_t0_real = pts_t0_real[inliers]
            obj_flow = obj_flow[inliers]

            # transform
            pts_t0_sensor = ego_to_sensor(pts_t0_real)

            print(f" obj: {obj_idx:03d} "
                  f"flow_mean={flow_mean.round(3)} flow_magnitude ={flow_magnitude:.3f}m "
                  f"inliers={len(obj_flow)}")

            # kdtree matching
            dists, indices = scene_tree.query(pts_t0_sensor, k=1, workers=-1)
            valid_matches = dists < 0.5

            matched_scene_idx = indices[valid_matches]
            matched_obj_flow = obj_flow[valid_matches]

            if len(matched_scene_idx) == 0:
                continue

            dyn_matched = dufo_label[matched_scene_idx] == 1

            # filter for dynamic points
            if dyn_matched.sum() < 20:
                print(f" obj: {obj_idx:03d} too few dynamic points ({dyn_matched.sum()})")
                continue

            # confidence score, used to reduce overrides of better objects
            confidence = dyn_matched.sum() / max(len(obj_flow), 1)

            better = confidence > flow_confidence[matched_scene_idx]

            flow_pred[matched_scene_idx[better]] = matched_obj_flow[better]
            flow_confidence[matched_scene_idx[better]] = confidence
            coverage_mask[matched_scene_idx[better]] = True

            # metrics 
            if dyn_matched.sum() > 0:
                pred_dyn = matched_obj_flow[dyn_matched]
                gt_dyn = flow_gt_comp[matched_scene_idx[dyn_matched]]
                obj_metrics = compute_metrics(pred_dyn, gt_dyn)
            else:
                obj_metrics = {"EPE3D": float("nan"), "ACC3DS": float("nan"),
                               "ACC3DR": float("nan"), "Outlier": float("nan"),
                               "n_points": 0}

            # object purity
            matched_instance_ids = instance_ids[matched_scene_idx]
            matched_categories = flow_cat[matched_scene_idx]

            if len(matched_instance_ids) > 0:
                valid_ids = matched_instance_ids >= 0
                if valid_ids.sum() > 0:
                    unique_instances, counts = np.unique(
                        matched_instance_ids[valid_ids],
                        return_counts=True
                    )
                    dominant_count = counts.max()
                    dominant_instance = unique_instances[counts.argmax()]
                    purity = float(dominant_count) / len(matched_scene_idx)
                    n_instances_hit = len(unique_instances)

                    dom_mask = matched_instance_ids == dominant_instance
                    dom_cats = matched_categories[dom_mask]
                    dominant_cat_idx = int(np.bincount(dom_cats).argmax()) if len(dom_cats) > 0 else 0
                    dominant_cat = INDEX_TO_CATEGORY.get(dominant_cat_idx, "UNKNOWN")
                else:
                    purity = 0.0
                    n_instances_hit = 0
                    dominant_cat = "NONE"
            else:
                purity = 0.0
                n_instances_hit = 0
                dominant_cat = "NONE"

            per_object_diag.append({
                "npz": f"{npz_path.name}::obj{obj_idx:03d}",
                "n_points": int(n1),
                "n_matched": int(valid_matches.sum()),
                "n_dynamic_matched": int(dyn_matched.sum()),
                "purity": float(purity),
                "n_instances_hit": int(n_instances_hit),
                "dominant_category": dominant_cat,
                "EPE3D": obj_metrics["EPE3D"],
                "ACC3DS": obj_metrics["ACC3DS"],
                "Outlier": obj_metrics["Outlier"],
            })

    diagnostics = {
        "per_object": per_object_diag,
        "missing_flow_files": missing_flow_count,
        "found_flow_files": found_flow_count,
    }

    return flow_pred, coverage_mask, diagnostics

def print_sam_diagnostics(ts_str, diagnostics, lidar_pts, dufo_label,
                           flow_cat, coverage_mask, dyn_mask):
    # Print per-timestamp SAM diagnostic summary

    print(f"\n SAM Diagnostic Timestamp {ts_str[:18]}")

    # Flow file availability
    total_files = diagnostics["found_flow_files"] + diagnostics["missing_flow_files"]
    print(f" Flow files: {diagnostics['found_flow_files']}/{total_files} found "
          f"({diagnostics['missing_flow_files']} missing)")

    if len(diagnostics["per_object"]) == 0:
        print(" No objects processed.")
        return

    # Per-object summary
    print(f" {'NPZ File':<45} {'Pts':>5} {'Dyn':>5} {'Purity':>7} "
          f"{'Instances':>9} {'Category':<20} {'EPE3D':>7} {'Outlier':>8}")
    

    for obj in diagnostics["per_object"]:
        epe_str = f"{obj['EPE3D']:.3f}" if not np.isnan(obj['EPE3D']) else "  nan"
        out_str = f"{obj['Outlier']:.3f}" if not np.isnan(obj['Outlier']) else "  nan"
        print(f"    {obj['npz']:<45} {obj['n_points']:>5} {obj['n_dynamic_matched']:>5} "
              f"  {obj['purity']:>5.2f}   {obj['n_instances_hit']:>9} "
              f"{obj['dominant_category']:<20} {epe_str:>7} {out_str:>8}")

    # Purity summary
    purities = [o["purity"] for o in diagnostics["per_object"] if o["n_dynamic_matched"] > 0]
    if purities:
        print(f"\n Purity: mean={np.mean(purities):.2f}, "
              f"min={np.min(purities):.2f}, max={np.max(purities):.2f}")
        chimera_count = sum(1 for p in purities if p < 0.8)
        print(f" Chimera objects (purity < 0.8): {chimera_count}/{len(purities)}")

    # Per-category coverage at this timestamp
    print("\n Per-category coverage:")
    for cat_name, cat_idx in EVAL_CATEGORIES.items():
        cat_dyn_mask = (flow_cat == cat_idx) & dyn_mask
        n_cat_dyn = cat_dyn_mask.sum()
        if n_cat_dyn == 0:
            continue
        n_cat_covered = (coverage_mask & cat_dyn_mask).sum()
        pct = 100.0 * n_cat_covered / n_cat_dyn
        print(f" {cat_name:<23} {n_cat_covered:>5}/{n_cat_dyn:<5} ({pct:.1f}%)")

    # Per-object EPE3D ranked worst to best
    obj_with_epe = [(o["npz"], o["EPE3D"], o["dominant_category"], o["n_dynamic_matched"])
                    for o in diagnostics["per_object"]
                    if not np.isnan(o["EPE3D"]) and o["n_dynamic_matched"] > 0]
    if obj_with_epe:
        obj_with_epe.sort(key=lambda x: x[1], reverse=True)
        print("\n Per-object EPE3D (worst first):")
        for npz_name, epe, cat, n_dyn in obj_with_epe:
            print(f" {npz_name:<45} EPE3D={epe:.3f}m  cat={cat:<20} dyn_pts={n_dyn}")



# Main evaluation

def main():
    print(f"Loading H5 ground truth: {H5_PATH}")
    print(f"Loading flow outputs: {FLOW_OUTPUT_DIR}")

    with open(EVAL_INDEX_PATH, "rb") as f:
        eval_index = pickle.load(f)
    eval_entries = [e for e in eval_index if e[0] == LOG_ID]
    print(f"Eval timestamps: {len(eval_entries)}")

    
    all_npz = sorted([
    f for f in FLOW_OUTPUT_DIR.glob("*.npz")
    if "_flow" not in f.name
    and "metadata" not in f.name
    ])
    print(f"Total SAM NPZ pairs: {len(all_npz)}")

    # Count flow files globally
    all_flow = [f for f in FLOW_OUTPUT_DIR.glob("*_flow.npz")]
    print(f"Total ICP-Flow output files: {len(all_flow)}")
    if len(all_flow) < len(all_npz):
        print(f"  Warning: {len(all_npz) - len(all_flow)} NPZ pairs have no flow output")

    npz_by_ts = defaultdict(list)
    for npz in all_npz:
        parts = npz.stem.split("_")
        ts0_short = parts[0]
        npz_by_ts[ts0_short].append(npz)

    all_results = defaultdict(lambda: {"flow_pred": [], "flow_gt": []})
    category_results = {cat: {"flow_pred": [], "flow_gt": []}
                        for cat in EVAL_CATEGORIES}

    per_timestamp_results = []

    # global diagnostic accumulators
    global_purity_scores = []
    global_chimera_count = 0
    global_total_objects = 0
    global_missing_flow = 0
    global_found_flow = 0
    category_coverage = {cat: {"covered": 0, "total": 0} for cat in EVAL_CATEGORIES}

    covered_results = {"flow_pred": [], "flow_gt": []}
    covered_category_results = {cat: {"flow_pred": [], "flow_gt": []}
                             for cat in EVAL_CATEGORIES}

    print(f"\n{'Timestamp':<22} {'Pts':>6} {'Covered':>8} {'EPE3D':>8} {'ACC3DS':>8}")

    with h5py.File(H5_PATH, "r") as f:
        for log_id_entry, ts_str in eval_entries:
            ts = int(ts_str)
            ts_short = ts_str[:15]

            if ts_str not in f:
                print(f" {ts_str}: Not in H5, skipping")
                continue

            sweep = f[ts_str]

            lidar_pts       = sweep["lidar"][:]
            flow_gt_raw     = sweep["flow"][:]
            ego_motion      = sweep["ego_motion"][:]
            dufo_label      = sweep["dufo_label"][:]
            flow_cat        = sweep["flow_category_indices"][:]
            instance_ids    = sweep["flow_instance_id"][:]
            eval_mask       = sweep["eval_mask"][:].squeeze() if "eval_mask" in sweep \
                              else np.ones(len(lidar_pts), dtype=bool)

            R, t = ego_motion[:3, :3], ego_motion[:3, 3]
            ego_flow = (R @ lidar_pts.T).T + t - lidar_pts
            flow_gt_comp = flow_gt_raw - ego_flow

            npz_files_for_ts = npz_by_ts.get(ts_short, [])

            flow_pred, coverage_mask, diagnostics = reconstruct_full_scene_flow(
                ts, lidar_pts, flow_gt_comp, instance_ids,
                flow_cat, dufo_label, npz_files_for_ts
            )

            dyn_mask = (dufo_label == 1) & eval_mask
            n_dyn = dyn_mask.sum()
            n_covered = (coverage_mask & dyn_mask).sum()

            if n_dyn == 0:
                continue

            metrics = compute_metrics(flow_pred[dyn_mask], flow_gt_comp[dyn_mask])

            # metrics on covered dynamic points only
            covered_dyn_mask = dyn_mask & coverage_mask
            if covered_dyn_mask.sum() > 0:
                metrics_covered = compute_metrics(
                    flow_pred[covered_dyn_mask],
                    flow_gt_comp[covered_dyn_mask]
                )
            else:
                metrics_covered = {"EPE3D": float("nan"), "ACC3DS": float("nan"),
                                "ACC3DR": float("nan"), "Outlier": float("nan"),
                                "n_points": 0}

            print(f" {ts_str[:18]:<22} {n_dyn:>6} {n_covered:>8} "
                f"{metrics['EPE3D']:>8.4f} {metrics['ACC3DS']:>8.4f} "
                f"covered_only: EPE3D={metrics_covered['EPE3D']:>8.4f} "
                f"ACC3DS={metrics_covered['ACC3DS']:>8.4f}")

            # Print SAM diagnostics per timestamp
            print_sam_diagnostics(ts_str, diagnostics, lidar_pts, dufo_label,
                                   flow_cat, coverage_mask, dyn_mask)

            global_missing_flow += diagnostics["missing_flow_files"]
            global_found_flow += diagnostics["found_flow_files"]

            for obj in diagnostics["per_object"]:
                global_total_objects += 1
                if obj["n_dynamic_matched"] > 0:
                    global_purity_scores.append(obj["purity"])
                    if obj["purity"] < 0.8:
                        global_chimera_count += 1

            # per-category coverage accumulation
            for cat_name, cat_idx in EVAL_CATEGORIES.items():
                cat_dyn_mask = (flow_cat == cat_idx) & dyn_mask
                n_cat = cat_dyn_mask.sum()
                n_cat_covered = (coverage_mask & cat_dyn_mask).sum()
                category_coverage[cat_name]["total"] += int(n_cat)
                category_coverage[cat_name]["covered"] += int(n_cat_covered)

            per_timestamp_results.append({
                "timestamp": ts_str,
                "n_dynamic": int(n_dyn),
                "n_covered": int(n_covered),
                "coverage_rate": float(n_covered / n_dyn) if n_dyn > 0 else 0.0,
                **metrics
            })

            all_results["dynamic"]["flow_pred"].append(flow_pred[dyn_mask])
            all_results["dynamic"]["flow_gt"].append(flow_gt_comp[dyn_mask])

            # accumulate covered-only results
            if covered_dyn_mask.sum() > 0:
                covered_results["flow_pred"].append(flow_pred[covered_dyn_mask])
                covered_results["flow_gt"].append(flow_gt_comp[covered_dyn_mask])

            # per-category covered-only
            for cat_name, cat_idx in EVAL_CATEGORIES.items():
                cat_covered_mask = (flow_cat == cat_idx) & covered_dyn_mask
                if cat_covered_mask.sum() > 0:
                    covered_category_results[cat_name]["flow_pred"].append(
                        flow_pred[cat_covered_mask])
                    covered_category_results[cat_name]["flow_gt"].append(
                        flow_gt_comp[cat_covered_mask])

    
    # Overall results
    
    
    print("Overall Results (Dynamic foreground points)")
   

    if all_results["dynamic"]["flow_pred"]:
        all_pred = np.concatenate(all_results["dynamic"]["flow_pred"])
        all_gt = np.concatenate(all_results["dynamic"]["flow_gt"])
        overall = compute_metrics(all_pred, all_gt)

        print(f" Total dynamic points evaluated: {overall['n_points']:,}")
        print(f" EPE3D: {overall['EPE3D']:.4f} m ")
        print(f" ACC3DS: {overall['ACC3DS']:.4f} ")
        print(f" ACC3DR: {overall['ACC3DR']:.4f} ")
        print(f" Outlier: {overall['Outlier']:.4f} ")
    else:
        overall = {}
        print(" No dynamic points found, check flow output directory")

    print("\nPer-Category Results")
    print(f"{'Category':<25} {'Pts':>8} {'EPE3D':>8} {'ACC3DS':>8} {'ACC3DR':>8} {'Outlier':>8}")

    print("\nOverall Results(Covered dynamic points only) ")


    if covered_results["flow_pred"]:
        covered_pred = np.concatenate(covered_results["flow_pred"])
        covered_gt = np.concatenate(covered_results["flow_gt"])
        overall_covered = compute_metrics(covered_pred, covered_gt)

        print(f" Total covered dynamic points: {overall_covered['n_points']:,}")
        print(f" EPE3D: {overall_covered['EPE3D']:.4f} m ")
        print(f" ACC3DS: {overall_covered['ACC3DS']:.4f} ")
        print(f" ACC3DR: {overall_covered['ACC3DR']:.4f} ")
        print(f" Outlier: {overall_covered['Outlier']:.4f} ")

    print("\nPer-Category Results (Covered points only)")
    print(f"{'Category':<25} {'Pts':>8} {'EPE3D':>8} {'ACC3DS':>8} "
        f"{'ACC3DR':>8} {'Outlier':>8}")
    for cat_name, data in covered_category_results.items():
        if data["flow_pred"]:
            pred = np.concatenate(data["flow_pred"])
            gt = np.concatenate(data["flow_gt"])
            m = compute_metrics(pred, gt)
            print(f" {cat_name:<23} {m['n_points']:>8} {m['EPE3D']:>8.4f} "
                f"{m['ACC3DS']:>8.4f} {m['ACC3DR']:>8.4f} {m['Outlier']:>8.4f}")
        else:
            print(f"  {cat_name:<23} {'no data':>8}")

    category_metrics = {}
    for cat_name, data in category_results.items():
        if data["flow_pred"]:
            pred = np.concatenate(data["flow_pred"])
            gt = np.concatenate(data["flow_gt"])
            m = compute_metrics(pred, gt)
            category_metrics[cat_name] = m
            print(f"  {cat_name:<23} {m['n_points']:>8} {m['EPE3D']:>8.4f} "
                  f"{m['ACC3DS']:>8.4f} {m['ACC3DR']:>8.4f} {m['Outlier']:>8.4f}")
        else:
            print(f"  {cat_name:<23} {'no data':>8}")

    # 
    # SAM Diagnostic Summary
    # 
    
    print("SAM Diagnostic Summary")
   

    print("\n Flow file availability:")
    print(f" Found: {global_found_flow}")
    print(f" Missing: {global_missing_flow}")
    if global_found_flow + global_missing_flow > 0:
        pct = 100.0 * global_found_flow / (global_found_flow + global_missing_flow)
        print(f" Coverage: {pct:.1f}% of NPZ pairs have ICP-Flow output")

    print("\n Object purity:")
    print(f" Total objects evaluated: {global_total_objects}")
    if global_purity_scores:
        print(f" Mean purity: {np.mean(global_purity_scores):.3f}")
        print(f" Median purity: {np.median(global_purity_scores):.3f}")
        print(f" Min purity: {np.min(global_purity_scores):.3f}")
        print(f" Chimera objects (purity < 0.8): "
              f"{global_chimera_count}/{len(global_purity_scores)} "
              f"({100.0*global_chimera_count/len(global_purity_scores):.1f}%)")

    print("\n Per-category coverage:")
    print(f" {'Category':<23} {'Covered':>8} {'Total':>8} {'Pct':>8}")
    for cat_name, counts in category_coverage.items():
        total = counts["total"]
        covered = counts["covered"]
        if total > 0:
            pct = 100.0 * covered / total
            print(f"    {cat_name:<23} {covered:>8} {total:>8} {pct:>7.1f}%")
        else:
            print(f"    {cat_name:<23} {'no data':>8}")

    if per_timestamp_results:
        coverage_rates = [r["coverage_rate"] for r in per_timestamp_results]
        print(f"\n Overall point coverage:")
        print(f" Mean: {np.mean(coverage_rates)*100:.1f}%")
        print(f" Min: {np.min(coverage_rates)*100:.1f}%")
        print(f" Max: {np.max(coverage_rates)*100:.1f}%")

    # Save results
    results = {
        "log_id": LOG_ID,
        "overall": overall,
        "per_category": category_metrics, ###
        "overall_covered": overall_covered if covered_results["flow_pred"] else {},
        "per_category": category_metrics,
        "per_category_covered": {
            cat: compute_metrics(
                np.concatenate(data["flow_pred"]),
                np.concatenate(data["flow_gt"])
            ) if data["flow_pred"] else {}
            for cat, data in covered_category_results.items()
        }, ##
        "per_timestamp": per_timestamp_results,
        "sam_diagnostics": {
            "found_flow_files": global_found_flow,
            "missing_flow_files": global_missing_flow,
            "total_objects": global_total_objects,
            "mean_purity": float(np.mean(global_purity_scores)) if global_purity_scores else None,
            "chimera_count": global_chimera_count,
            "category_coverage": {
                cat: {
                    "covered": counts["covered"],
                    "total": counts["total"],
                    "pct": float(100.0 * counts["covered"] / counts["total"])
                    if counts["total"] > 0 else 0.0
                }
                for cat, counts in category_coverage.items()
            }
        }
    }

    out_path = FLOW_OUTPUT_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()