"""
Custom dataloader for ICP-Flow to read SAM LiDAR pairs.
Expects files named: {ts0_short}_{ts1_short}.npz
as produced by lidarpipeline.py 
"""
import numpy as np
from pathlib import Path
import json


class Dataset_sam():
    def __init__(self, args):
        self.args = args
        self.data_root = Path(args.root)

        # Match new single-NPZ-per-timestamp format: {ts0}_{ts1}.npz
        # Exclude _flow.npz outputs and metadata.json
        self.npz_files = sorted([
            f for f in self.data_root.glob("*.npz")
            if "_flow" not in f.name
            and "metadata" not in f.name
        ])

        self.seq_paths = self.npz_files

        metadata_path = self.data_root / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        print(f'Dataset_sam: Loaded {len(self.npz_files)} point cloud pairs '
              f'from {args.root}')

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        # Returns data in ICP-Flow format.

        # Labels are set per-object using saved point counts so that
        # track() sees K distinct clusters without needing to re-run HDBSCAN internally.

        npz_path = self.npz_files[idx]
        npz_data = np.load(npz_path)

        pc1 = npz_data['pc1'].astype(np.float32) 
        pc2 = npz_data['pc2'].astype(np.float32)  

        counts_1 = npz_data['object_point_counts_1'].astype(np.int32) 
        counts_2 = npz_data['object_point_counts_2'].astype(np.int32)  

        ts0 = int(npz_data['ts0']) if 'ts0' in npz_data else 0
        ts1 = int(npz_data['ts1']) if 'ts1' in npz_data else 0

        # Build per-point object labels from saved counts
        labels_pc1 = np.repeat(
            np.arange(len(counts_1), dtype=np.int32), counts_1
        )
        labels_pc2 = np.repeat(
            np.arange(len(counts_2), dtype=np.int32), counts_2
        )

        data = {
            'raw_points': np.concatenate([pc2, pc1], axis=0),
            'time_indice': np.concatenate([
                np.zeros(len(pc2), dtype=np.int32),
                np.ones(len(pc1), dtype=np.int32)
            ]),
            'ego_poses': np.eye(4)[np.newaxis, :, :].repeat(2, axis=0),
            'data_path': str(npz_path),
            'scene_flow': np.zeros((len(pc1) + len(pc2), 3), dtype=np.float32),
            'ts0': ts0,
            'ts1': ts1,
        }

        points_dst = [pc2]
        points_src = [pc1]
        labels_dst = [labels_pc2]
        labels_src = [labels_pc1]

        # verify structure
        print(f" Dataset {npz_path.name}: "
              f"pc1={pc1.shape} pc2={pc2.shape} "
              f"n_objects={len(counts_1)} "
              f"labels_unique={np.unique(labels_pc1).tolist()}")

        return data, points_src, points_dst, labels_src, labels_dst