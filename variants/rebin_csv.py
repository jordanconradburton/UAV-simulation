"""
variants/rebin_csv.py

The flat-CSV half of elevation rebinning. Rebuilds the elevation-sector columns under a
different boundary scheme without re-simulating.

Run through the CLI, which calls `rebin.py`:
    python -m uav_sim rebin --dataset data/clean --out data/clean_E \
        --scheme E_fisher_optimal
"""
import ast
import glob
import os
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import elevation_schemes as ES
from ..schema import ZF_X, ZF_Z, N_SECTORS, PHI_N_SECTORS

PASSTHROUGH_FILES = [
    "drone_information.csv", "drone_trajectories.csv", "sensor_distances.csv",
    "sensor_sectors.csv", "sensor_information.csv", "toa_measurements.csv",
]


def _write_phi_sector_scheme_csv(csv_path, edges, centers):
    """
    The same 6-row reference table the generator writes
    (phi_sector_id, angle_start_deg, angle_end_deg, center_deg) for a
    given scheme.
    """
    rows = [(1, -90.0, 0.0, centers[0])]
    for i in range(len(edges) - 1):
        rows.append((i + 2, edges[i], edges[i + 1], centers[i + 1]))
    df = pd.DataFrame(rows, columns=["phi_sector_id", "angle_start_deg", "angle_end_deg", "center_deg"])
    df.to_csv(csv_path, index=False)


def _bin_index_vectorized(phi, edges_arr):
    """
    Elevation bin index for each angle, vectorized over the whole column.

    Returns a 0-indexed bin (0 = below horizon, 1-5 = the bins over [0, 90)); 
    the caller adds 1 because the CSV's phi_sector_id is 1-indexed.
    """
    bins = np.where(phi < 0, 0,
                     np.clip(np.searchsorted(edges_arr, phi, side="right") - 1, 0, N_SECTORS - 1) + 1)
    return np.clip(bins, 0, N_SECTORS - 1)


def sensor_xyz_lookup(graph_shard_path):
    """
    {experiment_id: [N, 3] sensor xyz} for one graph shard, keyed so the
    caller can look up by the CSV's own experiment_id.
    """
    graphs = torch.load(graph_shard_path, weights_only=False)
    offset = min(graphs.keys()) - 1
    return {exp_id - offset: glist[0].x[:, ZF_X:ZF_Z + 1].numpy() for exp_id, glist in graphs.items()}


def rebin_detection_summary(csv_path, out_path, sensor_xyz_by_exp, edges, num_sensors):
    edges_arr = np.asarray(edges, dtype=np.float64)
    df = pd.read_csv(csv_path)

    drone_xyz = np.array([ast.literal_eval(s)[0] for s in df["drone_positions"]], dtype=np.float64)  # [R,3]
    sensor_idx = (df["sensor_id"].to_numpy() - 1) % num_sensors

    sensor_xyz = np.zeros((len(df), 3), dtype=np.float64)
    for exp_id, group_idx in df.groupby("experiment_id").indices.items():
        arr = sensor_xyz_by_exp[exp_id]
        sensor_xyz[group_idx] = arr[sensor_idx[group_idx]]

    diff = drone_xyz - sensor_xyz
    horiz = np.hypot(diff[:, 0], diff[:, 1])
    phi = np.degrees(np.arctan2(diff[:, 2], np.maximum(horiz, 1e-6)))
    bins = _bin_index_vectorized(phi, edges_arr)  # 0-indexed
    phi_sector_ids = bins + 1  # the CSV is 1-indexed (1=below-horizon, 2-6=real bins)

    df["sensor_phi_sectors_detected"] = [f"[{b}]" for b in phi_sector_ids]

    powers = [float(ast.literal_eval(s)[0]) for s in df["individual_powers"]]
    df["elevation_bin_powers_dBm"] = [
        str([p if i == b else 0.0 for i in range(PHI_N_SECTORS)])
        for p, b in zip(powers, bins)]
    df.to_csv(out_path, index=False)


def run(scheme, graph_src_dir, csv_src_dir, out_dir, num_sensors, dir_glob="shard_*",
        graph_filename="graphs.pt", progress=True):
    """
    Rebin the flat-CSV output of a dataset into `out_dir` under `scheme`.

    graph_src_dir: the dataset's `graphs/` root. Read-only, and used ONLY to
        recover per-experiment sensor positions — it does not itself need to
        have been rebinned first.
    csv_src_dir:   the dataset's `csv/` root. Read-only.
    num_sensors:   the dataset's fixed `--sensors` count, needed to map each
        row's sensor_id to its graph node.
    """
    if scheme not in ES.SCHEME_BOUNDARIES:
        raise ValueError(f"unknown scheme {scheme!r}; "
                         f"choose from {sorted(ES.SCHEME_BOUNDARIES)}")
    for src in (graph_src_dir, csv_src_dir):
        if os.path.abspath(src) == os.path.abspath(out_dir):
            raise ValueError("out_dir must differ from every source dir — "
                             "rebinning never edits in place")

    edges, centers = ES.SCHEME_BOUNDARIES[scheme]

    subdirs = sorted(d for d in glob.glob(os.path.join(csv_src_dir, dir_glob))
                     if os.path.isdir(d))
    if not subdirs:
        raise FileNotFoundError(f"no shard directory matched {dir_glob!r} in {csv_src_dir}")
    print(f"Rebinning {len(subdirs)} shard directories under scheme={scheme} edges={edges}")

    for subdir in tqdm(subdirs, desc="Shards", disable=not progress):
        name = os.path.basename(subdir)
        out_subdir = os.path.join(out_dir, name)
        os.makedirs(out_subdir, exist_ok=True)

        for fname in PASSTHROUGH_FILES:
            src = os.path.join(subdir, fname)
            dst = os.path.join(out_subdir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        _write_phi_sector_scheme_csv(os.path.join(out_subdir, "phi_sector_scheme.csv"),
                                     edges, centers)

        graph_shard_path = os.path.join(graph_src_dir, name, graph_filename)
        if not os.path.exists(graph_shard_path):
            raise FileNotFoundError(
                f"need the graph shard for sensor positions but it is missing: {graph_shard_path}")

        rebin_detection_summary(
            os.path.join(subdir, "sensor_detection_summary.csv"),
            os.path.join(out_subdir, "sensor_detection_summary.csv"),
            sensor_xyz_lookup(graph_shard_path), edges, num_sensors=num_sensors,
        )

    print(f"Done. Rebinned CSVs written to {out_dir}")
    return len(subdirs)
