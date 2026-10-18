"""
variants/rebin_graphs.py

The tensor half of elevation rebinning. Refiles each detection into a different 
elevation bin over the already-exported graph shards — a pure tensor remap, no re-simulation.
Everything else (azimuth columns, sensor positions, edges, labels) is copied
through byte-for-byte.

Only the elevation power columns and, when the dataset has them, the
elevation angle-center columns change. Their positions come from the dataset's
manifest (uav_sim.schema.layout), since they depend on which sector options
the dataset was generated with.

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
from ..schema import ZF_X, ZF_Z, PHI_N_SECTORS


def normalize_timestamp(ts):
    """
    One canonical string form for a timestamp, so a key built from a labels
    CSV always matches a key built from a graph's `.timestamp`.
    """
    return str(ts).replace("T", " ")


def load_positions(labels_csv, exp_ids=None):
    """
    {(exp_id, ts_str): (x, y, z)} for the first drone of every labeled
    timestep, optionally restricted to `exp_ids`.
    """
    df = pd.read_csv(labels_csv, usecols=["exp_id", "timestamp", "drone_positions"])
    if exp_ids is not None:
        df = df[df["exp_id"].isin(set(exp_ids))]

    out = {}
    for exp_id, ts, positions in zip(df["exp_id"].tolist(),
                                     df["timestamp"].tolist(),
                                     df["drone_positions"].tolist()):
        first = ast.literal_eval(positions)[0]
        if tuple(first) == (0.0, 0.0, 0.0):
            continue
        out[(int(exp_id), normalize_timestamp(ts))] = tuple(first)
    return out


def rebin_shard(exp_graphs, exp_id, pos_dict, new_edges, new_centers_tensor, columns):
    """
    exp_graphs: list of Data objects for one experiment (build_experiment_graphs's
    per-exp_id list) — typically ~60 timesteps. Replaces each graph's x with a
    copy whose elevation columns are rebinned, and returns the list.
    columns: the dataset's feature layout, from uav_sim.schema.layout.
    """
    edges_arr = np.asarray(new_edges, dtype=np.float64)
    T = len(exp_graphs)
    N = exp_graphs[0].x.shape[0]

    x_stack = torch.stack([g.x for g in exp_graphs])          # [T, N, F]
    old_phi_power = x_stack[:, :, columns["phi_power"]]        # [T, N, 6]
    active = (old_phi_power != 0).any(dim=-1)                  # [T, N]
    current_power = old_phi_power.sum(dim=-1)                  # [T, N] (isolates the single nonzero, or 0)

    sensor_xyz = x_stack[0, :, ZF_X:ZF_Z + 1].numpy()           # [N, 3]

    drone_xyz_per_t = np.zeros((T, 3), dtype=np.float64)
    has_drone = np.zeros(T, dtype=bool)
    for t, g in enumerate(exp_graphs):
        pos = pos_dict.get((exp_id, normalize_timestamp(g.timestamp)))
        if pos is not None:
            drone_xyz_per_t[t] = pos
            has_drone[t] = True

    diff = drone_xyz_per_t[:, None, :] - sensor_xyz[None, :, :]        # [T, N, 3]
    horiz = np.hypot(diff[:, :, 0], diff[:, :, 1])                     # [T, N]
    phi = np.degrees(np.arctan2(diff[:, :, 2], np.maximum(horiz, 1e-6)))  # [T, N]

    bins = np.where(phi < 0, 0,
                     np.clip(np.searchsorted(edges_arr, phi, side="right") - 1, 0, PHI_N_SECTORS - 1) + 1)
    bins = np.clip(bins, 0, PHI_N_SECTORS - 1)                          # [T, N]

    valid = active.numpy() & has_drone[:, None]                        # [T, N]
    power_np = current_power.numpy()

    # One-hot scatter: new_phi_power[t,n,bins[t,n]] = power_np[t,n] where valid.
    one_hot = np.eye(PHI_N_SECTORS, dtype=np.float32)[bins]            # [T, N, 6]
    new_phi_power_np = one_hot * (power_np * valid)[:, :, None]        # [T, N, 6]
    new_phi_power = torch.from_numpy(new_phi_power_np).to(x_stack.dtype)

    x_new_stack = x_stack.clone()
    x_new_stack[:, :, columns["phi_power"]] = new_phi_power
    if "phi_angles" in columns:
        x_new_stack[:, :, columns["phi_angles"]] = new_centers_tensor.view(1, 1, PHI_N_SECTORS)
    for t, g in enumerate(exp_graphs):
        g.x = x_new_stack[t].clone()

    return exp_graphs


def run(scheme, src_dir, out_dir, columns, shard_glob="*.pt", progress=True):
    """
    Rebin every matching graph shard in `src_dir` into `out_dir` under
    `scheme`. `src_dir` is never written to. columns: the dataset's feature
    layout, from uav_sim.schema.layout.
    """
    if scheme not in ES.SCHEME_BOUNDARIES:
        raise ValueError(f"unknown scheme {scheme!r}; "
                         f"choose from {sorted(ES.SCHEME_BOUNDARIES)}")
    if os.path.abspath(src_dir) == os.path.abspath(out_dir):
        raise ValueError("out_dir must differ from src_dir — rebinning never edits in place")

    edges, centers = ES.SCHEME_BOUNDARIES[scheme]
    centers_t = torch.tensor(centers, dtype=torch.float32)

    labels_csv = next((os.path.join(src_dir, f) for f in sorted(os.listdir(src_dir))
                       if f.endswith("labels.csv")), None)
    if labels_csv is None:
        raise FileNotFoundError(f"no *labels.csv found in {src_dir}")

    os.makedirs(out_dir, exist_ok=True)

    out_labels_csv = os.path.join(out_dir, os.path.basename(labels_csv))
    if not os.path.exists(out_labels_csv):
        shutil.copy2(labels_csv, out_labels_csv)

    shard_paths = sorted(glob.glob(os.path.join(src_dir, shard_glob)))
    if not shard_paths:
        raise FileNotFoundError(f"no shard matched {shard_glob!r} in {src_dir}")
    print(f"Rebinning {len(shard_paths)} shard(s) under scheme={scheme} edges={edges}")

    for shard_path in tqdm(shard_paths, desc="Shards", disable=not progress):
        graphs = torch.load(shard_path, weights_only=False)
        width = next(iter(graphs.values()))[0].x.shape[1]
        if width != columns["width"]:
            raise ValueError(f"{shard_path}: feature width {width} does not match the "
                             f"manifest's layout ({columns['width']})")
        pos_dict = load_positions(labels_csv, exp_ids=graphs.keys())
        for exp_id, glist in graphs.items():
            rebin_shard(glist, exp_id, pos_dict, edges, centers_t, columns)
        torch.save(graphs, os.path.join(out_dir, os.path.basename(shard_path)))

    print(f"Done. Rebinned shards written to {out_dir}")
    return len(shard_paths)
