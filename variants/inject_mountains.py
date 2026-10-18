"""
variants/inject_mountains.py

Adds mountains to an already-generated mountain-free dataset, as pure
post-processing — no re-simulation. Both halves of the dataset (the graph .pt
shards and the flat CSVs) are rewritten from a single shared mountain draw per
experiment, so the two outputs are guaranteed consistent with each other rather
than independently sampled. The sampling and occlusion tests are in
`mountain_sampling.py`; this module reads and writes the dataset.

REQUIREMENTS ON THE SOURCE DATASET
──────────────────────────────────
  * single drone per experiment (the masking argument holds only for 1 drone)
  * a fixed `--sensors` count, not a randomized one
  * generated with `--mountains 0`

Usage:
    python -m uav_sim add-mountains --dataset data/clean --out data/mountains
"""
import csv
import json
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .. import layout as L
from ..errors import InjectionError
from .mountain_sampling import (
    SENSOR_PLACEMENT_HEIGHT, TARGET_HEIGHT, apply_blocked_mask_to_graphs,
    experiment_rng, filter_edges, sample_mountains_for_experiment,
    sensor_drone_blocked_matrix, sensor_pair_blocked,
)
from ..schema import ZF_X, ZF_Z, layout as feature_layout


RNG_DATASET_TAG = "mountains_v1"

CSV_PASSTHROUGH = [
    "drone_information.csv", "drone_trajectories.csv", "phi_sector_scheme.csv",
    "sensor_distances.csv", "sensor_sectors.csv", "sensor_information.csv",
]


def _fail(msg):
    raise InjectionError(msg)


def assert_non_destructive(src_paths, out_paths):
    """
    Hard guarantee that this cannot touch the source dataset.
    """
    for out in out_paths:
        out_abs = os.path.abspath(out)
        for src in src_paths:
            src_abs = os.path.abspath(src)
            if out_abs == src_abs or out_abs.startswith(src_abs + os.sep):
                _fail(f"output path {out_abs} is inside source tree {src_abs} — refusing to run")


# ---------------------------------------------------------------------------
# Reading the source geometry (full precision)
# ---------------------------------------------------------------------------
def load_sensor_positions(csv_dir, num_sensors):
    """
    {local_exp_id: [N, 3] float64} from the shard's sensor_distances.csv.
    """
    df = pd.read_csv(os.path.join(csv_dir, "sensor_distances.csv"),
                      usecols=["exp_id", "sensor1_id", "x1", "y1", "z1",
                               "sensor2_id", "x2", "y2", "z2"])
    a = df[["exp_id", "sensor1_id", "x1", "y1", "z1"]].to_numpy()
    b = df[["exp_id", "sensor2_id", "x2", "y2", "z2"]].to_numpy()
    both = np.unique(np.vstack([a, b]), axis=0)

    out = {}
    for exp_id, group in pd.DataFrame(
            both, columns=["exp_id", "sensor_id", "x", "y", "z"]).groupby("exp_id"):
        group = group.sort_values("sensor_id")

        idx = (group["sensor_id"].to_numpy().astype(np.int64) - 1) % num_sensors
        if not np.array_equal(idx, np.arange(len(group))):
            _fail(f"exp {int(exp_id)}: sensor_id -> node index is not 0..N-1 (got {idx})")
        out[int(exp_id)] = group[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return out


def load_trajectories(csv_dir):
    """
    {local_exp_id: (timestamps[T], seg_start[T,3], seg_end[T,3])} from the
    shard's drone_trajectories.csv, ordered by time.
    """
    df = pd.read_csv(os.path.join(csv_dir, "drone_trajectories.csv"),
                      usecols=["experiment_id", "drone_id", "timestamp",
                               "start_x", "start_y", "start_z", "end_x", "end_y", "end_z"])
    n_drones = df.groupby("experiment_id")["drone_id"].nunique()
    if (n_drones != 1).any():
        _fail("multi-drone experiment found — adding mountains as a mask over "
              "existing detections only holds for 1 drone")

    out = {}
    for exp_id, group in df.groupby("experiment_id"):
        group = group.sort_values("timestamp")
        out[int(exp_id)] = (
            group["timestamp"].to_numpy(),
            group[["start_x", "start_y", "start_z"]].to_numpy(dtype=np.float64),
            group[["end_x", "end_y", "end_z"]].to_numpy(dtype=np.float64),
        )
    return out


# ---------------------------------------------------------------------------
# Per-shard work
# ---------------------------------------------------------------------------
def build_masks(graphs, sensors_by_exp, traj_by_exp, shard, num_sensors, region_size,
                sensor_height=SENSOR_PLACEMENT_HEIGHT, target_height=TARGET_HEIGHT,
                limit=None, require_blocking=False, progress=True):
    """
    Sample mountains and compute both masks for every experiment in the shard.
    """
    offset = min(graphs.keys()) - 1

    pairs = np.array([(i, j) for i in range(num_sensors) for j in range(i + 1, num_sensors)])
    per_exp = {}
    stats = defaultdict(float)
    attempts = []

    exp_ids = sorted(graphs.keys())
    if limit is not None:
        exp_ids = exp_ids[:limit]

    for global_exp in tqdm(exp_ids, desc=f"{shard} sampling", disable=not progress):
        local_exp = global_exp - offset
        if local_exp not in sensors_by_exp or local_exp not in traj_by_exp:
            _fail(f"graph exp {global_exp} -> local {local_exp} missing from the CSVs "
                  f"(offset={offset} wrong?)")

        sensor_xyz = sensors_by_exp[local_exp]
        timestamps, seg_start, seg_end = traj_by_exp[local_exp]
        glist = graphs[global_exp]

        graph_xyz = glist[0].x[:, ZF_X:ZF_Z + 1].numpy().astype(np.float64)
        if graph_xyz.shape != sensor_xyz.shape or np.abs(graph_xyz - sensor_xyz).max() > 1e-2:
            _fail(f"exp {global_exp}/local {local_exp}: graph sensor xyz disagree with "
                  f"the CSV (max diff {np.abs(graph_xyz - sensor_xyz).max():.4g})")

        graph_ts = [str(g.timestamp) for g in glist]
        if graph_ts != list(timestamps):
            _fail(f"exp {global_exp}: graph timestamps differ from trajectory timestamps")

        hover = np.all(seg_start == seg_end, axis=1)
        rng = experiment_rng(RNG_DATASET_TAG, shard, local_exp)
        mountains, n_att = sample_mountains_for_experiment(
            rng, region_size, sensor_xyz, seg_start, seg_end, hover,
            sensor_height=sensor_height, target_height=target_height,
            require_blocked_detection=require_blocking,
        )
        attempts.append(n_att)

        blocked_tn = sensor_drone_blocked_matrix(sensor_xyz, seg_start, mountains)
        blocked_pair = sensor_pair_blocked(sensor_xyz, pairs, mountains)

        per_exp[global_exp] = {
            "local_exp": local_exp,
            "mountains": mountains,
            "blocked_tn": blocked_tn,
            "blocked_pair": blocked_pair,
            "ts_index": {ts: t for t, ts in enumerate(graph_ts)},
        }
        stats["n_mountains"] += len(mountains)
        stats["n_blocked_cells"] += int(blocked_tn.sum())
        stats["n_cells"] += blocked_tn.size
        stats["n_blocked_pairs"] += int(blocked_pair.sum())
        stats["n_pairs"] += len(blocked_pair)
        stats["n_exp_with_blocked"] += int(blocked_tn.any())
        stats["n_exp_with_dropped_edge"] += int(blocked_pair.any())

    stats["n_exp"] = len(per_exp)
    stats["mean_proposals_per_exp"] = float(np.mean(attempts))
    stats["max_proposals_per_exp"] = int(np.max(attempts))
    return per_exp, offset, dict(stats)


def write_graph_shard(graphs, per_exp, out_path, columns):
    """
    Apply both masks to the graph shard and save it. columns: the dataset's
    feature layout, from uav_sim.schema.layout.
    """
    removed = 0
    for global_exp, info in per_exp.items():
        glist = graphs[global_exp]
        removed += apply_blocked_mask_to_graphs(glist, info["blocked_tn"], columns["theta_power"],
                                                columns.get("phi_power"))

        edge_index, edge_attr = filter_edges(glist[0].edge_index, glist[0].edge_attr,
                                              info["blocked_pair"])
        for g in glist:
            g.edge_index = edge_index
            g.edge_attr = edge_attr

    keep = {e: graphs[e] for e in per_exp}
    torch.save(keep, out_path)
    return removed


def write_csv_dir(csv_src, csv_out, per_exp, num_sensors):
    """
    Copy the mountain-independent CSVs through untouched, and rewrite the two
    per-detection tables (sensor_detection_summary.csv, toa_measurements.csv)
    with the occluded rows removed.

    Returns (kept, dropped, skipped_exp) for the detection summary.
    """
    for fname in CSV_PASSTHROUGH:
        src = os.path.join(csv_src, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(csv_out, fname))

    by_local = {info["local_exp"]: info for info in per_exp.values()}
    counts = _filter_occluded_rows(os.path.join(csv_src, "sensor_detection_summary.csv"),
                                   os.path.join(csv_out, "sensor_detection_summary.csv"),
                                   by_local, num_sensors)

    toa_src = os.path.join(csv_src, "toa_measurements.csv")
    if os.path.exists(toa_src):
        toa_counts = _filter_occluded_rows(toa_src, os.path.join(csv_out, "toa_measurements.csv"),
                                           by_local, num_sensors)
        # One drone per experiment, so one TOA row per summary row.
        if toa_counts != counts:
            _fail(f"TOA rows (kept, dropped, skipped) {toa_counts} disagree with the "
                  f"detection summary {counts}")
    return counts


def _filter_occluded_rows(src, dst, by_local, num_sensors):
    """
    Copy a per-(experiment, sensor, timestep) CSV, deleting the rows whose line
    of sight a mountain now blocks.
    """
    kept = dropped = skipped_exp = 0
    with open(src, "r", newline="") as fin, open(dst, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            local_exp = int(row[col["experiment_id"]])
            info = by_local.get(local_exp)
            if info is None:          # only possible under --limit-experiments
                skipped_exp += 1
                continue
            node = (int(row[col["sensor_id"]]) - 1) % num_sensors
            t = info["ts_index"].get(row[col["timestamp"]])
            if t is None:
                _fail(f"exp {local_exp}: detection timestamp {row[col['timestamp']]!r} "
                      f"not in the graph timeline")
            if info["blocked_tn"][t, node]:
                dropped += 1
            else:
                writer.writerow(row)
                kept += 1
    return kept, dropped, skipped_exp


def write_mountains_csv(path, per_exp):
    """The exact mountains attached to each experiment.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment_id", "graph_exp_id", "mountain_index", "x", "y",
                    "side_length", "height"])
        for global_exp in sorted(per_exp):
            info = per_exp[global_exp]
            for i, (x, y, side, h) in enumerate(info["mountains"], start=1):
                w.writerow([info["local_exp"], global_exp, i, x, y, side, h])


# ---------------------------------------------------------------------------
def process_shard(dataset_root, out_root, shard, num_sensors, region_size, columns,
                  sensor_height=SENSOR_PLACEMENT_HEIGHT, target_height=TARGET_HEIGHT,
                  limit=None, require_blocking=False, overwrite=False, progress=True):
    """Inject mountains into one shard, writing both outputs into `out_root`."""
    t0 = time.time()
    graph_src = L.graph_shard_path(dataset_root, shard)
    labels_src = L.labels_path(dataset_root, shard)
    csv_src_dir = L.csv_dir(dataset_root, shard)
    graph_out_dir = L.graph_dir(out_root, shard)
    csv_out_dir = L.csv_dir(out_root, shard)
    graph_out = L.graph_shard_path(out_root, shard)

    assert_non_destructive([dataset_root], [graph_out_dir, csv_out_dir])

    for path in (graph_src, labels_src, csv_src_dir):
        if not os.path.exists(path):
            _fail(f"source missing: {path}")
    # Refuse to clobber an already-generated shard rather than silently redoing it.
    if os.path.exists(graph_out) and not overwrite:
        _fail(f"output already exists: {graph_out} (pass --overwrite to replace it)")

    os.makedirs(graph_out_dir, exist_ok=True)
    os.makedirs(csv_out_dir, exist_ok=True)

    sensors_by_exp = load_sensor_positions(csv_src_dir, num_sensors)
    traj_by_exp = load_trajectories(csv_src_dir)

    graphs = torch.load(graph_src, weights_only=False)
    if not graphs:
        _fail(f"graph shard {graph_src} is empty")
    width = next(iter(graphs.values()))[0].x.shape[1]
    if width != columns["width"]:
        _fail(f"graph feature width {width} does not match the manifest's layout ({columns['width']})")

    per_exp, offset, stats = build_masks(
        graphs, sensors_by_exp, traj_by_exp, shard, num_sensors, region_size,
        sensor_height=sensor_height, target_height=target_height,
        limit=limit, require_blocking=require_blocking, progress=progress)

    removed = write_graph_shard(graphs, per_exp, graph_out, columns)
    kept, dropped, _ = write_csv_dir(csv_src_dir, csv_out_dir, per_exp, num_sensors)

    shutil.copy2(labels_src, L.labels_path(out_root, shard))

    write_mountains_csv(os.path.join(csv_out_dir, "mountains.csv"), per_exp)

    if removed != dropped:
        _fail(f"graph removed {removed} detections but the CSV dropped {dropped} rows")

    report = {
        "shard": shard, "exp_id_offset": offset,
        "require_blocking": require_blocking, "rng_dataset_tag": RNG_DATASET_TAG,
        "experiments": int(stats["n_exp"]),
        "mountains_total": int(stats["n_mountains"]),
        "mountains_per_exp": stats["n_mountains"] / max(stats["n_exp"], 1),
        "detections_removed": removed,
        "detections_kept": kept,
        "pct_detections_removed": 100.0 * removed / max(removed + kept, 1),
        "pct_cells_blocked": 100.0 * stats["n_blocked_cells"] / max(stats["n_cells"], 1),
        "pct_edges_dropped": 100.0 * stats["n_blocked_pairs"] / max(stats["n_pairs"], 1),
        "pct_exp_with_blocked_detection": 100.0 * stats["n_exp_with_blocked"] / max(stats["n_exp"], 1),
        "pct_exp_with_dropped_edge": 100.0 * stats["n_exp_with_dropped_edge"] / max(stats["n_exp"], 1),
        "mean_proposals_per_exp": stats["mean_proposals_per_exp"],
        "max_proposals_per_exp": stats["max_proposals_per_exp"],
        "seconds": round(time.time() - t0, 1),
    }

    os.makedirs(L.reports_root(out_root), exist_ok=True)
    with open(os.path.join(L.reports_root(out_root), f"mountains_{shard}.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def consolidate_mountains(out_root, shards):
    """Merge the per-shard mountains.csv files into one combined_mountains.csv."""
    out = os.path.join(out_root, "combined_mountains.csv")
    n = 0
    with open(out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["shard", "experiment_id", "graph_exp_id", "mountain_index",
                    "x", "y", "side_length", "height"])
        for shard in shards:
            src = os.path.join(L.csv_dir(out_root, shard), "mountains.csv")
            if not os.path.exists(src):
                _fail(f"missing per-shard mountains.csv: {src}")
            with open(src, newline="") as fi:
                r = csv.reader(fi)
                next(r)
                for row in r:
                    w.writerow([shard] + row)
                    n += 1
    return out, n


def run(dataset_root, out_root, shards=None, limit=None, require_blocking=False,
        overwrite=False, progress=True):
    """
    Inject mountains into every shard of `dataset_root`, writing a complete new
    dataset to `out_root`.
    """
    manifest = L.read_manifest(dataset_root)
    params = manifest["parameters"]

    if params.get("drones") != 1:
        _fail("mountain injection requires a single-drone dataset "
              f"(this one has drones={params.get('drones')}) — see the module docstring")
    if params.get("sensors") is None:
        _fail("mountain injection requires a fixed --sensors count, not a randomized one")
    if params.get("mountains"):
        _fail(f"source dataset was generated with mountains={params['mountains']}; "
              "inject into a mountain-free dataset instead")

    if str(manifest.get("variant", "")).startswith("mountains"):
        _fail(f"{dataset_root} is already a mountains variant "
              f"(derived from {manifest.get('derived_from')}); "
              "inject into the mountain-free dataset instead")

    num_sensors = int(params["sensors"])
    region_size = tuple(params["region"])
    sensor_height = float(params.get("sensor_height", SENSOR_PLACEMENT_HEIGHT))
    target_height = float(params.get("target_height", TARGET_HEIGHT))
    columns = feature_layout(
        include_sector_angle_centers=params["include_azimuth_angle_centers"],
        include_phi_sectors=params["elevation_sectors"],
        include_phi_sector_angle_centers=params["include_elevation_angle_centers"])

    shards = shards or L.list_shards(dataset_root)
    if not shards:
        _fail(f"no shards found in {dataset_root}")

    reports = []
    for shard in shards:
        print(f"[{shard}] injecting mountains...", flush=True)
        rep = process_shard(dataset_root, out_root, shard, num_sensors, region_size, columns,
                            sensor_height=sensor_height, target_height=target_height,
                            limit=limit, require_blocking=require_blocking,
                            overwrite=overwrite, progress=progress)
        reports.append(rep)
        print(f"    experiments {rep['experiments']}   "
              f"mountains/exp {rep['mountains_per_exp']:.2f}   "
              f"detections removed {rep['detections_removed']:,} "
              f"({rep['pct_detections_removed']:.2f}%)   "
              f"edges dropped {rep['pct_edges_dropped']:.2f}%", flush=True)

    combined, n_mountains = consolidate_mountains(out_root, shards)

    out_manifest = dict(manifest)
    out_manifest["derived_from"] = os.path.abspath(dataset_root)
    out_manifest["variant"] = "mountains"
    out_manifest["variant_details"] = {
        "rng_dataset_tag": RNG_DATASET_TAG,
        "require_blocking": require_blocking,
        "mountains_total": n_mountains,
        "shards": reports,
    }
    L.write_manifest(out_root, out_manifest)

    total_removed = sum(r["detections_removed"] for r in reports)
    total_kept = sum(r["detections_kept"] for r in reports)
    print(f"\nDone. {len(shards)} shard(s), {n_mountains:,} mountains -> {out_root}")
    print(f"  detections removed  {total_removed:,} "
          f"({100.0 * total_removed / max(total_removed + total_kept, 1):.2f}%)")
    print(f"  mountain provenance {combined}")
    return reports
