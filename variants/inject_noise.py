"""
variants/inject_noise.py

Adds receiver noise to the received power of an already-generated dataset, as
post-processing — no re-simulation. The graph shards, the detection summary
and the time-of-arrival table are all rewritten from one shared noise draw per
experiment, so the outputs agree with each other.

THE NOISE MODEL
───────────────
One parameter, a noise floor F in dBm. The noise is added to the signal's
voltage, not to its power in dB, because a power is a voltage squared:

    sigma^2  = 10^(F/10) mW                     noise power, the floor itself
    P_meas   = |sqrt(P_signal) + n|^2,          n ~ complex N(0, sigma^2)
    RSSI     = 10 log10(P_meas)

so E[P_meas] = P_signal + sigma^2. The noise is therefore not a fixed number of
dB: at a -90 dBm floor it moves a -58 dBm reading by about +-0.15 dB and a
-90 dBm reading by about +-5 dB. It is drawn as complex (I/Q) noise, as real
receivers sample; a single real sample could cancel the signal exactly and
send the RSSI to -inf.

WHICH SENSORS DETECT THE DRONE
──────────────────────────────
Noise moves readings across the detection threshold both ways, so the whole
(timestep, sensor) grid is rebuilt, including sensors with no detection in the
source. Their clean power is not recorded, but it follows from Friis: the only
unrecorded input, the carrier frequency, is solved for from the recorded
detections, and that solution is checked on detections it was not fitted on
before it is used. A sensor then detects the drone under the simulator's rule
for one drone, with the noise floor as the noise term of the SINR:

    detected = (prx - F > sinr_threshold) and (prx > sensitivity)

--gate off keeps the source's detections exactly, so readings get noisier but
no detection is lost or added.

FALSE ALARMS
────────────
A false alarm is a detection the noisy dataset has and the clean source did
not: noise pushed a reading over the threshold. They are kept by default, as a
real receiver would report them; false_alarms=False (--no-false-alarms) removes
them, so noise can only weaken or remove the source's detections.

REQUIREMENTS ON THE SOURCE DATASET
──────────────────────────────────
  * one drone per experiment (the Friis solution and the detection rule above
    assume one transmitter)
  * a fixed `--sensors` count
  * no mountains: a new detection could lie behind one, and mountains are not
    in the export. Add noise first, then run `add-mountains` on the result.

Usage:
    python -m uav_sim add-noise --dataset data/clean --out data/noise_90 \
        --noise-floor-dbm -90
"""
import csv
import hashlib
import json
import math
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .. import layout as L
from ..core.detection import PHI_SECTOR_EDGES_DEG
from ..core.physics import (C_LIGHT, DEFAULT_SYNC_ERROR_NS, DEFAULT_TOA_NOISE_NS,
                            FRIIS_CONSTANT_DB, calculate_distance, sum_powers_dbm)
from ..errors import InjectionError
from ..schema import N_SECTORS, PHI_N_SECTORS, layout as feature_layout


RNG_DATASET_TAG = "noise_v1"
GATES = ("noise_floor", "off")

MAX_FRIIS_RESIDUAL_DB = 1e-3
MIN_LINEAR_POWER_MW = 1e-30

CSV_PASSTHROUGH = [
    "drone_information.csv", "drone_trajectories.csv", "phi_sector_scheme.csv",
    "sensor_distances.csv", "sensor_sectors.csv", "sensor_information.csv",
]

SUMMARY_COLUMNS = {
    "experiment_id", "sensor_id", "timestamp", "num_detections", "total_power_dBm",
    "azimuth_sector_powers_dBm", "elevation_bin_powers_dBm", "sensor_sectors_detected",
    "drone_ids", "drone_positions", "individual_powers", "sensor_phi_sectors_detected",
}


def _fail(msg):
    raise InjectionError(msg)


def assert_non_destructive(src_paths, out_paths):
    """Refuse to run if an output path lies inside a source tree."""
    for out in out_paths:
        out_abs = os.path.abspath(out)
        for src in src_paths:
            src_abs = os.path.abspath(src)
            if out_abs == src_abs or out_abs.startswith(src_abs + os.sep):
                _fail(f"output path {out_abs} is inside source tree {src_abs} — refusing to run")


def experiment_rng(*key_parts):
    """
    Deterministic numpy generator for one experiment, independent of process
    and run order. The noise floor is deliberately not part of the key, so
    every floor, gate and false-alarm setting with the same seed sees the same
    unit noise draws and differs only by the setting itself.
    """
    key = "|".join(str(p) for p in key_parts).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))


# ---------------------------------------------------------------------------
# Physics, vectorised over a [T, N] grid
# ---------------------------------------------------------------------------
def friis_prx(ptx_dbm, gt_db, gr_db, distance_m, freq_hz):
    """core.physics.calculate_signal_power over arrays."""
    return (ptx_dbm + gt_db + gr_db - 20.0 * np.log10(distance_m)
            - 20.0 * np.log10(freq_hz) + FRIIS_CONSTANT_DB)


def recover_carrier_frequency(prx_dbm, distance_m, ptx_dbm, gt_db, gr_db):
    """
    The carrier frequency, which no exported file records (drone_information
    has only the band), solved from Friis over the given detections:

        20 log10(f) = Ptx + Gt + Gr - 20 log10(d) + FRIIS_CONSTANT_DB - Prx

    One drone transmits on one carrier, so every detection gives the same f up
    to float32 storage; the mean is taken. Returns (f_hz, worst in-sample
    error in dB).
    """
    log20f = ptx_dbm + gt_db + gr_db - 20.0 * np.log10(distance_m) + FRIIS_CONSTANT_DB - prx_dbm
    f = float(np.mean(10.0 ** (log20f / 20.0)))
    residual = np.abs(friis_prx(ptx_dbm, gt_db, gr_db, distance_m, f) - prx_dbm)
    return f, float(residual.max())


def holdout_error(prx_dbm, distance_m, ptx_dbm, gt_db, gr_db):
    """
    Worst error, in dB, when a frequency fitted on the even-indexed detections
    predicts the odd-indexed ones. This is the check that makes it safe to
    compute powers for sensors with no recorded detection. Returns (error,
    number of held-out detections); 0 held out means fewer than two detections.
    """
    if prx_dbm.size < 2:
        return 0.0, 0
    fit = np.zeros(prx_dbm.size, dtype=bool)
    fit[::2] = True
    f, _ = recover_carrier_frequency(prx_dbm[fit], distance_m[fit], ptx_dbm, gt_db, gr_db[fit])
    predicted = friis_prx(ptx_dbm, gt_db, gr_db[~fit], distance_m[~fit], f)
    return float(np.abs(predicted - prx_dbm[~fit]).max()), int((~fit).sum())


def add_receiver_noise(prx_dbm, noise_floor_dbm, unit_i, unit_q):
    """
    The measured power, in dBm, of a signal of `prx_dbm` plus complex noise of
    power 10^(F/10) mW. unit_i / unit_q are standard normals from the caller,
    so the same draw can be reused at every floor.
    """
    amp = math.sqrt(10.0 ** (noise_floor_dbm / 10.0) / 2.0)   # per quadrature
    v_signal = np.sqrt(10.0 ** (prx_dbm / 10.0))
    p_meas = (v_signal + amp * unit_i) ** 2 + (amp * unit_q) ** 2
    return 10.0 * np.log10(np.maximum(p_meas, MIN_LINEAR_POWER_MW))


def detection_mask(prx_dbm, sensitivity_dbm, sinr_threshold_db, noise_dbm):
    """
    core.physics.calculate_sinr and is_detected for one drone (no
    interferers), over arrays: SINR above the threshold and power above the
    sensitivity, both strict.
    """
    sinr = prx_dbm - 10.0 * np.log10(10.0 ** (noise_dbm / 10.0))
    return (sinr > sinr_threshold_db) & (prx_dbm > sensitivity_dbm)


def azimuth_sector_index(drone_xyz, sensor_xyz, sector_starts):
    """
    [T, N] 1-based azimuth sector, as run_detection_simulation assigns it: the
    sector whose [start, end) holds degrees(atan2(dy, dx)) % 360. Each sensor's
    sectors are contiguous from 0, so the sector is the last start <= theta.
    sector_starts: one array of start angles per sensor.
    """
    dx = drone_xyz[:, None, 0] - sensor_xyz[None, :, 0]
    dy = drone_xyz[:, None, 1] - sensor_xyz[None, :, 1]
    theta = np.degrees(np.arctan2(dy, dx)) % 360.0
    idx = np.empty(theta.shape, dtype=np.int64)
    for n, starts in enumerate(sector_starts):
        idx[:, n] = np.searchsorted(starts, theta[:, n], side="right")
    return idx


def elevation_bin_index(drone_xyz, sensor_xyz, edges):
    """
    [T, N] 1-based elevation bin, as detection._phi_sector_index assigns it:
    1 below the horizon, then one bin per [edge_i, edge_i+1), and exactly 90
    degrees in the last bin.
    """
    dz = drone_xyz[:, None, 2] - sensor_xyz[None, :, 2]
    horiz = np.hypot(drone_xyz[:, None, 0] - sensor_xyz[None, :, 0],
                     drone_xyz[:, None, 1] - sensor_xyz[None, :, 1])
    phi = np.degrees(np.arctan2(dz, np.maximum(horiz, 1e-6)))
    idx = np.searchsorted(np.asarray(edges, dtype=np.float64), phi, side="right") + 1
    return np.where(phi < 0.0, 1, np.minimum(idx, len(edges)))


# ---------------------------------------------------------------------------
# Reading one shard's geometry and radio parameters (full precision)
# ---------------------------------------------------------------------------
def load_shard_inputs(csv_dir, num_sensors):
    """
    {local_exp_id: dict} of everything the noise model needs except the
    recorded powers, which come from the graph.
    """
    read = lambda name, **kw: pd.read_csv(os.path.join(csv_dir, name),
                                          float_precision="round_trip", **kw)

    sd = read("sensor_distances.csv", usecols=["exp_id", "sensor1_id", "x1", "y1", "z1",
                                               "sensor2_id", "x2", "y2", "z2"])
    pairs = np.vstack([sd[["exp_id", "sensor1_id", "x1", "y1", "z1"]].to_numpy(),
                       sd[["exp_id", "sensor2_id", "x2", "y2", "z2"]].to_numpy()])
    sensors = pd.DataFrame(np.unique(pairs, axis=0),
                           columns=["exp_id", "sensor_id", "x", "y", "z"])

    info = read("sensor_information.csv").set_index("sensor_id")
    sectors = read("sensor_sectors.csv").sort_values(["sensor_id", "sector_id"])
    starts_by_sensor = {int(s): g["angle_start"].to_numpy(dtype=np.float64)
                        for s, g in sectors.groupby("sensor_id")}

    drones = read("drone_information.csv")
    if drones.groupby("experiment_id").size().max() != 1:
        _fail("multi-drone experiment found — the noise model assumes one drone")
    drones = drones.set_index("experiment_id")

    traj = read("drone_trajectories.csv",
                usecols=["experiment_id", "timestamp", "start_x", "start_y", "start_z"],
                dtype={"timestamp": str})

    out = {}
    for exp_id, grp in sensors.groupby("exp_id"):
        exp_id = int(exp_id)
        grp = grp.sort_values("sensor_id")
        sids = grp["sensor_id"].to_numpy().astype(np.int64)

        if not np.array_equal((sids - 1) % num_sensors, np.arange(len(sids))):
            _fail(f"exp {exp_id}: sensor_id -> node index is not 0..N-1")
        props = info.loc[sids]
        d = drones.loc[exp_id]
        out[exp_id] = {
            "sensor_ids": sids,
            "sensor_xyz": grp[["x", "y", "z"]].to_numpy(dtype=np.float64),
            "gain": props["gain"].to_numpy(dtype=np.float64),
            "sensitivity": props["sensitivity"].to_numpy(dtype=np.float64),
            "sinr_threshold": props["sinr_threshold"].to_numpy(dtype=np.float64),
            "band_min": props["freq_min"].to_numpy(dtype=np.float64),
            "band_max": props["freq_max"].to_numpy(dtype=np.float64),
            "sector_starts": [starts_by_sensor[int(s)] for s in sids],
            "drone_id": int(d["drone_id"]),
            "ptx": float(d["drone_transmitter_power"]),
            "gt": float(d["transmitter_antenna_gain"]),
            "drone_band": (float(d["freq_min"]), float(d["freq_max"])),
        }

    for exp_id, grp in traj.groupby("experiment_id"):
        if int(exp_id) not in out:
            _fail(f"exp {int(exp_id)} has a trajectory but no sensors")
        grp = grp.sort_values("timestamp")
        out[int(exp_id)]["timestamps"] = grp["timestamp"].tolist()
        out[int(exp_id)]["drone_xyz"] = grp[["start_x", "start_y", "start_z"]].to_numpy(dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# One experiment
# ---------------------------------------------------------------------------
def rebuild_detections(exp, graph_list, power_slice, noise_floor_dbm, gate_noise_dbm,
                       false_alarms, rng, elevation_edges):
    """
    One experiment's noisy detection grid.

    Returns (detected [T, N] bool, noisy power [T, N] dBm, azimuth sector and
    elevation bin [T, N], source detections [T, N] bool, stats).
    """
    sxyz, dxyz = exp["sensor_xyz"], exp["drone_xyz"]
    T, N = len(dxyz), len(sxyz)
    dist = np.linalg.norm(dxyz[:, None, :] - sxyz[None, :, :], axis=2)
    gain = np.broadcast_to(exp["gain"][None, :], (T, N))

    src_prx = torch.stack([g.x[:, power_slice] for g in graph_list]).double().sum(dim=2).numpy()
    src_has = src_prx != 0.0

    in_sample = holdout = 0.0
    n_holdout = 0
    if src_has.any():
        freq, in_sample = recover_carrier_frequency(
            src_prx[src_has], dist[src_has], exp["ptx"], exp["gt"], gain[src_has])
        holdout, n_holdout = holdout_error(
            src_prx[src_has], dist[src_has], exp["ptx"], exp["gt"], gain[src_has])
        worst = max(in_sample, holdout)
        if worst > MAX_FRIIS_RESIDUAL_DB:
            _fail(f"Friis does not reproduce the recorded powers (worst error "
                  f"{worst:.3e} dB > {MAX_FRIIS_RESIDUAL_DB}); the CSVs and the "
                  f"graph shard probably describe different experiments")
    else:
        freq = 0.5 * sum(exp["drone_band"])

    prx = friis_prx(exp["ptx"], exp["gt"], gain, dist, freq)
    prx[src_has] = src_prx[src_has]

    unit_i = rng.standard_normal((T, N))
    unit_q = rng.standard_normal((T, N))
    noisy = add_receiver_noise(prx, noise_floor_dbm, unit_i, unit_q)

    if gate_noise_dbm is None:
        detected = src_has.copy()
    else:
        in_band = (exp["band_min"] <= freq) & (freq <= exp["band_max"])
        detected = detection_mask(noisy, exp["sensitivity"][None, :],
                                  exp["sinr_threshold"][None, :], gate_noise_dbm) & in_band[None, :]
        if not false_alarms:
            detected &= src_has

    azimuth = azimuth_sector_index(dxyz, sxyz, exp["sector_starts"])
    elevation = elevation_bin_index(dxyz, sxyz, elevation_edges)
    stats = {
        "source_detections": int(src_has.sum()),
        "detections": int(detected.sum()),
        "false_alarms": int((detected & ~src_has).sum()),
        "lost": int((src_has & ~detected).sum()),
        "cells": int(detected.size),
        "in_sample_friis_error_db": in_sample,
        "holdout_friis_error_db": holdout,
        "holdout_checked": n_holdout,
    }
    return detected, noisy, azimuth, elevation, src_has, stats


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def summary_row(header, exp_id, sensor_id, ts, drone_id, drone_pos, prx, sector, phi):
    """
    One detection-summary row for a single detection.
    """
    combined = sum_powers_dbm([prx])

    def bins(index, n):
        values = [0.0] * n
        values[index - 1] = round(combined, 2)
        return values

    values = {
        "experiment_id": exp_id, "sensor_id": sensor_id, "timestamp": ts,
        "num_detections": 1,
        "total_power_dBm": f"{combined:.2f}",
        "azimuth_sector_powers_dBm": bins(sector, N_SECTORS),
        "elevation_bin_powers_dBm": bins(phi, PHI_N_SECTORS),
        "sensor_sectors_detected": [sector], "drone_ids": [drone_id],
        "drone_positions": [drone_pos], "individual_powers": [f"{prx:.2f}"],
        "sensor_phi_sectors_detected": [phi],
    }
    return [values[c] for c in header]


def write_toa_rows(writer, src_rows, exp_id, exp, detected, rng, sync_error_s, toa_noise_s):
    """
    Write one experiment's TOA rows: the source row for every surviving
    detection, a new row for every false alarm, in the exporter's
    (timestamp, sensor) order. src_rows: {(timestamp, sensor_id): row}.
    """
    T, N = detected.shape

    fresh_emission = rng.uniform(0.0, 1.0, T)
    fresh_clock = rng.normal(0.0, sync_error_s, N)
    timing_noise = rng.normal(0.0, toa_noise_s, (T, N))

    col = {"ts": 2, "sensor": 1, "emission": 8, "clock": 10}
    emission = {row[col["ts"]]: float(row[col["emission"]]) for row in src_rows.values()}
    clock = {int(row[col["sensor"]]): float(row[col["clock"]]) for row in src_rows.values()}

    sids, sxyz, dxyz = exp["sensor_ids"], exp["sensor_xyz"], exp["drone_xyz"]
    for t, ts in enumerate(exp["timestamps"]):
        for n in np.flatnonzero(detected[t]):
            sid = int(sids[n])
            row = src_rows.get((ts, sid))
            if row is None:
                d_pos, s_pos = tuple(dxyz[t].tolist()), tuple(sxyz[n].tolist())
                distance = calculate_distance(d_pos, s_pos)
                e = emission.get(ts, float(fresh_emission[t]))
                c = clock.get(sid, float(fresh_clock[n]))
                toa = e + distance / C_LIGHT + c + float(timing_noise[t, n])
                row = [exp_id, sid, ts, exp["drone_id"], *s_pos,
                       toa, e, distance / C_LIGHT, c, *d_pos]
            writer.writerow(row)


def _read_toa_by_experiment(path):
    """
    Yield (experiment_id, {(timestamp, sensor_id): row}) from a TOA table
    sorted by experiment.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected = ["experiment_id", "sensor_id", "timestamp", "drone_id",
                    "sensor_x", "sensor_y", "sensor_z", "toa_s", "emission_offset_s",
                    "propagation_time_s", "clock_offset_s", "drone_x", "drone_y", "drone_z"]
        if header != expected:
            _fail(f"{path}: unexpected TOA header {header}")
        yield header
        current, rows = None, {}
        for row in reader:
            exp_id = int(row[0])
            if exp_id != current:
                if current is not None:
                    yield current, rows
                current, rows = exp_id, {}
            rows[(row[2], int(row[1]))] = row
        if current is not None:
            yield current, rows


# ---------------------------------------------------------------------------
# One shard
# ---------------------------------------------------------------------------
def process_shard(dataset_root, out_root, shard, params, noise_floor_dbm, gate,
                  false_alarms, seed, overwrite=False, progress=True):
    """Add noise to one shard, writing graphs, labels and CSVs into `out_root`."""
    t0 = time.time()
    graph_src, labels_src = L.graph_shard_path(dataset_root, shard), L.labels_path(dataset_root, shard)
    csv_src = L.csv_dir(dataset_root, shard)
    graph_out_dir, csv_out = L.graph_dir(out_root, shard), L.csv_dir(out_root, shard)
    graph_out = L.graph_shard_path(out_root, shard)

    assert_non_destructive([dataset_root], [graph_out_dir, csv_out])
    for path in (graph_src, labels_src, csv_src):
        if not os.path.exists(path):
            _fail(f"source missing: {path}")
    if os.path.exists(graph_out) and not overwrite:
        _fail(f"output already exists: {graph_out} (pass --overwrite to replace it)")
    os.makedirs(graph_out_dir, exist_ok=True)
    os.makedirs(csv_out, exist_ok=True)

    num_sensors = int(params["sensors"])
    columns = feature_layout(
        include_sector_angle_centers=params["include_azimuth_angle_centers"],
        include_phi_sectors=params["elevation_sectors"],
        include_phi_sector_angle_centers=params["include_elevation_angle_centers"])
    elevation_edges = params.get("elevation_sector_edges") or PHI_SECTOR_EDGES_DEG
    gate_noise_dbm = noise_floor_dbm if gate == "noise_floor" else None

    inputs = load_shard_inputs(csv_src, num_sensors)
    graphs = torch.load(graph_src, weights_only=False)
    if not graphs:
        _fail(f"graph shard {graph_src} is empty")
    width = next(iter(graphs.values()))[0].x.shape[1]
    if width != columns["width"]:
        _fail(f"graph feature width {width} does not match the manifest's layout ({columns['width']})")
    offset = min(graphs) - 1

    with open(os.path.join(csv_src, "sensor_detection_summary.csv"), newline="") as f:
        header = next(csv.reader(f))
    unknown = [c for c in header if c not in SUMMARY_COLUMNS]
    if unknown:
        _fail(f"unknown detection-summary columns {unknown}")

    totals = {k: 0 for k in ("source_detections", "detections", "false_alarms", "lost",
                             "cells", "holdout_checked")}
    worst_in_sample = worst_holdout = 0.0
    detected_by_exp = {}

    with open(os.path.join(csv_out, "sensor_detection_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for graph_exp in tqdm(sorted(graphs), desc=f"{shard} noise", disable=not progress):
            local = graph_exp - offset
            exp = inputs.get(local)
            glist = graphs[graph_exp]
            if exp is None or "timestamps" not in exp:
                _fail(f"graph exp {graph_exp} -> local {local} is missing from the CSVs")
            if [str(g.timestamp) for g in glist] != exp["timestamps"]:
                _fail(f"exp {local}: graph timestamps differ from the trajectory timestamps")
            graph_xyz = glist[0].x[:, columns["x"]:columns["z"] + 1].numpy().astype(np.float64)
            if np.abs(graph_xyz - exp["sensor_xyz"]).max() > 1e-2:
                _fail(f"exp {local}: graph sensor positions disagree with the CSVs")

            rng = experiment_rng(RNG_DATASET_TAG, seed, shard, local)
            detected, noisy, azimuth, elevation, _, stats = rebuild_detections(
                exp, glist, columns["theta_power"], noise_floor_dbm, gate_noise_dbm,
                false_alarms, rng, elevation_edges)
            detected_by_exp[local] = detected
            for k in totals:
                totals[k] += stats[k]
            worst_in_sample = max(worst_in_sample, stats["in_sample_friis_error_db"])
            worst_holdout = max(worst_holdout, stats["holdout_friis_error_db"])

            theta = torch.zeros(len(glist), num_sensors, N_SECTORS)
            phi = torch.zeros(len(glist), num_sensors, PHI_N_SECTORS)
            for t, ts in enumerate(exp["timestamps"]):
                pos = tuple(exp["drone_xyz"][t].tolist())
                for n in np.flatnonzero(detected[t]):
                    p = float(noisy[t, n])
                    combined = sum_powers_dbm([p])
                    theta[t, n, azimuth[t, n] - 1] = combined
                    phi[t, n, elevation[t, n] - 1] = combined
                    writer.writerow(summary_row(header, local, int(exp["sensor_ids"][n]), ts,
                                                exp["drone_id"], pos, p,
                                                int(azimuth[t, n]), int(elevation[t, n])))
            for t, g in enumerate(glist):
                g.x[:, columns["theta_power"]] = theta[t]
                if "phi_power" in columns:
                    g.x[:, columns["phi_power"]] = phi[t]

    torch.save(graphs, graph_out)
    shutil.copy2(labels_src, L.labels_path(out_root, shard))
    for name in CSV_PASSTHROUGH:
        src = os.path.join(csv_src, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(csv_out, name))

    toa_src = os.path.join(csv_src, "toa_measurements.csv")
    if os.path.exists(toa_src):
        sync_error_s = float(params.get("sync_error_ns", DEFAULT_SYNC_ERROR_NS)) * 1e-9
        toa_noise_s = float(params.get("toa_noise_ns", DEFAULT_TOA_NOISE_NS)) * 1e-9
        groups = _read_toa_by_experiment(toa_src)
        toa_header = next(groups)
        pending = next(groups, None)
        with open(os.path.join(csv_out, "toa_measurements.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(toa_header)
            for local in sorted(detected_by_exp):
                src_rows = {}
                if pending is not None and pending[0] == local:
                    src_rows = pending[1]
                    pending = next(groups, None)
                rng = experiment_rng(RNG_DATASET_TAG, "toa", seed, shard, local)
                write_toa_rows(writer, src_rows, local, inputs[local], detected_by_exp[local],
                               rng, sync_error_s, toa_noise_s)
        if pending is not None:
            _fail(f"{toa_src} has rows for experiment {pending[0]}, which the graph shard lacks")

    report = {
        "shard": shard, "noise_floor_dbm": noise_floor_dbm, "gate": gate,
        "false_alarms": false_alarms, "seed": seed, "rng_dataset_tag": RNG_DATASET_TAG,
        "experiments": len(detected_by_exp),
        "source_detections": totals["source_detections"],
        "detections": totals["detections"],
        "false_alarms_added": totals["false_alarms"],
        "detections_lost": totals["lost"],
        "pct_cells_detected_source": 100.0 * totals["source_detections"] / max(totals["cells"], 1),
        "pct_cells_detected": 100.0 * totals["detections"] / max(totals["cells"], 1),
        "max_in_sample_friis_error_db": worst_in_sample,
        "max_holdout_friis_error_db": worst_holdout,
        "holdout_detections_checked": totals["holdout_checked"],
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(L.reports_root(out_root), exist_ok=True)
    with open(os.path.join(L.reports_root(out_root), f"noise_{shard}.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def run(dataset_root, out_root, noise_floor_dbm, gate="noise_floor", false_alarms=True,
        seed=0, shards=None, overwrite=False, progress=True):
    """
    Add receiver noise at `noise_floor_dbm` to every shard of `dataset_root`,
    writing a complete new dataset (graphs, labels, CSVs and a manifest) to
    `out_root`.
    """
    if gate not in GATES:
        raise ValueError(f"unknown gate {gate!r}; choose from {list(GATES)}")
    if not math.isfinite(noise_floor_dbm):
        raise ValueError("the noise floor must be a finite number of dBm")
    if gate == "off" and not false_alarms:
        raise ValueError("--no-false-alarms has no effect with --gate off, "
                         "which never adds a detection")

    manifest = L.read_manifest(dataset_root)
    params = manifest["parameters"]
    if params.get("drones") != 1:
        _fail(f"adding noise requires a single-drone dataset (this one has drones={params.get('drones')})")
    if params.get("sensors") is None:
        _fail("adding noise requires a fixed --sensors count, not a randomized one")
    if params.get("mountains") != 0 or str(manifest.get("variant", "")).startswith("mountains"):
        _fail("the source has mountains, which are not in the export, so a new detection "
              "could lie behind one; add noise to the mountain-free dataset, then run "
              "add-mountains on the result")
    if str(manifest.get("variant", "")).startswith("noise"):
        _fail(f"{dataset_root} already has noise added; add noise to the clean dataset instead")

    shards = shards or L.list_shards(dataset_root)
    if not shards:
        _fail(f"no shards found in {dataset_root}")

    print(f"Adding noise: floor {noise_floor_dbm} dBm, gate {gate}, "
          f"false alarms {'kept' if false_alarms else 'removed'}, seed {seed}", flush=True)
    reports = []
    for shard in shards:
        print(f"[{shard}] adding noise...", flush=True)
        rep = process_shard(dataset_root, out_root, shard, params, noise_floor_dbm, gate,
                            false_alarms, seed, overwrite=overwrite, progress=progress)
        reports.append(rep)
        print(f"    experiments {rep['experiments']}   detections "
              f"{rep['source_detections']:,} -> {rep['detections']:,}   "
              f"(+{rep['false_alarms_added']:,} false alarms, -{rep['detections_lost']:,} lost)",
              flush=True)

    out_manifest = dict(manifest)
    out_manifest["derived_from"] = os.path.abspath(dataset_root)
    out_manifest["variant"] = "noise"
    out_manifest["variant_details"] = {
        "noise_floor_dbm": noise_floor_dbm, "gate": gate, "false_alarms": false_alarms,
        "seed": seed, "rng_dataset_tag": RNG_DATASET_TAG,
        "sync_error_ns": params.get("sync_error_ns", DEFAULT_SYNC_ERROR_NS),
        "toa_noise_ns": params.get("toa_noise_ns", DEFAULT_TOA_NOISE_NS),
        "shards": reports,
    }
    L.write_manifest(out_root, out_manifest)

    source = sum(r["source_detections"] for r in reports)
    total = sum(r["detections"] for r in reports)
    print(f"\nDone. {len(shards)} shard(s) -> {out_root}")
    print(f"  detections {source:,} -> {total:,}")
    return reports
