"""
generate.py

Turns simulation parameters into a dataset on disk.

One shard at a time: simulate N experiments into a fresh SQLite database,
export both halves of the shard from that database, verify what was written,
then discard the database and move on. A failed shard leaves its database
behind for inspection and does not touch the shards already written.
"""
import contextlib
import io
import os
import random
import shutil
import sys
import time

from . import layout as L
from .errors import VerificationError
from .schema import layout
from .core.database import DatabaseManager
from .core.physics import TOA_RNG
from .core.experiment import (DEFAULT_MIN_SENSOR_DISTANCE, DEFAULT_NUM_STEPS,
                              DEFAULT_SENSOR_HEIGHT, DEFAULT_TARGET_HEIGHT,
                              run_simulation_experiments)
from .core.csv_export import export_csv_tables
from .core.graph_export import export_all_graphs_to_one_file, export_drone_labels

DEFAULT_SHARD_SIZE = 5000


@contextlib.contextmanager
def muffled(quiet):
    """
    Silence the exporters' own progress chatter when `quiet` is set.
    """
    if not quiet:
        yield 
        return
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            yield
    except BaseException:
        sys.stderr.write(buffer.getvalue())
        raise


def seed_everything(seed):
    """
    Seed the simulator's random number generator.

    Every random draw the simulator makes (sensor placement, trajectories,
    noise, detection) comes from Python's built-in `random` module, except the
    time-of-arrival draws, which come from their own generator
    (physics.TOA_RNG) seeded here too. Seeding both makes a run reproducible end
    to end. The add-mountains variant does not depend on this: it uses its own
    generator, seeded per shard.
    """
    if seed is None:
        return
    random.seed(seed)
    TOA_RNG.seed(f"toa:{seed}")


def verify_shard(root, shard, n_exp, expect_width, require_ptx=True): 
    """
    Re-open everything the shard wrote and check it is complete.
    """
    import pandas as pd
    import torch

    def fail(msg):
        raise VerificationError(f"[{shard}] {msg}")

    labels_csv = L.labels_path(root, shard)
    if not os.path.exists(labels_csv):
        fail(f"labels CSV not found: {labels_csv}")
    labels = pd.read_csv(labels_csv)
    n_unique = labels["exp_id"].nunique()
    if n_unique != n_exp:
        fail(f"labels CSV has {n_unique} experiments, expected {n_exp} — export truncated")
    if require_ptx:
        if "drone_transmitter_powers" not in labels.columns:
            fail("drone_transmitter_powers column missing from labels CSV")
        if labels["drone_transmitter_powers"].isna().any():
            fail("drone_transmitter_powers has null values")

    graph_path = L.graph_shard_path(root, shard)
    if not os.path.exists(graph_path):
        fail(f"graph shard not found: {graph_path}")
    graphs = torch.load(graph_path, weights_only=False)
    if len(graphs) != n_exp:
        fail(f"graph shard has {len(graphs)} experiments, expected {n_exp} — export truncated")

    for exp_id, snapshots in graphs.items():
        if not snapshots:
            fail(f"experiment {exp_id} has no timesteps")
        for snap in snapshots:
            width = snap.x.shape[1]
            if width != expect_width:
                fail(f"experiment {exp_id}: graph feature width is {width}, expected {expect_width}")
            if torch.isinf(snap.edge_attr).any():
                fail(f"experiment {exp_id}: edge_attr contains inf — mountain-blocked "
                     "pairs must be dropped, not marked infinite")

    csv_out = L.csv_dir(root, shard)
    for fname in L.CSV_FILENAMES:
        path = os.path.join(csv_out, fname)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            fail(f"CSV missing or empty: {path}")

    del graphs
    return {"experiments": n_exp, "feature_width": expect_width, "label_rows": len(labels)}


def simulate_shard(db, n_exp, params, progress=True):
    """Run `n_exp` experiments into `db`. No files are written."""
    return run_simulation_experiments(
        db, num_exp=n_exp, progress=progress,
        num_drones=params["drones"],
        num_sensors=params["sensors"],
        region_size=tuple(params["region"]),
        sensor_height=params["sensor_height"],
        target_height=params["target_height"],
        num_mountains=params["mountains"],
        num_steps=params["steps"],
        num_sectors=params["sectors"],
        min_sensor_distance=params["min_sensor_distance"],
        phi_sector_edges=params["elevation_sector_edges"],
        sync_error_ns=params["sync_error_ns"],
        toa_noise_ns=params["toa_noise_ns"],
    )


def export_shard(db, root, shard, params):
    """
    Write graph and csv output files of one shard from an already-simulated database.

    Both come from the same database, which is what guarantees the graph
    tensors and the CSVs describe the same experiments.
    """
    export_all_graphs_to_one_file(
        db, out_pt=L.graph_shard_path(root, shard),
        include_phi_sectors=params["elevation_sectors"],
        include_sector_angle_centers=params["include_azimuth_angle_centers"],
        include_phi_sector_angle_centers=params["include_elevation_angle_centers"],
        phi_sector_centers=params["elevation_sector_centers"],
    )
    export_drone_labels(db, csv_path=L.labels_path(root, shard))

    export_csv_tables(db, L.csv_dir(root, shard),
                      elevation_sector_edges=params["elevation_sector_edges"],
                      elevation_sector_centers=params["elevation_sector_centers"])


def generate_shard(root, shard, n_exp, params, scratch_dir, keep_db=False, progress=True):
    """Simulate, export and verify one shard. Returns its verification summary."""
    t0 = time.time()
    os.makedirs(L.graph_dir(root, shard), exist_ok=True)
    os.makedirs(L.csv_dir(root, shard), exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    db_path = os.path.join(scratch_dir, f"{shard}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    db = DatabaseManager(db_name=db_path)
    try:
        with muffled(not progress):
            simulate_shard(db, n_exp, params, progress=progress)
            export_shard(db, root, shard, params)
    finally:
        db.conn.close()

    summary = verify_shard(
        root, shard, n_exp,
        expect_width=layout(
            include_sector_angle_centers=params["include_azimuth_angle_centers"],
            include_phi_sectors=params["elevation_sectors"],
            include_phi_sector_angle_centers=params["include_elevation_angle_centers"])["width"],
    )
    summary["shard"] = shard
    summary["seconds"] = round(time.time() - t0, 1)

    # Only remove the scratch database after verification passes — a failed
    # shard keeps its database so it can be inspected instead of vanishing.
    if not keep_db:
        os.remove(db_path)
    return summary


def clear_dataset(root):
    """
    Remove a previous dataset's contents from `root` before regenerating.

    Only the four directories this package creates are removed, and only from a
    directory already confirmed to hold a manifest.json. Nothing else under
    `root` is touched, so pointing --out at a directory with other contents in
    it cannot destroy them.
    """
    for sub in (L.GRAPHS_SUBDIR, L.CSV_SUBDIR, L.REPORTS_SUBDIR, "_scratch"):
        path = os.path.join(root, sub)
        if os.path.isdir(path):
            shutil.rmtree(path)
    os.remove(L.manifest_path(root))


def generate(root, experiments, params, shard_size=DEFAULT_SHARD_SIZE, seed=None,
             keep_db=False, overwrite=False, progress=True):
    """
    Generate a complete dataset at `root`.

    Splits `experiments` into shards of at most `shard_size`; the final shard
    holds the remainder. Writes manifest.json last, so a directory with a
    manifest is a directory whose shards all verified.
    """
    existing = os.path.exists(L.manifest_path(root))
    if existing and not overwrite:
        raise FileExistsError(
            f"{root} already contains a dataset (manifest.json). "
            "Pass --overwrite to regenerate it, or choose another --out.")
    if existing and overwrite:
        clear_dataset(root)

    seed_everything(seed)

    sizes = [shard_size] * (experiments // shard_size)
    if experiments % shard_size:
        sizes.append(experiments % shard_size)

    scratch_dir = os.path.join(root, "_scratch")
    print(f"Generating {experiments:,} experiments in {len(sizes)} shard(s) -> {root}")

    summaries = []
    t0 = time.time()
    for i, size in enumerate(sizes):
        shard = L.shard_name(i)
        print(f"\n[{shard}] {size:,} experiments ({i + 1}/{len(sizes)})", flush=True)
        summaries.append(generate_shard(root, shard, size, params, scratch_dir,
                                        keep_db=keep_db, progress=progress))
        print(f"[{shard}] verified in {summaries[-1]['seconds']}s", flush=True)

    if not keep_db and os.path.isdir(scratch_dir) and not os.listdir(scratch_dir):
        os.rmdir(scratch_dir)

    manifest = {
        "format_version": 1,
        "generator": "uav_sim",
        "experiments": experiments,
        "shard_size": shard_size,
        "seed": seed,
        "parameters": params,
        "shards": summaries,
        "seconds": round(time.time() - t0, 1),
    }
    L.write_manifest(root, manifest)

    print(f"\nDone in {manifest['seconds']}s. {experiments:,} experiments, "
          f"{len(sizes)} shard(s), feature width {summaries[0]['feature_width']}.")
    print(f"Manifest: {L.manifest_path(root)}")
    return manifest
