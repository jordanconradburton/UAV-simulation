"""
variants/rebin.py

Re-sectorizes an existing dataset's elevation ("phi") bins under a different
boundary scheme, without re-simulating.

The elevation angle from a sensor to the drone is fully determined by two
positions: sensor xyz in the graph node features, drone xyz in the labels. 
So the bin index each detection should have had under a different set of 
boundaries is recomputable, and the received power is unaffected by which bin 
it is filed under. Rebinning moves the same power into a different column and 
rewrites the angle-center columns to match.

Requires a single-drone dataset (`--drones 1`) with a fixed `--sensors` count.
"""
import os

from .. import layout as L
from ..schema import layout as feature_layout
from . import elevation_schemes as ES
from . import rebin_csv, rebin_graphs


def rebin_dataset(dataset_root, out_root, scheme, shards=None, progress=True):
    """
    Rebin both output types of `dataset_root` into `out_root` under `scheme`.

    `dataset_root` is opened read-only throughout.
    """
    if scheme not in ES.SCHEME_BOUNDARIES:
        raise ValueError(f"unknown scheme {scheme!r}; "
                         f"choose from {sorted(ES.SCHEME_BOUNDARIES)}")
    if os.path.abspath(dataset_root) == os.path.abspath(out_root):
        raise ValueError("out must differ from dataset — rebinning never edits in place")

    manifest = L.read_manifest(dataset_root)
    params = manifest["parameters"]
    if not params.get("elevation_sectors"):
        raise ValueError("this dataset was generated without elevation sectors "
                         "(--no-elevation-sectors), so there is nothing to rebin")
    if params.get("sensors") is None:
        raise ValueError("rebinning requires a fixed --sensors count, not a randomized one")
    if params.get("drones") != 1:
        raise ValueError("rebinning requires a single-drone dataset (--drones 1): elevation "
                         "bins are recomputed from each detection's drone position")

    shards = shards or L.list_shards(dataset_root)
    if not shards:
        raise FileNotFoundError(f"no shards found in {dataset_root}")

    columns = feature_layout(
        include_sector_angle_centers=params["include_azimuth_angle_centers"],
        include_phi_sectors=params["elevation_sectors"],
        include_phi_sector_angle_centers=params["include_elevation_angle_centers"])

    edges, centers = ES.SCHEME_BOUNDARIES[scheme]
    print(f"Rebinning {len(shards)} shard(s) to scheme={scheme}")
    print(f"  edges   {edges}")
    print(f"  centers {centers}")

    for shard in shards:
        rebin_graphs.run(scheme,
                         src_dir=L.graph_dir(dataset_root, shard),
                         out_dir=L.graph_dir(out_root, shard),
                         columns=columns,
                         shard_glob=L.GRAPH_SHARD_FILENAME,
                         progress=progress)

    rebin_csv.run(scheme,
                  graph_src_dir=L.graphs_root(dataset_root),
                  csv_src_dir=L.csv_root(dataset_root),
                  out_dir=L.csv_root(out_root),
                  dir_glob="shard_*",
                  graph_filename=L.GRAPH_SHARD_FILENAME,
                  num_sensors=int(params["sensors"]),
                  progress=progress)

    out_manifest = dict(manifest)
    out_params = dict(params)
    out_params["elevation_scheme"] = scheme
    out_params["elevation_sector_edges"] = list(edges)
    out_params["elevation_sector_centers"] = list(centers)
    out_manifest["parameters"] = out_params
    out_manifest["derived_from"] = os.path.abspath(dataset_root)
    out_manifest["variant"] = f"rebin:{scheme}"
    L.write_manifest(out_root, out_manifest)

    print(f"\nDone. Rebinned dataset written to {out_root}")
    return out_manifest
