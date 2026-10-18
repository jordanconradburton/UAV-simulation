"""
uav_sim/cli.py

The one entry point: `uav-sim <command>`, or `python -m uav_sim <command>`.

    generate        simulate a new dataset
    add-mountains   derive a terrain-occluded variant of an existing dataset
    add-noise       derive a variant with receiver noise on the received power
    rebin           derive a variant with different elevation-sector boundaries
    schemes         list the available elevation-sector boundary schemes
    inspect         summarize a dataset on disk
"""
import argparse
import os
import sys

from . import layout as L
from .errors import InjectionError, PlacementError, VerificationError
from .core.physics import DEFAULT_SYNC_ERROR_NS, DEFAULT_TOA_NOISE_NS
from .schema import MAX_DRONES, N_SECTORS


def _add_generate_args(p):
    sim = p.add_argument_group("scenario")
    sim.add_argument("--experiments", "-n", type=int, required=True,
                     help="Total number of experiments to simulate.")
    sim.add_argument("--out", "-o", required=True, help="Output dataset directory.")
    sim.add_argument("--sensors", type=int, default=10,
                     help="Sensors per experiment. Pass 0 to randomize it per "
                          "experiment (3-20); the variant tools then cannot be used.")
    sim.add_argument("--drones", type=int, default=1,
                     help="Drones per experiment, 1-8 (default: 1). Pass 0 to randomize "
                          "it per experiment (1-8); the variant tools require 1.")
    sim.add_argument("--region", type=float, nargs=2, default=[5000.0, 5000.0],
                     metavar=("WIDTH", "DEPTH"),
                     help="Region size in meters (default: 5000 5000).")
    sim.add_argument("--steps", type=int, default=60,
                     help="Timesteps per experiment, one second apart (default: 60).")
    sim.add_argument("--mountains", type=int, default=0,
                     help="Mountains per experiment (default: 0). Pass -1 to "
                          "randomize it per experiment (1-5). To add mountains to "
                          "an existing clean dataset instead, use `add-mountains`.")
    sim.add_argument("--sensor-height", type=float, default=100.0,
                     help="Upper bound on sensor height, meters (default: 100). "
                          "Each sensor's height is drawn uniformly from [0, this].")
    sim.add_argument("--target-height", type=float, default=5000.0,
                     help="Ceiling of the drone flight envelope, meters (default: 5000).")
    sim.add_argument("--min-sensor-distance", type=float, default=1000.0,
                     help="Minimum spacing between sensors, meters (default: 1000). "
                          "Lower it if many sensors will not fit in a small region.")

    toa = p.add_argument_group("time of arrival")
    toa.add_argument("--sync-error-ns", type=float, default=DEFAULT_SYNC_ERROR_NS,
                     help="Standard deviation of each sensor's residual clock offset, "
                          "nanoseconds (default: %(default)g, GPS-disciplined clocks).")
    toa.add_argument("--toa-noise-ns", type=float, default=DEFAULT_TOA_NOISE_NS,
                     help="Standard deviation of the timing noise on each time-of-arrival "
                          "measurement, nanoseconds (default: %(default)g).")

    sec = p.add_argument_group("sectorisation")
    sec.add_argument("--sectors", type=int, default=6,
                     help="Azimuth sectors per sensor, 1-6 (default: 6). Pass 0 to "
                          "randomize per sensor from [1, 2, 3, 6]. Sensors with "
                          "fewer than 6 sectors are zero-padded to 6 feature columns.")
    sec.add_argument("--elevation-sectors", dest="elevation_sectors",
                     action="store_true", default=True,
                     help="Record elevation ('phi') sectors as well as azimuth "
                          "(default: on). There are always 6 elevation sectors per "
                          "sensor: one below-horizon catch-all plus five over [0, 90) "
                          "degrees; --elevation-scheme sets their boundaries, not "
                          "their number.")
    sec.add_argument("--no-elevation-sectors", dest="elevation_sectors",
                     action="store_false",
                     help="Record azimuth sectors only. The elevation power and "
                          "center-angle features are then both left out.")
    sec.add_argument("--elevation-scheme", default="A_uniform_angle",
                     help="Elevation bin boundaries (default: A_uniform_angle, "
                          "five equal 18-degree bins). See `uav-sim schemes`.")
    sec.add_argument("--sector-angle-centers", dest="include_azimuth_angle_centers",
                     action="store_true", default=True,
                     help="Include azimuth sector center angles as features (default: on).")
    sec.add_argument("--no-sector-angle-centers", dest="include_azimuth_angle_centers",
                     action="store_false")
    sec.add_argument("--no-elevation-angle-centers", dest="include_elevation_angle_centers",
                     action="store_false", default=True,
                     help="Omit the elevation sector center-angle features. They "
                          "are included by default whenever elevation sectors are "
                          "on; with --no-elevation-sectors they are never included.")

    run = p.add_argument_group("run")
    run.add_argument("--shard-size", type=int, default=5000,
                     help="Experiments per shard (default: 5000). Each shard is "
                          "simulated and verified independently.")
    run.add_argument("--seed", type=int, default=None,
                     help="Seed for reproducibility. Omit for a nondeterministic run.")
    run.add_argument("--keep-db", action="store_true",
                     help="Keep each shard's intermediate SQLite database.")
    run.add_argument("--overwrite", action="store_true",
                     help="Regenerate over an existing dataset directory.")
    run.add_argument("--quiet", action="store_true", help="Suppress progress bars.")


def _build_parameters(args):
    """Normalize CLI arguments into the manifest's `parameters` block."""
    from .variants import elevation_schemes as ES

    if args.elevation_scheme not in ES.SCHEME_BOUNDARIES:
        raise SystemExit(f"unknown --elevation-scheme {args.elevation_scheme!r}; "
                         f"run `uav-sim schemes` to list them")
    edges, centers = ES.SCHEME_BOUNDARIES[args.elevation_scheme]

    if args.sensors and args.sensors < 3:
        raise SystemExit("--sensors must be 0 (randomize) or at least 3")
    if not 0 <= args.drones <= MAX_DRONES:
        raise SystemExit(f"--drones must be 0 (randomize) or between 1 and {MAX_DRONES}")
    if not 0 <= args.sectors <= N_SECTORS:
        raise SystemExit(f"--sectors must be 0 (randomize) or between 1 and {N_SECTORS}")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.region[0] <= 0 or args.region[1] <= 0:
        raise SystemExit("--region width and depth must both be greater than 0")
    if args.target_height <= 0:
        raise SystemExit("--target-height must be greater than 0")
    if args.sensor_height < 0:
        raise SystemExit("--sensor-height must be at least 0")
    if args.min_sensor_distance < 0:
        raise SystemExit("--min-sensor-distance must be at least 0")
    if args.mountains < -1:
        raise SystemExit("--mountains must be -1 (randomize) or at least 0")
    if args.sync_error_ns < 0:
        raise SystemExit("--sync-error-ns must be at least 0")
    if args.toa_noise_ns < 0:
        raise SystemExit("--toa-noise-ns must be at least 0")
    if args.experiments < 1:
        raise SystemExit("--experiments must be at least 1")
    if args.shard_size < 1:
        raise SystemExit("--shard-size must be at least 1")

    return {
        "sensors": args.sensors or None,
        "drones": args.drones or None,
        "region": [float(args.region[0]), float(args.region[1])],
        "steps": args.steps,
        "mountains": None if args.mountains < 0 else args.mountains,
        "sensor_height": args.sensor_height,
        "target_height": args.target_height,
        "min_sensor_distance": args.min_sensor_distance,
        "sync_error_ns": args.sync_error_ns,
        "toa_noise_ns": args.toa_noise_ns,
        "sectors": args.sectors or None,
        "elevation_sectors": args.elevation_sectors,
        "elevation_scheme": args.elevation_scheme if args.elevation_sectors else None,
        "elevation_sector_edges": list(edges) if args.elevation_sectors else None,
        "elevation_sector_centers": list(centers) if args.elevation_sectors else None,
        "include_azimuth_angle_centers": args.include_azimuth_angle_centers,
        "include_elevation_angle_centers": args.include_elevation_angle_centers,
    }


def cmd_generate(args):
    from .generate import generate
    params = _build_parameters(args)
    generate(args.out, args.experiments, params,
             shard_size=args.shard_size, seed=args.seed, keep_db=args.keep_db,
             overwrite=args.overwrite, progress=not args.quiet)
    return 0


def cmd_add_mountains(args):
    from .variants import inject_mountains
    inject_mountains.run(args.dataset, args.out, shards=args.shards,
                         limit=args.limit_experiments,
                         require_blocking=args.require_blocking,
                         overwrite=args.overwrite, progress=not args.quiet)
    return 0


def cmd_add_noise(args):
    from .variants import inject_noise
    inject_noise.run(args.dataset, args.out, args.noise_floor_dbm, gate=args.gate,
                     false_alarms=args.false_alarms, seed=args.seed, shards=args.shards,
                     overwrite=args.overwrite, progress=not args.quiet)
    return 0


def cmd_rebin(args):
    from .variants.rebin import rebin_dataset
    rebin_dataset(args.dataset, args.out, args.scheme, shards=args.shards,
                  progress=not args.quiet)
    return 0


def cmd_schemes(args):
    from .variants import elevation_schemes as ES
    print("Elevation-sector boundary schemes\n")
    print("Each defines one below-horizon catch-all bin plus five bins over "
          "[0, 90) degrees.\n")
    for name, (edges, centers) in ES.SCHEME_BOUNDARIES.items():
        doc = {
            "A_uniform_angle": "equal 18-degree slices (default)",
            "B_uniform_sin": "equal steps in sin(elevation)",
            "C_uniform_tan": "equal altitude bands up to a cap",
            "D_quantile": "equal occupancy under the reference geometry",
            "E_fisher_optimal": "maximizes elevation Fisher information",
        }.get(name, "")
        print(f"  {name:<18} {doc}")
        print(f"  {'':<18} edges   {[round(e, 3) for e in edges]}")
        print(f"  {'':<18} centers {[round(c, 3) for c in centers]}\n")
    return 0


def cmd_inspect(args):
    manifest = L.read_manifest(args.dataset)
    shards = L.list_shards(args.dataset)
    print(f"{args.dataset}\n")
    if manifest.get("variant"):
        print(f"  variant       {manifest['variant']}")
        print(f"  derived from  {manifest.get('derived_from')}")
    print(f"  experiments   {manifest.get('experiments'):,}")
    print(f"  shards        {len(shards)} on disk, {len(manifest.get('shards', []))} in manifest")
    print(f"  seed          {manifest.get('seed')}")
    print("\n  parameters")
    for key, value in sorted(manifest.get("parameters", {}).items()):
        print(f"    {key:<24} {value}")
    widths = {s.get("feature_width") for s in manifest.get("shards", [])}
    if widths:
        print(f"\n  feature width {sorted(w for w in widths if w)}")
    missing = [s for s in manifest.get("shards", []) if s.get("shard") not in shards]
    if missing:
        print(f"\n  WARNING: {len(missing)} shard(s) in the manifest are not on disk: "
              f"{[s.get('shard') for s in missing]}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="uav-sim",
        description="Simulate RF sensor networks detecting airborne targets, "
                    "and export the result as graph tensors and flat CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", metavar="<command>")

    gen = sub.add_parser("generate", help="Simulate a new dataset.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_generate_args(gen)
    gen.set_defaults(func=cmd_generate)

    mnt = sub.add_parser("add-mountains",
                         help="Derive a terrain-occluded variant of a dataset.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mnt.add_argument("--dataset", required=True, help="Source dataset (read-only).")
    mnt.add_argument("--out", "-o", required=True, help="Output dataset directory.")
    mnt.add_argument("--shards", nargs="+", default=None,
                     help="Restrict to these shard names (default: all).")
    mnt.add_argument("--limit-experiments", dest="limit_experiments", type=int, default=None,
                     help="Process only the first N experiments per shard.")
    mnt.add_argument("--require-blocking", dest="require_blocking", action="store_true",
                     help="Re-draw until every experiment occludes at least one "
                          "detection. NOT distribution-faithful.")
    mnt.add_argument("--overwrite", action="store_true")
    mnt.add_argument("--quiet", action="store_true")
    mnt.set_defaults(func=cmd_add_mountains)

    noi = sub.add_parser("add-noise",
                         help="Derive a variant with receiver noise on the received power.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    noi.add_argument("--dataset", required=True, help="Source dataset (read-only).")
    noi.add_argument("--out", "-o", required=True, help="Output dataset directory.")
    noi.add_argument("--noise-floor-dbm", dest="noise_floor_dbm", type=float, required=True,
                     help="Receiver noise floor F in dBm; the noise power is 10^(F/10) mW.")
    noi.add_argument("--gate", choices=["noise_floor", "off"], default="noise_floor",
                     help="noise_floor: re-decide every detection with F as the SINR's "
                          "noise term. off: keep the source's detections exactly, so "
                          "readings get noisier but none are lost or added.")
    noi.add_argument("--false-alarms", dest="false_alarms", action="store_true", default=True,
                     help="Keep false alarms: detections the source did not have, "
                          "made by noise pushing a reading over the threshold.")
    noi.add_argument("--no-false-alarms", dest="false_alarms", action="store_false",
                     help="Remove false alarms, so noise can only weaken or remove "
                          "the source's detections.")
    noi.add_argument("--seed", type=int, default=0,
                     help="Seed for the noise draws. The same seed reproduces the output "
                          "exactly and gives every noise floor the same draws.")
    noi.add_argument("--shards", nargs="+", default=None,
                     help="Restrict to these shard names (default: all).")
    noi.add_argument("--overwrite", action="store_true")
    noi.add_argument("--quiet", action="store_true")
    noi.set_defaults(func=cmd_add_noise)

    reb = sub.add_parser("rebin",
                         help="Derive a variant with different elevation-sector boundaries.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    reb.add_argument("--dataset", required=True, help="Source dataset (read-only).")
    reb.add_argument("--out", "-o", required=True, help="Output dataset directory.")
    reb.add_argument("--scheme", required=True, help="See `uav-sim schemes`.")
    reb.add_argument("--shards", nargs="+", default=None,
                     help="Restrict to these shard names (default: all).")
    reb.add_argument("--quiet", action="store_true")
    reb.set_defaults(func=cmd_rebin)

    sch = sub.add_parser("schemes", help="List the elevation-sector boundary schemes.")
    sch.set_defaults(func=cmd_schemes)

    ins = sub.add_parser("inspect", help="Summarize a dataset on disk.")
    ins.add_argument("dataset")
    ins.set_defaults(func=cmd_inspect)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except (FileNotFoundError, FileExistsError, ValueError,
            PlacementError, VerificationError, InjectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:                                   
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":                                     
    sys.exit(main())
