# uav-sim

A simulator for **RF sensor networks localizing airborne targets**.

Place a network of ground sensors in a region, fly drones through it, and
record what each sensor hears: received power per antenna sector, per second,
subject to a full link budget, interference, sensitivity limits and terrain
occlusion. The result is exported in two forms from the same simulation —
temporal graphs for graph neural networks, and flat CSV tables for anything
else.

The simulator is deterministic given a seed, so a published result can be
regenerated exactly.

---

## Install

```bash
pip install -e .
```

Or, without installing the package:

```bash
pip install -r requirements.txt
python -m uav_sim --help
```

Requires Python 3.9+. The simulator is CPU-only — it never moves a tensor to a
device — so the CPU build of PyTorch is sufficient.

## Quickstart

```bash
# 100 experiments with the default configuration
uav-sim generate --experiments 100 --out data/clean --seed 0

# what did that produce?
uav-sim inspect data/clean
```

```python
import torch

graphs = torch.load("data/clean/graphs/shard_000/graphs.pt", weights_only=False)
snapshots = graphs[1]           # experiment 1, one Data object per second
print(snapshots[0].x.shape)     # [10 sensors, 34 features]
```

---

## The model

**Signal.** Each drone transmits at a power drawn from 10–30 dBm on a randomly
chosen band. Received power follows the Friis free-space equation,

```
Prx(dBm) = Ptx + Gt + Gr − 20·log₁₀(d) − 20·log₁₀(f) + 147
```

so power falls 6 dB per doubling of distance.

**Detection.** A sensor registers a drone only if *both* conditions hold: the
received power exceeds the sensor's sensitivity (drawn from −100 to −80 dBm),
and the SINR exceeds its threshold (5–15 dB). Each drone is judged on its own
SINR: thermal noise (−174 dBm/Hz + 10·log₁₀(BW) + noise figure) plus
interference from the other drones **in the same azimuth sector**. Each sector
antenna hears its own direction, so drones in other sectors do not interfere
(leakage between sectors is not modeled). Sensors are tuned to a frequency band
and hear nothing that transmits on a frequency outside of it.

**Direction.** Each sensor divides the horizon into equal **azimuth** sectors
and reports power per sector, so the data carries coarse bearing information
rather than range alone. Independently, it bins the **elevation** angle into
one below-horizon catch-all plus five bins over [0°, 90°). Which elevation
boundaries to use is a design question the `rebin` command exists to explore.

**Terrain.** Mountains are axis-aligned square prisms. A mountain between a
sensor and a drone removes that detection entirely — there is no partially
attenuated case — and a mountain between two sensors removes the graph edge
joining them. Sensors are never placed inside a mountain, and drones never fly
through one.

**Movement.** Drones fly random trajectories. Horizontal and vertical motion are
independent: each step may change the heading, the horizontal speed (5–50 m/s)
or the vertical speed (climb rate, ±10 m/s), each with a fixed probability and
by a bounded amount. A drone that reaches the ground climbs back up.

**Time of arrival.** Every detection also records when the signal arrived, for
use by an external TDOA solver. Sensors are modeled as GPS-disciplined, so clock 
skew is corrected away and what remains is a small per-sensor offset:

    toa = emission_offset + distance / c + clock_offset + noise

`emission_offset` is when within the 1-second step the drone transmitted
(uniform, shared by every sensor, unknown to them, and canceled by any TDOA
difference). `clock_offset` is drawn once per sensor from N(0, σ_sync²) and
`noise` once per detection from N(0, σ_n²), with σ_sync = 20 ns and σ_n = 10 ns
by default (`--sync-error-ns`, `--toa-noise-ns`). These draws use their own
random generator, so changing them leaves every other output unchanged.

---

## Output

A dataset is one directory:

```
data/clean/
├── manifest.json               every parameter used, plus per-shard counts
├── graphs/
│   ├── shard_000/
│   │   ├── graphs.pt           {exp_id: [Data, ...]}, one Data per second
│   │   └── labels.csv          ground truth per (experiment, timestep)
│   └── shard_001/ ...
└── csv/
    ├── shard_000/              eight flat tables
    └── shard_001/ ...
```

Both halves of a shard come from one simulation database, so they can never
disagree about what was simulated.

### Sharding

Experiments are generated in shards of `--shard-size` (default 5000). Each
shard is simulated into its own database, exported, verified, and then the
database is discarded. A 200,000-experiment run in one database is slow and
loses everything if it dies at 90%; the same run in 40 shards is resumable and
parallelizable. **`exp_id` restarts at 1 in every shard** — an experiment is
identified by `(shard, exp_id)`, never by `exp_id` alone.

### Graph tensors

Each `Data` object is one second of one experiment:

| field | meaning |
|---|---|
| `x` | `[N_sensors, F]` node features |
| `edge_index` | `[2, E]` sensor pairs, both directions; occluded pairs absent |
| `edge_attr` | `[E, 1]` distance between the pair, in meters |
| `timestamp` | the second this snapshot covers |

With the default options `F` is 34:

| columns | block |
|---|---|
| 0–6 | sensor type, sensitivity, SINR threshold, gain, freq min, freq max, sector count |
| 7–9 | sensor x, y, z (meters) |
| 10–15 | azimuth sector center angles (degrees) |
| 16–21 | **received power per azimuth sector (dBm)** |
| 22–27 | elevation sector center angles (degrees) |
| 28–33 | **received power per elevation sector (dBm)** |

> **A zero in either power block means "no detection", not "0 dBm".** Real
> received powers are around −75 dBm. Anything that perturbs these columns must
> leave the zeros exactly zero, or it invents detections the simulator never
> produced.

Each azimuth block is always 6 columns wide. A sensor with fewer sectors
(`--sectors 3`, or a randomized count) fills the first columns and leaves the
rest zero, so every experiment in a dataset has the same width. The width
changes only when a block is switched off (`--no-elevation-sectors`,
`--no-sector-angle-centers`, `--no-elevation-angle-centers`) — see
`schema.py`, whose `layout()` computes the blocks for any
configuration.

### Tables

| file | one row per |
|---|---|
| `labels.csv` | (experiment, timestep) — drone positions, transmit powers, heading / horizontal-speed / vertical-speed change flags |
| `sensor_information.csv` | sensor — sensitivity, SINR threshold, gain, band, sector count |
| `sensor_sectors.csv` | (sensor, azimuth sector) — angular extent |
| `sensor_distances.csv` | sensor pair — both positions and their separation |
| `sensor_detection_summary.csv` | (experiment, sensor, timestep) — power per azimuth sector and per elevation bin (summed as in the graphs), each detection's power and sectors, ground truth |
| `phi_sector_scheme.csv` | elevation bin — its boundaries and center |
| `drone_trajectories.csv` | (drone, timestep) — flight segment, horizontal and vertical speed, change flags |
| `drone_information.csv` | drone — transmit power, antenna gain, band, bandwidth |
| `toa_measurements.csv` | detection — sensor position, measured time of arrival, and its ground-truth parts (emission time, flight time, clock offset, drone position) |


## Commands

### `generate`

```bash
uav-sim generate --experiments 200000 --out data/train --seed 0
```

| flag | default |                                                                   |
|---|---|-------------------------------------------------------------------|
| `--experiments`, `-n` | — | total experiments to simulate                                     |
| `--out`, `-o` | — | output dataset directory                                          |
| `--sensors` | 10 | per experiment; `0` randomizes it (3–20)                          |
| `--drones` | 1 | per experiment; `0` randomizes it (1–8)                           |
| `--region` | 5000 5000 | region size (x,y) in meters                                       |
| `--steps` | 60 | timesteps per experiment, one second apart                        |
| `--mountains` | 0 | per experiment; `-1` randomizes it (1–5)                          |
| `--sensor-height` | 100 | **upper bound**; each sensor's height is drawn from [0, this]     |
| `--target-height` | 5000 | ceiling of the drone flight envelope                              |
| `--min-sensor-distance` | 1000 | minimum spacing between sensors                                   |
| `--sync-error-ns` | 20 | std of each sensor's residual clock offset (GPS-disciplined)      |
| `--toa-noise-ns` | 10 | std of the timing noise on each time of arrival                   |
| `--sectors` | 6 | azimuth sectors per sensor, 1–6; `0` randomizes from [1, 2, 3, 6] |
| `--elevation-scheme` | `A_uniform_angle` | see `uav-sim schemes`                                             |
| `--no-elevation-sectors` | | omit the elevation blocks entirely                                |
| `--no-sector-angle-centers` | | omit the azimuth center-angle features                            |
| `--no-elevation-angle-centers` | | omit the elevation center-angle features                          |
| `--shard-size` | 5000 | experiments per shard                                             |
| `--seed` | none | omit for a nondeterministic run                                   |
| `--keep-db` | | keep each shard's intermediate SQLite database                    |
| `--overwrite` | | regenerate over an existing dataset                               |

Every shard is verified after it is written — experiment count, the feature
width of every snapshot of every experiment, label completeness, no infinite edge attributes, no empty CSV. `manifest.json`
is written **last**, so a directory that has one is a directory whose shards
all passed.

> `--min-sensor-distance` is enforced by rejection sampling with no retry cap.
> Many sensors at a large spacing in a small region can stall: 20 sensors at
> 1000 m in a 5000×5000 region occupies ~63% of the area with exclusion disks,
> past the ~55% jamming limit for random hard-disk packing. Lower the spacing
> rather than raising the count against a fixed spacing.

### `rebin` — different elevation boundaries

```bash
uav-sim rebin --dataset data/clean --out data/clean_E --scheme E_fisher_optimal
```

Recomputes which elevation bin each detection falls in under a different set of
boundaries. The elevation angle is fully determined by the sensor and drone
positions, both already in the dataset, and the received power does not depend
on which bin it is filed under — so this is exact, not approximate. Every
scheme's dataset therefore describes the *same* physics, which makes comparing
schemes a controlled experiment rather than a comparison of independent random
draws. `uav-sim schemes` lists the five available. Requires a single-drone
dataset with a fixed `--sensors` count.

### `add-mountains` — terrain occlusion

```bash
uav-sim add-mountains --dataset data/clean --out data/mountains
```

Adds mountains to a mountain-free dataset without re-simulating. For a
single-drone experiment, occlusion enters the recorded data as a pure **mask**:
a blocked line of sight deletes a detection, and a blocked sensor pair deletes
an edge. Nothing is ever attenuated, moved, or added. The one conditioning this
introduces (only mountains that would not have deflected the drone are drawn)
is described in `variants/mountain_sampling.py`.

Mountains are sampled with an importance correction, because the naive approach
is measurably biased: in the true generative order mountains are placed first
and sensors are placed around them, so rejecting proposals that swallow a
sensor under-weights large mountains. Requires a single-drone dataset with a
fixed `--sensors` count.

### `add-noise` — receiver noise

```bash
uav-sim add-noise --dataset data/clean --out data/noise_90 --noise-floor-dbm -90
```

Adds receiver noise to the received power without re-simulating. One parameter,
a noise floor F in dBm, sets the noise power to 10^(F/10) mW. The noise is
added to the signal's voltage rather than to its dB value, so it barely moves
a strong reading and moves a weak one by several dB: at a -90 dBm floor, about
±0.15 dB on a -58 dBm reading and ±5 dB on a -90 dBm one.

Noise also changes which sensors detect the drone: every (timestep, sensor) is
re-decided with F as the noise term of the SINR, so readings can drop below
the threshold and readings that were below it can rise above it. Options:

| Option | Effect |
|---|---|
| `--gate off` | keep the source's detections exactly; only the powers get noisier |
| `--no-false-alarms` | remove detections the source did not have, so noise can only remove detections |
| `--seed` | the noise draw; every floor with the same seed uses the same draw |

Detections that survive keep their `toa_measurements.csv` row unchanged; a
false alarm gets a new row from the simulator's TOA model. Requires a
single-drone dataset with a fixed `--sensors` count and no mountains; to have
both, add noise first and then run `add-mountains` on the result.

### `inspect`

```bash
uav-sim inspect data/clean
```

Prints the manifest and flags shards that the manifest lists but the disk does
not have.

---

## Reproducibility

`--seed` seeds every source of randomness the simulator draws from. The same
seed and the same parameters produce byte-identical output, and the parameters
are recorded in `manifest.json` so a dataset always carries its own recipe.

The variant tools are deterministic too, keyed on the source dataset — any
shard can be regenerated in isolation and will match.

## Layout

```
uav_sim/            installed as the `uav_sim` package (see pyproject.toml)
├── pyproject.toml    packaging: name, dependencies, the `uav-sim` console script
├── requirements.txt  dependency floors, for use without installing the package
├── __init__.py       package docstring and __version__
├── __main__.py       lets `python -m uav_sim` run the CLI
├── cli.py            the command line
├── generate.py       parameters -> dataset on disk, with verification
├── layout.py         the on-disk shape of a dataset, in one place
├── errors.py         the package's exception types
├── schema.py         the node-feature column layout and label list length
├── core/             the simulator
│   ├── __init__.py
│   ├── physics.py        Friis, thermal noise, SINR, time of arrival
│   ├── mountains.py      mountain geometry and occlusion
│   ├── environment.py    region, terrain and sensor placement
│   ├── trajectories.py   drone flight paths
│   ├── detection.py      what each sensor hears each second
│   ├── experiment.py     one experiment, end to end
│   ├── database.py       the SQLite schema everything is recorded into
│   ├── graph_export.py   database -> graph tensors
│   └── csv_export.py     database -> flat tables
└── variants/         post-processing that derives a dataset from a dataset
    ├── __init__.py
    ├── elevation_schemes.py  the five elevation boundary schemes
    ├── rebin.py              re-sectorise elevation bins; calls the two below
    ├── rebin_csv.py              ...the flat-CSV half of that rebinning
    ├── rebin_graphs.py           ...the graph-tensor half of that rebinning
    ├── inject_mountains.py   add terrain occlusion
    ├── mountain_sampling.py  draw those mountains, and what they block
    └── inject_noise.py       add receiver noise
```

Variants read their source read-only and write a complete new dataset, so the
original stays re-derivable and the two can be compared experiment by
experiment. Writing anywhere inside the source tree is refused.

