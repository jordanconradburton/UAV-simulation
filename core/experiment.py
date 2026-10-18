"""
core/experiment.py

One experiment, end to end: place the terrain and the sensors, fly the drones,
record what each sensor heard.

An "experiment" is a single scenario — one random sensor layout, one random
terrain, one set of drone trajectories — observed over a sequence of one-second
timesteps. A dataset is many independent experiments.

Everything an experiment produces lands in the SQLite database passed in. The
exporters in `graph_export` and `csv_export` then read that database.
"""
import random

from tqdm import tqdm

from .detection import (create_drones, run_detection_simulation,
                        simulate_sensor_up_times)
from .csv_export import build_sensor_distances_table
from .physics import DEFAULT_SYNC_ERROR_NS, DEFAULT_TOA_NOISE_NS
from ..schema import MAX_DRONES
from .environment import (link_experiment_environment, setup_environment_record,
                          setup_experiment, setup_sensors, setup_topography)
from .trajectories import (generate_random_trajectories,
                           get_earliest_and_latest_times,
                           insert_drone_trajectories_and_frequencies)

DEFAULT_NUM_STEPS = 60
DEFAULT_SENSOR_HEIGHT = 100.0            # UPPER BOUND on sensor height; each
                                         # sensor's z is drawn from [0, this]
DEFAULT_TARGET_HEIGHT = 5000.0           # ceiling of the drone flight envelope
DEFAULT_MIN_SENSOR_DISTANCE = 1000.0     # meters between any two sensors


def run_simulation(
    db,
    num_drones=None,
    num_sensors=None,
    region_size=None,
    sensor_height=None,
    target_height=None,
    num_mountains=None,
    num_steps=DEFAULT_NUM_STEPS,
    num_sectors=None,
    min_sensor_distance=DEFAULT_MIN_SENSOR_DISTANCE,
    phi_sector_edges=None,
    sync_error_ns=DEFAULT_SYNC_ERROR_NS,
    toa_noise_ns=DEFAULT_TOA_NOISE_NS,
):
    """
    Run one experiment and record it. Returns True if it was recorded.

    Any argument left as None is randomized per experiment.

    num_steps:          timesteps to simulate, one second apart. A drone that
                        leaves the region stops early; the sensors stop once
                        the last drone has left.
    num_sectors:        equal azimuth sectors per sensor. None picks randomly
                        from [1, 2, 3, 6].
    min_sensor_distance: minimum spacing between sensors. Lower it for sensor
                        counts that cannot be placed at this spacing in the
                        given region.
    phi_sector_edges:   elevation-bin boundaries. None uses the uniform
                        18-degree scheme; pass a scheme's edge list from
                        `uav_sim.variants.elevation_schemes` to generate under
                        an alternative sectorization.
    sync_error_ns:      std of each sensor's residual clock offset (GPS-
                        disciplined clocks), nanoseconds.
    toa_noise_ns:       std of the timing noise on each time-of-arrival
                        measurement, nanoseconds.

    Returns False and rolls the experiment back if any drone produced no
    trajectory.
    """
    if num_sensors is None:
        num_sensors = random.randint(3, 20)
    if num_drones is None:
        num_drones = random.randint(1, MAX_DRONES)
    if region_size is None:
        region_size = (random.randint(100, 5000), random.randint(100, 5000))
    if sensor_height is None:
        sensor_height = DEFAULT_SENSOR_HEIGHT
    if num_mountains is None:
        num_mountains = random.randint(1, 5)
    if target_height is None:
        target_height = DEFAULT_TARGET_HEIGHT

    db.cursor.execute("SAVEPOINT run_simulation")

    exp_id = setup_experiment(db, region_size=region_size)
    topo_id = setup_topography(db, region_size=region_size, num_mountains=num_mountains)
    env_id = setup_environment_record(db, disturbance=random.uniform(0, 1),
                                      visibility=random.uniform(10, 100))
    link_experiment_environment(db, exp_id, topo_id, env_id)
    setup_sensors(db, exp_id, num_sensors=num_sensors, region_size=region_size,
                  height=sensor_height, num_sectors=num_sectors,
                  min_sensor_distance=min_sensor_distance,
                  sync_error_s=sync_error_ns * 1e-9)

    drone_info = create_drones(db, exp_id, num_drones)
    drone_ids = list(drone_info.keys())
    drone_freq_ranges = {did: info['freq_range'] for did, info in drone_info.items()}

    trajectories = generate_random_trajectories(
        db, exp_id, drone_ids,
        num_steps=num_steps,
        drone_freq_ranges=drone_freq_ranges,
        height=target_height,
        region=region_size,
    )

    if not all(trajectories.values()):
        db.cursor.execute("ROLLBACK TO SAVEPOINT run_simulation")
        db.cursor.execute("RELEASE SAVEPOINT run_simulation")
        return False

    db.cursor.execute("RELEASE SAVEPOINT run_simulation")

    insert_drone_trajectories_and_frequencies(db, exp_id, trajectories)

    start, end = get_earliest_and_latest_times(trajectories)
    simulate_sensor_up_times(db, exp_id, start=start, end=end)
    run_detection_simulation(db, exp_id, trajectories, phi_sector_edges=phi_sector_edges,
                             toa_noise_s=toa_noise_ns * 1e-9)

    build_sensor_distances_table(db, exp_id)
    return True


def run_simulation_experiments(db, num_exp, progress=True, commit_every=200, **kwargs):
    """
    Run experiments until `num_exp` have been recorded.
    """
    bar = tqdm(total=num_exp) if progress else None
    recorded = 0
    try:
        while recorded < num_exp:
            if run_simulation(db, **kwargs):
                recorded += 1
                if bar is not None:
                    bar.update(1)
                if recorded % commit_every == 0:
                    db.conn.commit()
    finally:
        if bar is not None:
            bar.close()
    db.conn.commit()
    return recorded
