import random
from datetime import datetime

from Simulator.calculations import calculate_distance
from Simulator.MountainSightCalculations import can_place_sensor_mountains
from Simulator.simulation_utils import FREQUENCY_BANDS_HZ

def setup_experiment(db, test_type="practice", region_size=(1000, 1000)):
    """
    Creates an experiment record and sets up the sensors associated with it.
    """
    parameters = (
        f"region_size: {region_size}")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_id = db.insert_experiment(date_str, test_type, parameters)
    return exp_id


def setup_topography(db, region_size, num_mountains=0):
    """
    Creates a topography record and, if num_mountains > 0, creates mountains for the topography.
    """

    center_x = region_size[0] / 2.0
    center_y = region_size[1] / 2.0
    topography_id = db.insert_topography(center_x, center_y)

    if num_mountains > 0:
        mountain_ids = _setup_mountains(db, num_mountains, region_size)
        db.cursor.executemany(
            "INSERT INTO topo_mountains (topography_id, mountain_id) VALUES (?, ?)",
            [(topography_id, mid) for mid in mountain_ids]
        )
    return topography_id


def setup_environment_record(db, disturbance, visibility):
    """
    Creates an environment record with the given disturbance and visibility.
    """

    env_id = db.insert_environment(disturbance, visibility)
    return env_id

def link_experiment_environment(db, exp_id, topography_id, env_id):
    """
    Links an experiment to a topography and environment by inserting a record into exp_env.
    """
    db.cursor.execute("INSERT OR REPLACE INTO exp_env (exp_id, topography_id, env_id) VALUES (?, ?, ?)",
                      (exp_id, topography_id, env_id))

def setup_sensors(db, exp_id, region_size, num_sensors=5, height=500, sensor_type="RF",
                  sensitivity_range=(-100, -80), sinr_threshold_range=(5, 15)):

    mountains = db.get_mountains_by_experiment(exp_id)
    placed_sensors = []
    sensors_data = []
    sector_counts = []

    for _ in range(num_sensors):
        # pick x,y,z until not inside any mountain:
        while True:
            x = random.uniform(0, region_size[0])
            y = random.uniform(0, region_size[1])
            z = random.uniform(0, height)

            if _can_place_sensor((x, y, z), mountains, placed_sensors, min_distance=1000):
                break

        sensitivity    = random.uniform(*sensitivity_range)
        sinr_threshold = random.uniform(*sinr_threshold_range)
        freq_min, freq_max = random.choice(FREQUENCY_BANDS_HZ)
        # sector_count = random.randint(1, 6)
        # sector_count = random.choice([1, 2, 3, 6])
        sector_count = 6
        sensor_gain = random.uniform(-5, 5)

        sensors_data.append((sensor_type, sensitivity, sinr_threshold, sensor_gain,
                                  freq_min, freq_max, sector_count))
        sector_counts.append(sector_count)
        placed_sensors.append((x, y, z))

    sensor_ids = []
    for data in sensors_data:
        db.cursor.execute(
            "INSERT INTO sensors (type, sensitivity, sinr_threshold, gain, freq_min, freq_max, sector_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            data
        )
        sensor_ids.append(db.cursor.lastrowid)


    sensor_exp_to_insert = [
        (exp_id, sid, x, y, z)
        for sid, (x, y, z) in zip(sensor_ids, placed_sensors)
    ]

    db.cursor.executemany(
        "INSERT INTO sensor_exp (exp_id, sensor_id, x, y, z) VALUES (?, ?, ?, ?, ?)",
        sensor_exp_to_insert
    )

    for sid, scount in zip(sensor_ids, sector_counts):
        _setup_sensor_gain_sectors(db, sid, exp_id, scount)


def _can_place_sensor(point, mountain_list, placed_sensors, min_distance=100):
    """
    Return True if `point` is not inside any mountain and not too close to other sensors.
    """
    # Check against mountains
    if not can_place_sensor_mountains(point, mountain_list):
        return False

    # Check against other sensors
    for (sx, sy, sz) in placed_sensors:
        dist = calculate_distance((point[0], point[1], point[2]), (sx, sy, sz))
        if dist < min_distance:
            return False

    return True

def _rects_overlap(r1, r2):
    # r = (xmin, ymin, xmax, ymax)
    return not (r1[2] <= r2[0] or r1[0] >= r2[2] or
                r1[3] <= r2[1] or r1[1] >= r2[3])

def _setup_mountains(db, num_mountains, region_size):
    """
    Generates `num_mountains` mountains with square bases.
      - (mx, my) randomly in [0,region_size]
      - side_length randomly in [50,1000]
      - height randomly in [50,500]

    Inserts into mountains(id, x, y, side_length, height)
    and returns the list of inserted mountain IDs.
    """
    mountains = []
    existing = []

    for _ in range(num_mountains):

        while True:
            mx = random.uniform(0, region_size[0])
            my = random.uniform(0, region_size[1])
            side_length = random.uniform(50, 1000)
            half = side_length / 2.0

            xmin, ymin = mx - half, my - half
            xmax, ymax = mx + half, my + half
            rect = (xmin, ymin, xmax, ymax)

            if not any(_rects_overlap(rect, other) for other in existing):
                existing.append(rect)
                break

        mheight = random.uniform(50, 500)
        mountains.append((mx, my, side_length, mheight))

    db.cursor.executemany(
        "INSERT INTO mountains (x, y, side_length, height) VALUES (?, ?, ?, ?)",
        mountains
    )

    start_id = db.cursor.lastrowid - len(mountains) + 1 #TODO: there is a bug here
    mountain_ids = list(range(start_id, start_id + len(mountains)))

    return mountain_ids

def _setup_sensor_gain_sectors(db, sensor_id, exp_id, sector_count):
    sectors = []
    for sector_idx in range(1, sector_count + 1):
        # carve up 360° into equal slices
        angle_start = 360 * (sector_idx - 1) / sector_count
        angle_end = 360 * sector_idx / sector_count
        sectors.append((sensor_id, exp_id, sector_idx, angle_start, angle_end))

    db.cursor.executemany(
        "INSERT INTO sensor_gain_sector (sensor_id, exp_id, sector_id, angle_start, angle_end) "
        "VALUES (?, ?, ?, ?, ?)",
        sectors
    )