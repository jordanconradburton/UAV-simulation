import math
import random
from MountainSightCalculations import can_place_sensor

def setup_experiment(db, exp_id, test_type="circle", region_size=(1000, 1000), height=500):
    """
    Creates an experiment record and sets up the sensors associated with it.
    """
    parameters = (
        f"region_size: {region_size}, height: {height}")
    db.set_experiment(exp_id, "2025-03-23", test_type, parameters=parameters)
    db.conn.commit()


def setup_topography(db, topography_id, region_size, num_mountains=0):
    """
    Creates a topography record and, if num_mountains > 0, creates mountains for the topography.
    """

    center_x = region_size[0] / 2.0
    center_y = region_size[1] / 2.0
    db.cursor.execute("INSERT OR REPLACE INTO topography (id, x, y) VALUES (?, ?, ?)",
                      (topography_id, center_x, center_y))

    if num_mountains > 0:
        mountain_ids = _setup_mountains(db, num_mountains, region_size)
        for mountain_id in mountain_ids:
            db.cursor.execute("INSERT OR REPLACE INTO topo_mountains (topography_id, mountain_id) VALUES (?, ?)",
                              (topography_id, mountain_id))
    db.conn.commit()


def setup_environment_record(db, env_id, disturbance, visibility):
    """
    Creates an environment record with the given disturbance and visibility.
    """

    db.cursor.execute("INSERT OR REPLACE INTO environment (id, disturbance, visibility) VALUES (?, ?, ?)",
                      (env_id, disturbance, visibility))
    db.conn.commit()

def link_experiment_environment(db, exp_id, topography_id, env_id):
    """
    Links an experiment to a topography and environment by inserting a record into exp_env.
    """
    db.cursor.execute("INSERT OR REPLACE INTO exp_env (exp_id, topography_id, env_id) VALUES (?, ?, ?)",
                      (exp_id, topography_id, env_id))
    db.conn.commit()

def setup_sensors(db, exp_id, region_size, num_sensors=5, height=500, sensor_type = "RF",
                  sensitivity_range=(-100, -80), sinr_threshold_range=(5, 15)):

    mountains = db.get_mountains_by_experiment(exp_id)
    for sensor_id in range(1, num_sensors+1):
        # pick x,y,z until not inside any mountain:
        while True:
            x = random.uniform(0, region_size[0])
            y = random.uniform(0, region_size[1])
            z = random.uniform(0, height)

            if can_place_sensor((x, y, z), mountains):
                break

        sensitivity    = random.uniform(*sensitivity_range)
        sinr_threshold = random.uniform(*sinr_threshold_range)

        db.set_sensor(sensor_id, sensor_type, sensitivity, sinr_threshold)
        db.set_sensor_exp(exp_id, sensor_id, sensor_id, x, y, z)
    db.conn.commit()

def _setup_mountains(db, num_mountains, region_size):
    """
    Generates `num_mountains` mountains with square bases.
      - (mx, my) randomly in [0,region_size]
      - side_length randomly in [1,50]
      - height randomly in [50,500]

    Inserts into mountains(id, x, y, side_length, height)
    and returns the list of inserted mountain IDs.
    """
    mountain_ids = []
    for mountain_id in range(1, num_mountains + 1):
        mx = random.uniform(0, region_size[0])
        my = random.uniform(0, region_size[1])
        side_length = random.uniform(50, 1000)
        mheight = random.uniform(50, 500)

        db.cursor.execute(
            "INSERT OR REPLACE INTO mountains (id, x, y, side_length, height) VALUES (?, ?, ?, ?, ?)",
            (mountain_id, mx, my, side_length, mheight)
        )
        mountain_ids.append(mountain_id)

    db.conn.commit()
    return mountain_ids