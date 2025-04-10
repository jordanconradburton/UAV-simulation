import math
import random

def setup_experiment(db, exp_id, test_type="circle", region_size=(1000, 1000), height=500,
                     sensor_type="RF", sensitivity=-90, sinr_threshold=10, num_sensors=5):
    """
    Creates an experiment record and sets up the sensors associated with it.
    """
    parameters = (
        f"region_size: {region_size}, height: {height}, sensor_type: {sensor_type}, "
        f"sensitivity: {sensitivity}, sinr_threshold: {sinr_threshold}, num_sensors: {num_sensors}"
    )
    db.set_experiment(exp_id, "2025-03-23", test_type, parameters=parameters)
    _setup_sensors(db, exp_id, num_sensors, region_size, height, sensor_type, sensitivity, sinr_threshold)

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

def _setup_sensors(db, exp_id, num_sensors, region_size, height, sensor_type, sensitivity, sinr_threshold):

    for sensor_id in range(1, num_sensors + 1):
        x = random.uniform(0, region_size[0])
        y = random.uniform(0, region_size[1])
        z = random.uniform(0, height) # TODO: can z be negative? What are the limits?
        db.set_sensor(sensor_id, sensor_type, sensitivity, sinr_threshold)
        db.set_sensor_exp(exp_id, sensor_id, sensor_id, x, y, z)

def _setup_mountains(db, num_mountains, region_size):
    """
    Generates a specified number of mountains.
      - The x and y positions (mx, my) are chosen randomly within the region.
      - The mountain's height is randomly chosen between 50 and 500.
      - The base is modeled as a rectangle with independent random dimensions:
            width (x-dimension) between 1 and 50,
            length (y-dimension) between 1 and 50.
      - This rectangular area is converted into an equivalent circular base using:
            r = sqrt((width * length) / π)
    """
    mountain_ids = []
    for mountain_id in range(1, num_mountains + 1):
        mx = random.uniform(1, region_size[0])
        my = random.uniform(1, region_size[1])
        mheight = random.uniform(50, 500)

        width = random.uniform(1, 50)
        length = random.uniform(1, 50)

        area = width * length
        mradius = math.sqrt(area / math.pi)
        db.cursor.execute(
            "INSERT OR REPLACE INTO mountains (id, x, y, height, radius) VALUES (?, ?, ?, ?, ?)",
            (mountain_id, mx, my, mheight, mradius)
        )
        mountain_ids.append(mountain_id)

    return mountain_ids