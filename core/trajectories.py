import math
import random
from datetime import datetime, timedelta
from .mountains import point_clear_of_mountains, reject_if_mountain
from .physics import calculate_distance

MAX_VSPEED_MPS = 10.0     # climb/descent rate bound
VSPEED_STEP_MPS = 2.0     # largest change to the climb rate in one perturbation


def simulate_random_trajectory(db, exp_id, drone_id,
                               freq_range, region=(5000, 5000), height=500, num_steps=60, start_time=None,
                               change_heading_prob=0.1, change_horizontal_speed_prob=0.1,
                               change_vertical_speed_prob=0.1, sensor_safe_distance=5):
    """
    freq_range: the drone's (min, max) band in Hz, as chosen by create_drones.
    """

    mountains = db.get_mountains_by_experiment(exp_id)
    sensors  = db.get_sensors_by_exp_id(exp_id)
    sensor_positions = {s[2]:(s[3],s[4],s[5]) for s in sensors}

    if start_time is None:
        start_time = datetime.now()

    time_interval = timedelta(seconds=1)

    # Start at a random position inside the region that is not a mountain
    while True:
        x, y = random.uniform(0, region[0]), random.uniform(0, region[1])
        z = random.uniform(0, height)
        if point_clear_of_mountains((x, y, z), mountains):
            break
    initial_angle = random.uniform(0, 2 * math.pi)

    # The carrier frequency is fixed for the whole flight. It is still recorded
    # on every step (and as time intervals in drone_radio_freq) so frequency
    # changes during flight, e.g. hopping, can be added later without changing
    # the storage format. Detection assumes one frequency per flight and checks
    # it.
    carrier_freq_hz = random.uniform(*freq_range)
    speed = random.uniform(5, 50)
    vspeed = random.uniform(-MAX_VSPEED_MPS, MAX_VSPEED_MPS)  # independent of speed, see the motion model above

    trajectory = []
    current_angle = initial_angle
    max_turn = math.pi / 8  # About 22.5 degrees

    for i in range(num_steps):
        is_heading_change = False
        is_vertical_speed_change = False
        is_horizontal_speed_change = False

        current_speed = speed
        current_vspeed = vspeed

        if random.random() < change_horizontal_speed_prob:
            delta_v = random.uniform(-0.1 * speed, 0.1 * speed)
            current_speed = max(5, min(50, speed + delta_v))
            is_horizontal_speed_change = True

        if random.random() < change_heading_prob:
            delta = random.uniform(-max_turn, max_turn)
            current_angle += delta
            current_angle = (current_angle + math.pi) % (2 * math.pi) - math.pi
            is_heading_change = True

        if random.random() < change_vertical_speed_prob:
            dv = random.uniform(-VSPEED_STEP_MPS, VSPEED_STEP_MPS)
            current_vspeed = max(-MAX_VSPEED_MPS, min(MAX_VSPEED_MPS, vspeed + dv))
            is_vertical_speed_change = True

        dx = current_speed * math.cos(current_angle)
        dy = current_speed * math.sin(current_angle)
        dz = current_vspeed

        new_x = x + dx
        new_y = y + dy
        new_z = z + dz

        t = start_time + i * time_interval

        # Exit map boundaries → end this drone's trajectory
        if new_x < 0 or new_x > region[0] or new_y < 0 or new_y > region[1] or new_z > height:
            break

        # Hit the ground → reflect vertical speed upward so drone climbs
        if new_z < 0:
            new_z = 0
            current_vspeed = abs(current_vspeed)

        # 1) Mountain check
        if reject_if_mountain((x, y, z), (new_x, new_y, new_z), mountains):
            new_x, new_y, new_z = x, y, z  # hover
            current_speed = speed
            current_vspeed = vspeed
            is_heading_change = False
            is_vertical_speed_change = False
            is_horizontal_speed_change = False

        # 2) Sensor check
        elif _reject_if_too_close_to_sensors((new_x, new_y, new_z), sensor_positions, sensor_safe_distance):
            new_x, new_y, new_z = x, y, z
            current_speed = speed
            current_vspeed = vspeed
            is_heading_change = False
            is_vertical_speed_change = False
            is_horizontal_speed_change = False

        trajectory.append((drone_id, exp_id, t, current_speed, x, y, z, new_x, new_y, new_z,
                           is_heading_change, is_vertical_speed_change, is_horizontal_speed_change,
                           current_vspeed, carrier_freq_hz))

        x = new_x
        y = new_y
        z = new_z
        speed = current_speed
        vspeed = current_vspeed

    return trajectory


def generate_random_trajectories(db, exp_id, drone_ids, drone_freq_ranges, num_steps=60,
                                 region=(5000, 5000), height=1000):
    all_trajectories = {}
    start_time = datetime.now()

    for drone_id in drone_ids:
        fmin, fmax = drone_freq_ranges[drone_id]
        traj = simulate_random_trajectory(db,
            exp_id, drone_id,
            freq_range=(fmin, fmax),
            region=region,
            height=height,
            num_steps=num_steps,
            start_time=start_time,
        )
        all_trajectories[drone_id] = traj

    return all_trajectories


def extract_frequency_intervals(trajectory):
    """
    Split a trajectory into (frequency, start_time, end_time) runs of constant
    carrier frequency (the last field of each row).

    Today every drone keeps one frequency for its whole flight, so this always
    returns a single interval. It is written for the general case so that
    frequency changes during flight can be added later.
    """
    if not trajectory:
        return []

    intervals = []
    current_freq = trajectory[0][-1]
    interval_start = trajectory[0][2]

    for i in range(1, len(trajectory)):
        freq = trajectory[i][-1]
        timestamp = trajectory[i][2]

        if freq != current_freq:
            interval_end = trajectory[i - 1][2]
            intervals.append((current_freq, interval_start, interval_end))

            current_freq = freq
            interval_start = timestamp

    interval_end = trajectory[-1][2]
    intervals.append((current_freq, interval_start, interval_end))
    return intervals


def insert_drone_trajectories_and_frequencies(db, exp_id, drone_trajectories):
    """
    Store each drone's trajectory per step in drone_exp_time, and its carrier
    frequency as constant-frequency intervals in drone_radio_freq.

    entry[:14] is the row without its last field, the carrier frequency, which
    extract_frequency_intervals reads instead. The frequency must stay last.
    """
    exp_time_rows = []
    freq_rows = []

    for drone_id, trajectory in drone_trajectories.items():
        for entry in trajectory:
            exp_time_rows.append(entry[:14])

        freq_intervals = extract_frequency_intervals(trajectory)
        for freq, start_time, end_time in freq_intervals:
            freq_rows.append((exp_id, drone_id, drone_id, freq,
                                    start_time, end_time))

    db.cursor.executemany(
        "INSERT INTO drone_exp_time (drone_id, exp_id, time, horizontal_speed, start_x, start_y, start_z, "
        "end_x, end_y, end_z, is_heading_change, is_vertical_speed_change, is_horizontal_speed_change, "
        "vertical_speed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        exp_time_rows
    )

    db.cursor.executemany(
        "INSERT INTO drone_radio_freq (exp_id, drone_id, drone_exp_id, radio_freq, start_time, end_time) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        freq_rows
    )

def get_earliest_and_latest_times(drone_trajectories: dict):
    """
    (first, last) timestamp flown by any drone: the window the sensors are on.
    Drones that leave the region stop early, so the end is the last step of
    whichever drone stays longest.
    """
    start = min(traj[0][2] for traj in drone_trajectories.values())
    end = max(traj[-1][2] for traj in drone_trajectories.values())
    return start, end

def _reject_if_too_close_to_sensors(point, sensor_positions, min_dist):
    for sx, sy, sz in sensor_positions.values():
        if calculate_distance(point, (sx, sy, sz)) < min_dist:
            return True
    return False
