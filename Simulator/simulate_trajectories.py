import random
from datetime import datetime, timedelta
from Simulator.MountainSightCalculations import segment_intersects_mountain, point_in_mountain
from Simulator.calculations import *


def simulate_random_trajectory(db, exp_id, drone_id,
                               region=(5000, 5000), height=500, num_steps=60, start_time=None,
                               freq_range=(900e6, 5e9), change_dir_prob=0.1, change_speed_prob=0.1,
                               change_pitch_prob=0.1, big_change_prob=0.1, sensor_safe_distance=5):

    mountains = db.get_mountains_by_experiment(exp_id)
    sensors  = db.get_sensors_by_exp_id(exp_id)
    sensor_positions = {s[2]:(s[3],s[4],s[5]) for s in sensors}

    if start_time is None:
        start_time = datetime.now()

    time_interval = timedelta(seconds=1)

    # Start at a random position in the region that is not a mountain
    while True:
        x, y = random.uniform(0, region[0]), random.uniform(0, region[1])
        z = random.uniform(0, height)
        if not any(
            point_in_mountain((x, y, z), (m_id, mx, my, side, h))
            for m_id, mx, my, side, h in mountains
        ):
            break

    current_freq = random.uniform(*freq_range)
    speed = random.uniform(5, 50)

    trajectory = []
    current_angle = random.uniform(0, 2 * math.pi)
    max_turn = math.pi / 8  # About 22.5 degrees

    # Vertical movement: start with a random pitch (vertical angle)
    current_pitch = random.uniform(-math.pi/16, math.pi/16)
    vertical_max_turn = math.pi / 16  # Maximum change in vertical angle per time step


    for i in range(num_steps):
        is_anomaly = False
        is_large_anomaly = False

        # if random.random() < change_freq_prob:
        #     current_freq = random.uniform(*freq_range)

        if random.random() < change_speed_prob:
            big_speed = random.random() < big_change_prob
            if big_speed:
                speed = random.uniform(5, 50)
                is_large_anomaly = True
            else:
                delta_v = random.uniform(-0.1 * speed, 0.1 * speed)
                speed = max(5, min(50, speed + delta_v))

            is_anomaly = True

        if random.random() < change_dir_prob:
            if random.random() < big_change_prob:
                current_angle = random.uniform(0, 2 * math.pi)
                is_large_anomaly = True
            else:
                delta = random.uniform(-max_turn, max_turn)
                current_angle += delta

            is_anomaly = True

        dx = speed * math.cos(current_angle)
        dy = speed * math.sin(current_angle)

        if random.random() < change_pitch_prob:
            if random.random() < big_change_prob:
                current_pitch = random.uniform(0, 2 * math.pi)
                is_large_anomaly = True
            else:
                dp = random.uniform(-vertical_max_turn, vertical_max_turn)
                current_pitch += dp

            is_anomaly = True

        dz = speed * math.sin(current_pitch)

        # Update position while keeping within region bounds
        new_x = max(0, min(region[0], x + dx))
        new_y = max(0, min(region[1], y + dy))
        new_z = max(0, min(height, z + dz))

        t = start_time + i * time_interval

        # 1) Mountain check
        if _reject_if_mountain((x, y, z), (new_x, new_y, new_z), mountains):
            new_x, new_y, new_z = x, y, z  # hover

        # 2) Sensor check
        elif _reject_if_too_close_to_sensors((new_x, new_y, new_z), sensor_positions, sensor_safe_distance):
            new_x, new_y, new_z = x, y, z

        trajectory.append((drone_id, exp_id, t, speed, x, y, z,
                           new_x, new_y, new_z, is_anomaly, is_large_anomaly, current_freq))

        x = new_x
        y = new_y
        z = new_z

    return trajectory


def generate_random_trajectories(db, exp_id, drone_ids, drone_freq_ranges, num_steps=60,
                                 region=(5000, 5000), height=1000):

    all_trajectories = {}
    start_time = datetime.now()

    for drone_id in drone_ids:
        fmin, fmax = drone_freq_ranges[drone_id]
        traj = simulate_random_trajectory(db,
            exp_id, drone_id,
            region=region,
            height=height,
            num_steps=num_steps,
            start_time=start_time,
            freq_range=(fmin, fmax)
        )
        all_trajectories[drone_id] = traj

    return all_trajectories


def extract_frequency_intervals(trajectory):
    if not trajectory:
        return []

    intervals = []
    # Get initial frequency and start time.
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

    exp_time_rows = []
    freq_rows = []

    for drone_id, trajectory in drone_trajectories.items():
        for entry in trajectory:
            exp_time_rows.append(entry[:12])

        freq_intervals = extract_frequency_intervals(trajectory)
        for freq, start_time, end_time in freq_intervals:
            freq_rows.append((exp_id, drone_id, drone_id, freq,
                                    start_time, end_time))

    db.cursor.executemany(
        "INSERT INTO drone_exp_time (drone_id, exp_id, time, speed, start_x, start_y, start_z, end_x, end_y, end_z, is_anomaly, is_large_anomaly) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        exp_time_rows
    )

    db.cursor.executemany(
        "INSERT INTO drone_radio_freq (exp_id, drone_id, drone_exp_id, radio_freq, start_time, end_time) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        freq_rows
    )

def get_earliest_and_latest_times(drone_trajectories: dict):
    start = min(traj[0][2] for traj in drone_trajectories.values())
    end = max(traj[-1][2] for traj in drone_trajectories.values())
    return start, end

def _reject_if_mountain(a, b, mountains):
    """
    Returns True if the segment a→b intersects any mountain in the list.
    """
    for m in mountains or []:
        if segment_intersects_mountain(a, b, m):
            return True
    return False

def _reject_if_too_close_to_sensors(point, sensor_positions, min_dist):
    """
    Returns True if `point` is within min_dist of ANY sensor in sensor_positions.
    """
    for sx, sy, sz in sensor_positions.values():
        if calculate_distance(point, (sx, sy, sz)) < min_dist:
            return True
    return False