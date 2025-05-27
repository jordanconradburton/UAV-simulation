import random
import math
from datetime import datetime, timedelta
from MountainSightCalculations import segment_intersects_mountain
from calculations import *

def simulate_random_trajectory(db, exp_id, drone_id, region=(5000, 5000), height=500, num_steps=60, start_time=None,
                               freq_range=(900e6, 5e9), change_prob=0.1, sensor_safe_distance=5):

    mountains = db.get_mountains_by_experiment(exp_id)
    sensors  = db.get_sensors_by_exp_id(exp_id)
    sensor_positions = {s[1]:(s[3],s[4],s[5]) for s in sensors}

    if start_time is None:
        start_time = datetime(2025, 3, 23, 12, 0, 0)

    time_interval = timedelta(seconds=1)
    # Start at a random position in the region
    x, y = random.uniform(0, region[0]), random.uniform(0, region[1])
    z = random.uniform(0, height)
    current_freq = random.uniform(*freq_range)

    trajectory = []
    # Start with a random initial direction (angle in radians)
    current_angle = random.uniform(0, 2 * math.pi)
    # Maximum change in angle per time step (in radians)
    max_turn = math.pi / 8  # About 22.5 degrees

    # Vertical movement: start with a random pitch (vertical angle)
    current_pitch = random.uniform(-math.pi/16, math.pi/16)
    vertical_max_turn = math.pi / 16  # Maximum change in vertical angle per time step

    for i in range(num_steps):
        if random.random() < change_prob:
            current_freq = random.uniform(*freq_range)

        # Use a random speed within the allowed range
        speed = random.uniform(5, 50)
        # Slightly change the current angle instead of choosing a completely new one
        delta_angle = random.uniform(-max_turn, max_turn)
        current_angle += delta_angle
        dx = speed * math.cos(current_angle)
        dy = speed * math.sin(current_angle)

        # Update vertical (altitude) direction with a small random pitch change
        delta_pitch = random.uniform(-vertical_max_turn, vertical_max_turn)
        current_pitch += delta_pitch
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

        trajectory.append((100 * drone_id * exp_id + i, drone_id, exp_id, t, speed, x, y, z,
                           new_x, new_y, new_z, current_freq))

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
    current_freq = trajectory[0][11]
    interval_start = trajectory[0][3]

    for i in range(1, len(trajectory)):
        freq = trajectory[i][11]
        timestamp = trajectory[i][3]

        if freq != current_freq:
            interval_end = trajectory[i - 1][3]
            intervals.append((current_freq, interval_start, interval_end))

            current_freq = freq
            interval_start = timestamp

    interval_end = trajectory[-1][3]
    intervals.append((current_freq, interval_start, interval_end))
    return intervals


def insert_drone_trajectories_and_frequencies(db, exp_id, drone_trajectories):

    for drone_id, trajectory in drone_trajectories.items():
        for entry in trajectory:
            db.set_drone_exp_time(*entry[:11])

        freq_intervals = extract_frequency_intervals(trajectory)
        for interval in freq_intervals:
            freq, start_time, end_time = interval
            db.set_drone_radio_freq(exp_id, drone_id, drone_exp_id=drone_id, radio_freq=freq,
                                    start_time=start_time, end_time=end_time)
    db.conn.commit()

def get_earliest_and_latest_times(drone_trajectories: dict):
    start = min(traj[0][3] for traj in drone_trajectories.values())
    end = max(traj[-1][3] for traj in drone_trajectories.values())
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