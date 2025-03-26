import random
import math
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from calculations import calculate_signal_power
import DatabaseManager

matplotlib.use("TkAgg")

#TODO: check all functions for accuracy

def setup_environment(db, exp_id, num_sensors=5, region_size=(1000, 1000), test_type="circle", sensor_type="RF",
                      sensitivity=-90, sinr_threshold=10):
    db.set_experiment(exp_id, "2025-03-23", test_type, "radius:100")

    for i in range(1, num_sensors + 1):
        x = random.uniform(0, region_size[0])
        y = random.uniform(0, region_size[1])
        z = 0  # assume ground
        db.set_sensor(i, sensor_type, sensitivity, sinr_threshold)
        db.set_sensor_exp(exp_id, i, i, x, y, z)


def simulate_circle_trajectory(exp_id, drone_id, radius=3000, center=(500, 500), z=100, num_steps=60, start_time=None, speed=15):
    if start_time is None:
        start_time = datetime(2025, 3, 23, 12, 0, 0)
    time_interval = timedelta(seconds=1)
    trajectory = []

    for i in range(num_steps):
        angle = 2 * math.pi * i / num_steps
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        t = start_time + i * time_interval
        trajectory.append((drone_id, exp_id, t, speed, x, y, z, x, y, z))

    return trajectory

def simulate_random_trajectory(exp_id, drone_id, region=(5000, 5000), z=100, num_steps=60, start_time=None):
    from datetime import datetime, timedelta
    if start_time is None:
        start_time = datetime(2025, 3, 23, 12, 0, 0)

    time_interval = timedelta(seconds=1)
    # Start at a random position in the region
    x, y = random.uniform(0, region[0]), random.uniform(0, region[1])
    trajectory = []

    # Start with a random initial direction (angle in radians)
    current_angle = random.uniform(0, 2 * math.pi)
    # Maximum change in angle per time step (in radians)
    max_turn = math.pi / 8  # About 22.5 degrees

    for i in range(num_steps):
        # Use a random speed within the allowed range
        speed = random.uniform(5, 50)
        # Slightly change the current angle instead of choosing a completely new one
        delta_angle = random.uniform(-max_turn, max_turn)
        current_angle += delta_angle

        dx = speed * math.cos(current_angle)
        dy = speed * math.sin(current_angle)

        # Update position while keeping within region bounds
        x = max(0, min(region[0], x + dx))
        y = max(0, min(region[1], y + dy))
        t = start_time + i * time_interval

        trajectory.append((drone_id, exp_id, t, speed, x, y, z, x, y, z))

    return trajectory



# def simulate_random_trajectory(exp_id, drone_id, region=(5000, 5000), z=100, num_steps=60, start_time=None):
#     if start_time is None:
#         from datetime import datetime
#         start_time = datetime(2025, 3, 23, 12, 0, 0)
#
#     time_interval = timedelta(seconds=1)
#     x, y = random.uniform(0, region[0]), random.uniform(0, region[1])
#     trajectory = []
#
#     for i in range(num_steps):
#         speed = random.uniform(20, 50)
#         angle = random.uniform(0, 2 * math.pi)
#         dx = speed * math.cos(angle)
#         dy = speed * math.sin(angle)
#
#         x = max(0, min(region[0], x + dx))
#         y = max(0, min(region[1], y + dy))
#         t = start_time + i * time_interval
#
#         trajectory.append((drone_id, exp_id, t, speed, x, y, z, x, y, z))
#
#     return trajectory

def generate_mixed_trajectories(exp_id, num_drones=4, num_circular=2, num_steps=60):
    all_trajectories = {}
    start_time = datetime(2025, 3, 23, 12, 0, 0)

    for i in range(1, num_drones + 1):
        if i <= num_circular:
            traj = simulate_circle_trajectory(
                exp_id, i,
                radius=1000 + 50 * i,
                center=(2500, 2500),
                num_steps=num_steps,
                start_time=start_time,
                speed=random.randint(5, 50)
            )
        else:
            traj = simulate_random_trajectory(
                exp_id, i,
                region=(5000, 5000),
                num_steps=num_steps,
                start_time=start_time
            )
        all_trajectories[i] = traj

    return all_trajectories


def simulate_sensor_up_times(db, exp_id, sensor_ids, start, end):
    current_time = start
    while current_time <= end:
        for i, sensor_id in enumerate(sensor_ids):
            db.set_sensor_up_time(
                exp_id, sensor_id, sensor_up_id = 100 + sensor_id + int(current_time.timestamp()),
                start_time=current_time, end_time=current_time + timedelta(seconds=1),
                angle=45, sensor_noise=5, receiver_antenna_gain=0.0,
                received_thermal_noise_power=None, sinr=None, active=True
            )
        current_time += timedelta(seconds=1)


def insert_all_drones(db, exp_id, drone_trajectories: dict, freq_mhz=10):
    for drone_id, trajectory in drone_trajectories.items():
        # Add drone
        db.set_drone(
            drone_id=drone_id,
            drone_type="Quad",
            drone_transmitter_power=20,
            transmitter_antenna_gain=0.0,
            properties="Autonomous"
        ) #TODO: transmitter power, antenna gain determined how? random?

        # Link drone to experiment
        db.set_drone_exp(exp_id, drone_id, drone_exp_id=drone_id)

        # Set frequency usage (from first to last timestamp in trajectory)
        start_time = trajectory[0][2]
        end_time = trajectory[-1][2]
        db.set_drone_radio_freq(
            exp_id, drone_id, drone_exp_id=drone_id,
            radio_freq=freq_mhz, start_time=start_time, end_time=end_time
        )
        # Insert trajectory points
        for entry in trajectory:
            db.set_drone_exp_time(*entry)

def get_earliest_and_latest_times(drone_trajectories: dict):
    start = min(traj[0][2] for traj in drone_trajectories.values())
    end = max(traj[-1][2] for traj in drone_trajectories.values())
    return start, end


def run_detection_simulation(db, exp_id, drone_trajectories: dict):
    sensors = db.get_sensors_by_exp_id(exp_id)
    sensor_positions = {s[1]: (s[3], s[4], s[5]) for s in sensors}

    drones = {}
    for drone_entry in db.get_drones_by_exp_id(exp_id):
        drone_id = drone_entry[1]
        drone_info = db.get_drone_by_id(drone_id)
        drones[drone_id] = drone_info

    freqs = {(f[0], f[1]): f[3] for f in db.get_drone_radio_freq()}

    # Preprocess drone positions by time
    # for each timestamp, we have a list of the drones' positions
    drone_positions_by_time = {}
    for drone_id, trajectory in drone_trajectories.items():
        for entry in trajectory:
            _, _, timestamp, _, x, y, z, *_ = entry
            if timestamp not in drone_positions_by_time:
                drone_positions_by_time[timestamp] = []
            drone_positions_by_time[timestamp].append((drone_id, x, y, z))

    # Main simulation loop - loop according to time
    for timestamp in sorted(drone_positions_by_time.keys()):
        print(f"Time: {timestamp}")
        drone_positions = drone_positions_by_time[timestamp]
        sensor_up_times = db.get_active_sensor_up_times(exp_id, timestamp) # only entries for this timestamp

        for up in sensor_up_times:
            up_exp_id, sensor_id, sensor_up_id, start_str, end_str, _, _, rx_gain, *_ = up

            sx, sy, sz = sensor_positions[sensor_id]

            for drone_id, dx, dy, dz in drone_positions:
                drone = drones[drone_id]
                tx_power = drone[2]
                tx_gain = drone[3]
                freq_mhz = freqs[(exp_id, drone_id)]
                freq_hz = freq_mhz * 1e6

                print(f"  Drone {drone_id} position: ({dx:.1f}, {dy:.1f}, {dz:.1f})")

                distance = math.sqrt((dx - sx) ** 2 + (dy - sy) ** 2 + (dz - sz) ** 2)
                prx = calculate_signal_power(tx_power, distance, freq_hz, tx_gain, rx_gain)

                print(f"  Sensor {sensor_id} at ({sx:.1f}, {sy:.1f}, {sz:.1f}) → Prx: {prx:.2f} dBm")

                if prx > -100: # -100 < prx < -80
                    db.set_sensor_detection(sensor_up_id=sensor_up_id, drone_id=drone_id, received_power=prx,
                                            received_signal_bandwidth=5) #TODO: how to simulate received bw?
                    print("      → Detection occurred")

                else:
                    print("      → No detection")


def visualize_simulation(db, exp_id, drone_trajectories, region=(1000, 1000)):
    plt.ion()  # Enable interactive mode for PyCharm animations

    sensors = db.get_sensors_by_exp_id(exp_id)

    fig, ax = plt.subplots()
    ax.set_xlim(0, region[0])
    ax.set_ylim(0, region[1])
    ax.set_aspect('equal')

    # Assign unique colors to each sensor
    sensor_colors = plt.cm.get_cmap("tab10", len(sensors))

    # Plot sensors with unique colors
    sensor_dots = {}
    sensor_original_colors = {}  # Store the original colors
    for i, s in enumerate(sensors):
        _, sensor_id, _, sx, sy, _ = s
        color = sensor_colors(i)  # Get a unique color
        dot, = ax.plot(sx, sy, 'o', color=color, markersize=8, label=f"Sensor {sensor_id}")  # Default sensor marker
        sensor_dots[sensor_id] = dot
        sensor_original_colors[sensor_id] = color  # Store original color

    # Plot drone
    drone_fixed_colors = ['red', 'green']  # cycle through these for multiple drones
    drone_dots = {}
    drone_trails = {}  # recent trail (e.g., last 10 points)
    drone_full_trails = {}  # full past trajectory (dotted line)
    for i, (drone_id, traj) in enumerate(drone_trajectories.items()):
        color = drone_fixed_colors[i % len(drone_fixed_colors)]
        dot, = ax.plot([], [], 'o', color=color, markersize=8, label=f"Drone {drone_id}")
        trail, = ax.plot([], [], '-', color=color, alpha=0.4)  # recent trail
        # full_trail, = ax.plot([], [], 'k--', linewidth=1, label=f"Drone {drone_id} path")  # full dotted line
        drone_dots[drone_id] = dot
        drone_trails[drone_id] = trail
        # drone_full_trails[drone_id] = full_trail

    # Store persistent detection markers (triangles) and detection lines
    detection_markers = []  # List of all detection markers
    detection_lines = []  # List of detection lines

    def update(frame):
        all_artists = []

        for drone_id, traj in drone_trajectories.items():
            if frame >= len(traj):
                continue  # skip if frame exceeds drone's trajectory

            _, _, timestamp, _, x, y, z, *_ = traj[frame]
            drone_dots[drone_id].set_data([x], [y])

            trail_length = 10
            trail_x = [traj[max(0, frame - i)][4] for i in range(trail_length) if (frame - i) >= 0]
            trail_y = [traj[max(0, frame - i)][5] for i in range(trail_length) if (frame - i) >= 0]
            drone_trails[drone_id].set_data(trail_x, trail_y)

            all_artists.extend([drone_dots[drone_id], drone_trails[drone_id]])

        # Add sensor visualizations if needed
        all_artists.extend(sensor_dots.values())
        all_artists.extend(detection_markers)
        all_artists.extend(detection_lines)

        return all_artists

    # Choose frame count based on longest trajectory
    max_frames = max(len(traj) for traj in drone_trajectories.values())

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=300, blit=False)

    plt.legend()
    plt.title(f"Drone Flight Simulation - Exp {exp_id}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.show(block=True)  # Keep the animation running in PyCharm