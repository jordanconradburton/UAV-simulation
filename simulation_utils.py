import random
import math
from datetime import datetime, timedelta

from calculations import calculate_signal_power
from MountainSightCalculations import segment_intersects_mountain

def simulate_sensor_up_times(db, exp_id, start, end):
    sensors = db.get_sensors_by_exp_id(exp_id)
    sensor_ids = [s[1] for s in sensors]
    current_time = start
    while current_time <= end:
        for sensor_id in sensor_ids:
            sensor_up_id = 100 * sensor_id + int(current_time.timestamp()) #TODO: maybe change this somehow? autoincrement in DB?
            db.set_sensor_up_time(
                exp_id, sensor_id, sensor_up_id,
                start_time=current_time, end_time=current_time + timedelta(seconds=1),
                angle=45, sensor_noise=5, receiver_antenna_gain=0.0,
                received_thermal_noise_power=None, sinr=None, active=True
            )
        current_time += timedelta(seconds=1)


def create_drones(db, exp_id, num_drones,
                  power_range=(10, 30),
                  gain_range=(0.0, 5.0),
                  drone_types=("Quad", "Hex", "Octo"),
                  properties_list=("Autonomous", "Manually Controlled", "Hybrid")):

    drone_info = {}
    for drone_id in range(1, num_drones + 1):
        transmitter_power = random.uniform(*power_range)
        transmitter_gain = random.uniform(*gain_range)
        drone_type = random.choice(drone_types)
        properties = random.choice(properties_list)

        # Store the properties in our drone_info mapping.
        drone_info[drone_id] = {
            "transmitter_power": transmitter_power,
            "gain": transmitter_gain,
            "drone_type": drone_type,
            "properties": properties
        }

        # Insert the drone record into the database.
        db.set_drone(
            drone_id=drone_id,
            drone_type=drone_type,
            drone_transmitter_power=transmitter_power,
            transmitter_antenna_gain=transmitter_gain,
            properties=properties
        )
        # Link the drone to the experiment.
        db.set_drone_exp(exp_id, drone_id, drone_exp_id=drone_id)
    db.conn.commit()
    return drone_info


# Get sensor positions as a dictionary: sensor_id -> (x, y, z)
def _get_sensor_positions(db, exp_id):
    sensors = db.get_sensors_by_exp_id(exp_id)
    return {s[1]: (s[3], s[4], s[5]) for s in sensors}

# Get drone information as a dictionary: drone_id -> drone_info
def _get_drone_info(db, exp_id):
    drones = {}
    for drone_entry in db.get_drones_by_exp_id(exp_id):
        drone_id = drone_entry[1]
        drones[drone_id] = db.get_drone_by_id(drone_id)
    return drones

# Get radio frequencies as a dictionary: (exp_id, drone_id) -> freq_mhz
def _get_drone_freqs(db, exp_id):
    return {(f[0], f[1]): f[3] for f in db.get_drone_radio_freq()}

# Retrieve mountains from DB.
def _get_mountains(db, exp_id):

    mountains = db.get_mountains_by_experiment(exp_id)
    mountain_list = []
    for m in mountains:
        # m is (id, x, y, side_length, height)
        mountain_list.append(m)
    return mountain_list

# Preprocess drone positions by timestamp.
# Returns a dictionary: timestamp -> list of (drone_id, x, y, z)
def _preprocess_drone_positions(drone_trajectories):
    drone_positions_by_time = {}
    for drone_id, trajectory in drone_trajectories.items():
        for entry in trajectory:
            _, _, _, timestamp, _, x, y, z, *_ = entry
            drone_positions_by_time.setdefault(timestamp, []).append((drone_id, x, y, z))
    return drone_positions_by_time

# Check if a line of sight is blocked by any mountain.
def _line_of_sight_blocked(sx, sy, sz, dx, dy, dz, mountain_list):
    """
    Returns True if the straight segment from (sx,sy,sz) to (dx,dy,dz)
    intersects any mountain in mountain_list.
    """
    a = (sx, sy, sz)
    b = (dx, dy, dz)

    for mount in mountain_list:
        if segment_intersects_mountain(a, b, mount):
            return True

    return False

# Main simulation function.
def run_detection_simulation(db, exp_id, drone_trajectories: dict):
    sensor_positions = _get_sensor_positions(db, exp_id)
    drones = _get_drone_info(db, exp_id)
    freqs = _get_drone_freqs(db, exp_id)
    mountain_list = _get_mountains(db, exp_id)
    drone_positions_by_time = _preprocess_drone_positions(drone_trajectories)

    # Main simulation loop: iterate in sorted order by timestamp.
    for timestamp in sorted(drone_positions_by_time.keys()):
        print(f"Time: {timestamp}")
        drone_positions = drone_positions_by_time[timestamp]
        sensor_up_times = db.get_active_sensor_up_times(exp_id, timestamp)  # Only entries for this timestamp

        for up in sensor_up_times:
            # Unpack sensor up time entry.
            # up: (exp_id, sensor_id, sensor_up_id, start_time, end_time, ..., rx_gain, ...)
            _, sensor_id, sensor_up_id, start_str, end_str, _, _, rx_gain, *_ = up
            sx, sy, sz = sensor_positions[sensor_id]

            for drone_id, dx, dy, dz in drone_positions:
                drone = drones[drone_id]
                tx_power = drone[2]    # transmitter power
                tx_gain = drone[3]       # transmitter antenna gain
                freq_hz = freqs[(exp_id, drone_id)]

                # Check mountain blockage.
                blocked = _line_of_sight_blocked(sx, sy, sz, dx, dy, dz, mountain_list)
                if blocked:
                    print(f"      → Line of sight blocked by a mountain for sensor {sensor_id} and drone {drone_id}")

                print(f"  Drone {drone_id} position: ({dx:.1f}, {dy:.1f}, {dz:.1f})")
                distance = math.sqrt((dx - sx) ** 2 + (dy - sy) ** 2 + (dz - sz) ** 2)
                prx = calculate_signal_power(tx_power, distance, freq_hz, tx_gain, rx_gain)
                print(f"  Sensor {sensor_id} at ({sx:.1f}, {sy:.1f}, {sz:.1f}) → Prx: {prx:.2f} dBm")
                print("Distance: ", distance)

                if not blocked and (-100 < prx < -80):
                    db.set_sensor_detection(sensor_up_id=sensor_up_id, drone_id=drone_id,
                                            received_power=prx, received_signal_bandwidth=5)
                    print("      → Detection occurred")
                else:
                    print("      → No detection")