import random
from datetime import timedelta

from Simulator.calculations import *
from Simulator.MountainSightCalculations import segment_intersects_mountain

FREQUENCY_BANDS_HZ = [
    (433e6, 435e6),    # Band 1
    # (902e6, 928e6),    # Band 2
    # (1710e6,1880e6),   # Band 3
    # (2400e6,2500e6),   # Band 4
    # (3655e6,3695e6),   # Band 5
    # (4819e6,5989e6),   # Band 6
    # (5160e6,5895e6),   # Band 7
]

def simulate_sensor_up_times(db, exp_id, start, end):
    sensors = db.get_sensors_by_exp_id(exp_id)
    sensor_ids = [s[1] for s in sensors]
    current_time = start
    while current_time <= end:
        for sensor_id in sensor_ids:
            db.set_sensor_up_time(
                exp_id, sensor_id,
                start_time=current_time, end_time=current_time + timedelta(seconds=1),
                sensor_noise=5, received_thermal_noise_power=None, sinr=None, active=True
            ) #TODO: what are values for sensor noise
        current_time += timedelta(seconds=1)


def create_drones(db, exp_id, num_drones,
                  power_range=(10, 30),
                  gain_range=(0.0, 5.0),
                  drone_types=("Quad", "Hex", "Octo"),
                  properties_list=("Autonomous", "Manually Controlled", "Hybrid")):

    drone_info = {}
    for drone_id in range(1, num_drones + 1):
        transmitter_power = random.uniform(*power_range)
        # transmitter_gain = random.uniform(*gain_range)
        transmitter_gain = 0.0
        drone_type = random.choice(drone_types)
        properties = random.choice(properties_list)

        fmin, fmax = random.choice(FREQUENCY_BANDS_HZ)
        bandwidth = fmax - fmin

        # Store the properties in our drone_info mapping.
        drone_info[drone_id] = {
            "transmitter_power": transmitter_power,
            "gain": transmitter_gain,
            "drone_type": drone_type,
            "properties": properties,
            "freq_range": (fmin, fmax),
            "bandwidth": bandwidth
        }

        # Insert the drone record into the database.
        db.set_drone(
            drone_id=drone_id,
            drone_type=drone_type,
            drone_transmitter_power=transmitter_power,
            transmitter_antenna_gain=transmitter_gain,
            properties=properties,
            freq_min=fmin,
            freq_max=fmax,
            bandwidth=bandwidth
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

def _drone_frequency_in_sensor_range(drone_frequency, sensor_range):
    fmin, fmax = sensor_range
    return (fmin <= drone_frequency <= fmax)


# Main simulation function.
def run_detection_simulation(db, exp_id, drone_trajectories: dict):
    sensor_positions = {s[1]: (s[3], s[4], s[5]) for s in db.get_sensors_by_exp_id(exp_id)}

    sensor_info = db.get_sensor_info_by_experiment(exp_id)
    # sensor_info rows: (id, type, sens, sinr_th, freq_min, freq_max, sector_count)

    patterns = db.load_sensor_patterns(exp_id) # sensor angle gain patterns
    sensor_sens = {sid: sens for sid, _, sens, _, _, _, _ in sensor_info}
    sensor_sinr_threshold = {sid: sinr_th for sid, _, _, sinr_th, _, _, _ in sensor_info}
    freq_ranges = {sid: (fmin, fmax)
                   for sid, _, _, _, fmin, fmax, _ in sensor_info}

    drones = _get_drone_info(db, exp_id)
    freqs = _get_drone_freqs(db, exp_id)
    mountain_list = _get_mountains(db, exp_id)
    drone_positions_by_time = _preprocess_drone_positions(drone_trajectories)

    # Main simulation loop: iterate in sorted order by timestamp.
    for timestamp in sorted(drone_positions_by_time.keys()):
        # print(f"Time: {timestamp}")
        drone_positions = drone_positions_by_time[timestamp]
        sensor_up_times = db.get_active_sensor_up_times(exp_id, timestamp)  # Only entries for this timestamp

        for up in sensor_up_times:
            # Unpack sensor up time entry.
            # up: (exp_id, sensor_id, sensor_up_id, start_time, end_time, ...)
            _, sensor_id, sensor_up_id, start_str, end_str, sensor_noise, thermal_power, sinr, active = up
            sx, sy, sz = sensor_positions[sensor_id]
            sensitivity = sensor_sens[sensor_id]
            sinr_threshold = sensor_sinr_threshold[sensor_id]
            rx_gain = 0
            signal_powers = []

            for drone_id, dx, dy, dz in drone_positions:
                drone = drones[drone_id]
                tx_power = drone[2]    # transmitter power
                tx_gain = drone[3]       # transmitter antenna gain
                bandwidth = drone[7]
                freq_hz = freqs[(exp_id, drone_id)]

                # Check drone frequency in sensor range
                in_range = _drone_frequency_in_sensor_range(freq_hz, freq_ranges[sensor_id])
                if not in_range:
                    # print(f"      → Drone {drone_id} freq {freq_hz/1e6:,.2f} MHz  outside sensor {sensor_id} range "
                    #       f"{fmin/1e6:,.2f}–{fmax/1e6:,.2f} MHz ")
                    break

                # Check mountain blockage.
                mountain_blocked = _line_of_sight_blocked(sx, sy, sz, dx, dy, dz, mountain_list)
                if mountain_blocked:
                    # print(f"      → Line of sight blocked by a mountain for sensor {sensor_id} and drone {drone_id}")
                    break

                # Calculate G_r according to sensor sector
                dif_x, dif_y = dx - sx, dy - sy
                theta = math.degrees(math.atan2(dif_y, dif_x)) % 360

                for start, end, gain_db in patterns[sensor_id]:
                    if start <= theta < end:
                        rx_gain = gain_db
                        break

                distance = calculate_distance((dx, dy, dz), (sx, sy, sz))
                prx = calculate_signal_power(tx_power, distance, freq_hz, tx_gain, rx_gain)
                signal_powers.append((drone_id, prx, bandwidth))

                # print(f"  Drone {drone_id} position: ({dx:.1f}, {dy:.1f}, {dz:.1f})")
                # print(f"  Sensor {sensor_id} at ({sx:.1f}, {sy:.1f}, {sz:.1f}) → Prx: {prx:.2f} dBm")
                # print("Distance: ", distance)

            if not signal_powers:
                break

            lead = max(signal_powers, key=lambda x: x[1])
            prx_lead = lead[1]
            bandwidth = lead[2]
            pthermal = calculate_thermal_noise(bandwidth)
            prx_others = [prx for drone_id, prx, _ in signal_powers if prx != prx_lead]
            sinr = calculate_sinr(prx_lead, pthermal, prx_others)
            active = is_sensor_active(sinr, sinr_threshold, prx_lead, sensitivity)

            if active:
                for drone_id, prx, bandwidth in signal_powers:
                    if prx > sensitivity:
                        db.set_sensor_detection(sensor_up_id=sensor_up_id, drone_id=drone_id,
                                        received_power=prx, received_signal_bandwidth=bandwidth)
                    #     print(f"      → Detection occurred: Sensor {sensor_id}, Drone {drone_id}")
                    # else:
                    #     print(f"      → No detection: Prx: {prx:.2f} dBm, Sensor Sensitivity: {sensitivity:.2f} dBm")

                db.update_sensor_metrics(sensor_up_id=sensor_up_id, thermal_noise=pthermal, sinr=sinr, active=active)
            # else:
            #     print(f"      → No detection: Sensor active: {active}")