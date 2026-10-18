import math
import random
from datetime import timedelta, datetime
from collections import defaultdict

from .physics import (C_LIGHT, TOA_RNG, calculate_distance, calculate_signal_power,
                      calculate_sinr, calculate_thermal_noise,
                      calculate_toa, is_detected)
from .mountains import segment_intersects_mountain


PHI_SECTOR_EDGES_DEG = [0.0, 18.0, 36.0, 54.0, 72.0, 90.0]

FREQUENCY_BANDS_HZ = [
    (433e6, 435e6),    # Band 1
    # (902e6, 928e6),    # Band 2
    # (1710e6,1880e6),   # Band 3
    # (2400e6,2500e6),   # Band 4
    # (3655e6,3695e6),   # Band 5
    # (4819e6,5989e6),   # Band 6
    # (5160e6,5895e6),   # Band 7
]


def _phi_sector_index(dz, horizontal_dist, edges=None):
    """
    Returns the 1-based phi (elevation-angle) sector index for a drone seen
    from a sensor, given dz = drone_z - sensor_z and the horizontal distance
    between them.
    """
    edges = edges if edges is not None else PHI_SECTOR_EDGES_DEG
    phi_deg = math.degrees(math.atan2(dz, max(horizontal_dist, 1e-6)))
    if phi_deg < 0.0:
        return 1
    for i in range(len(edges) - 1):
        if edges[i] <= phi_deg < edges[i + 1]:
            return i + 2
    return 6  # phi_deg == 90.0 exactly (boundary) -> last bin

def simulate_sensor_up_times(db, exp_id, start, end):
    sensors = db.get_sensors_by_exp_id(exp_id)
    sensor_ids = [s[2] for s in sensors]
    data = []
    current_time = start

    while current_time <= end:
        next_time = current_time + timedelta(seconds=1)
        for sensor_id in sensor_ids:
            data.append((exp_id, sensor_id, current_time, next_time, 5, None, None, False))
        current_time = next_time

    db.cursor.executemany(
        "INSERT INTO sensor_up_time (exp_id, sensor_id, start_time, end_time, sensor_noise, "
        "received_thermal_noise_power, sinr, active) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        data
    )


def create_drones(db, exp_id, num_drones,
                  power_range=(10, 30),
                  gain_range=(0.0, 5.0),
                  drone_types=("Quad", "Hex", "Octo"),
                  properties_list=("Autonomous", "Manually Controlled", "Hybrid")):

    drone_info = {}
    drone_rows = []
    drone_exp_rows = []

    for i in range(num_drones):
        transmitter_power = random.uniform(*power_range)
        # transmitter_gain = random.uniform(*gain_range)
        transmitter_gain = 0.0
        drone_type = random.choice(drone_types)
        properties = random.choice(properties_list)

        fmin, fmax = random.choice(FREQUENCY_BANDS_HZ)
        bandwidth = fmax - fmin

        drone_rows.append((drone_type, transmitter_power, transmitter_gain, properties,
                           fmin, fmax, bandwidth))

    drone_ids = []
    for data in drone_rows:
        db.cursor.execute(
            "INSERT INTO drones (drone_type, drone_transmitter_power, transmitter_antenna_gain, properties, freq_min, freq_max, bandwidth) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            data
        )
        drone_ids.append(db.cursor.lastrowid)

    for did, (drone_type, tx_power, tx_gain, properties, fmin, fmax, bw) in zip(drone_ids, drone_rows):
        drone_info[did] = {
            "transmitter_power": tx_power,
            "gain": tx_gain,
            "drone_type": drone_type,
            "properties": properties,
            "freq_range": (fmin, fmax),
            "bandwidth": bw
        }

    # link drones to experiment
    drone_exp_rows = [(exp_id, did, did) for did in drone_ids]
    db.cursor.executemany(
        "INSERT INTO drone_exp (exp_id, drone_id, drone_exp_id) VALUES (?, ?, ?)",
        drone_exp_rows
    )

    return drone_info


# Get sensor positions as a dictionary: sensor_id -> (x, y, z)
def _get_sensor_positions(db, exp_id):
    sensors = db.get_sensors_by_exp_id(exp_id)
    return {s[2]: (s[3], s[4], s[5]) for s in sensors}

# Get drone information as a dictionary: drone_id -> drone_info
def _get_drone_info(db, exp_id):
    drones = {}
    for drone_entry in db.get_drones_by_exp_id(exp_id):
        drone_id = drone_entry[1]
        drones[drone_id] = db.get_drone_by_id(drone_id)
    return drones

def _get_drone_freqs(db, exp_id):
    """
    Carrier frequency of each drone in the experiment, in Hz:
    {(exp_id, drone_id): freq}.

    Detection uses one frequency per drone for the whole flight. drone_radio_freq
    can hold several time intervals per drone, kept for future in-flight
    frequency changes.
    """
    freqs = {}
    for exp, drone_id, _, freq, _, _ in db.get_drone_radio_freq(exp_id):
        if (exp, drone_id) in freqs:
            raise NotImplementedError(
                f"drone {drone_id} in experiment {exp} changes frequency during "
                f"flight; detection supports one frequency per drone")
        freqs[(exp, drone_id)] = freq
    return freqs

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
            _, _, timestamp, _, x, y, z, *_ = entry
            drone_positions_by_time.setdefault(timestamp, []).append((drone_id, x, y, z))
    return drone_positions_by_time

def _preprocess_sensor_up_times(db, exp_id, timestamps):
    all_up_times = db.get_sensor_up_time_by_experiment(exp_id)
    grouped = defaultdict(list)

    for ts in timestamps:
        for row in all_up_times:
            start_time = datetime.fromisoformat(row[3])
            end_time = datetime.fromisoformat(row[4])

            if start_time <= ts < end_time:
                grouped[ts].append(row)

    return grouped

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


def detect_drones(signals, sensitivity, sinr_threshold):
    """
    Decide which drones one sensor detects in one time step.

    Each drone is judged on its own. Its SINR counts thermal noise over its own
    bandwidth plus every OTHER drone in the same azimuth sector as interference.
    Drones in other sectors do not interfere, because each sector antenna hears
    its own direction (leakage between sectors is not modeled). A drone is
    detected when its SINR clears `sinr_threshold` and its received power clears
    `sensitivity`.

    signals: (drone_id, prx_dbm, bandwidth_hz, sector_idx, ...) tuples.
    Returns (detected, sinr, pthermal): the detected subset of `signals` in
    input order, and the SINR and thermal noise of the drone with the highest
    SINR, which are recorded as the sensor's metrics for the step.
    """
    detected = []
    best_sinr = best_pthermal = None
    for k, (_, prx, bandwidth, sector_idx, *_) in enumerate(signals):
        interference = [other[1] for j, other in enumerate(signals)
                        if j != k and other[3] == sector_idx]
        pthermal = calculate_thermal_noise(bandwidth)
        sinr = calculate_sinr(prx, pthermal, interference)
        if best_sinr is None or sinr > best_sinr:
            best_sinr, best_pthermal = sinr, pthermal
        if is_detected(sinr, sinr_threshold, prx, sensitivity):
            detected.append(signals[k])
    return detected, best_sinr, best_pthermal


# Main simulation function.
def run_detection_simulation(db, exp_id, drone_trajectories: dict, phi_sector_edges=None,
                             toa_noise_s=0.0):
    sensor_positions = {s[2]: (s[3], s[4], s[5]) for s in db.get_sensors_by_exp_id(exp_id)}

    sensor_info = db.get_sensor_info_by_experiment(exp_id)
    # sensor_info rows: (id, type, sens, sinr_th, gain, freq_min, freq_max, sector_count, clock_offset)

    sensor_sectors = db.load_sensor_sectors(exp_id)  # azimuth sector boundaries per sensor
    sensor_sens          = {sid: sens   for sid, _, sens,  _,    _,    _,    _,    _, _  in sensor_info}
    sensor_sinr_threshold= {sid: sinr_th for sid, _, _,    sinr_th, _, _,   _,    _, _  in sensor_info}
    sensor_gains         = {sid: gain   for sid, _, _,    _,    gain, _,    _,    _, _  in sensor_info}
    freq_ranges          = {sid: (fmin, fmax)
                            for sid, _, _, _, _, fmin, fmax, _, _ in sensor_info}
    sensor_clock_offsets = {sid: co     for sid, _, _,    _,    _,    _,    _,    _, co in sensor_info}

    drones = _get_drone_info(db, exp_id)
    freqs = _get_drone_freqs(db, exp_id)
    mountain_list = _get_mountains(db, exp_id)

    drone_positions_by_time = _preprocess_drone_positions(drone_trajectories)
    timestamps = sorted(drone_positions_by_time.keys())
    sensor_up_times_by_time = _preprocess_sensor_up_times(db, exp_id, timestamps)

    detections_to_insert = []
    metrics_to_update = []

    # Main simulation loop: iterate in sorted order by timestamp.
    for timestamp in timestamps:
        drone_positions = drone_positions_by_time[timestamp]
        sensor_up_times = sensor_up_times_by_time[timestamp]
        # When within this 1-second step each drone transmits.
        emission_offsets = {drone_id: TOA_RNG.uniform(0.0, 1.0) for drone_id, *_ in drone_positions}

        for up in sensor_up_times:
            _, sensor_id, sensor_up_id, start_str, end_str, sensor_noise, thermal_power, sinr, active = up
            sx, sy, sz = sensor_positions[sensor_id]
            sensitivity = sensor_sens[sensor_id]
            sinr_threshold = sensor_sinr_threshold[sensor_id]
            rx_gain = sensor_gains[sensor_id]
            clock_offset = sensor_clock_offsets[sensor_id]
            signal_powers = []

            for drone_id, dx, dy, dz in drone_positions:
                drone = drones[drone_id]
                tx_power = drone[2]    # transmitter power
                tx_gain = drone[3]       # transmitter antenna gain
                bandwidth = drone[7]
                freq_hz = freqs[(exp_id, drone_id)]
                sector_idx = None

                # Check drone frequency in sensor range
                in_range = _drone_frequency_in_sensor_range(freq_hz, freq_ranges[sensor_id])
                if not in_range:
                    continue

                # Check mountain blockage.
                mountain_blocked = _line_of_sight_blocked(sx, sy, sz, dx, dy, dz, mountain_list)
                if mountain_blocked:
                    continue

                dif_x, dif_y, dif_z = dx - sx, dy - sy, dz - sz
                theta = math.degrees(math.atan2(dif_y, dif_x)) % 360

                for i, (start, end) in enumerate(sensor_sectors.get(sensor_id, []), start=1):
                    if start <= theta < end:
                        sector_idx = i
                        break

                phi_sector_idx = _phi_sector_index(dif_z, math.hypot(dif_x, dif_y), edges=phi_sector_edges)

                distance = calculate_distance((dx, dy, dz), (sx, sy, sz))
                prx = calculate_signal_power(tx_power, distance, freq_hz, tx_gain, rx_gain)
                signal_powers.append((drone_id, prx, bandwidth, sector_idx, distance, dx, dy, dz, phi_sector_idx))

            if not signal_powers:
                continue

            detected, sinr, pthermal = detect_drones(signal_powers, sensitivity, sinr_threshold)
            for drone_id, prx, bandwidth, sector_idx, distance, dx, dy, dz, phi_sector_idx in detected:
                emission = emission_offsets[drone_id]
                toa = calculate_toa(distance, emission, clock_offset, toa_noise_s)
                detections_to_insert.append(
                    (sensor_up_id, drone_id, prx, bandwidth, sector_idx, toa, phi_sector_idx,
                     emission, distance / C_LIGHT)
                )

            metrics_to_update.append((pthermal, sinr, bool(detected), sensor_up_id))

    db.cursor.executemany(
        "INSERT INTO sensor_detections (sensor_up_id, drone_id, received_power, "
        "received_signal_bandwidth, sector_detected, toa, phi_sector_detected, emission_offset, "
        "propagation_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        detections_to_insert
    )

    db.cursor.executemany(
        "UPDATE sensor_up_time SET received_thermal_noise_power = ?, sinr = ?, active = ? "
        "WHERE sensor_up_id = ?",
        metrics_to_update
    )
