import math
import DatabaseManager

def calculate_distance(drone_pos, sensor_pos):
    return math.sqrt(sum((dp - sp) ** 2 for dp, sp in zip(drone_pos, sensor_pos)))

def calculate_signal_power(Ptx, distance, frequency_hz, Gt=0, Gr=0):
    if distance <= 0 or frequency_hz <= 0:
        return None
    return Ptx + Gt + Gr - 20 * math.log10(distance) - 20 * math.log10(frequency_hz) + 147

def calculate_thermal_noise(bw_hz, nf=5):
    if bw_hz <= 0:
        return None
    return -174 + 10 * math.log10(bw_hz) + nf

def calculate_sinr(prx_lead, pthermal, prx_others):
    # NOTE: all input received is in dB, and so must be converted to linear units (mW) before being summed

    # SINR (dB) = 10log10(P_lead / (P_interference + P_noise)) = P_lead(dB) - 10log10(P_interference + P_noise)
    # P_noise = 10 ** (P_noise(dB) / 10)
    # P_interference = sum (10 ** P_interference_i(dB) / 10)

    if prx_others:
        interference_linear = sum(10 ** (p / 10) for p in prx_others)  # Convert to linear and sum
    else:
        interference_linear = 0  # No interference if prx_others is empty

    pthermal_linear = 10 ** (pthermal / 10)
    sinr = prx_lead - 10 * math.log10(interference_linear + pthermal_linear)

    return sinr

def is_sensor_active(sinr, sinr_threshold, prx_lead, sensitivity):
    return sinr > sinr_threshold and prx_lead > sensitivity

def calculate_sensor_metrics(exp_id, sensor_id, sensor_up_id, db: DatabaseManager):
    # Get related sensor info
    sensors = {s[0]: s for s in db.get_sensors()}
    sensor_info = sensors[sensor_id]
    sensitivity = sensor_info[2]
    sinr_threshold = sensor_info[3]

    up_entry = next(row for row in db.get_sensor_up_time() if row[2] == sensor_up_id)
    Gr = up_entry[7]

    # Get sensor position
    sensor_exp = next(row for row in db.get_sensor_exp() if row[0] == exp_id and row[1] == sensor_id)
    sensor_pos = sensor_exp[3:6]

    # Get detections for this sensor_up_id
    detections = [d for d in db.get_sensor_detections() if d[1] == sensor_up_id]

    drones = {d[0]: d for d in db.get_drones()}
    freqs = {(f[0], f[1]): f[3] for f in db.get_drone_radio_freq()}
    positions = db.get_drone_exp_time()

    signal_powers = []

    for detection in detections:
        _, _, drone_id, received_power, bandwidth = detection
        drone = drones[drone_id]
        Ptx = drone[2]
        Gt = drone[3]
        freq_mhz = freqs.get((exp_id, drone_id), None)

        if freq_mhz is None:
            continue

        # Get drone position closest to start_time
        pos_matches = [p for p in positions if p[0] == drone_id and p[1] == exp_id]
        drone_pos = None
        for p in pos_matches:
            drone_pos = p[4:7]  # start_x, start_y, start_z
            break
        if drone_pos is None:
            continue

        distance = calculate_distance(drone_pos, sensor_pos)
        freq_hz = freq_mhz * 1e6
        Prx = calculate_signal_power(Ptx, distance, freq_hz, Gt, Gr)
        signal_powers.append((drone_id, Prx, bandwidth))

    if not signal_powers:
        return None

    # Identify lead UAV
    lead = max(signal_powers, key=lambda x: x[1])
    prx_lead = lead[1]
    bandwidth = lead[2] * 1e6  # MHz to Hz
    pthermal = calculate_thermal_noise(bandwidth)
    prx_others = [prx for drone_id, prx, _ in signal_powers if prx != prx_lead]
    sinr = calculate_sinr(prx_lead, pthermal, prx_others)
    active = is_sensor_active(sinr, sinr_threshold, prx_lead, sensitivity)

    return {
        "sensor_up_id": sensor_up_id,
        "thermal_noise": round(pthermal, 2),
        "sinr": round(sinr, 2),
        "active": int(active)
    }

# TODO: calculating metrics for all sensor_up_times - after experiment run
def compute_metrics(db, exp_id):
    for up_time in db.get_sensor_up_time():
        if up_time[0] == exp_id:
            sensor_id = up_time[1]
            sensor_up_id = up_time[2]
            metrics = calculate_sensor_metrics(exp_id, sensor_id, sensor_up_id, db)
            if metrics:
                db.update_sensor_metrics(
                    sensor_up_id=metrics["sensor_up_id"],
                    thermal_noise=metrics["thermal_noise"],
                    sinr=metrics["sinr"],
                    active=metrics["active"]
                )