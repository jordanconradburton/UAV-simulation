import math

def calculate_distance(pos1, pos2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))

def sum_powers_dbm(prx_list, floor_dbm=-120.0):
    if not prx_list:
        return floor_dbm
    lin_sum = sum(10 ** (p / 10) for p in prx_list)
    return 10 * math.log10(lin_sum)

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
