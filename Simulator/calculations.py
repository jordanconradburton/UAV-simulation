import math

def calculate_distance(pos1, pos2):
    """
    Calculate the Euclidean distance between two points in N-dimensional space.

    Parameters
    ----------
    pos1 : iterable of float
        Coordinates of the first point.
    pos2 : iterable of float
        Coordinates of the second point.

    Returns
    -------
    float
        Euclidean distance between `pos1` and `pos2`.
    """
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))

def sum_powers_dbm(prx_list, floor_dbm=-120.0):
    """
    Sum received powers expressed in dBm.

    Parameters
    ----------
    prx_list : list of float
        List of received powers in dBm.
    floor_dbm : float, optional
        Default power level (in dBm) returned if `prx_list` is empty.
        Default is -120.0 dBm.

    Returns
    -------
    float
        Total power in dBm.

    Notes
    -----
    Power values are converted from dBm to linear scale (mW), summed,
    and then converted back to dBm.
    """
    if not prx_list:
        return floor_dbm
    lin_sum = sum(10 ** (p / 10) for p in prx_list)
    return 10 * math.log10(lin_sum)

def calculate_signal_power(Ptx, distance, frequency_hz, Gt=0, Gr=0):
    """
    Compute the received signal power using the Friis transmission equation.

    Parameters
    ----------
    Ptx : float
        Transmit power in dBm.
    distance : float
        Distance between transmitter and receiver in meters.
    frequency_hz : float
        Carrier frequency in Hz.
    Gt : float, optional
        Transmitter antenna gain in dB. Default is 0.
    Gr : float, optional
        Receiver antenna gain in dB. Default is 0.

    Returns
    -------
    float or None
        Received power in dBm, or `None` if inputs are invalid
        (non-positive distance or frequency).

    Notes
    -----
    Uses the Friis free-space path loss formula:
        Prx(dBm) = Ptx + Gt + Gr - 20 log10(d) - 20 log10(f) + 147
    """
    if distance <= 0 or frequency_hz <= 0:
        return None
    return Ptx + Gt + Gr - 20 * math.log10(distance) - 20 * math.log10(frequency_hz) + 147

def calculate_thermal_noise(bw_hz, nf=5):
    """
    Calculate thermal noise power in dBm.

    Parameters
    ----------
    bw_hz : float
        System bandwidth in Hz.
    nf : float, optional
        Receiver noise figure in dB. Default is 5 dB.

    Returns
    -------
    float or None
        Thermal noise power in dBm, or `None` if `bw_hz` is non-positive.

    Notes
    -----
    Thermal noise is computed as:
        N(dBm) = -174 + 10 log10(BW) + NF

    where -174 dBm/Hz is the thermal noise density at room temperature.
    """
    if bw_hz <= 0:
        return None
    return -174 + 10 * math.log10(bw_hz) + nf

def calculate_sinr(prx_lead, pthermal, prx_others):
    """
    Calculate the Signal-to-Interference-plus-Noise Ratio (SINR) in dB.

    Parameters
    ----------
    prx_lead : float
        Desired signal power in dBm.
    pthermal : float
        Thermal noise power in dBm.
    prx_others : list of float
        List of interfering signal powers in dBm.

    Returns
    -------
    float
        SINR in dB.

    Notes
    -----
    The SINR is calculated as:
        SINR(dB) = P_lead(dB) - 10 log10(Σ(10^(P_interference/10)) + 10^(P_noise/10))
    """
    if prx_others:
        interference_linear = sum(10 ** (p / 10) for p in prx_others)  # Convert to linear and sum
    else:
        interference_linear = 0  # No interference if prx_others is empty

    pthermal_linear = 10 ** (pthermal / 10)
    sinr = prx_lead - 10 * math.log10(interference_linear + pthermal_linear)

    return sinr

def is_sensor_active(sinr, sinr_threshold, prx_lead, sensitivity):
    """
    Determine whether a sensor is active based on SINR and sensitivity thresholds.

    Parameters
    ----------
    sinr : float
        Signal-to-Interference-plus-Noise Ratio (dB).
    sinr_threshold : float
        Minimum SINR required for activation (dB).
    prx_lead : float
        Received signal power (dBm).
    sensitivity : float
        Receiver sensitivity threshold (dBm).

    Returns
    -------
    bool
        True if the sensor is active, False otherwise.
    """
    return sinr > sinr_threshold and prx_lead > sensitivity
