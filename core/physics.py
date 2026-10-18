import math
import random

C_LIGHT = 299_792_458.0  # speed of light, m/s
TOA_RNG = random.Random()

# Defaults for GPS-disciplined sensors, in nanoseconds.
DEFAULT_SYNC_ERROR_NS = 20.0   # std of each sensor's residual clock offset
DEFAULT_TOA_NOISE_NS = 10.0    # std of the per-detection timing noise

# Free-space path-loss constant in calculate_signal_power, dB.
FRIIS_CONSTANT_DB = 147

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
    return Ptx + Gt + Gr - 20 * math.log10(distance) - 20 * math.log10(frequency_hz) + FRIIS_CONSTANT_DB

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

def calculate_sinr(prx, pthermal, prx_interference):
    """
    Calculate the Signal-to-Interference-plus-Noise Ratio (SINR) in dB.

    Parameters
    ----------
    prx : float
        Desired signal power in dBm.
    pthermal : float
        Thermal noise power in dBm.
    prx_interference : list of float
        Interfering signal powers in dBm (empty for none).

    Returns
    -------
    float
        SINR in dB.

    Notes
    -----
    The SINR is calculated as:
        SINR(dB) = P_lead(dB) - 10 log10(Σ(10^(P_interference/10)) + 10^(P_noise/10))
    """
    if prx_interference:
        interference_linear = sum(10 ** (p / 10) for p in prx_interference)  # Convert to linear and sum
    else:
        interference_linear = 0  # no interferers

    pthermal_linear = 10 ** (pthermal / 10)
    sinr = prx - 10 * math.log10(interference_linear + pthermal_linear)

    return sinr

def is_detected(sinr, sinr_threshold, prx, sensitivity):
    """
    Determine whether a sensor detects one drone's signal, from its SINR and power.

    Parameters
    ----------
    sinr : float
        Signal-to-Interference-plus-Noise Ratio (dB).
    sinr_threshold : float
        Minimum SINR required for detection (dB).
    prx : float
        Received signal power (dBm).
    sensitivity : float
        Receiver sensitivity threshold (dBm).

    Returns
    -------
    bool
        True if the signal is detected, False otherwise.
    """
    return sinr > sinr_threshold and prx > sensitivity


def calculate_toa(distance, emission_offset, clock_offset, sigma_n):
    """
    Measured time of arrival at a sensor, in seconds from the start of the step.

        toa = emission_offset + distance / c + clock_offset + n,   n ~ N(0, sigma_n²)

    Parameters
    ----------
    distance : float
        Drone-to-sensor distance in meters.
    emission_offset : float
        When within the step the drone transmitted, in seconds. Unknown to the
        sensors; it cancels in any TDOA difference.
    clock_offset : float
        The sensor's residual synchronization error, in seconds.
    sigma_n : float
        Standard deviation of the timing measurement noise, in seconds.
    """
    return emission_offset + distance / C_LIGHT + clock_offset + TOA_RNG.gauss(0.0, sigma_n)
