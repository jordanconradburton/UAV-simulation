"""
schema.py

Node-feature column layout of the exported graph tensors.

The core exporter writes these tensors and takes the azimuth block width
(N_SECTORS) from here; the variant tools re-open an already-generated dataset
and rewrite specific columns. Both read the layout from this one module.

LAYOUT
──────
`build_experiment_graphs` lays out `Data.x` as [N_sensors, F] in fixed blocks.
With the CLI defaults (`--sectors 6 --elevation-sectors --sector-angle-centers`)
F is 34 and the blocks are:

    index  block                     meaning
    ────────────────────────────────────────────────────────────────────────
     0     sensor_type               integer code, dense-ranked per experiment
     1     sensitivity               dBm, minimum detectable power
     2     sinr_threshold            dB
     3     gain                      dBi, receive antenna gain
     4     freq_min                  Hz, low edge of the sensor's tuned band
     5     freq_max                  Hz, high edge
     6     sector_count              azimuth sectors on this sensor
     7     x                         meters
     8     y                         meters
     9     z                         meters
    10-15  theta_angle_centers       degrees, azimuth sector center per column
    16-21  theta_sector_power        dBm received per azimuth sector, 0 = none
    22-27  phi_angle_centers         degrees, elevation sector center per column
    28-33  phi_sector_power          dBm received per elevation sector, 0 = none

A **zero** in either power block means "no detection in that sector", not
"0 dBm".

The two angle-center blocks are constant per sensor.

WHEN THE WIDTH IS NOT 34
────────────────────────
The theta blocks are always `N_SECTORS` (6) wide, whatever the sensors' sector
counts: a sensor with fewer sectors (`--sectors 3`, or a randomized count) is
zero-padded, so every experiment in a dataset has the same width. The width
changes only with the export options: the phi blocks are absent without
`--elevation-sectors`, and either angle-center block can be switched off. Use
`layout()` rather than the bare constants whenever the dataset was not
generated with the defaults.
"""

N_SECTORS = 6           # azimuth columns per block; sensors with fewer sectors are zero-padded
PHI_N_SECTORS = 6       # elevation bins: 1 below-horizon catch-all + 5 over [0, 90)

N_STATIC = 10           # columns 0..9: the static block above
MAX_DRONES = 8          # drones per experiment; labels.csv pads every drone list to this length

# --- The 34-wide default layout, as flat constants ---------------------------
ZF_X, ZF_Y, ZF_Z = 7, 8, 9
ZF_THETA_ANGLES = slice(10, 10 + N_SECTORS)                       # 10:16
ZF_SEC = slice(ZF_THETA_ANGLES.stop, ZF_THETA_ANGLES.stop + N_SECTORS)          # 16:22
ZF_PHI_ANGLES = slice(ZF_SEC.stop, ZF_SEC.stop + PHI_N_SECTORS)                 # 22:28
ZF_PHI_SEC = slice(ZF_PHI_ANGLES.stop, ZF_PHI_ANGLES.stop + PHI_N_SECTORS)      # 28:34

EXPECTED_WIDTH = ZF_PHI_SEC.stop        # 34


def layout(include_sector_angle_centers=True,
           include_phi_sectors=True, include_phi_sector_angle_centers=True):
    """
    Column layout for a dataset generated with non-default sector options.

    Returns a dict of slices/ints with the same keys as the flat constants
    above, plus "width". Blocks that were not exported are absent from the
    dict.
    """
    out = {"x": 7, "y": 8, "z": 9}
    cur = N_STATIC
    if include_sector_angle_centers:
        out["theta_angles"] = slice(cur, cur + N_SECTORS)
        cur += N_SECTORS
    out["theta_power"] = slice(cur, cur + N_SECTORS)
    cur += N_SECTORS
    if include_phi_sectors:
        if include_phi_sector_angle_centers:
            out["phi_angles"] = slice(cur, cur + PHI_N_SECTORS)
            cur += PHI_N_SECTORS
        out["phi_power"] = slice(cur, cur + PHI_N_SECTORS)
        cur += PHI_N_SECTORS
    out["width"] = cur
    return out
