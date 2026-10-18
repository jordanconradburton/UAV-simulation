from .physics import calculate_distance, sum_powers_dbm
from ..schema import N_SECTORS, PHI_N_SECTORS
import csv
import pandas as pd
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import os

def build_sensor_distances_table(db, exp_id):
    """
    Computes pairwise distances among sensors in experiment exp_id and builds an SQL table.
    """
    rows = db.get_sensors_by_exp_id(exp_id)
    sensors = [(r[2], r[3], r[4], r[5]) for r in rows]  # (sensor_id, x, y, z)

    distances = []
    n = len(sensors)

    for i in range(n):
        sid_i, x1, y1, z1 = sensors[i]
        for j in range(i + 1, n):
            sid_j, x2, y2, z2 = sensors[j]
            d = calculate_distance((x1, y1, z1), (x2, y2, z2))
            distances.append((exp_id, sid_i, x1, y1, z1, sid_j, x2, y2, z2, d))

    db.cursor.executemany(
        """INSERT INTO sensor_distances
           (exp_id, sensor1_id, sensor1_x, sensor1_y, sensor1_z,
            sensor2_id, sensor2_x, sensor2_y, sensor2_z, distance)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        distances
    )

    return distances


def export_sensor_information_to_csv(db, csv_path):
    print("Exporting all sensor information to CSV...")
    # The clock offset is TOA ground truth, exported in toa_measurements.csv
    # instead of here.
    rows = db.cursor.execute(
        "SELECT id, type, sensitivity, sinr_threshold, gain, freq_min, freq_max, sector_count "
        "FROM sensors ORDER BY id").fetchall()
    df = pd.DataFrame(
        rows,
        columns=["sensor_id", "sensor_type", "sensitivity", "sinr_threshold", "gain", "freq_min", "freq_max", "sector_count"]
    )
    df.to_csv(csv_path, index=False)
    print(f"Exported sensors to {csv_path}")

def export_sensor_sectors_to_csv(db, csv_path):
    print("Exporting all sensor sectors to CSV...")
    rows = db.get_sensor_sectors()
    df = pd.DataFrame(
        rows,
        columns=["sensor_id", "exp_id", "sector_id", "angle_start", "angle_end"]
    )
    df.to_csv(csv_path, index=False)
    print(f"Exported sensor_sectors to {csv_path}")


def export_sensor_distances_to_csv(db, csv_path):
    print("Exporting all sensor distances to CSV...")
    rows = db.get_sensors_distances()
    df = pd.DataFrame(
        rows,
        columns=["exp_id", "sensor1_id", "x1", "y1", "z1", "sensor2_id", "x2", "y2", "z2", "d"]
    )
    df.to_csv(csv_path, index=False)
    print(f"Exported sensor_distances to {csv_path}")

def export_all_sensor_detection_summary(db, csv_path, combine_signals=True):
    """
    For every experiment in the DB, writes detection data to CSV.

    combine_signals=True (default): one row per (sensor, timestamp).
      Columns: experiment_id, sensor_id, timestamp, num_detections,
               azimuth_sector_powers_dBm, elevation_bin_powers_dBm,
               sensor_sectors_detected, drone_ids, drone_positions,
               individual_powers, sensor_phi_sectors_detected
      The two power columns are what the sensor measured: one value per
      azimuth sector (N_SECTORS long) and one per elevation bin
      (PHI_N_SECTORS long), each the sum of the detections landing in it and
      0.0 where nothing was detected — the same values as the graph features.
      The remaining list columns give each detection separately.

    combine_signals=False: one row per individual detection; powers are not combined.
      Columns: experiment_id, sensor_id, timestamp, num_detections_at_timestamp,
               drone_id, received_power, sector_detected, phi_sector_detected, drone_position
      num_detections_at_timestamp is included so each row retains context about how
      many signals were present simultaneously.
    """
    exp_ids = [row[0] for row in db.get_experiments()]
    print("Exporting all sensor detection summaries...")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        if combine_signals:
            writer.writerow([
                "experiment_id",
                "sensor_id",
                "timestamp",
                "num_detections",
                "azimuth_sector_powers_dBm",
                "elevation_bin_powers_dBm",
                "sensor_sectors_detected",
                "drone_ids",
                "drone_positions",
                "individual_powers",
                "sensor_phi_sectors_detected",
            ])
        else:
            writer.writerow([
                "experiment_id",
                "sensor_id",
                "timestamp",
                "num_detections_at_timestamp",
                "drone_id",
                "received_power",
                "sector_detected",
                "drone_position",
                "phi_sector_detected",
            ])

        for exp_id in tqdm(exp_ids):
            det_rows = db.cursor.execute("""
                SELECT d.sensor_up_id, d.drone_id, d.received_power, d.sector_detected,
                       s.sensor_id, s.start_time, d.phi_sector_detected
                  FROM sensor_detections AS d
                  JOIN sensor_up_time AS s ON d.sensor_up_id = s.sensor_up_id
                 WHERE s.exp_id = ?
            """, (exp_id,)).fetchall()

            drone_positions = db.cursor.execute("""
                SELECT drone_id, time, start_x, start_y, start_z
                  FROM drone_exp_time
                 WHERE exp_id = ?
            """, (exp_id,)).fetchall()
            pos_map = {(d_id, t): (x, y, z) for d_id, t, x, y, z in drone_positions}

            det_group = defaultdict(list)
            for sensor_up_id, drone_id, prx, sector, sensor_id, ts, phi_sector in det_rows:
                pos = pos_map.get((drone_id, ts))
                det_group[(sensor_id, ts)].append((drone_id, prx, pos, sector, phi_sector))

            if combine_signals:
                for (sensor_id, ts), entries in det_group.items():
                    prx_list = [e[1] for e in entries]
                    writer.writerow([
                        exp_id,
                        sensor_id,
                        ts,
                        len(entries),
                        _per_bin_powers(entries, sector_index=3, n_bins=N_SECTORS),
                        _per_bin_powers(entries, sector_index=4, n_bins=PHI_N_SECTORS),
                        [e[3] for e in entries],
                        [e[0] for e in entries],
                        [e[2] for e in entries],
                        [f"{p:.2f}" for p in prx_list],
                        [e[4] for e in entries],
                    ])
            else:
                for (sensor_id, ts), entries in det_group.items():
                    num = len(entries)
                    for drone_id, prx, pos, sector, phi_sector in entries:
                        writer.writerow([
                            exp_id,
                            sensor_id,
                            ts,
                            num,
                            drone_id,
                            f"{prx:.2f}",
                            sector,
                            pos,
                            phi_sector,
                        ])

    print(f"Exported sensor detection summary with ground truth to {csv_path}")


def _per_bin_powers(entries, sector_index, n_bins):
    """
    Received power per sector: detections in the same sector summed in linear
    power (sum_powers_dbm), 0.0 for a sector with no detection.
    """
    by_bin = [[] for _ in range(n_bins)]
    for entry in entries:
        by_bin[entry[sector_index] - 1].append(entry[1])
    return [round(sum_powers_dbm(p), 2) if p else 0.0 for p in by_bin]


def export_phi_sector_scheme_to_csv(csv_path, edges=None, centers=None):
    """
    Writes the phi (elevation-angle) sector scheme as a small reference
    CSV — analogous to export_sensor_sectors_to_csv's theta boundaries,
    but phi sectors are the same for every sensor/experiment.

    Columns: phi_sector_id, angle_start_deg, angle_end_deg, center_deg
    Sector 1 (below-horizon) has angle_start_deg=-90 as a nominal outer bound
    (the true physical bound), even though the center bin's -9 is
    just a representative feature value, not a true centroid.
    """
    from .detection import PHI_SECTOR_EDGES_DEG
    from .graph_export import PHI_SECTOR_CENTERS_DEG

    edges = edges if edges is not None else PHI_SECTOR_EDGES_DEG
    centers = centers if centers is not None else PHI_SECTOR_CENTERS_DEG

    print("Exporting phi sector scheme reference to CSV...")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    edges = [-90.0] + list(edges)  # sector 1's nominal outer bound prepended
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["phi_sector_id", "angle_start_deg", "angle_end_deg", "center_deg"])
        for i in range(len(edges) - 1):
            writer.writerow([i + 1, edges[i], edges[i + 1], centers[i]])
    print(f"Exported phi sector scheme to {csv_path}")


def export_drone_trajectories_to_csv(db, csv_path):
    """
    Export all drone trajectories into a CSV file.

    Columns:
      id, experiment_id, drone_id, timestamp,
      horizontal_speed (m/s), vertical_speed (m/s, + = climbing),
      start_x, start_y, start_z,
      end_x, end_y, end_z,
      is_heading_change, is_vertical_speed_change, is_horizontal_speed_change
    """
    print("Exporting all drone trajectories to CSV...")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    query = """
        SELECT id, exp_id, drone_id, time, horizontal_speed, vertical_speed,
               start_x, start_y, start_z,
               end_x, end_y, end_z,
               is_heading_change, is_vertical_speed_change, is_horizontal_speed_change
          FROM drone_exp_time
         ORDER BY exp_id, drone_id, time
    """

    # Fetch all trajectories at once for speed
    rows = db.cursor.execute(query)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "experiment_id", "drone_id", "timestamp",
            "horizontal_speed", "vertical_speed",
            "start_x", "start_y", "start_z",
            "end_x", "end_y", "end_z",
            "is_heading_change", "is_vertical_speed_change", "is_horizontal_speed_change"
        ])
        for row in tqdm(rows, desc="Writing trajectories"):
            writer.writerow(row)

    print(f"Exported drone trajectories to {csv_path}")


def export_drone_information_to_csv(db, csv_path):
    """
    Export drone identity and transmitting-power information per experiment.

    Joins `drones` (global drone attributes) with `drone_exp` (per-experiment
    membership) so each row can be matched to the corresponding detection/
    trajectory rows via (experiment_id, drone_id).

    Columns:
      experiment_id, drone_id, drone_type, drone_transmitter_power,
      transmitter_antenna_gain, freq_min, freq_max, bandwidth
    """
    print("Exporting all drone information to CSV...")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    query = """
        SELECT de.exp_id, d.id, d.drone_type, d.drone_transmitter_power,
               d.transmitter_antenna_gain, d.freq_min, d.freq_max, d.bandwidth
          FROM drone_exp de
          JOIN drones d ON d.id = de.drone_id
         ORDER BY de.exp_id, d.id
    """
    rows = db.cursor.execute(query)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_id", "drone_id", "drone_type", "drone_transmitter_power",
            "transmitter_antenna_gain", "freq_min", "freq_max", "bandwidth"
        ])
        for row in tqdm(rows, desc="Writing drone information"):
            writer.writerow(row)

    print(f"Exported drone information to {csv_path}")



def export_toa_measurements_to_csv(db, csv_path):
    """
    One row per recorded detection: the time of arrival a sensor measured, for
    use by a TDOA solver, plus the ground truth needed to score one.

    Columns
    -------
    experiment_id, sensor_id, timestamp, drone_id
    sensor_x, sensor_y, sensor_z      sensor position, meters
    toa_s                             measured arrival time, seconds from the
                                      start of the step:
                                      emission_offset_s + propagation_time_s
                                      + clock_offset_s + timing noise
    emission_offset_s                 ground truth: when within the step the
                                      drone transmitted (unknown to sensors;
                                      cancels in any TDOA difference)
    propagation_time_s                ground truth: distance / c
    clock_offset_s                    ground truth: the sensor's residual
                                      clock synchronization error
    drone_x, drone_y, drone_z         ground truth: drone position, meters
    """
    print("Exporting time-of-arrival measurements to CSV...")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = db.cursor.execute("""
        SELECT u.exp_id, u.sensor_id, u.start_time, d.drone_id,
               se.x, se.y, se.z,
               d.toa, d.emission_offset, d.propagation_time, s.clock_offset,
               t.start_x, t.start_y, t.start_z
          FROM sensor_detections d
          JOIN sensor_up_time u  ON u.sensor_up_id = d.sensor_up_id
          JOIN sensor_exp se     ON se.exp_id = u.exp_id AND se.sensor_id = u.sensor_id
          JOIN sensors s         ON s.id = u.sensor_id
          JOIN drone_exp_time t  ON t.exp_id = u.exp_id AND t.drone_id = d.drone_id
                                AND t.time = u.start_time
         ORDER BY u.exp_id, u.start_time, d.drone_id, u.sensor_id
    """)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_id", "sensor_id", "timestamp", "drone_id",
            "sensor_x", "sensor_y", "sensor_z",
            "toa_s", "emission_offset_s", "propagation_time_s", "clock_offset_s",
            "drone_x", "drone_y", "drone_z",
        ])
        for row in tqdm(rows, desc="Writing TOA measurements"):
            writer.writerow(row)
    print(f"Exported TOA measurements to {csv_path}")

def export_csv_tables(db, csv_dir, elevation_sector_edges, elevation_sector_centers):
    """
    Write a shard's eight flat CSV tables into `csv_dir`.

    elevation_sector_edges/centers: the elevation scheme the shard was
    generated with, documented in phi_sector_scheme.csv.
    """
    export_sensor_information_to_csv(db, os.path.join(csv_dir, "sensor_information.csv"))
    export_sensor_sectors_to_csv(db, os.path.join(csv_dir, "sensor_sectors.csv"))
    export_sensor_distances_to_csv(db, os.path.join(csv_dir, "sensor_distances.csv"))
    export_all_sensor_detection_summary(
        db, os.path.join(csv_dir, "sensor_detection_summary.csv"), combine_signals=True)
    export_phi_sector_scheme_to_csv(
        os.path.join(csv_dir, "phi_sector_scheme.csv"),
        edges=elevation_sector_edges, centers=elevation_sector_centers)
    export_drone_trajectories_to_csv(db, os.path.join(csv_dir, "drone_trajectories.csv"))
    export_drone_information_to_csv(db, os.path.join(csv_dir, "drone_information.csv"))
    export_toa_measurements_to_csv(db, os.path.join(csv_dir, "toa_measurements.csv"))
