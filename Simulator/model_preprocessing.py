from Simulator.calculations import calculate_distance, sum_powers_dbm
import csv
import pandas as pd
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

def build_sensor_distances_table(db, exp_id):
    """
    Computes pairwise distances among sensors in experiment exp_id and builds an SQL table.
    """
    rows = db.get_sensors_by_exp_id(exp_id)
    sensors = [(r[1], r[3], r[4], r[5]) for r in rows]  # (sensor_id, x, y, z)

    distances = []
    n = len(sensors)

    for i in range(n):
        sid_i, x1, y1, z1 = sensors[i]
        for j in range(i + 1, n):
            sid_j, x2, y2, z2 = sensors[j]
            d = calculate_distance((x1, y1, z1), (x2, y2, z2))
            distances.append((sid_i, (x1, y1, z1), sid_j, (x2, y2, z2), d))

    for sid_i, (x1, y1, z1), sid_j, (x2, y2, z2), d in distances:
        db.set_sensors_distances(exp_id, sid_i, x1, y1, z1, sid_j, x2, y2, z2, d)

    return distances


def export_sensor_information_to_csv(db, csv_path):
    print("Exporting all sensor information to CSV...")
    rows = db.get_sensors()
    df = pd.DataFrame(
        rows,
        columns=["sensor_id", "sensor_type", "sensitivity", "sinr_threshold", "freq_min", "freq_max", "sector_count"]
    )
    df.to_csv(csv_path, index=False)
    print(f"Exported sensors to {csv_path}")

def export_sensor_gain_sectors_to_csv(db, csv_path):
    print("Exporting all sensor gain sectors to CSV...")
    rows = db.get_sensor_gain_sectors()
    df = pd.DataFrame(
        rows,
        columns=["sensor_id", "exp_id", "sector_id", "angle_start", "angle_end", "gain_db"]
    )
    df.to_csv(csv_path, index=False)
    print(f"Exported sensor_gain_sectors to {csv_path}")


def export_sensor_distances_to_csv(db, csv_path):
    print("Exporting all sensor distances to CSV...")
    rows = db.get_sensors_distances()
    df = pd.DataFrame(
        rows,
        columns=["exp_id", "sensor1_id", "x1", "y1", "z1", "sensor2_id", "x2", "y2", "z2", "d"]
    )
    df.to_csv(csv_path, index=False)
    print(f"Exported sensor_distances to {csv_path}")

def export_all_sensor_detection_summary(db, csv_path):
    """
    For every experiment in the DB, builds the per‐sensor, per‐timestamp
    sum‐of‐powers & detection counts, plus ground truth (drone ids, positions,
    individual powers), and writes a single CSV with columns:
      experiment_id, sensor_id, timestamp,
      num_detections, total_power_dBm,
      drone_ids, drone_positions, individual_powers
    """
    # fetch all experiment IDs
    exp_ids = [row[0] for row in db.get_experiments()]
    print("Exporting all sensor detection summaries...")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_id",
            "sensor_id",
            "timestamp",
            "num_detections",
            "total_power_dBm",
            "drone_ids",
            "drone_positions",
            "individual_powers"
        ])

        for exp_id in tqdm(exp_ids):
            # build up_time mapping: sensor_up_id -> (sensor_id, timestamp)
            up_rows = db.get_sensor_up_time_by_experiment(exp_id)
            up_map = {row[2]: (row[1],
                               # ensure datetime
                               (row[3] if not isinstance(row[3], str)
                                else datetime.fromisoformat(row[3])) )
                      for row in up_rows}

            # collect detections grouped by (sensor_id, timestamp)
            det_group = defaultdict(list)
            for det in db.get_sensor_detections():
                _, sup_id, drone_id, prx, _ = det
                if sup_id not in up_map:
                    continue
                sensor_id, ts = up_map[sup_id]
                # lookup drone position at this timestamp
                pos = db.cursor.execute(
                    "SELECT end_x, end_y, end_z FROM drone_exp_time"
                    " WHERE exp_id=? AND drone_id=? AND time=?",
                    (exp_id, drone_id, ts)
                ).fetchone()
                det_group[(sensor_id, ts)].append((drone_id, prx, pos))

            # write one row per group
            for (sensor_id, ts), entries in det_group.items():
                prx_list = [e[1] for e in entries]
                total_dbm = sum_powers_dbm(prx_list)

                drone_ids = [e[0] for e in entries]
                positions = [e[2] for e in entries]
                writer.writerow([
                    exp_id,
                    sensor_id,
                    ts.isoformat(),
                    len(entries),
                    f"{total_dbm:.2f}",
                    drone_ids,
                    positions,
                    [f"{p:.2f}" for p in prx_list]
                ])
    print(f"Exported sensor detection summary with ground truth to {csv_path}")
