from .mountains import segment_intersects_mountain
from .physics import sum_powers_dbm
import torch
from torch_geometric.data import Data

import csv
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from ..schema import MAX_DRONES, N_SECTORS

PHI_N_SECTORS = 6
PHI_SECTOR_CENTERS_DEG = [-9.0, 9.0, 27.0, 45.0, 63.0, 81.0]

#=== GRAPH FORMAT ===
# Each graph snapshot is represented as a PyTorch-Geometric Data object:
#   • x           : Tensor[N, F] of node features (static + dynamic)
#   • edge_index  : LongTensor[2, E] listing source/target node indices
#   • edge_attr   : Tensor[E, D] of edge attributes (distance; mountain-blocked
#                   pairs have no edge)


def build_experiment_graphs(db, exp_id, *, include_phi_sectors,
                             include_sector_angle_centers,
                             include_phi_sector_angle_centers,
                             phi_sector_centers):
    """
    Build a time-series of graph snapshots for experiment `exp_id`.

    Nodes: sensors
    Node feature layout: [x_static(10) | sector_angle_centers(S, only if include_sector_angle_centers) | sector_signals | phi_features (if include_phi_sectors)]
      x_static(10): type, sensitivity, sinr_threshold, gain, fmin, fmax, sector_count, x, y, z
      sector_angle_centers(S): center angle (degrees) of each sector, where
        S = N_SECTORS (6). Sensors with fewer sectors are
        zero-padded. E.g. 6 sectors: [30,90,150,210,270,330]; 3 sectors:
        [60,180,300,0,0,0].
        Included only when include_sector_angle_centers is set. When it is
        absent, columns 10:10+S hold per-sector POWER; when it is present, they
        hold angle centers and the power block starts S columns later.
      sector_signals: one summed power per sector → width S, zero-padded the
        same way (all detections
        landing in the same sector are summed via sum_powers_dbm).
      phi_features (only present if include_phi_sectors=True):
        an elevation-angle axis, not fused with the theta sectors
        above into a joint grid — same summed-power aggregation applied to
        PHI_N_SECTORS=6 bins:
          phi_sector_centers(6): the scheme's bin centers, same for every sensor
            (unlike theta's centers, which depend on that sensor's own sector_count)
          phi_sector_signals: width 6, same aggregation rule as sector_signals above.

    Edges: undirected between every sensor pair not blocked by a mountain
           (a blocked pair has no edge at all)
    Edge attribute: Euclidean distance from `sensor_distances` table.
    """

    sensors = db.get_sensors_by_exp_id(exp_id)
    sensor_info_rows = db.get_sensor_info_by_experiment(exp_id)
    distances = db.get_sensors_distances_by_exp(exp_id)  # (exp_id, sid1, x1, y1, z1, sid2, x2, y2, z2, dist)
    mountains = db.get_mountains_by_experiment(exp_id) # (m_id, x, y, side_length, height)
    ups = db.get_sensor_up_time_by_experiment(exp_id)
    # (exp_id, sensor_id, sensor_up_id, start_time, end_time, sensor_noise, received_thermal_noise, sinr, active)
    detections = db.get_sensor_detections_by_exp(exp_id)

    sensor_info = {r[0]: r for r in sensor_info_rows}  # sid → info tuple
    id_to_idx = {s[2]: i for i, s in enumerate(sensors)}  # sid → index
    sensor_ids = list(id_to_idx.keys())
    N = len(sensor_ids)

    largest = max(int(sensor_info[sid][7]) for sid in sensor_ids)
    if largest > N_SECTORS:
        raise ValueError(f"experiment {exp_id} has a sensor with {largest} azimuth sectors; "
                         f"at most {N_SECTORS} fit in the feature layout")

    # map sensor_up_id → list of detections
    det_by_sup = defaultdict(list)
    for det in detections:
        sup_id = det[1]
        det_by_sup[sup_id].append(det)

    # group ups by timestamp
    ups_by_time = defaultdict(list)
    for u in ups:
        ts = u[3]
        ups_by_time[ts].append(u)
    times = sorted(ups_by_time.keys())

    # 1) Static node features
    type2idx = {t: i for i, t in enumerate(sorted({r[1] for r in sensor_info_rows}))}
    static_feats = []
    sector_angle_feats = []
    for _, _, sid, x, y, z in sensors:
        typ, sens, sinr_th, gain, fmin, fmax, scount, _ = sensor_info[sid][1:]  # _ = clock_offset (not a node feature)
        static_feats.append([
            type2idx[typ], sens, sinr_th, gain, fmin, fmax, scount, x, y, z
        ])
        step = 360.0 / int(scount)
        centers = [(k + 0.5) * step for k in range(int(scount))]
        centers += [0.0] * (N_SECTORS - int(scount))  # zero-pad
        sector_angle_feats.append(centers)
    x_static = torch.tensor(static_feats, dtype=torch.float)
    x_sector_angles = torch.tensor(sector_angle_feats, dtype=torch.float)  # [N, N_SECTORS]

    if include_phi_sectors:
        x_phi_angles = torch.tensor([phi_sector_centers for _ in sensor_ids], dtype=torch.float)  # [N, 6]

    # 2) Edges
    edges, attrs = [], []
    for rec in distances:
        _, sid1, x1, y1, z1, sid2, x2, y2, z2, dist = rec

        i, j = id_to_idx[sid1], id_to_idx[sid2]

        p1 = (x1, y1, z1)
        p2 = (x2, y2, z2)
        blocked = any(segment_intersects_mountain(p1, p2, m) for m in mountains)
        if blocked:
            continue

        edges.append([i, j]); attrs.append([dist]) # undirected: add both directions
        edges.append([j, i]); attrs.append([dist])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=torch.float)

    # 3) Dynamic node features over time
    graphs = []
    for t in times:
        sectors_power = torch.zeros(N, N_SECTORS, dtype=torch.float)
        if include_phi_sectors:
            phi_sectors_power = torch.zeros(N, PHI_N_SECTORS, dtype=torch.float)

        for u in ups_by_time[t]:
            _, sid, sup_id, *_ = u
            idx = id_to_idx[sid]
            dets = det_by_sup.get(sup_id, [])
            per_sector_powers = [[] for _ in range(N_SECTORS)]
            if include_phi_sectors:
                per_phi_sector_powers = [[] for _ in range(PHI_N_SECTORS)]

            for det in dets:
                per_sector_powers[det[5] - 1].append(det[3])
                if include_phi_sectors:
                    per_phi_sector_powers[det[7] - 1].append(det[3])

            for s in range(N_SECTORS):
                if not per_sector_powers[s]:
                    continue
                sectors_power[idx, s] = sum_powers_dbm(per_sector_powers[s])

            if include_phi_sectors:
                for s in range(PHI_N_SECTORS):
                    if not per_phi_sector_powers[s]:
                        continue
                    phi_sectors_power[idx, s] = sum_powers_dbm(per_phi_sector_powers[s])

        feature_blocks = [x_static]
        if include_sector_angle_centers:
            feature_blocks.append(x_sector_angles)
        feature_blocks.append(sectors_power)
        if include_phi_sectors:
            if include_phi_sector_angle_centers:
                feature_blocks.append(x_phi_angles)
            feature_blocks.append(phi_sectors_power)
        x = torch.cat(feature_blocks, dim=1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.timestamp = t
        graphs.append(data)

    return graphs

def export_all_graphs_to_one_file(db, out_pt, *, include_phi_sectors,
                                   include_sector_angle_centers,
                                   include_phi_sector_angle_centers, phi_sector_centers):
    exp_ids = [row[0] for row in db.get_experiments()]

    all_graphs = {}
    for exp_id in tqdm(exp_ids):
        graphs = build_experiment_graphs(db, exp_id,
                                          include_phi_sectors=include_phi_sectors,
                                          include_sector_angle_centers=include_sector_angle_centers,
                                          include_phi_sector_angle_centers=include_phi_sector_angle_centers,
                                          phi_sector_centers=phi_sector_centers)
        all_graphs[exp_id] = graphs

    print(f"Saving all {len(exp_ids)} experiments into `{out_pt}`")
    torch.save(all_graphs, out_pt)

def export_drone_labels(db, csv_path, max_drones=MAX_DRONES):
    """
    Writes a CSV with one row per (experiment, timestamp):
      exp_id, timestamp, num_drones,
      drone_ids (padded list of length max_drones),
      drone_positions (padded list of length max_drones of (x,y,z) tuples),
      drone_transmitter_powers (padded list of length max_drones, aligned with
        drone_ids; each drone's transmitter power is static over time but can
        differ between drones in the same experiment)
    """
    exp_ids = [row[0] for row in db.get_experiments()]
    power_by_drone_id = {row[0]: row[2] for row in db.get_drones()}

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id", "timestamp", "num_drones",
            "drone_ids", "drone_positions", "drone_transmitter_powers",
            "is_heading_change", "is_vertical_speed_change", "is_horizontal_speed_change"
        ])

        for exp_id in exp_ids:
            rows = db.cursor.execute("""
                SELECT drone_id, time, start_x, start_y, start_z,
                       is_heading_change, is_vertical_speed_change, is_horizontal_speed_change
                  FROM drone_exp_time WHERE exp_id = ?""", (exp_id,)).fetchall()

            by_time = defaultdict(list)
            change_flags = {}

            for drone_id, ts, sx, sy, sz, is_heading_change, is_vertical_speed_change, is_horizontal_speed_change in rows:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)

                by_time[ts].append((drone_id, (sx, sy, sz)))

                if ts not in change_flags:
                    change_flags[ts] = {
                        'is_heading_change': is_heading_change,
                        'is_vertical_speed_change': is_vertical_speed_change,
                        'is_horizontal_speed_change': is_horizontal_speed_change,
                    }
                else:
                    change_flags[ts]['is_heading_change'] |= is_heading_change
                    change_flags[ts]['is_vertical_speed_change'] |= is_vertical_speed_change
                    change_flags[ts]['is_horizontal_speed_change'] |= is_horizontal_speed_change

            sensor_ts_rows = db.cursor.execute(
                "SELECT DISTINCT start_time FROM sensor_up_time WHERE exp_id = ?", (exp_id,)
            ).fetchall()
            for (ts_str,) in sensor_ts_rows:
                ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else ts_str
                if ts not in by_time:
                    by_time[ts]
                    change_flags[ts] = {
                        'is_heading_change': False, 'is_vertical_speed_change': False,
                        'is_horizontal_speed_change': False,
                    }

            for ts in sorted(by_time):
                entries = by_time[ts]
                entries.sort(key=lambda x: x[0])

                num = len(entries)
                ids = [d for d, _ in entries] + [-1] * (max_drones - num)
                pos = [p for _, p in entries] + [(0.0, 0.0, 0.0)] * (max_drones - num)
                powers = [power_by_drone_id.get(d, 0.0) for d, _ in entries] + [0.0] * (max_drones - num)

                flags = change_flags[ts]

                writer.writerow([
                    exp_id,
                    ts.isoformat(),
                    num,
                    ids,
                    pos,
                    powers,
                    flags['is_heading_change'],
                    flags['is_vertical_speed_change'],
                    flags['is_horizontal_speed_change']
                ])

    print(f"Exported drone label CSV to {csv_path}")
