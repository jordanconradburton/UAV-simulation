from Simulator.MountainSightCalculations import segment_intersects_mountain
from Simulator.calculations import sum_powers_dbm
import torch
from torch_geometric.data import Data

import csv
from collections import defaultdict
from datetime import datetime

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MAX_DRONES = 8
MAX_SECTORS = 6

#=== GRAPH FORMAT ===
# Each graph snapshot is represented as a PyTorch-Geometric Data object:
#   • x           : Tensor[N, F] of node features (static + dynamic)
#   • edge_index  : LongTensor[2, E] listing source/target node indices
#   • edge_attr   : Tensor[E, D] of edge attributes (e.g. distance or inf if blocked)


def build_experiment_graphs(db, exp_id):
    """
    Build a time-series of graph snapshots for experiment `exp_id`.

    Nodes: sensors
    Static node features: [sensitivity, sinr_threshold, freq_min, freq_max, sector_count]
    Dynamic node feature per snapshot: sum of received power at that timestamp

    Edges: undirected between every sensor pair
    Edge attribute: Euclidean distance from `sensor_distances` table,
                    set to float('inf') if a mountain blocks line of sight.
    """

    sensors = db.get_sensors_by_exp_id(exp_id)
    sensor_info = db.get_sensor_info_by_experiment(exp_id)
    sensor_ids = [row[1] for row in sensors]           # row = (exp_id, sensor_id, sens_exp_id, x, y, z)
    id_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}
    N = len(sensor_ids)

    types = [info[1] for info in sensor_info]
    unique_types = sorted(set(types))
    type2idx = {t:i for i,t in enumerate(unique_types)}

    # 1) Static node features
    static_feats = []
    for (_, sid, _, x, y, z) in sensors:
        # each row from get_sensor_info_by_experiment: (sensor_id, type, sens, sinr_th, gain, fmin, fmax, sector_count)
        info = next(r for r in sensor_info if r[0] == sid)
        _, type, sens, sinr_th, gain, fmin, fmax, scount = info
        ti = type2idx[type]

        static_feats.append([ti, sens, sinr_th, gain, fmin, fmax, scount, x, y, z])
    x_static = torch.tensor(static_feats, dtype=torch.float)

    # 2) Edges are the distance between sensors
    rows = db.get_sensors_distances_by_exp(exp_id)  # (exp_id, sid1, x1, y1, z1, sid2, x2, y2, z2, dist)
    edges, attrs = [], []
    mountains = db.get_mountains_by_experiment(exp_id) # (m_id, x, y, side_length, height)
    for rec in rows:
        _, sid1, x1, y1, z1, sid2, x2, y2, z2, dist = rec

        i, j = id_to_idx[sid1], id_to_idx[sid2]

        p1 = (x1, y1, z1)
        p2 = (x2, y2, z2)
        blocked = any(segment_intersects_mountain(p1, p2, m) for m in mountains)
        val = float('inf') if blocked else dist

        edges.append([i, j]); attrs.append([val]) # undirected: add both directions
        edges.append([j, i]); attrs.append([val])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(attrs, dtype=torch.float)

    # 3) Dynamic node features and time-dependent graphs
    ups = db.get_sensor_up_time_by_experiment(exp_id)
    # (exp_id, sensor_id, sensor_up_id, start_time, end_time, sensor_noise, received_thermal_noise, sinr, active)
    times = sorted({row[3] for row in ups})

    graphs = []
    for t in times:
        prx_sum = [0.0]*N

        prx_sum = torch.zeros(N, dtype=torch.float)
        sectors_mat = torch.zeros(N, MAX_SECTORS, dtype=torch.float)

        active_ups = [u for u in ups if u[3] == t]
        for (_, sid, sup_id, *_ ) in active_ups:
            dets = db.get_sensor_detections_by_sensor_up_id(sup_id) # (det_id, sup_id, drone_id, rec_power, rec_bw, sector)
            idx = id_to_idx[sid]

            prx_list = [det[3] for det in dets]
            prx_sum[idx] = sum_powers_dbm(prx_list)

            for det in dets:
                sector = det[5]
                sectors_mat[idx, sector-1] = 1.0


        x_dyn = torch.cat([prx_sum.view(N,1), sectors_mat], dim=1)       # [N, 1+MAX_SECTORS]
        x     = torch.cat([x_static, x_dyn], dim=1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.timestamp = t
        graphs.append(data)

    return graphs

def export_all_graphs_to_one_file(db, out_pt="all_experiments_graphs.pt"):
    exp_ids = [row[0] for row in db.get_experiments()]

    all_graphs = {}
    for exp_id in exp_ids:
        print(f"Building graphs for experiment {exp_id}…")
        graphs = build_experiment_graphs(db, exp_id)
        all_graphs[exp_id] = graphs

    print(f"Saving all {len(exp_ids)} experiments into `{out_pt}`")
    torch.save(all_graphs, out_pt)


def export_drone_labels(db, csv_path, max_drones=MAX_DRONES):
    """
    Writes a CSV with one row per (experiment, timestamp):
      exp_id, timestamp, num_drones,
      drone_ids (padded list of length max_drones),
      drone_positions (padded list of length max_drones of (x,y,z) tuples)
    """
    # collect all experiments
    exp_ids = [row[0] for row in db.get_experiments()]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id", "timestamp", "num_drones",
            "drone_ids", "drone_positions"
        ])

        for exp_id in exp_ids:
            rows = db.get_drone_exp_time_by_exp_id(exp_id)
            # group by timestamp
            by_time = defaultdict(list)
            for _, drone_id, _, ts, _, ex, ey, ez, *_ in rows:
                # ensure datetime
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                by_time[ts].append((drone_id, (ex, ey, ez)))

            for ts in sorted(by_time):
                entries = by_time[ts]
                # sort by drone_id for consistency
                entries.sort(key=lambda x: x[0])

                num = len(entries)
                # pad ids and positions
                ids = [d for d,_ in entries] + [-1]*(max_drones - num)
                pos = [p for _,p in entries] + [(0.0,0.0,0.0)]*(max_drones - num)

                writer.writerow([
                    exp_id,
                    ts.isoformat(),
                    num,
                    ids,
                    pos
                ])

    print(f"Exported drone label CSV to {csv_path}")


def visualize_graph_series(graphs):
    """
    Animate a time series of PyTorch-Geometric Data graphs.

    Nodes are sensors, positioned by their static x,y coords (features 6,7).
    Each frame shows:
      - Node labels (sensor indices) drawn below nodes.
      - Dynamic received power (last feature) drawn above nodes.
      - Edge labels indicating distance (edge_attr).
    """
    if not graphs:
        raise ValueError("Empty graph list")

    # Build fixed Graph structure from first snapshot
    data0 = graphs[0]
    G = nx.Graph()
    N = data0.x.shape[0]
    # Positions from static features: indices 6 & 7
    pos = { i: (float(data0.x[i,6]), float(data0.x[i,7])) for i in range(N) }

    # Add nodes
    for i in range(N):
        G.add_node(i)
    # Add edges (undirected)
    edge_idx = data0.edge_index.t().tolist()
    edge_attr = data0.edge_attr.tolist()
    for (u, v), attr in zip(edge_idx, edge_attr):
        # add only one direction
        if u < v:
            G.add_edge(u, v, weight=attr[0])

    # Determine vertical offset for labels
    y_vals = [y for _, y in pos.values()]
    y_range = max(y_vals) - min(y_vals) if y_vals else 1.0
    offset = y_range * 0.05

    fig, ax = plt.subplots()
    ax.set_axis_off()

    def update(frame_idx):
        ax.clear()
        ax.set_title(f"Time: {graphs[frame_idx].timestamp}")
        ax.set_axis_off()

        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=300)

        # Static node labels (sensor index) below
        lbls = {i: str(i) for i in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels=lbls, font_size=8,
            verticalalignment='bottom', ax=ax
        )

        # Dynamic PRX labels above
        prx_vals = graphs[frame_idx].x[:, -1].tolist()
        pos_prx = { i: (x, y + offset) for i, (x, y) in pos.items() }
        prx_lbls = {i: f"{prx_vals[i]:.1f}" for i in G.nodes()}
        nx.draw_networkx_labels(
            G, pos_prx, labels=prx_lbls, font_size=8,
            verticalalignment='top', ax=ax
        )

        # Edge distance labels
        edge_labels = { (u,v): f"{d['weight']:.1f}" for u,v,d in G.edges(data=True) }
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=6, ax=ax
        )

        # Set equal aspect
        ax.set_aspect('equal')

    ani = FuncAnimation(
        fig, update,
        frames=len(graphs),
        interval=500,
        repeat=True
    )
    plt.show()
