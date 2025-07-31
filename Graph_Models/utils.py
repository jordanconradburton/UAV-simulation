import ast
import csv
from datetime import datetime
import torch
from torch.utils.data import random_split
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from Simulator.simulation_utils import FREQUENCY_BANDS_HZ

def create_dataloaders(dataset, train_ratio=0.7, val_ratio=0.15, batch_size=1, shuffle=True):
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("Dataloaders created")

    return train_loader, val_loader, test_loader

def graph_reconstruction_loss(pred_node_feats, true_node_feats, pred_node_count_logits, true_node_counts, max_nodes):
    """
    Args:
        pred_node_feats: [B, max_nodes, D]
        true_node_feats: list of [N_i, D] (variable sizes)
        pred_node_count_logits: [B, max_nodes]
        true_node_counts: [B] (ints)
    Returns:
        scalar loss
    """
    batch_size = len(true_node_feats)
    node_loss = 0.0
    count_loss = 0.0

    for i in range(batch_size):
        n = true_node_counts[i]

        pred_feats = pred_node_feats[i, :n, :]
        true_feats = true_node_feats[i].to(pred_feats.device)
        node_loss += F.mse_loss(pred_feats, true_feats)

        count_target = torch.tensor([n], dtype=torch.long, device=pred_node_count_logits.device)
        count_loss += F.cross_entropy(pred_node_count_logits[i:i+1], count_target)

    return (node_loss + count_loss) / batch_size


def extract_graph_targets(data_batch):
    """
    Assumes data_batch is a list of unbatched graphs (batch_size = 1), or a PyG `Batch` you split manually.
    Returns:
        - true_node_feats: list of [N_i, F] tensors
        - true_node_counts: list of ints [N1, N2, ..., NB]
    """
    if isinstance(data_batch, list):
        true_node_feats = [d.x for d in data_batch]
        true_node_counts = [d.num_nodes for d in data_batch]
    else:
        # If it's a PyG Batch object (from batch_size > 1), split it
        from torch_geometric.data import Batch
        true_node_feats = Batch.to_data_list(data_batch)
        true_node_counts = [d.num_nodes for d in true_node_feats]
        true_node_feats = [d.x for d in true_node_feats]

    return true_node_feats, true_node_counts

def get_band_index(fmin, fmax):
    for i, (lo, hi) in enumerate(FREQUENCY_BANDS_HZ):
        if lo <= fmin <= hi or lo <= fmax <= hi:
            return i + 1  # IDs from 1 to 6
    return 0

def normalize_features(x):
    """
    Normalize node features tensor x. Assumes layout:
    [type, sensitivity, sinr, fmin, fmax, sector_count, x, y, z, gains..., prx]
    """
    x_norm = x.clone()

    # Feature index map
    sens_idx = 1
    sinr_idx = 2
    fmin_idx = 3
    fmax_idx = 4
    band_idx = fmin_idx
    scount_idx = 5
    x_idx, y_idx, z_idx = 6, 7, 8
    gain_start = 9
    prx_idx = -1  # assuming prx is last

    # Normalize static features
    x_norm[:, sens_idx] = (x[:, sens_idx] + 100) / 20.0 # (x-min)/(max-min)
    x_norm[:, sinr_idx] = (x[:, sinr_idx] - 5) / 10.0   # (x-min)/(max-min)
    x_norm[:, scount_idx] = x[:, scount_idx] / 6.0      # x/max
    x_norm[:, x_idx] = x[:, x_idx] / 5000.0             # x/max
    x_norm[:, y_idx] = x[:, y_idx] / 5000.0             # x/max
    x_norm[:, z_idx] = x[:, z_idx] / 100.0              # x/max

    # Replace fmin/fmax with normalized band ID
    for i in range(x.shape[0]):
        fmin = x[i, fmin_idx].item()
        fmax = x[i, fmax_idx].item()
        band_id = get_band_index(fmin, fmax)
        x_norm[i, band_idx] = band_id / 6.0

    # Drop fmax (its info is redundant now)
    x_norm = torch.cat([x_norm[:, :fmax_idx], x_norm[:, fmax_idx+1:]], dim=1)

    # Normalize gains: [-1, 3] → [0, 1]
    gain_end = x.shape[1] - 1
    x_norm[:, gain_start:gain_end] = (x[:, gain_start:gain_end] + 1.0) / 4.0

    # Normalize received power (linear method)
    x_norm[:, prx_idx] = (x[:, prx_idx] + 90) / 40.0 #TODO: check again for each dataset

    return x_norm

def check_feature_ranges(dataset):
    all_feats = []

    for data in dataset:
        all_feats.append(data.x)

    all_feats = torch.cat(all_feats, dim=0)

    mins = torch.min(all_feats, dim=0)[0]
    maxs = torch.max(all_feats, dim=0)[0]
    print("Feature mins:", mins)
    print("Feature maxs:", maxs)

    if not torch.isfinite(all_feats).all():
        print("⚠️ Dataset contains nan or inf in node features!")

def compare_received_power_normalizations(dataset):
    """
    Extract and compare different normalization strategies applied to
    the received power values in the dataset. Assumes received power
    is the last column of data.x.
    """
    raw_powers = []
    scaled_linear = []
    scaled_log = []
    scaled_zscore = []

    all_powers = []
    for data in dataset:
        prx = data.x[:, -1]  # last column is received power
        raw_powers.extend(prx.tolist())
        all_powers.append(prx)

    all_powers_tensor = torch.cat(all_powers)
    mean = all_powers_tensor.mean().item()
    std = all_powers_tensor.std().item()

    raw_powers = [p for p in raw_powers if np.isfinite(p)]

    for p in raw_powers:
        scaled_linear.append((p + 90) / 40.0)
        p_mW = 10 ** (p / 10)
        scaled_log.append(np.log10(p_mW + 1e-10))
        scaled_zscore.append((p - mean) / std if std > 0 else 0.0)

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(raw_powers, bins=50, color='gray')
    plt.title("Raw Received Power (dBm)")

    plt.subplot(2, 2, 2)
    plt.hist(scaled_linear, bins=50, color='blue')
    plt.title("Linear Normalized: (P + 90) / 40")

    plt.subplot(2, 2, 3)
    plt.hist(scaled_log, bins=50, color='green')
    plt.title("Log-mW Normalized: log10(mW + 1e-10)")

    plt.subplot(2, 2, 4)
    plt.hist(scaled_zscore, bins=50, color='red')
    plt.title("Z-Score Normalized")

    plt.tight_layout()
    plt.show()

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # On SLURM cluster with NVIDIA GPUs
    else:
        device = torch.device("cpu")  # Fallback

    return device

def load_labels(csv_path, max_drones=8):
    labels = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_id = int(row["exp_id"])
            ts = row["timestamp"]

            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")

            num = int(row["num_drones"])

            positions = ast.literal_eval(row["drone_positions"]) #TODO: check
            if len(positions) != max_drones:
                positions += [(0.0, 0.0, 0.0)] * (max_drones - len(positions))

            labels[(exp_id, ts)] = {
                "num_drones": num,
                "positions": positions
            }
    return labels
