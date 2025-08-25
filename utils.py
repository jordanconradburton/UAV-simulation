import ast
import csv
from datetime import datetime
import torch
from torch_geometric.data import Batch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pandas as pd
import json

from Simulator.simulation_utils import FREQUENCY_BANDS_HZ
from Phase2_Decoder.LabeledEmbeddingDataset import LabeledEmbeddingDataset

def create_encoder_dataloaders(dataset, train_ratio=0.7, val_ratio=0.15, batch_size=1, shuffle=True):
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

def create_decoder_dataloaders(train_embeddings_file, val_embeddings_file, test_embeddings_file, batch_size=16, shuffle=True):
    train_dataset = LabeledEmbeddingDataset(train_embeddings_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_dataset = LabeledEmbeddingDataset(val_embeddings_file)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = LabeledEmbeddingDataset(test_embeddings_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Dataloaders created")

    embedding_dim = torch.load(train_embeddings_file)["embeddings"].shape[1]

    return train_loader, val_loader, test_loader, embedding_dim


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


def inspect_embeddings(path, save_dir=None, make_plots=True):
    d = torch.load(path)
    E = d["embeddings"]       # [N,D]
    counts = d["counts"]      # [N]
    pos = d["positions"]      # [N, max_drones, 3]

    E_np = E.numpy()
    means = E_np.mean(0)
    stds = E_np.std(0)
    overall_std = E_np.std()

    # Correlation with position of drone-0 (if you currently have one drone)
    p0 = pos[:,0,:].numpy()   # [N,3]
    corr_x = np.corrcoef(E_np.T, p0[:,0], rowvar=True)[-1,:-1]
    corr_y = np.corrcoef(E_np.T, p0[:,1], rowvar=True)[-1,:-1]
    corr_z = np.corrcoef(E_np.T, p0[:,2], rowvar=True)[-1,:-1]

    med_abs_corr = {
        "x": float(np.median(np.abs(corr_x))),
        "y": float(np.median(np.abs(corr_y))),
        "z": float(np.median(np.abs(corr_z))),
    }

    print("Embeddings mean/std per dim:")
    print(np.round(means, 4))
    print(np.round(stds, 4))
    print("Overall std:", overall_std)
    print("Median |corr(emb, x)|:", med_abs_corr["x"])
    print("Median |corr(emb, y)|:", med_abs_corr["y"])
    print("Median |corr(emb, z)|:", med_abs_corr["z"])

    # ---- Save to disk ----
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        df = pd.DataFrame({
            "dim": np.arange(E_np.shape[1]),
            "mean": means,
            "std": stds,
            "corr_x": corr_x,
            "corr_y": corr_y,
            "corr_z": corr_z,
            "abs_corr_x": np.abs(corr_x),
            "abs_corr_y": np.abs(corr_y),
            "abs_corr_z": np.abs(corr_z),
        })

        per_dim_csv = os.path.join(save_dir, f"embeddings_per_dim{timestamp}.csv")
        df.to_csv(per_dim_csv, index=False)

        summary = {
            "n_samples": int(E_np.shape[0]),
            "embedding_dim": int(E_np.shape[1]),
            "overall_std": float(overall_std),
            "median_abs_corr": med_abs_corr,
            "top5_dims_by_abs_corr_x": df["abs_corr_x"].nlargest(5).index.tolist(),
            "top5_dims_by_abs_corr_y": df["abs_corr_y"].nlargest(5).index.tolist(),
            "top5_dims_by_abs_corr_z": df["abs_corr_z"].nlargest(5).index.tolist(),
        }

        with open(os.path.join(save_dir, f"embeddings_summary_{timestamp}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if make_plots:
            # Histograms of |corr| per axis
            for axis_name, arr in [("x", np.abs(corr_x)), ("y", np.abs(corr_y)), ("z", np.abs(corr_z))]:
                plt.hist(arr, bins=30)
                plt.title(f"|corr(embedding dim, {axis_name})|")
                plt.xlabel("absolute correlation")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"abs_corr_hist_{axis_name}_{timestamp}.png"), dpi=200)
                plt.close()

            # Mean/std per dim (quick diagnostic)
            plt.plot(means)
            plt.title("Embedding means per dimension")
            plt.xlabel("dim")
            plt.ylabel("mean")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"means_per_dim_{timestamp}.png"), dpi=200)
            plt.close()

            plt.plot(stds)
            plt.title("Embedding stds per dimension")
            plt.xlabel("dim")
            plt.ylabel("std")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"stds_per_dim_{timestamp}.png"), dpi=200)
            plt.close()

        print(f"📝 Saved metrics to {save_dir} (CSV + JSON{', plots' if make_plots else ''}).")

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
            ts = datetime.fromisoformat(ts)
            ts = ts.strftime("%Y-%m-%d %H:%M:%S.%f")

            num = int(row["num_drones"])

            positions = ast.literal_eval(row["drone_positions"])
            if len(positions) != max_drones:
                positions += [(0.0, 0.0, 0.0)] * (max_drones - len(positions))

            labels[(exp_id, ts)] = {
                "num_drones": num,
                "positions": positions
            }
    return labels

def _reshape_pos_from_batch(pos_gt, num_graphs: int, max_drones: int):
    """
    PyG's Batch concatenates custom fields across graphs.
    We want [B, max_drones, 3] per batch.
    """
    if pos_gt.ndim == 3 and pos_gt.shape[0] == num_graphs:
        # Already [B, max_drones, 3]
        return pos_gt
    if pos_gt.ndim == 2 and pos_gt.shape[-1] == 3:
        total = pos_gt.shape[0]
        expected = num_graphs * max_drones
        if total == expected:
            return pos_gt.view(num_graphs, max_drones, 3)
        if num_graphs == 1 and total == max_drones:
            return pos_gt.unsqueeze(0)  # [1, max_drones, 3]
        raise RuntimeError(f"Cannot reshape pos_gt with shape {tuple(pos_gt.shape)} "
                           f"to [B={num_graphs}, {max_drones}, 3].")
    raise RuntimeError(f"Unexpected pos_gt shape {tuple(pos_gt.shape)}")

@torch.no_grad()
def save_embeddings_with_labels(encoder, dataloader, device, save_path, max_drones=8):
    encoder.eval()
    all_embeds, all_counts, all_positions = [], [], []

    for data in dataloader:
        data = data.to(device)
        emb = encoder(data)                 # [B, D]
        B = emb.size(0)

        # 2) counts: depending on how you stored y, batched y is often [B] or [B,1]
        y = data.y
        if y.ndim == 2 and y.size(1) == 1:
            y = y.view(-1)                  # [B]
        elif y.ndim == 0:                   # single scalar
            y = y.unsqueeze(0)
        assert y.numel() == B, f"y has {y.numel()} elements, expected {B}"

        # 3) positions: fix PyG concatenation -> [B, max_drones, 3]
        pos = data.pos_gt
        pos = _reshape_pos_from_batch(pos, num_graphs=B, max_drones=max_drones)

        # move to CPU and append
        all_embeds.append(emb.detach().cpu())      # [B, D]
        all_counts.append(y.detach().cpu())        # [B]
        all_positions.append(pos.detach().cpu())   # [B, max_drones, 3]

    out = {
        "embeddings": torch.cat(all_embeds, dim=0),         # [N, D]
        "counts":     torch.cat(all_counts, dim=0),         # [N]
        "positions":  torch.cat(all_positions, dim=0),      # [N, max_drones, 3]
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(out, save_path)
    print(f"✅ Saved embeddings+labels to {save_path} with shapes:",
          tuple(out['embeddings'].shape), tuple(out['counts'].shape), tuple(out['positions'].shape))


def sanity_check_embeddings_file(path, max_drones=8):
    import torch
    d = torch.load(path, map_location="cpu")
    emb, cnt, pos = d["embeddings"], d["counts"], d["positions"]
    print(f"[sanity] embeddings: {tuple(emb.shape)}  counts: {tuple(cnt.shape)}  positions: {tuple(pos.shape)}")
    if pos.ndim == 2 and pos.shape[-1] == 3:
        # Classic collapse: [N, 3] instead of [N, max_drones, 3]
        N = pos.shape[0]
        if N % max_drones == 0:
            print(f"⚠️ positions collapsed; reshaping [{N}, 3] -> [{N//max_drones}, {max_drones}, 3]")
            pos = pos.view(N // max_drones, max_drones, 3)
            d["positions"] = pos
            torch.save(d, path)
            print("✅ File rewritten with corrected positions shape:", tuple(pos.shape))
        else:
            raise RuntimeError(
                f"positions has shape {tuple(pos.shape)} and N%max_drones != 0 "
                f"(N={N}, max_drones={max_drones}) — cannot safely reshape."
            )
    elif pos.ndim != 3 or pos.shape[-2:] != (max_drones, 3):
        raise RuntimeError(f"Unexpected positions shape {tuple(pos.shape)} — expected [N, {max_drones}, 3].")
