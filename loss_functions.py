import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import torch.nn as nn

from utils import *

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

def graph_reconstruction_loss_weighted(pred_node_feats, true_node_feats, pred_node_count_logits,
                                       true_node_counts, feat_weights=None):

    batch_size = len(true_node_feats)
    node_loss = 0.0
    count_loss = 0.0

    D = pred_node_feats.size(-1)

    if feat_weights is None:
        w = torch.ones(D, device=pred_node_feats.device, dtype=pred_node_feats.dtype)
    else:
        w = torch.as_tensor(feat_weights, dtype=pred_node_feats.dtype, device=pred_node_feats.device)

    for i in range(batch_size):
        n = true_node_counts[i]

        pred_feats = pred_node_feats[i, :n, :]
        true_feats = true_node_feats[i].to(pred_feats.device)

        # weighted MSE over features
        diff2 = (pred_feats - true_feats) ** 2
        weighted = diff2 * w
        node_loss += weighted.mean()

        count_target = torch.tensor([n], dtype=torch.long, device=pred_node_count_logits.device)
        count_loss += F.cross_entropy(pred_node_count_logits[i:i + 1], count_target)

    return (node_loss + count_loss) / batch_size


def hungarian_match_loss(pred_positions, gt_positions):
    """
    Compute MSE loss between matched predicted and GT positions using Hungarian algorithm.
    pred_positions: (P, 3) tensor
    gt_positions: (G, 3) tensor
    Returns:
        mse_loss (scalar tensor)
    """
    if pred_positions.size(0) == 0 or gt_positions.size(0) == 0:
        return torch.tensor(0.0, device=pred_positions.device)

    # Compute cost matrix (Euclidean distances)
    diff = pred_positions.unsqueeze(1) - gt_positions.unsqueeze(0)  # [P, G, 3]
    cost_matrix = torch.norm(diff, dim=2)  # [P, G]

    # Hungarian algorithm
    cost_matrix_cpu = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_cpu)

    matched_pred = pred_positions[row_ind]
    matched_gt = gt_positions[col_ind]

    mse_loss = nn.MSELoss()(matched_pred, matched_gt)
    return mse_loss

def temporal_reconstruction_loss_weighted(pred_feats_list, pred_counts_list, seq, feat_weights=None, time_weights=None):
    """
    pred_feats_list: list[T] of [Nmax, F]
    pred_counts_list: list[T] of [Nmax+1]
    seq: list[T] of PyG Data
    feat_weights: None or 1D array/tensor of length F (same as your existing function)
    time_weights: None or list/1D tensor of length T (optional emphasis on later steps)
    """
    T = len(seq)
    losses = []

    for t in range(T):
        pred_feats_t = pred_feats_list[t].unsqueeze(0)        # [1, Nmax, F] to satisfy your function's [B,...]
        pred_counts_t = pred_counts_list[t].unsqueeze(0)      # [1, Nmax+1]
        true_feats_t, true_count_t = extract_graph_targets(seq[t])

        # adapt to your API: expects list of true feats/counts (length B)
        loss_t = graph_reconstruction_loss_weighted(
            pred_node_feats=pred_feats_t,
            true_node_feats=true_feats_t,
            pred_node_count_logits=pred_counts_t,
            true_node_counts=true_count_t,
            feat_weights=feat_weights
        )
        if time_weights is not None:
            w = float(time_weights[t])
            loss_t = loss_t * w
        losses.append(loss_t)

    if time_weights is None:
        return torch.stack(losses).mean()
    else:
        tw = torch.tensor(time_weights, dtype=losses[0].dtype,
                          device=losses[0].device)
        return (torch.stack(losses) * tw).sum() / tw.sum()
