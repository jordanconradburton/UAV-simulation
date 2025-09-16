import torch
import torch.nn as nn

class SequenceGraphDecoder(nn.Module):
    """
    Wraps the BasicGraphDecoder to process a sequence of graph embeddings.
    Accepts either:
      - Tensor [T, D]  (single sequence, no batch)
      - Tensor [B, T, D]  (optional future extension; here we stick to B=1 in loaders)
    Returns:
      - pred_feats_list: list of length T with tensors [Nmax, F]
      - pred_count_logits_list: list of length T with tensors [Nmax+1]
    """
    def __init__(self, base_decoder: nn.Module):
        super().__init__()
        self.base = base_decoder  # your BasicGraphDecoder

    def forward(self, g_seq):
        # Normalize to [T, D]
        if g_seq.dim() == 3:  # [B, T, D]
            assert g_seq.size(0) == 1, "This wrapper expects batch_size=1. Use a proper collate for >1."
            g_seq = g_seq[0]  # -> [T, D]

        T = g_seq.size(0)
        pred_feats_list = []
        pred_count_list = []
        for t in range(T):
            g_t = g_seq[t].unsqueeze(0)   # [1, D]
            node_feats_t, count_logits_t = self.base(g_t)   # [1, Nmax, F], [1, Nmax+1]
            pred_feats_list.append(node_feats_t[0])         # [Nmax, F]
            pred_count_list.append(count_logits_t[0])       # [Nmax+1]
        return pred_feats_list, pred_count_list
