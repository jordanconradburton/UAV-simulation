import torch.nn as nn

class TemporalAutoencoder(nn.Module):
    """
    Many-to-many: sequence in -> per-step graph embeddings -> per-step reconstructions
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, seq):
        """
        seq: list of T PyG Data items
        returns:
          pred_feats_list: list[T] of [Nmax, F]
          pred_counts_list: list[T] of [Nmax+1]
        """
        g_seq = self.encoder(seq)  # [T, D]
        pred_feats_list, pred_counts_list = self.decoder(g_seq)  # lists length T
        return pred_feats_list, pred_counts_list
