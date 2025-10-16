import torch.nn as nn

class ASTGCNAutoencoder(nn.Module):
    """
    Many-to-many: sequence in -> per-step graph embeddings -> per-step reconstructions
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, A):

        g_emb = self.encoder(x, A)
        pred_feats_list, pred_counts_list = self.decoder(g_emb)  # lists length T
        return pred_feats_list, pred_counts_list
