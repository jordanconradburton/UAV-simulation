import torch
import torch.nn as nn

class ASTGCNDecoder(nn.Module):

    def __init__(self, embed_dim, max_nodes, dyn_feat_dim):
        super().__init__()
        self.max_nodes = max_nodes
        self.dyn_feat_dim = dyn_feat_dim

        self.node_feature_generator = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_nodes * dyn_feat_dim),
            nn.Tanh()
        )

    def forward(self, g_emb):
        """
        g_emb: [B, embed_dim]
        Returns: [B, max_nodes, dyn_feat_dim]
        """
        out = self.node_feature_generator(g_emb)          # [B, max_nodes * dyn_feat_dim]
        out = out.view(-1, self.max_nodes, self.dyn_feat_dim)  # [B, N_max, F_dyn]

        return out