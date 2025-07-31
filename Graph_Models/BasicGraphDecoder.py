import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicGraphDecoder(torch.nn.Module):
    def __init__(self, embed_dim, max_nodes, out_node_dim):
        super().__init__()
        self.max_nodes = max_nodes
        self.out_node_dim = out_node_dim

        self.node_count_predictor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_nodes + 1)  # classification over 0 to max_nodes
        )

        self.node_feature_generator = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_nodes * out_node_dim),
            nn.Tanh()
        )

    def forward(self, graph_embedding):
        """
        Args:
            graph_embedding: [B, D]
        Returns:
            node_feature_preds: [B, max_nodes, out_node_dim]
            node_count_logits: [B, max_nodes]
        """
        B = graph_embedding.size(0)

        count_logits = self.node_count_predictor(graph_embedding)             # [B, max_nodes]
        raw_node_feats = self.node_feature_generator(graph_embedding)         # [B, max_nodes * D]
        node_feats = raw_node_feats.view(B, self.max_nodes, self.out_node_dim)

        return node_feats, count_logits
