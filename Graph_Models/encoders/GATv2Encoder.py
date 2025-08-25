import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import torch_geometric

class BasicGATv2Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, attn_heads, embed_dim=128, dropout=0.3, activation=F.relu):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=attn_heads, concat=True, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_channels * attn_heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.pool = torch_geometric.nn.global_mean_pool

        self.embedding_head = torch.nn.Linear(hidden_channels, embed_dim)
        self.activation = activation

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.activation(self.gat1(x, edge_index))
        x = self.activation(self.gat2(x, edge_index))
        g = self.pool(x, batch)  # [B, hidden_channels]
        embedding = self.embedding_head(g)  # [B, embed_dim]
        return embedding