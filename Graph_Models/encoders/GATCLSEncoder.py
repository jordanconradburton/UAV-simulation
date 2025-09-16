import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import CLSPool

class GATCLSEncoder(nn.Module):
    """
    2× GATv2 layers + GlobalAttention pooling.
    Returns graph embedding [B, embed_dim].
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        attn_heads: int = 4,
        embed_dim: int = 64,
        dropout: float = 0.2,
        activation = F.elu
    ):
        super().__init__()
        # concat=False keeps feature dims == hidden_channels
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=attn_heads, dropout=dropout, concat=False)
        self.gat2 = GATv2Conv(hidden_channels, hidden_channels, heads=attn_heads, dropout=dropout, concat=False)

        # CLS Pool
        self.pool = CLSPool.CLSPool(in_dim=hidden_channels, out_dim=embed_dim, attn_dim=hidden_channels, dropout=0.1)
        self.activation = activation

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.activation(self.g1(x, edge_index))
        x = self.activation(self.g2(x, edge_index))
        g = self.cls_pool(x, batch)  # [B, embed_dim]
        return g
