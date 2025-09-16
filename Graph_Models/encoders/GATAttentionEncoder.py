import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, AttentionalAggregation

class GATAttentionEncoder(nn.Module):
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

        # Global attention needs a gate network: R^H -> R (per-node gate)
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.pool = AttentionalAggregation(self.gate_nn)

        self.proj = nn.Linear(hidden_channels, embed_dim)
        self.activation = activation

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = self.activation(self.gat1(x, edge_index))
        x = self.activation(self.gat2(x, edge_index))
        hg = self.pool(x, batch)          # [B, hidden_channels]
        z  = self.proj(hg)                 # [B, embed_dim]
        return z
