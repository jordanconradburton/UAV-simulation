import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class BasicGAT(torch.nn.Module):
    """
    A simple 2-layer Graph Attention Network.
    - in_channels: dimensionality of input node features (static + dynamic)
    - hidden_channels: number of hidden units per head
    - out_channels: dimensionality of output per node
    - heads: number of attention heads in first layer
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        # First GATConv: produces hidden_channels * heads features
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        # Second GATConv: outputs out_channels features per node
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )

    def forward(self, x, edge_index, edge_attr=None):
        # x: [N, in_channels]
        # edge_index: [2, E]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # [N, out_channels]