import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class TGATEncoder(nn.Module):
    """
    Temporal graph encoder:
      1) Add sinusoidal time encodings (shared per-graph-frame) to each node's features.
      2) Spatial GATv2 over the batched graph at each frame (weights shared across time) -> spatial node embeddings
      3) Temporal GRU across frames, per node (learn historical dynamics) -> temporal node embeddings
      4) Pool nodes -> graph embedding for the *last* frame in the window.
    Returns: graph embedding [B_graphs, embed_dim]
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        embed_dim: int = 64,
        heads: int = 4,
        time_dim: int = 16,         # dimensionality of time encoding per frame
        gru_hidden: int = 128,
        dropout: float = 0.2,
        window: int = 2
        # edge_dim: int | None = None # set to 1 if you pass distance as edge_attr, else None
    ):
        super().__init__()
        self.hidden = hidden_channels
        self.gru_hidden = gru_hidden or hidden_channels
        self.time_dim = time_dim
        self.dropout = dropout
        self.window = window
        self.feature_count = in_channels

        self.gat1 = GATv2Conv(
            in_channels = in_channels + time_dim,   # number of node features + timestamp encoding
            out_channels = hidden_channels,
            heads = heads,
            concat = True,
            dropout = dropout
        )
        self.gat2 = GATv2Conv(
            in_channels = hidden_channels * heads,
            out_channels = hidden_channels,
            heads = 1,
            concat = False,
            dropout = dropout
        )

        # Temporal GRU over frames, per node
        # treats each node as a "batch"
        self.gru = nn.GRU(
            input_size = hidden_channels,
            hidden_size = self.gru_hidden,
            batch_first = True   # [N_total_nodes, T, H] in -> out same
        )

        self.pool = global_mean_pool

        # Final projection to embedding dim
        self.embedding_head = nn.Sequential(
            nn.Linear(self.gru_hidden, embed_dim),
        )

        self.activation = F.elu


    def _build_time_encoding(self, T: int, device):
        """
        Sinusoidal time encoding for relative frame indices in the window.
        t_norm in [-1, 0], equally spaced; produces [T, time_dim].
        """
        if T == 1 or self.time_dim == 0:
            return torch.zeros(1, 0, device=device)  # empty time feat

        # relative positions: -1 .. 0 (older -> newer)
        t_idx = torch.arange(T, device=device, dtype=torch.float)
        t_norm = (t_idx - (T - 1)) / max(1, (T - 1))  # [-1, 0]

        # build frequencies (log-spaced)
        half = self.time_dim // 2
        if half == 0:
            omegas = torch.empty(0, device=device)
        else:
            # periods range ~[2, T*2] frames; tweak as you like
            periods = torch.logspace(math.log10(2.0), math.log10(max(2.0, 2.0*T)), steps=half, device=device)
            omegas = 2 * math.pi / periods  # [half]

        # [T, half]
        if half > 0:
            args = t_norm.unsqueeze(1) * omegas.unsqueeze(0)
            sin = torch.sin(args)
            cos = torch.cos(args)
            tenc = torch.cat([sin, cos], dim=1)  # [T, 2*half]
        else:
            tenc = torch.zeros(T, 0, device=device)

        # if time_dim odd, pad one zero column
        if tenc.size(1) < self.time_dim:
            pad = self.time_dim - tenc.size(1)
            tenc = torch.cat([tenc, torch.zeros(T, pad, device=device)], dim=1)

        return tenc  # [T, time_dim]


    def forward(self, data):
        """
        Inputs:
          data.x:     [N_total_nodes, F * T]  (from Temporal Dataset)
          data.edge_index, data.edge_attr (optional)
          data.batch: [N_total_nodes]  graph ids for pooling

        Output:
          graph_embedding: [B_graphs, embed_dim]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch_vec = data.batch

        T = self.window
        NB = x.size(0)

        # reshape to [N_total_nodes, T, F]
        x_seq = x.view(NB, T, self.feature_count).contiguous()

        # build shared time enc for the window: [T, time_dim] -> expand to [N_total_nodes, T, time_dim]
        tenc = self._build_time_encoding(T, x.device)                 # [T, time_dim]
        tenc = tenc.unsqueeze(0).expand(NB, -1, -1)                   # [NB, T, time_dim]
        x_seq = torch.cat([x_seq, tenc], dim=2)               # [NB, T, F + time_dim]

        # ----- spatial over each frame -----
        h_per_t = []
        for t in range(T):
            x_t = x_seq[:, t, :]  # [NB, F_eff]
            h = self.activation(self.gat1(x_t, edge_index, edge_attr=edge_attr))
            # h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.activation(self.gat2(h, edge_index, edge_attr=edge_attr))
            h_per_t.append(h)

        # stack to [NB, T, H] for GRU
        h_seq = torch.stack(h_per_t, dim=1)  # [NB, T, H]

        # ----- temporal over frames (per node) -----
        h_temporal, _ = self.gru(h_seq)      # [NB, T, H_gru]
        h_last = h_temporal[:, -1, :]        # [NB, H_gru]  <-- embedding for the *current* frame informed by history

        g = self.pool(h_last, batch_vec)

        # projection
        embedding = self.embedding_head(g)                 # [B, D]
        return embedding
