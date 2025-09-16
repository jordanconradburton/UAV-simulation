import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from torch_geometric_temporal.nn.recurrent import EvolveGCNO, EvolveGCNH

class EvolveGCNEncoderMean(nn.Module):
    """
    EvolveGCN (O or H variant) + mean pooling → per-timestep graph embeddings.

    Forward input:
        sequence: List[Data] (one experiment's full sequence, sorted)

    Forward output:
        graph_embs: Tensor [T, embed_dim]  (one embedding per snapshot)
    """
    def __init__(self,
                 in_channels: int,
                 embed_dim: int = 64,
                 variant: str = "O",  # "O" or "H"
                 activation=F.relu):
        super().__init__()
        if variant.upper() == "O":
            self.cell = EvolveGCNO(in_channels)
        elif variant.upper() == "H":
            self.cell = EvolveGCNH() #num nodes, num in channels
        else:
            raise ValueError("variant must be 'O' or 'H'")

        self.activation = activation
        self.pool = global_mean_pool
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, sequence: List[Data]) -> torch.Tensor:
        """
        Processes the entire experiment sequence step-by-step.
        Returns [T, embed_dim].
        """
        device = next(self.parameters()).device
        graph_embs = []

        # EvolveGCN maintains its parameters/hidden internally step-by-step.
        # We simply iterate time-forward.
        for d in sequence:
            x = d.x.to(device)                       # [N_t, F]

            batch = getattr(d, "batch", None)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
            else:
                batch = batch.to(device)

            edge_index = d.edge_index.to(device)     # [2, E_t]
            edge_weight = getattr(d, "edge_attr", None)
            #TODO: check
            if edge_weight is not None and edge_weight.dim() == 2 and edge_weight.size(1) == 1:
                edge_weight = edge_weight.view(-1).to(device)  # [E_t]
            else:
                edge_weight = None

            # One EvolveGCN step → node embeddings [N_t, H]
            x = self.cell(x, edge_index, edge_weight=edge_weight)
            x = self.activation(x)

            g = global_mean_pool(x, batch)          # [1, H] for single graph
            g = self.proj(g)                        # [1, D]
            graph_embs.append(g)

        return torch.cat(graph_embs, dim=0)
