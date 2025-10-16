import torch
import torch.nn as nn
from torch_geometric_temporal.nn.attention.astgcn import ASTGCNBlock
from torch_geometric.nn import global_mean_pool

#TODO: play with the num_timesteps - start with 12, but try {2, 5, 10}

class ASTGCNEncoder(nn.Module):
    """
    Wraps the ASTGCN block to produce a graph-level embedding per window.
    Input:
      x:     [B, N_max, F_in, T]
      edge_index:
    Output:
      g_emb: [B, embed_dim]
    """
    def __init__(self,
                 in_channels: int,
                 num_nodes: int = 20,
                 num_timesteps: int = 12,
                 K: int = 3,
                 nb_chev_filter: int = 64,
                 nb_time_filter: int = 64, #nb_time_filter is also the output embedding size
                 time_strides: int = 1):
        super().__init__()

        self.block = ASTGCNBlock(
            in_channels=in_channels,
            K=K,
            nb_chev_filter=nb_chev_filter,
            nb_time_filter=nb_time_filter,
            time_strides=time_strides,
            num_of_timesteps=num_timesteps,
            num_of_vertices=num_nodes,
        )


    def forward(self, x, A):
        """
        x: [B, N, F_in, T]
        A: [N, N] adjacency
        returns: [B, nb_time_filter]
        """
        # 1. Run ASTGCN block
        z = self.block(x, A)  # [B, N, nb_time_filter, T_out]

        # 2. Mean over time
        z = z.mean(-1)  # [B, N, nb_time_filter]

        # 3. Mean over nodes (graph-level embedding)
        g_emb = z.mean(1)  # [B, nb_time_filter]

        return g_emb

