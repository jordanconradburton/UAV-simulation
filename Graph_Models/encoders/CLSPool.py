import math
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

class CLSPool(nn.Module):
    """
    Learnable CLS-style pooling for graph batches.

    Given node embeddings X (shape [N, F]) and batch vector (shape [N]),
    this module produces one pooled embedding per graph (shape [B, out_dim])
    via single-head attention with a learnable CLS query vector.

    Scores:   s_i = <W_k x_i, q> / sqrt(attn_dim)
    Weights:  a_i = softmax_graphwise(s_i)
    Output:   g_b = sum_{i in graph b} a_i * W_v x_i

    Args:
        in_dim      : node embedding size F
        out_dim     : output size of pooled graph embedding (defaults to F)
        attn_dim    : attention/key/query size (defaults to F)
        dropout     : dropout applied to attention weights (0.0 by default)
        normalize_q : if True, keep q on unit sphere (helps stability)
        return_alpha: if True, forward() also returns attention weights per node
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        attn_dim: int = None,
        dropout: float = 0.0,
        normalize_q: bool = True,
        return_alpha: bool = False,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        attn_dim = attn_dim or in_dim

        # Key & Value projections
        self.key   = nn.Linear(in_dim, attn_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim,  bias=True)

        # Learnable CLS query (shared across graphs)
        self.q = nn.Parameter(torch.randn(attn_dim) * 0.02)

        self.scale = 1.0 / math.sqrt(attn_dim)
        self.drop  = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.normalize_q = normalize_q
        self.return_alpha = return_alpha

        # Optional tiny init for value bias (often helps)
        if hasattr(self.value, "bias") and self.value.bias is not None:
            nn.init.zeros_(self.value.bias)

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        """
        x     : [N, F] node embeddings
        batch : [N] graph index per node (0..B-1)

        returns:
            pooled : [B, out_dim]
            (optional) alpha : [N] attention weights per node
        """
        if x.numel() == 0:
            # Rare corner case: empty batch
            B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            pooled = x.new_zeros(B, self.value.out_features)
            return (pooled, x.new_zeros(0)) if self.return_alpha else pooled

        K = self.key(x)                     # [N, attn_dim]
        V = self.value(x)                   # [N, out_dim]

        q = self.q
        if self.normalize_q:
            q = q / (q.norm(p=2) + 1e-8)    # keep q on (approx) unit sphere

        # Scores per node, graph-wise softmax
        scores = (K * q).sum(dim=-1) * self.scale   # [N]
        alpha  = softmax(scores, batch)             # [N] softmax within each graph
        alpha  = self.drop(alpha)

        # Weighted sum per graph
        pooled = scatter_add(alpha.unsqueeze(-1) * V, batch, dim=0)  # [B, out_dim]

        return (pooled, alpha) if self.return_alpha else pooled
