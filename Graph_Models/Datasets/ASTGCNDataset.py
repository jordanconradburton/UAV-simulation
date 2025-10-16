import torch
from torch.utils.data import Dataset
from torch_geometric.utils import to_dense_adj
from utils import normalize_adj, normalize_features, load_labels

class ASTGCNDataset(Dataset):
    """
    Prepares sequences of graph snapshots for temporal GNNs like ASTGCN.

    Each item = one sequence of `window_size` graphs (and optionally a target graph)
    Returned tensors:
        x_seq: [B=1, N_max, F_in, T_in]
        A_hat: [N_max, N_max]  (normalized adjacency)
        mask:  [N_max] (1 for real nodes, 0 for padded ones)
    """

    def __init__(self, graphs_path, label_file, window_size=12, stride=1, max_nodes=None, max_drones=8):
        """
        Args:
            graphs: list of `torch_geometric.data.Data` (time-ordered)
            window_size: number of snapshots per sequence
            stride: step size between sequences
            max_nodes: pad all graphs to this number of nodes (auto-computed if None)
            normalize_adj: whether to row-normalize adjacency
        """
        super().__init__()
        self.all_graphs = torch.load(graphs_path, weights_only=False)
        self.labels = load_labels(label_file)
        self.window_size = window_size
        self.stride = stride
        self.max_drones = max_drones

        # Normalize node features for each graph
        for exp_id, graph_list in self.all_graphs.items():
            for g in graph_list:
                g.x = normalize_features(g.x)

        # Determine N_max (largest graph size)
        self.N_max = max_nodes or max(g.x.size(0) for g in self.all_graphs)

        # Precompute adjacency (dense) for each snapshot and normalize
        self.adjs = {}
        for exp_id, graphs in self.all_graphs.items():
            A_list = [to_dense_adj(g.edge_index, max_num_nodes=self.N_max)[0] for g in graphs]
            A_list = [normalize_adj(A) for A in A_list]
            self.adjs[exp_id] = A_list

        # Build sequence index list (exp_id, start_idx)
        self.sequence_index = []
        for exp_id, graphs in self.all_graphs.items():
            num_graphs = len(graphs)
            for start in range(0, num_graphs - window_size + 1, stride):
                self.sequence_index.append((exp_id, start))

    def __len__(self):
        # Number of sliding windows
        return max(0, len(self.all_graphs) - self.window_size + 1) // self.stride

    def __getitem__(self, idx):
        """
        Returns:
            x_seq: [1, N_max, F_in, T_in]
            A_hat: [N_max, N_max]
            mask:  [N_max]
        """
        exp_id, start = self.sequence_index[idx]
        graphs = self.all_graphs[exp_id]
        end = start + self.window_size
        window = graphs[start:end]
        T_in = len(window)

        F_in = window[0].x.size(1)
        x_seq = torch.zeros((1, self.N_max, F_in, T_in), dtype=torch.float)
        mask  = torch.zeros((self.N_max,), dtype=torch.float)

        for t, g in enumerate(window):
            N = g.x.size(0)
            x_seq[0, :N, :, t] = g.x
            mask[:N] = 1  # mark valid nodes

        A_hat = self.adjs[0]

        # labels for Phase 2
        ts = str(window[-1].timestamp).strip()
        key = (exp_id, ts)
        if key not in self.labels:
            raise KeyError(f"No label found for (exp_id={exp_id}, ts={ts})")

        label_info = self.labels[key]
        count = label_info["num_drones"]
        pos_tensor = torch.tensor(label_info["positions"], dtype=torch.float)
        mask_gt = torch.zeros(self.max_drones)
        mask_gt[:count] = 1.0

        label = {
            "count": torch.tensor([count], dtype=torch.float),
            "positions": pos_tensor,
            "mask_gt": mask_gt,
            "exp_id": exp_id,
            "timestamp": ts,
        }

        return x_seq, A_hat, mask, label
