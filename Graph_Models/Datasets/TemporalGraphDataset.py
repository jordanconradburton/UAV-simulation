import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Dict, List, Tuple
from utils import normalize_features, load_labels

class TemporalGraphDataset(Dataset):
    """
    Each item is a *full sequence* of graphs for a single experiment:
      returns: (exp_id, [Data_t0, Data_t1, ..., Data_t{T-1}])
    - No windowing or prefixing.
    - Graphs are sorted by timestamp.
    - Features are normalized once here for consistency.
    """
    def __init__(self, graphs_path: str, label_file: str, max_drones=8):
        super().__init__()
        # your build_experiment_graphs saved: Dict[int, List[Data]] keyed by exp_id
        self.exp_to_graphs: Dict[int, List[Data]] = torch.load(graphs_path, weights_only=False)
        self.exp_ids = sorted(self.exp_to_graphs.keys())
        self.labels = load_labels(label_file)
        self.max_drones = max_drones

        # sort and normalize per experiment
        self.sequences: Dict[int, List[Data]] = {}
        for exp_id in self.exp_ids:
            seq = sorted(self.exp_to_graphs[exp_id], key=lambda d: str(getattr(d, "timestamp")))
            # normalize node features for each snapshot
            for d in seq:
                d.x = normalize_features(d.x)
            self.sequences[exp_id] = seq

    def __len__(self):
        return len(self.exp_ids)

    def _attach_labels_and_norm(self, exp_id: int, d: Data) -> Data:
        d = d.clone()
        d.x = normalize_features(d.x)

        ts = str(d.timestamp).strip()
        key = (exp_id, ts)
        label = self.labels.get(key)
        if label is None:
            raise KeyError(f"No label found for (exp_id={exp_id}, ts={ts})")

        count = int(label["num_drones"])
        pos_tensor = torch.tensor(label["positions"], dtype=torch.float)

        d.y = torch.tensor([count], dtype=torch.float)
        d.pos_gt = pos_tensor
        d.mask_gt = torch.zeros(self.max_drones)
        d.mask_gt[:count] = 1.0

        d.exp_id = exp_id
        d.timestamp = ts
        return d

    def __getitem__(self, idx) -> Tuple[int, List[Data]]:
        exp_id = self.exp_ids[idx]
        seq = [self._attach_labels_and_norm(exp_id, d) for d in self.exp_to_graphs[exp_id]]

        return exp_id, seq
