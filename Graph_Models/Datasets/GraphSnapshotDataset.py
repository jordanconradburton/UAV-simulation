import torch
from torch.utils.data import Dataset
from utils import normalize_features, load_labels

class GraphSnapshotDataset(Dataset):
    """
    Flattens your time-series graphs into independent snapshots.
    Each item is a single Data(...) with its labels attached.
    """
    def __init__(self, graphs_path: str, label_file: str, max_drones=8):
        super().__init__()
        self.all_graphs = torch.load(graphs_path, weights_only=False)
        self.labels = load_labels(label_file)
        self.max_drones = max_drones

        # build a flat list of (exp_id, Data)
        self.samples = []
        for exp_id, graph_list in self.all_graphs.items():
            for data in graph_list:
                data.x = normalize_features(data.x)
                self.samples.append((exp_id, data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        exp_id, data = self.samples[idx]

        ts = str(data.timestamp).strip()

        key = (exp_id, ts)
        if key not in self.labels:
            raise KeyError(f"No label found for (exp_id={exp_id}, ts={ts})")

        label = self.labels[key]
        count = label["num_drones"]
        pos_tensor = torch.tensor(label["positions"], dtype=torch.float)

        data.y = torch.tensor([count], dtype=torch.float)
        data.pos_gt = pos_tensor
        data.mask_gt = torch.zeros(self.max_drones)
        data.mask_gt[:count] = 1.0

        data.exp_id = exp_id
        data.timestamp = ts

        return data