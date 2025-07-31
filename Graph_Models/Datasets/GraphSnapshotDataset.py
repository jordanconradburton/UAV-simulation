import torch
from torch.utils.data import Dataset
from Graph_Models.utils import normalize_features

class GraphSnapshotDataset(Dataset):
    """
    Flattens your time-series graphs into independent snapshots.
    Each item is a single Data(...) with its labels attached.
    """
    def __init__(self, graphs_path: str):
        super().__init__()
        self.all_graphs = torch.load(graphs_path, weights_only=False)

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

        return data
