import torch
from torch.utils.data import Dataset
from utils import load_labels

class GraphSnapshotDataset(Dataset):
    """
    Flattens your time-series graphs into independent snapshots.
    Each item is a single Data(...) with its labels attached.
    """
    def __init__(self, graphs_path: str, label_file: str):
        super().__init__()
        self.all_graphs = torch.load(graphs_path)
        self.labels = load_labels(label_file)

        # build a flat list of (exp_id, Data)
        self.samples = []
        for exp_id, graph_list in self.all_graphs.items():
            for data in graph_list:
                self.samples.append((exp_id, data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        exp_id, data = self.samples[idx]
        ts = data.timestamp

        count_gt, pos_gt = self.labels[(exp_id, ts)]

        data.y = torch.tensor([count_gt], dtype=torch.float)
        data.pos_gt = pos_gt

        return data
