import torch
from torch.utils.data import Dataset
from utils import *

class TimeSeriesGraphDataset(Dataset):
    """
    A Dataset of time-series graphs, one per experiment.

    Expects a file (e.g. 'all_experiments_graphs.pt') containing
    a dict: {exp_id: List[torch_geometric.data.Data]} and a file containing
    a dict: {(exp_id, ts): data}.

    Each Data object in the list is a snapshot at one timestamp, with:
      - x: [N, F_static + 1]  (#nodes × [static_feats..., dynamic_prx])
      - edge_index, edge_attr, timestamp
    """
    def __init__(self, graphs_path: str, label_file: str):
        super().__init__()
        self.all_graphs = torch.load(graphs_path)
        self.labels = load_labels(label_file)
        self.exp_ids = sorted(self.all_graphs.keys())


    def __len__(self):
        return len(self.exp_ids)

    def __getitem__(self, idx):
        exp_id = self.exp_ids[idx]
        graph_list = self.all_graphs[exp_id]

        x0 = graph_list[0].x  # shape [N, F]
        static_feats = x0[:, :-1]  # [N, F_static]

        dyn_list = [g.x[:, -1] for g in graph_list]
        dynamic_prx = torch.stack(dyn_list, dim=1)  # [N, T]

        edge_index = graph_list[0].edge_index
        edge_attr  = graph_list[0].edge_attr
        timestamps = [g.timestamp for g in graph_list]

        count_gt, pos_gt = zip(*(self.labels[(exp_id, ts)] for ts in timestamps))
        count_gt = torch.tensor(count_gt, dtype=torch.float)         # [T]
        pos_gt   = torch.stack(pos_gt, dim=0)                        # [T, MAX_D, 3]

        return {
            'exp_id': exp_id,
            'static_feats': static_feats,
            'dynamic_prx': dynamic_prx,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'timestamps': timestamps,
            'snapshots': graph_list,
            'target_counts': count_gt,
            'target_positions': pos_gt
        }