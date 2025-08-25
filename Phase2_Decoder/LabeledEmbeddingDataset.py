import torch
from torch.utils.data import Dataset

class LabeledEmbeddingDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        data = torch.load(path)
        self.embeddings = data['embeddings']         # [N, D]
        self.counts = data['counts']                 # [N]
        self.positions = data['positions']           # [N, max_drones, 3]

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],       # [D]
            'count': self.counts[idx],               # scalar
            'positions': self.positions[idx]         # [max_drones, 3]
        }
