import torch
import GraphSnapshotDataset as gd
from torch_geometric.loader import DataLoader


def train_model(graphs_path, label_file):
    simple_dataset = gd.GraphSnapshotDataset(graphs_path, label_file)
    loader = DataLoader(simple_dataset, batch_size=1, shuffle=True)







if __name__ == '__main__':
    graphs_path = "Data_Files_and_Databases/1st_Trial_Data_Files_Graphs/1000_simple_simulations_graphs.pt"
    label_file = "Data_Files_and_Databases/1st_Trial_Data_Files_Graphs/1000_simple_simulations_labels.csv"

    train_model(graphs_path, label_file)



