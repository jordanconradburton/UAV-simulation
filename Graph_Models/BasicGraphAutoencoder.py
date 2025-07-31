import torch

class BasicGraphAutoencoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder  # GAT-based graph encoder
        self.decoder = decoder  # MLP decoder for reconstruction

    def forward(self, data):
        embedding = self.encoder(data)           # [B, D]
        node_feats, node_count = self.decoder(embedding)
        return node_feats, node_count
