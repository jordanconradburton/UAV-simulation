import torch
import torch.nn as nn

class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128, max_drones=8, use_transformer=False, n_heads=4, n_layers=2):
        super().__init__()
        self.max_drones = max_drones
        self.use_transformer = use_transformer

        if use_transformer:
            self.query_embed = nn.Parameter(torch.randn(max_drones, embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
            self.project_lstm = nn.Linear(hidden_dim, embed_dim)

        self.count_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_drones + 1)  # Predict 0 to max_drones
        )

        self.position_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_drones * 3)  # x, y, z for each drone
        )

    def forward(self, graph_embeddings):  # [B, D]
        B, D = graph_embeddings.shape

        if self.use_transformer:
            queries = self.query_embed.unsqueeze(0).repeat(B, 1, 1)
            x = self.transformer(queries.transpose(0, 1)).transpose(0, 1)  # [B, max_drones, D]
            pooled = torch.mean(x, dim=1)  # [B, D]
        else:
            input_seq = graph_embeddings.unsqueeze(1).repeat(1, self.max_drones, 1)
            lstm_out, _ = self.lstm(input_seq)
            pooled = self.project_lstm(lstm_out[:, -1, :])

        count_logits = self.count_head(pooled)  # [B, max_drones+1]
        pos_output = self.position_head(pooled).view(B, self.max_drones, 3)  # [B, max_drones, 3]
        return count_logits, pos_output
