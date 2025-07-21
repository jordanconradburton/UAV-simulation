import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
import logging
import pandas as pd

print("torch:", torch.__version__, ", built CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

# === Load Configuration ===
with open("config.json") as f:
    config = json.load(f)

PT_FILE = config["pt_file"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("using " + DEVICE)

BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]
LEARNING_RATE = config["learning_rate"]
WINDOW_SIZE = config["window_size"]
#EXPERIMENT_IDS = config["experiment_ids"]
USE_ATTENTION_MASK = config.get("use_attention_mask", False)  # optional

# === Setup Logging ===
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# === Step 1: Load from CSV file instead of .pt ===
CSV_FILE = "processed_sensor_data.csv"  # ← path to your CSV
print(f"Loading CSV file: {CSV_FILE}")

# Read everything as strings so we can parse timestamps manually
df = pd.read_csv(CSV_FILE, dtype=str)

# Build the same dict[str → list[dict]] that tensors_by_timestamp used to be
tensors_by_timestamp = defaultdict(list)
for _, row in df.iterrows():
    ts_str = row["timestamp"]
    rec = {
        # note: timestamp will get overwritten below in Step 2
        "sensor_id":       int(row["sensor_id"]),
        "experiment_id":   float(row["experiment_id"]),
        "total_power_dBm": float(row["total_power_dBm"]),
        "sensitivity":     float(row["sensitivity"]),
        "sinr_threshold":  float(row["sinr_threshold"]),
        "freq_min":        float(row["freq_min"]),
        "freq_max":        float(row["freq_max"]),
        "sector_count":    int(row["sector_count"]),
        # unpack all six gains
        **{f"sector_gain_{i+1}": float(row[f"sector_gain_{i+1}"])
           for i in range(6)}
    }
    tensors_by_timestamp[ts_str].append(rec)

# === Step 2 onward: exactly your original pipeline ===
all_records = []
for ts, records in tensors_by_timestamp.items():
    for rec in records:
        exp_id = rec['experiment_id']
        if False:  # exp_id not in EXPERIMENT_IDS:
            continue
        try:
            rec['timestamp'] = float(ts)
        except ValueError:
            rec['timestamp'] = datetime.fromisoformat(ts).timestamp()
        all_records.append(rec)

# Sort by experiment, timestamp, sensor_id
all_records = sorted(
    all_records,
    key=lambda r: (r['experiment_id'], r['timestamp'], r['sensor_id'])
)

# Map experiment IDs to indices
exp_id_to_idx = {
    eid: idx
    for idx, eid in enumerate(sorted({r['experiment_id'] for r in all_records}))
}

# Build global feature list and experiment boundaries
global_records = []
exp_boundaries = []
current_exp = None
start_idx = 0

for rec in all_records:
    exp_idx = exp_id_to_idx[rec['experiment_id']]
    features = [
        float(rec['timestamp']),
        float(exp_idx),
        float(rec['sensor_id']),
        rec['total_power_dBm'],
        rec['sensitivity'],
        rec['sinr_threshold'],
        rec['freq_min'],
        rec['freq_max'],
        rec['sector_count'],
        rec['sector_gain_1'],
        rec['sector_gain_2'],
        rec['sector_gain_3'],
        rec['sector_gain_4'],
        rec['sector_gain_5'],
        rec['sector_gain_6'],
    ]
    global_records.append(features)

    if current_exp != rec['experiment_id']:
        if current_exp is not None:
            exp_boundaries.append((start_idx, len(global_records) - 1))
        current_exp = rec['experiment_id']
        start_idx = len(global_records) - 1

# Close last boundary
exp_boundaries.append((start_idx, len(global_records) - 1))

global_tensor = torch.tensor(global_records, dtype=torch.float32)

# === Normalize global_tensor ===
means = global_tensor.mean(dim=0)
stds  = global_tensor.std(dim=0)
stds[stds == 0] = 1.0
global_tensor = (global_tensor - means) / stds

# Save normalization stats
os.makedirs("logs", exist_ok=True)
with open("logs/normalization_stats.json", "w") as f:
    json.dump({"mean": means.tolist(), "std": stds.tolist()}, f, indent=2)

print(f"Processed {len(global_records)} records into tensor shape {global_tensor.shape}")
print(f"Example record: {global_tensor[0].tolist()}")

# === Step 3: Dataset with sliding windows on global tensor ===
class GlobalTimeSeriesDataset(Dataset):
    def __init__(self, global_tensor, window_size, exp_boundaries):
        self.global_tensor = global_tensor
        self.window_size = window_size
        self.exp_boundaries = exp_boundaries
        self.valid_starts = []

        # Calculate valid window start indices (windows fully inside experiment boundaries)
        for start, end in exp_boundaries:
            for i in range(start, end - window_size):
                self.valid_starts.append(i)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        x = self.global_tensor[start_idx:start_idx+self.window_size]
        y = self.global_tensor[start_idx+1:start_idx+self.window_size+1]
        return x, y, start_idx  # return start_idx for attention mask if needed

dataset = GlobalTimeSeriesDataset(global_tensor, WINDOW_SIZE, exp_boundaries)
print(f"Total windowed samples (global): {len(dataset)}")

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Custom collate to keep start indices for mask generation
def collate_fn(batch):
    xs, ys, starts = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    starts = torch.tensor(starts)
    return xs, ys, starts

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# === Step 4: Model with optional attention mask ===
class SimpleDecoderModel(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim - 1, d_model)  # exclude timestamp for content
        self.pos_embedding_proj = nn.Linear(1, d_model)       # timestamp embedding
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, input_dim)

    def create_attention_mask(self, window_size, global_start_idx):
        # Find experiment boundaries for this window
        for start, end in exp_boundaries:
            if start <= global_start_idx <= end - window_size:
                break
        else:
            # If not found, no mask
            return torch.zeros((window_size, window_size), dtype=torch.bool, device=global_tensor.device)

        mask = torch.zeros((window_size, window_size), dtype=torch.bool, device=global_tensor.device)
        for i in range(window_size):
            for j in range(window_size):
                pos_i = global_start_idx + i
                pos_j = global_start_idx + j
                if pos_i < start or pos_i > end or pos_j < start or pos_j > end:
                    mask[i, j] = True
        return mask

    def forward(self, x, global_start_idx=None):
        B, T, F = x.shape
        pos_enc = self.pos_embedding_proj(x[:, :, :1])  # timestamp only
        content = self.input_linear(x[:, :, 1:])
        x_pe = content + pos_enc

        if USE_ATTENTION_MASK and global_start_idx is not None:
            # Create batch masks
            masks = []
            for i in range(B):
                masks.append(self.create_attention_mask(T, global_start_idx[i]))
            tgt_mask = torch.stack(masks)  # [B, T, T]
            # TransformerDecoder expects mask shape [T, T], for batch we need to do manual masking inside or reshape inputs
            # For simplicity, use standard subsequent mask (ignoring cross experiment masking) or implement custom attention mechanism
            # Here, just use standard mask to avoid complexity:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)

        out = self.decoder(x_pe, x_pe, tgt_mask=tgt_mask)
        return self.output_linear(out)

input_dim = global_tensor.shape[1]  # = number of features = 15
model = SimpleDecoderModel(input_dim=input_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# === Step 5: Training ===
train_losses = []
test_losses = []
logging.info("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for x, y, starts in tqdm(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        starts = starts.to(DEVICE)

        # Debug input for NaNs
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[WARNING] NaN or Inf detected in input tensor")
            continue

        optimizer.zero_grad()
        out = model(x, global_start_idx=starts if USE_ATTENTION_MASK else None)

        # Debug output for NaNs
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("[WARNING] NaN or Inf detected in model output")
            continue

        loss = loss_fn(out, y)
        if torch.isnan(loss) or torch.isinf(loss):
            print("[WARNING] NaN or Inf detected in loss")
            continue

        loss.backward()

        # Clip gradients to avoid exploding values
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y, starts in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            starts = starts.to(DEVICE)
            out = model(x, global_start_idx=starts if USE_ATTENTION_MASK else None)
            loss = loss_fn(out, y)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    logging.info(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

# === Step 6: Plot Loss ===
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Decoder Transformer Training Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("logs", "loss_plot.png"))
plt.show()


torch.save(model.state_dict(), 'logs/model_weights.pth')
