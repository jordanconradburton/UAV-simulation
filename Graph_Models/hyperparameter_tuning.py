import torch
from torch.utils.data import Subset
import random
from itertools import product
from tqdm import tqdm
from torch_geometric.loader import DataLoader

def hyperparameter_grid():
    return {
        'batch_size': [8, 16, 32],
        'lr': [1e-3, 5e-4],
        'embedding_dim': [64, 128, 256],
        'attn_heads': [2, 4, 8],
        'dropout': [0.2, 0.3],
        'hidden_channels': [32, 64, 128, 256],
        'activation': ['elu', 'relu']
    }

def get_activation(name):
    if name == 'relu':
        return torch.nn.ReLU()
    elif name == 'elu':
        return torch.nn.ELU()
    else:
        raise ValueError(f"Unknown activation function: {name}")

def run_hyperparameter_search(dataset, model_class, decoder_class, autoencoder_class, extract_targets_fn,
                              loss_fn, max_nodes, device, max_epochs=30, patience=5):
    grid = hyperparameter_grid()
    keys, values = zip(*grid.items())
    best_config = None
    best_val_loss = float('inf')
    results = []

    for config_values in product(*values):
        config = dict(zip(keys, config_values))
        print(f"\n🔍 Testing config: {config}")
        #TODO: add in counter to see which config out of total

        # Subsample dataset for faster tuning
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        train_idx = indices[:int(0.7 * len(indices))]
        val_idx = indices[int(0.7 * len(indices)):int(0.85 * len(indices))]

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'])

        in_channels = dataset[0].x.size(1)
        act_fn = get_activation(config['activation'])

        encoder = model_class(
            in_channels=in_channels,
            hidden_channels=config['hidden_channels'],
            embed_dim=config['embedding_dim'],
            attn_heads=config['attn_heads'],
            dropout=config['dropout'],
            activation=act_fn
        ).to(device)

        decoder = decoder_class(embed_dim=config['embedding_dim'], max_nodes=max_nodes, out_node_dim=in_channels).to(device)
        model = autoencoder_class(encoder, decoder).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        no_improve = 0
        best_val = float('inf')
        best_model_state = None

        for epoch in range(max_epochs):
            model.train()
            for data in tqdm(train_loader, desc=f"[Train] Epoch {epoch + 1}"):
                data = data.to(device)
                optimizer.zero_grad()
                pred_feats, pred_counts = model(data)
                true_feats, true_counts = extract_targets_fn(data)
                loss = loss_fn(pred_feats, true_feats, pred_counts, true_counts, max_nodes)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for data in tqdm(val_loader, desc=f"[Val] Epoch {epoch + 1}"):
                    data = data.to(device)
                    pred_feats, pred_counts = model(data)
                    true_feats, true_counts = extract_targets_fn(data)
                    loss = loss_fn(pred_feats, true_feats, pred_counts, true_counts, max_nodes)
                    val_loss_total += loss.item()
            avg_val_loss = val_loss_total / len(val_loader)

            if avg_val_loss < best_val:
                best_val = avg_val_loss
                best_model_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏹ Early stopping at epoch {epoch + 1}")
                    break

        print(f"→ Finished with val loss: {best_val:.6f}")
        results.append((config, best_val))
        if best_val < best_val_loss:
            best_val_loss = best_val
            best_config = config

    print(f"\n✅ Best config: {best_config} with val loss {best_val_loss:.6f}")
    return best_config, results