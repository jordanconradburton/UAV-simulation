from Graph_Models.Datasets import GraphSnapshotDataset as gd
from Graph_Models.Datasets import ASTGCNDataset

from Graph_Models.encoders import VanillaGATEncoder
from Graph_Models.encoders import GATv2Encoder
from Graph_Models.encoders import GATAttentionEncoder
from Graph_Models.encoders import ASTGCNEncoder

from Graph_Models import BasicGraphDecoder
from Graph_Models import ASTGCNDecoder
from Graph_Models import BasicGraphAutoencoder
from Graph_Models import ASTGCNAutoencoder


from loss_functions import *
from utils import *
from tqdm import tqdm
from Graph_Models.visualization_utils import *

MAX_NODES = 20
MAX_SECTORS = 6

"""
TODO
To tune:
1. number of attention heads per layer
2. dropout rate
3. GAT hidden layer size (same or different?)
4. number of convolutional layers
5. learning rate
6. loss function weights
7. batch size
8. embedding size
10. activation function (relu, elu, ...)
"""


def run_encoder_pipeline(graphs_path, label_file, batch_size=16, lr=1e-3, epochs=50, feat_weights=None,
                         embeddings_save_path="Graph_Models/outputs/embeddings/",
                         plots_save_path="Graph_Models/outputs/plots/", temporal=False,
                         model_name=None):

    # ------ 1. Data preparation ------
    device = get_device()
    embedding_dim = 64

    graph_dataset = gd.GraphSnapshotDataset(graphs_path, label_file)
    train_loader, val_loader, test_loader = create_encoder_dataloaders(graph_dataset, batch_size=batch_size)
    num_features = graph_dataset[0].x.size(1)

    # ------ 2. Model creation ------
    decoder = BasicGraphDecoder.BasicGraphDecoder(embed_dim=embedding_dim, max_nodes=MAX_NODES,
                                    out_node_dim=num_features).to(device)

    if model_name == 'VanillaGAT':
        encoder = VanillaGATEncoder.BasicGATEncoder(in_channels=num_features, hidden_channels=128, embed_dim=embedding_dim,
                              attn_heads=4, dropout=0.2).to(device)
    elif model_name == 'GATv2Encoder':
        encoder = GATv2Encoder.BasicGATv2Encoder(in_channels=num_features, hidden_channels=128, embed_dim=embedding_dim,
                               attn_heads=4, dropout=0.2).to(device)
    elif model_name == 'GATAttentionEncoder':
        encoder = GATAttentionEncoder.GATAttentionEncoder(in_channels=num_features, hidden_channels=128, embed_dim=embedding_dim,
                                          attn_heads=4, dropout=0.2).to(device)
    else:
        raise ValueError("No encoder with name {}".format(model_name))

    model = BasicGraphAutoencoder.BasicGraphAutoencoder(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------ 3. Model training ------
    train_losses, val_losses = train_model(train_loader, val_loader, model, optimizer, device,
                                                         num_epochs=epochs, feat_weights=feat_weights)
    plot_losses(train_losses, val_losses, save_path=plots_save_path + "loss_curve.png")

    # ------ 4. Model testing ------
    avg_loss, all_losses = evaluate_on_test(model, test_loader, device=device, feat_weights=feat_weights)
    plot_test_loss_distribution(all_losses, save_path=plots_save_path + "test_loss_distribution.png")

    # ------ 5. Save Embeddings ------
    save_embeddings_with_labels(encoder, train_loader, device, save_path=embeddings_save_path + "train_embeddings_with_labels.pt")
    save_embeddings_with_labels(encoder, val_loader, device, save_path=embeddings_save_path + "val_embeddings_with_labels.pt")
    save_embeddings_with_labels(encoder, test_loader, device, save_path=embeddings_save_path + "test_embeddings_with_labels.pt")


def run_ASTGCN_encoder_pipeline(graphs_path, label_file, lr=1e-3, epochs=50,
                         embeddings_save_path="Graph_Models/outputs/embeddings/",
                         plots_save_path="Graph_Models/outputs/plots/",
                         test_csv_path="Graph_Models/outputs/test_set_metrics/"):

    device = get_device()
    embedding_dim = 64

    # ------ 1. Data preparation ------
    # TODO: try both GMAN and ASTGCN with different poolings

    graph_dataset = ASTGCNDataset.ASTGCNDataset(graphs_path, label_file, window_size=12)
    train_loader, val_loader, test_loader = create_encoder_dataloaders(graph_dataset, batch_size=1)
    num_features = graph_dataset[0][1][0].x.size(1)
    print("num features: ", num_features)

    # ------ 2. Model creation ------
    encoder = ASTGCNEncoder.ASTGCNEncoder(in_channels=num_features, num_nodes=3, num_timesteps=12)

    # output of the decoder is only the reconstructed dynamic node features of power per sector
    decoder = ASTGCNDecoder.ASTGCNDecoder(embed_dim=embedding_dim, max_nodes=3, dyn_feat_dim=MAX_SECTORS)

    model = ASTGCNAutoencoder.ASTGCNAutoencoder(encoder=encoder, decoder=decoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------ 3. Model training ------
    train_losses, val_losses = train_ASTGCN_model(train_loader, val_loader, model, optimizer,
                                                                  device, num_epochs=epochs)

    plot_losses(train_losses, val_losses, save_path=plots_save_path + "loss_curve.png")

    # ------ 4. Model testing ------
    avg_loss, all_losses = evaluate_ASTGCN_on_test(model, test_loader, device=device,
                                                   csv_path=test_csv_path + ".csv")
    plot_test_loss_distribution(all_losses, save_path=plots_save_path + "test_loss_distribution.png")

    # ------ 5. Save Embeddings ------
    save_ASTGCN_embeddings_with_labels(encoder, train_loader, device, save_path=embeddings_save_path + "train_embeddings_with_labels.pt")
    save_ASTGCN_embeddings_with_labels(encoder, val_loader, device, save_path=embeddings_save_path + "val_embeddings_with_labels.pt")
    save_ASTGCN_embeddings_with_labels(encoder, test_loader, device, save_path=embeddings_save_path + "test_embeddings_with_labels.pt")


def train_ASTGCN_model(train_loader, val_loader, model, optimizer, device,
                         num_epochs=50, patience=5):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for x_seq, A_hat, mask, _ in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            x_seq, A_hat, mask = x_seq.to(device), A_hat.to(device), mask.to(device)
            x_target = x_seq[:, :, -MAX_SECTORS:, -1]

            optimizer.zero_grad()

            x_pred = model(x_seq, A_hat) #TODO: check output type

            loss = nn.MSELoss(x_pred * mask.unsqueeze(-1), x_target * mask.unsqueeze(-1))

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- Validation ------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_seq, A_hat, mask, _ in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}"):
                x_seq, A_hat, mask = x_seq.to(device), A_hat.to(device), mask.to(device)
                x_target = x_seq[:, :, -MAX_SECTORS:, -1]
                # for all feature reconstruction: x_target = x_seq[:, :, :, -1]

                optimizer.zero_grad()

                x_pred = model(x_seq, A_hat)  # TODO: check output type

                loss = nn.MSELoss(x_pred * mask.unsqueeze(-1), x_target * mask.unsqueeze(-1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"\nEpoch {epoch+1:02d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss, no_improve_epochs = avg_val_loss, 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            no_improve_epochs += 1
            print(f"  ↳ no improvement ({no_improve_epochs}/{patience})")
            if no_improve_epochs >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1} — best val: {best_val_loss:.6f}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return train_losses, val_losses


@torch.no_grad()
def evaluate_ASTGCN_on_test(model, test_loader, device, csv_path):
    model.eval()
    total_loss = 0.0
    all_losses = []

    preds, trues = [], []

    for x_seq, A_hat, mask, _ in tqdm(test_loader, desc="[Test]"):
        x_seq, A_hat, mask = x_seq.to(device), A_hat.to(device), mask.to(device)
        x_target = x_seq[:, :, -MAX_SECTORS:, -1]
        x_pred = model(x_seq, A_hat)

        preds.append(x_pred.cpu().numpy())
        trues.append(x_target.cpu().numpy())

        loss = nn.MSELoss(x_pred * mask.unsqueeze(-1), x_target * mask.unsqueeze(-1))

        total_loss += loss.item()
        all_losses.append(loss.item())
    avg = total_loss / len(test_loader)
    print(f"Test Set Avg Loss: {avg:.6f}")

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    # Flatten per node and feature for CSV export
    df = pd.DataFrame({
        'true': trues.flatten(),
        'pred': preds.flatten()
    })

    mse = ((trues - preds) ** 2).mean()
    mae = np.abs(trues - preds).mean()
    r2 = 1 - ((trues - preds) ** 2).sum() / ((trues - trues.mean()) ** 2).sum()

    print(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}, R2: {r2:.4f}")

    metrics_row = pd.DataFrame({'true': ['MSE', 'MAE', 'R2'],
                                'pred': [mse, mae, r2]})
    df = pd.concat([metrics_row, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"Saved predictions + metrics to {csv_path}")

    return avg, all_losses

def train_model(train_loader, val_loader, model, optimizer, device, num_epochs=50, patience=5, feat_weights=None):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for data in tqdm(train_loader, desc=f"[Train] Epoch {epoch + 1}"):
            data = data.to(device)
            optimizer.zero_grad()

            pred_feats, pred_counts = model(data)
            true_feats, true_counts = extract_graph_targets(data)

            loss = graph_reconstruction_loss_weighted(
                pred_feats,
                true_feats,
                pred_counts,
                true_counts,
                feat_weights=feat_weights,
            )

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"[Val] Epoch {epoch + 1}"):
                data = data.to(device)
                pred_feats, pred_counts = model(data)
                true_feats, true_counts = extract_graph_targets(data)

                loss = graph_reconstruction_loss_weighted(
                    pred_feats,
                    true_feats,
                    pred_counts,
                    true_counts,
                    feat_weights=feat_weights,
                )

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"\nEpoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- Early stopping logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            best_model_state = model.state_dict()
        else:
            no_improve_epochs += 1
            print(f"  ↳ No improvement. ({no_improve_epochs}/{patience} patience)")

        if no_improve_epochs >= patience:
            print(f"⏹ Early stopping at epoch {epoch + 1} — best val loss: {best_val_loss:.6f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


@torch.no_grad()
def evaluate_on_test(model, test_loader, device, feat_weights=None):
    model.eval()
    total_loss = 0.0
    all_losses = []

    for data in tqdm(test_loader, desc="[Test]"):
        data = data.to(device)
        pred_feats, pred_counts = model(data)
        true_feats, true_counts = extract_graph_targets(data)

        # default (unweighted)
        loss = graph_reconstruction_loss_weighted(
            pred_feats,
            true_feats,
            pred_counts,
            true_counts,
            feat_weights=feat_weights,
        )

        total_loss += loss.item()
        all_losses.append(loss.item())

    avg_loss = total_loss / len(test_loader)
    print(f"Test Set Avg Loss: {avg_loss:.6f}")
    return avg_loss, all_losses