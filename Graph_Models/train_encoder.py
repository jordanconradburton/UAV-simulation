from Graph_Models.Datasets import GraphSnapshotDataset as gd
# from Graph_Models.encoders import VanillaGATEncoder as enc
# from Graph_Models.encoders import GATv2Encoder as enc
from Graph_Models.encoders import GATAttentionEncoder as enc
from Graph_Models import BasicGraphDecoder as dec
from Graph_Models import BasicGraphAutoencoder as autoencoder
from loss_functions import *
from utils import *
from tqdm import tqdm
from Graph_Models.visualization_utils import *

MAX_NODES = 20

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
10. activation function (using elu)
"""


def run_encoder_pipeline(graphs_path, label_file, batch_size=16, lr=1e-3, epochs=50, feat_weights=None,
                         embeddings_save_path="Graph_Models/outputs/embeddings/",
                         plots_save_path="Graph_Models/outputs/plots/"):
    # ✅ Best config (GAT non-weighted): {'batch_size': 16, 'lr': 0.001, 'embedding_dim': 64, 'attn_heads': 4,
    # 'dropout': 0.2, 'hidden_channels': 128, 'activation': 'relu'}

    # ------ 1. Data preparation ------
    simple_dataset = gd.GraphSnapshotDataset(graphs_path, label_file)
    device = get_device()

    # check_feature_ranges(simple_dataset)
    # best_config, results = run_hyperparameter_search(simple_dataset, enc.BasicGATEncoder, dec.BasicGraphDecoder,
    #                                                  autoencoder.BasicGraphAutoencoder, extract_graph_targets,
    #                                                  graph_reconstruction_loss, max_nodes=MAX_NODES, device=device)

    train_loader, val_loader, test_loader = create_encoder_dataloaders(simple_dataset, batch_size=batch_size)

    # ------ 2. Autoencoder model creation ------
    num_features = simple_dataset[0].x.size(1)
    print("num features", num_features)
    embedding_dim = 64

    # encoder = enc.BasicGATEncoder(in_channels=num_features, hidden_channels=128, embed_dim=embedding_dim,
    #                               attn_heads=4, dropout=0.2).to(device)
    encoder = enc.GATAttentionEncoder(in_channels=num_features, hidden_channels=128, embed_dim=embedding_dim,
                                  attn_heads=4, dropout=0.2).to(device)
    # encoder = enc.BasicGATv2Encoder(in_channels=num_features, hidden_channels=128, embed_dim=embedding_dim,
    #                               attn_heads=4, dropout=0.2).to(device)
    decoder = dec.BasicGraphDecoder(embed_dim=embedding_dim, max_nodes=MAX_NODES, out_node_dim=num_features).to(device)
    model = autoencoder.BasicGraphAutoencoder(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------ 3. Model training ------
    train_losses, val_losses, feat_weights = train_model(train_loader, val_loader, model, optimizer, device,
                                                         num_epochs=epochs, feat_weights=feat_weights)
    plot_losses(train_losses, val_losses, save_path=plots_save_path + "loss_curve.png")

    # ------ 4. Model testing ------
    avg_loss, all_losses = evaluate_on_test(model, test_loader, device=device, feat_weights=feat_weights)
    plot_test_loss_distribution(all_losses, save_path=plots_save_path + "test_loss_distribution.png")

    # ------ 5. Save Embeddings ------
    save_embeddings_with_labels(encoder, train_loader, device, save_path=embeddings_save_path + "train_embeddings_with_labels.pt")
    save_embeddings_with_labels(encoder, val_loader, device, save_path=embeddings_save_path + "val_embeddings_with_labels.pt")
    save_embeddings_with_labels(encoder, test_loader, device, save_path=embeddings_save_path + "test_embeddings_with_labels.pt")


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

            # default (unweighted)
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

                # default (unweighted)
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

        print(f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

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

    return train_losses, val_losses, feat_weights


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