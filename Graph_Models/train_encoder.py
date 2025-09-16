from Graph_Models.Datasets import GraphSnapshotDataset as gd
from Graph_Models.Datasets import TemporalGraphDataset as tgd

from Graph_Models.encoders import VanillaGATEncoder
from Graph_Models.encoders import GATv2Encoder
from Graph_Models.encoders import GATAttentionEncoder
from Graph_Models.encoders import EvolveGCNEncoder

from Graph_Models import BasicGraphDecoder
from Graph_Models import SequenceGraphDecoder
from Graph_Models import BasicGraphAutoencoder
from Graph_Models import SequenceGraphAutoencoder


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


def run_temporal_encoder_pipeline(graphs_path, label_file, lr=1e-3, epochs=50, feat_weights=None,
                         embeddings_save_path="Graph_Models/outputs/embeddings/",
                         plots_save_path="Graph_Models/outputs/plots/",
                         model_name=None):

    device = get_device()
    embedding_dim = 64

    # ------ 1. Data preparation ------
    # (batch size is always 1 sequence of graphs)
    graph_dataset = tgd.TemporalGraphDataset(graphs_path, label_file)
    train_loader, val_loader, test_loader = create_temporal_encoder_dataloaders(graph_dataset)
    num_features = graph_dataset[0][1][0].x.size(1)
    print("num features: ", num_features)

    # ------ 2. Model creation ------
    one_step_decoder = BasicGraphDecoder.BasicGraphDecoder(embed_dim=embedding_dim,
                                                           max_nodes=MAX_NODES, out_node_dim=num_features)
    decoder = SequenceGraphDecoder.SequenceGraphDecoder(one_step_decoder)

    if model_name == 'EvolveGCNEncoderO':
        print("EvolveGCNEncoderO")
        encoder = EvolveGCNEncoder.EvolveGCNEncoderMean(num_features, embedding_dim, variant='O')
    elif model_name == 'EvolveGCNEncoderH':
        encoder = EvolveGCNEncoder.EvolveGCNEncoderMean(num_features, embedding_dim, variant='H')
    elif model_name == 'EvolveGCNAttentionEncoder':
        encoder = None
    else:
        raise ValueError("No temporal encoder with name {}".format(model_name))

    model = SequenceGraphAutoencoder.TemporalAutoencoder(encoder=encoder, decoder=decoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------ 3. Model training ------
    train_losses, val_losses = train_temporal_model(train_loader, val_loader, model, optimizer,
                                                                  device, num_epochs=epochs,
                                                                  feat_weights=feat_weights)

    plot_losses(train_losses, val_losses, save_path=plots_save_path + "loss_curve.png")

    # ------ 4. Model testing ------
    avg_loss, all_losses = evaluate_temporal_on_test(model, test_loader, device=device, feat_weights=feat_weights)
    plot_test_loss_distribution(all_losses, save_path=plots_save_path + "test_loss_distribution.png")

    # ------ 5. Save Embeddings ------
    save_temporal_embeddings_with_labels(encoder, train_loader, device, save_path=embeddings_save_path + "train_embeddings_with_labels.pt")
    save_temporal_embeddings_with_labels(encoder, val_loader, device, save_path=embeddings_save_path + "val_embeddings_with_labels.pt")
    save_temporal_embeddings_with_labels(encoder, test_loader, device, save_path=embeddings_save_path + "test_embeddings_with_labels.pt")


def train_temporal_model(train_loader, val_loader, model, optimizer, device,
                         num_epochs=50, patience=5, feat_weights=None, time_weights=None):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for exp_id, seq in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            # seq is a list length 1 (batch_size=1), containing a list of T Data
            sequence = [d.to(device) for d in seq]
            optimizer.zero_grad()

            # model returns a list of predicted counts and features for each graph in the sequence
            pred_feats_list, pred_counts_list = model(sequence)

            loss = temporal_reconstruction_loss_weighted(
                pred_feats_list, pred_counts_list, sequence,
                feat_weights=feat_weights,
                time_weights=time_weights
            )
            loss.backward(retain_graph=True)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- Validation ------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for seq in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}"):
                sequence = [d.to(device) for d in seq]
                pred_feats_list, pred_counts_list = model(sequence)
                loss = temporal_reconstruction_loss_weighted(
                    pred_feats_list, pred_counts_list, sequence,
                    feat_weights=feat_weights,
                    time_weights=time_weights
                )
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
def evaluate_temporal_on_test(model, test_loader, device, feat_weights=None, time_weights=None):
    model.eval()
    total_loss = 0.0
    all_losses = []

    for seq in tqdm(test_loader, desc="[Test]"):
        sequence = [d.to(device) for d in seq]

        pred_feats_list, pred_counts_list = model(sequence)
        loss = temporal_reconstruction_loss_weighted(
            pred_feats_list, pred_counts_list, sequence,
            feat_weights=feat_weights,
            time_weights=time_weights
        )
        total_loss += loss.item()
        all_losses.append(loss.item())
    avg = total_loss / len(test_loader)
    print(f"Test Set Avg Loss: {avg:.6f}")
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