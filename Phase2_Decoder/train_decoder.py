from Phase2_Decoder.SimpleDecoder import SimpleDecoder
from utils import *
from Graph_Models.visualization_utils import *
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from loss_functions import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def run_decoder_pipeline(train_embeddings_file, val_embeddings_file, test_embeddings_file,
                         plots_save_path="Phase2_Decoder/outputs/plots/",
                         metrics_save_path = "Phase2_Decoder/outputs/metrics",lr=1e-3, epochs=50):

    # ------ 1. Data preparation ------
    train_loader, val_loader, test_loader, embedding_dim = create_decoder_dataloaders(train_embeddings_file, val_embeddings_file, test_embeddings_file)
    device = get_device()

    # ------ 2. Decoder model creation ------
    model = SimpleDecoder(embed_dim=embedding_dim, use_transformer=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------ 3. Model training ------
    train_losses, val_losses = train_decoder(train_loader, val_loader, model, optimizer, device, num_epochs=epochs)
    plot_losses(train_losses, val_losses, save_path=plots_save_path + "loss_curve.png")

    # ------ 4. Evaluations of Decoder ------
    metrics = evaluate_decoder_on_test(model, test_loader, device, plots_dir=plots_save_path,
                                       metrics_dir=metrics_save_path)
    print(metrics)



def train_decoder(train_loader, val_loader, model, optimizer, device, num_epochs=50, patience=5):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None

    scale_factor = 1 / 1000.0  # meters → kilometers

    loss_fn_mse = torch.nn.MSELoss()
    loss_fn_count = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for data in tqdm(train_loader, desc=f"[Train] Epoch {epoch + 1}"):

            emb = data['embedding'].to(device)  # [B, D]
            count_gt = data['count'].long().to(device)  # [B]
            pos_gt = data['positions'].to(device)

            pred_count_logits, pred_positions = model(emb)

            scaled_pos_gt = pos_gt * scale_factor
            scaled_pred_positions = pred_positions * scale_factor

            # loss_pos = loss_fn_mse(scaled_pred_positions, scaled_pos_gt)
            loss_count = loss_fn_count(pred_count_logits, count_gt)

            loss_pos_total = 0.0
            for b in range(emb.size(0)):
                gt_n = count_gt[b].item()
                pred_n = torch.argmax(pred_count_logits[b]).item()

                gt_positions = scaled_pos_gt[b, :gt_n, :]  # only true targets
                pred_positions = scaled_pred_positions[b, :pred_n, :]  # only predicted targets
                loss_pos_total += hungarian_match_loss(pred_positions, gt_positions)

            if emb.size(0) > 0:
                loss_pos_total = loss_pos_total / emb.size(0)

            total_loss = loss_count + loss_pos_total

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"[Val] Epoch {epoch + 1}"):

                emb = data['embedding'].to(device)  # [B, D]
                count_gt = data['count'].long().to(device)  # [B]
                pos_gt = data['positions'].to(device)

                pred_count_logits, pred_positions = model(emb)
                scaled_pos_gt = pos_gt * scale_factor
                scaled_pred_positions = pred_positions * scale_factor

                # loss_pos = loss_fn_mse(scaled_pred_positions, scaled_pos_gt)
                loss_count = loss_fn_count(pred_count_logits, count_gt)

                loss_pos_total = 0.0
                for b in range(emb.size(0)):
                    gt_n = count_gt[b].item()
                    pred_n = torch.argmax(pred_count_logits[b]).item()

                    gt_positions = scaled_pos_gt[b, :gt_n, :]  # only true targets
                    pred_positions = scaled_pred_positions[b, :pred_n, :]  # only predicted targets

                    loss_pos_total += hungarian_match_loss(pred_positions, gt_positions)

                if emb.size(0) > 0:
                    loss_pos_total = loss_pos_total / emb.size(0)

                total_loss = loss_count + loss_pos_total
                total_val_loss += total_loss.item()

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
def evaluate_decoder_on_test(model, test_loader, device, plots_dir="Phase2_Decoder/outputs/plots",
                             metrics_dir="Phase2_Decoder/outputs/metrics"):
    model.eval()
    scale_factor = 1 / 1000.0

    loss_fn_count = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_losses = []

    all_true_counts = []
    all_pred_counts = []

    matched_true_positions = []
    matched_pred_positions = []

    for data in tqdm(test_loader, desc="[Test]"):
        emb = data['embedding'].to(device)
        count_gt = data['count'].long().to(device)
        pos_gt = data['positions'].to(device)

        pred_count_logits, pred_positions = model(emb)
        scaled_pos_gt = pos_gt * scale_factor
        scaled_pred_positions = pred_positions * scale_factor

        # Count loss
        loss_count = loss_fn_count(pred_count_logits, count_gt)

        # Position loss
        loss_pos_total = 0.0
        for b in range(emb.size(0)):
            gt_n = count_gt[b].item()
            pred_n = torch.argmax(pred_count_logits[b]).item()  # correct predicted count

            gt_positions = scaled_pos_gt[b, :gt_n, :]
            pred_positions_b = scaled_pred_positions[b, :pred_n, :]

            loss_pos_total += hungarian_match_loss(pred_positions_b, gt_positions)

            # Store matched positions for metrics
            # Use Hungarian again for metric computation
            if pred_positions_b.size(0) > 0 and gt_positions.size(0) > 0:
                diff = pred_positions_b.unsqueeze(1) - gt_positions.unsqueeze(0)
                cost_matrix = torch.norm(diff, dim=2).detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                matched_pred_positions.append(pred_positions_b[row_ind].cpu().numpy())
                matched_true_positions.append(gt_positions[col_ind].cpu().numpy())

        loss_pos_total = loss_pos_total / emb.size(0) if emb.size(0) > 0 else 0
        total_loss += (loss_count.item() + loss_pos_total)
        all_losses.append((loss_count.item() + loss_pos_total))

        # Count metrics
        pred_count_cls = torch.argmax(pred_count_logits, dim=1)
        all_true_counts.extend(count_gt.cpu().numpy())
        all_pred_counts.extend(pred_count_cls.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    print(f"Test Set Avg Loss: {avg_loss:.6f}")

    # ---------- Count metrics ----------
    all_true_counts = np.array(all_true_counts)
    all_pred_counts = np.array(all_pred_counts)
    count_accuracy = np.mean(all_true_counts == all_pred_counts)
    count_mae = np.mean(np.abs(all_true_counts - all_pred_counts))
    count_bias = np.mean(all_pred_counts - all_true_counts)

    print(f"Count Accuracy: {count_accuracy:.4f}")
    print(f"Count MAE: {count_mae:.4f}")
    print(f"Count Bias: {count_bias:.4f}")

    # ---------- Position metrics ----------
    matched_true_positions = np.concatenate(matched_true_positions, axis=0)
    matched_pred_positions = np.concatenate(matched_pred_positions, axis=0)

    pos_errors = np.linalg.norm(matched_pred_positions - matched_true_positions, axis=1)
    pos_rmse = np.sqrt(np.mean(pos_errors**2))
    pos_mae = np.mean(pos_errors)

    axis_bias = np.mean(matched_pred_positions - matched_true_positions, axis=0)
    axis_rmse = np.sqrt(np.mean((matched_pred_positions - matched_true_positions)**2, axis=0))

    print(f"Position RMSE: {pos_rmse:.6f} km")
    print(f"Position MAE: {pos_mae:.6f} km")
    print(f"Axis Bias (x,y,z): {axis_bias}")
    print(f"Axis RMSE (x,y,z): {axis_rmse}")

    # ---------- Bias Plots ----------
    axis_labels = ["X", "Y", "Z"]
    for i in range(3):
        plt.hist((matched_pred_positions[:, i] - matched_true_positions[:, i]), bins=30)
        plt.title(f"{axis_labels[i]} Error Distribution")
        plt.xlabel("Error (km)")
        plt.ylabel("Frequency")
        plt.savefig(f"{plots_dir}/axis_{axis_labels[i]}_error_hist.png", dpi=300)
        plt.close()

        plt.scatter(matched_true_positions[:, i], matched_pred_positions[:, i], alpha=0.5, s=5)
        plt.xlabel(f"True {axis_labels[i]} (km)")
        plt.ylabel(f"Pred {axis_labels[i]} (km)")
        plt.title(f"{axis_labels[i]} Predicted vs True")
        plt.savefig(f"{plots_dir}/axis_{axis_labels[i]}_scatter.png", dpi=300)
        plt.close()

    metrics = {
        "avg_loss": float(avg_loss),
        "count_accuracy": float(count_accuracy),
        "count_mae": float(count_mae),
        "count_bias": float(count_bias),
        "pos_rmse": float(pos_rmse),
        "pos_mae": float(pos_mae),
        "axis_bias": axis_bias.tolist(),
        "axis_rmse": axis_rmse.tolist()
    }

    os.makedirs(metrics_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    with open(os.path.join(metrics_dir, f"decoder_test_summary_{timestamp}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

