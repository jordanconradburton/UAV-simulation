import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os


def plot_losses(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Node Feature and Count Reconstruction Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved loss curve to {save_path}")
    else:
        plt.show()

# TODO: add in save path here as well
def plot_test_loss_distribution(test_losses):
    plt.hist(test_losses, bins=30)
    plt.xlabel('Per-Graph MSE')
    plt.ylabel('Count')
    plt.title('Test Reconstruction Loss Distribution')
    plt.grid(True)
    plt.show()


# TODO: add in save path
def plot_graph_embeddings(model, dataloader, device, method='tsne'):
    """
    Visualize graph-level embeddings using t-SNE, UMAP, or PCA.

    Assumes model.encoder returns a [B, D] graph embedding.
    """
    model.eval()
    embeddings, labels = [], []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            z = model.encoder(data)  # [B, D]
            embeddings.append(z.cpu())

    embeddings = torch.cat(embeddings).numpy()

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)

    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], cmap='viridis', s=40)
    plt.colorbar(scatter, label='Drone Count')
    plt.title(f'{method.upper()} of Graph Embeddings Colored by Drone Count')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()