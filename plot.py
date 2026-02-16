import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from model import MatrixSSLResNet
from data_utils import get_cifar10_loaders

def plot_tsne():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize and load model
    model = MatrixSSLResNet(feature_dim=128).to(device)
    checkpoint_path = "matrix_ssl_cifar10_epoch10.pth"

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Loaded: {checkpoint_path}")
    except Exception as e:
        print(f"Checkpoint error: {e}. Ensure the filename is correct.")
        return

    model.eval()

    # 2. FIX: Handling the 'not subscriptable' error
    # We capture everything returned into a variable called 'all_output'
    all_output = get_cifar10_loaders(batch_size=128)

    # If it's a tuple/list, the test loader is usually the second one
    if isinstance(all_output, (list, tuple)):
        test_loader = all_output[1]
    else:
        # If the function only returned one thing, we'll use that
        test_loader = all_output

    features = []
    labels = []

    print("Extracting features (Subset of 1000 images)...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i > 8: # Roughly 1000 images (8 * 128)
                break

            # Unpack batch: handle ( (v1, v2), labels ) OR ( img, labels )
            data, target = batch[0], batch[1]

            if isinstance(data, (list, tuple)):
                img = data[0] # Take first augmented view
            else:
                img = data

            z = model(img.to(device))
            features.append(z.cpu())
            labels.append(target)

    # Convert to numpy
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    print(f"Computing t-SNE for {len(features)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    reduced = tsne.fit_transform(features)

    # 3. Plotting with Class Names
    plt.figure(figsize=(12, 8))
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(10):
        indices = np.where(labels == i)
        plt.scatter(reduced[indices, 0], reduced[indices, 1], label=classes[i], alpha=0.7, s=20)

    plt.legend(loc='best', markerscale=1.5, fontsize='small', ncol=2)
    plt.title(f"Matrix-SSL: Latent Space Clusters (Accuracy: 34.92%)")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")

    save_name = "tsne_final_plot.png"
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved as {save_name}")
    plt.show()

if __name__ == "__main__":
    plot_tsne()
