import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MatrixSSLResNet
from tqdm import tqdm

@torch.no_grad()
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k=200, knn_t=0.5):
    """Checks the K-nearest neighbors in the feature space."""
    # Compute cosine similarity
    sim_matrix = torch.mm(feature, feature_bank)    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # Counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label.scatter_(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(-1), dim=1)

    return pred_scores.argsort(dim=-1, descending=True)[:, 0]

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Clear VRAM before starting
    torch.cuda.empty_cache()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, transform=transform)

    # LOWER BATCH SIZE to save VRAM
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    model = MatrixSSLResNet().to(device)

    # Fix the warning and load
    state_dict = torch.load("matrix_ssl_cifar10_epoch100.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    feature_bank = []
    feature_labels = []

    with torch.no_grad(): # Ensure no gradients are being tracked
        for data, target in tqdm(train_loader, desc="Building Feature Bank"):
            data = data.to(device)
            feature = model(data)
            feature = nn.functional.normalize(feature, dim=1)
            # Move to CPU immediately to keep VRAM free!
            feature_bank.append(feature.cpu())
            feature_labels.append(target.cpu())

    # Cat on CPU, then move to GPU only for the final math
    feature_bank = torch.cat(feature_bank, dim=0).t().to(device)
    feature_labels = torch.cat(feature_labels, dim=0).to(device)

    # ... (rest of the evaluation code remains the same)

    # 4. Evaluate on Test Set
    total_top1 = 0
    total_num = 0
    for data, target in tqdm(test_loader, desc="Evaluating"):
        data, target = data.to(device), target.to(device)
        feature = model(data)
        feature = nn.functional.normalize(feature, dim=1)

        preds = knn_predict(feature, feature_bank, feature_labels, 10)
        total_top1 += (preds == target).float().sum().item()
        total_num += data.size(0)

    print(f"KNN Test Accuracy: {total_top1 / total_num * 100:.2f}%")

if __name__ == '__main__':
    test()
