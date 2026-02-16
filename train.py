import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from data_utils import get_cifar10_loaders
from model import MatrixSSLResNet
from loss import MatrixSSLLoss
import math

def train():
    # --- Hyperparameters ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    accumulation_steps = 4
    epochs = 100 # Increased epochs - SSL loves to "cook" longer
    lr = 3e-3
    feature_dim = 128
    warmup_epochs = 5

    # --- Setup ---
    model = MatrixSSLResNet(feature_dim=feature_dim).to(device)

    # Using AdamW with slightly higher weight decay to prevent "lazy" features
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Increase lambd slightly to force more feature diversity (Redundancy Reduction)
    criterion = MatrixSSLLoss(lambd=1e-2).to(device)

    scaler = GradScaler('cuda')
    loader = get_cifar10_loaders(batch_size=batch_size)

    # --- Learning Rate Scheduler ---
    # Cosine Annealing: Starts high, ends near zero.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

    print(f"Starting intensive training on {torch.cuda.get_device_name(0)}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Linear Warmup Logic
        if epoch < warmup_epochs:
            curr_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

        for i, ((view1, view2), _) in enumerate(loader):
            view1, view2 = view1.to(device), view2.to(device)

            with autocast('cuda'):
                z1 = model(view1)
                z2 = model(view2)
                loss = criterion(z1, z2)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{i}/{len(loader)}] | Loss: {loss.item()*accumulation_steps:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Update scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"matrix_ssl_cifar10_epoch{epoch+1}.pth")
            print(f"--- Checkpoint saved at Epoch {epoch+1} ---")

    torch.save(model.state_dict(), f"matrix_ssl_cifar10_final.pth")
    print("Training Complete!")

if __name__ == '__main__':
    train()
