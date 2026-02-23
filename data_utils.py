import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SiameseTransform:
    """Applies two different random augmentations to the same image."""

    def __init__(self, size=32):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def get_cifar10_loaders(batch_size=128):
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=SiameseTransform(size=32)
    )

    # Notice: we don't shuffle for the "validation" to keep t-SNE plots consistent
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader
