from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_data_loaders(
    data_dir="./data",
    train_per_class=200,
    test_per_class=50,
    batch_size=16
):
    """Load MNIST and prepare balanced datasets for digits 0 and 1."""

    transform = transforms.ToTensor()

    # Load training dataset
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Load testing dataset
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Select the requested number of examples from each class
    def select_indices(labels, per_class):
        labels = labels.tolist()

        zeros = [
            i for i, label in enumerate(labels)
            if label == 0
        ][:per_class]

        ones = [
            i for i, label in enumerate(labels)
            if label == 1
        ][:per_class]

        return zeros + ones

    train_indices = select_indices(
        train_dataset.targets,
        train_per_class
    )

    test_indices = select_indices(
        test_dataset.targets,
        test_per_class
    )

    # Create balanced subsets
    train_subset = Subset(
        train_dataset,
        train_indices
    )

    test_subset = Subset(
        test_dataset,
        test_indices
    )

    # Prepare batches
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader