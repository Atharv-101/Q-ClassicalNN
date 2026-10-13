import random
import numpy as np
import torch
from torch import nn

from qclassicalnn.model import QClassicalNN
from qclassicalnn.data import get_data_loaders


# ---------------- CONFIGURATION ----------------

SEED = 7
EPOCHS = 3
LEARNING_RATE = 0.01
BATCH_SIZE = 16

TRAIN_PER_CLASS = 200
TEST_PER_CLASS = 50

MODEL_PATH = "models/qclassicalnn_mnist_0_vs_1.pt"


# ---------------- REPRODUCIBILITY ----------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------- EVALUATION ----------------

def evaluate(model, data_loader, loss_function):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            predictions = model(images)

            loss = loss_function(predictions, labels)

            total_loss += loss.item() * labels.size(0)

            predicted_labels = predictions.argmax(dim=1)

            correct += (
                predicted_labels == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    return average_loss, accuracy


# ---------------- MAIN TRAINING ----------------

def main():

    print("=" * 50)
    print("Q-ClassicalNN Training")
    print("=" * 50)

    # Load data
    train_loader, test_loader = get_data_loaders(
        train_per_class=TRAIN_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
        batch_size=BATCH_SIZE
    )

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Testing images: {len(test_loader.dataset)}")

    # Initialize model
    model = QClassicalNN()

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("Quantum simulator: PennyLane default.qubit")
    print("Starting training...\n")

    # Training loop
    for epoch in range(EPOCHS):

        model.train()

        total_training_loss = 0.0
        total_images = 0

        for images, labels in train_loader:

            optimizer.zero_grad()

            predictions = model(images)

            loss = loss_function(
                predictions,
                labels
            )

            loss.backward()

            optimizer.step()

            total_training_loss += (
                loss.item() * labels.size(0)
            )

            total_images += labels.size(0)

        average_training_loss = (
            total_training_loss / max(total_images, 1)
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            loss_function
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {average_training_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_accuracy * 100:.2f}%"
        )

    # Save model
    import os
    os.makedirs("models", exist_ok=True)

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\nTraining completed!")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()