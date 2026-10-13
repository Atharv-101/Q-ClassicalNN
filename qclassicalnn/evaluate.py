import torch
from torchvision import datasets, transforms

from qclassicalnn.model import QClassicalNN


MODEL_PATH = "models/qclassicalnn_mnist_0_vs_1.pt"


def main():

    print("Q-ClassicalNN Model Evaluation")
    print("-" * 35)

    # Initialize the model architecture
    model = QClassicalNN()

    # Load trained parameters
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    # Load the test dataset
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # Find one example of each class
    examples = []

    for image, label in test_dataset:

        if label in (0, 1):
            examples.append((image, label))

        if len(examples) == 2:
            break

    # Make predictions
    with torch.no_grad():

        for image, actual_label in examples:

            output = model(image.unsqueeze(0))

            predicted_label = output.argmax(dim=1).item()

            print(
                f"Actual: {actual_label} | "
                f"Predicted: {predicted_label}"
            )


if __name__ == "__main__":
    main()