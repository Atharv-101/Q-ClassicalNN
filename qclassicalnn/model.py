import torch
from torch import nn
import pennylane as qml


class QClassicalNN(nn.Module):

    def __init__(self, n_qubits=4, n_quantum_layers=2, n_classes=2):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_quantum_layers = n_quantum_layers

        # 1. Classical feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, n_qubits),
            nn.Tanh()
        )

        # 2. Quantum simulator
        dev = qml.device("default.qubit", wires=n_qubits)

        # 3. Quantum circuit
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):

            # Encode classical features into qubits
            qml.AngleEmbedding(
                inputs,
                wires=range(n_qubits),
                rotation="Y"
            )

            # Trainable quantum layers
            for layer in range(n_quantum_layers):

                for wire in range(n_qubits):
                    qml.RY(weights[layer, wire, 0], wires=wire)
                    qml.RZ(weights[layer, wire, 1], wires=wire)

                # Entangle neighboring qubits
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])

                # Connect the last qubit to the first
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            # Measure each qubit
            return [
                qml.expval(qml.PauliZ(wires=i))
                for i in range(n_qubits)
            ]

        # 4. Trainable quantum layer
        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_circuit,
            {
                "weights": (n_quantum_layers, n_qubits, 2)
            }
        )

        # 5. Classical output layer
        self.classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, images):

        # Extract classical features
        features = self.feature_extractor(images)

        # Encode features as quantum rotation angles
        angles = features * (torch.pi / 2)

        # Run the quantum circuit
        quantum_values = self.quantum_layer(angles)

        # Generate class scores
        predictions = self.classifier(quantum_values)

        return predictions