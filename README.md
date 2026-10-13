# Quantum-Classical Neural Network (Q-ClassicalNN)

> **Teaching Machines to See Through Quantum Intelligence**

A beginner-friendly hybrid machine learning project that combines a classical neural network with a trainable quantum circuit to classify handwritten digits.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Project Objectives](#3-project-objectives)
- [4. How It Works](#4-how-it-works)
- [5. Model Architecture](#5-model-architecture)
- [6. Technologies Used](#6-technologies-used)
- [7. Dataset](#7-dataset)
- [8. Project Structure](#8-project-structure)
- [9. Requirements](#9-requirements)
- [10. Installation](#10-installation)
- [11. Run the Model](#11-run-the-model)
- [12. Expected Output](#12-expected-output)
- [13. Understanding the Quantum Circuit](#13-understanding-the-quantum-circuit)
- [14. Evaluation Plan](#14-evaluation-plan)
- [15. Current Scope and Limitations](#15-current-scope-and-limitations)
- [16. Future Enhancements](#16-future-enhancements)
- [17. Troubleshooting](#17-troubleshooting)
- [18. Team](#18-team)
- [19. Academic Integrity and Reporting](#19-academic-integrity-and-reporting)
- [20. References](#20-references)

---

## 1. Project Overview

The **Quantum-Classical Neural Network (Q-ClassicalNN)** is a hybrid machine learning approach that combines conventional neural-network layers with a parameterized quantum circuit.

In this starter implementation, the model learns to distinguish handwritten images of the digits **0 and 1** from the MNIST dataset. A classical neural-network layer converts each image into four compact numerical features. These values are encoded as rotation angles for a four-qubit quantum circuit. The circuit applies trainable quantum gates and entangling operations, then returns measurement values to a classical output layer for classification.

The first version runs on a **local quantum simulator** using PennyLane. This allows the model to be developed and tested without requiring access to a physical quantum computer. IBM Quantum hardware may be explored in a later stage.

### Project tagline

**Teaching Machines to See Through Quantum Intelligence**

---

## 2. Problem Statement

Traditional machine-learning models process data using classical computing operations. Quantum machine learning explores whether quantum circuits can be integrated into learning pipelines to provide alternative ways of representing and processing information.

The challenge in this project is to implement a small, trainable hybrid model that combines classical feature processing and quantum-circuit operations, then evaluate its ability to classify handwritten digits. Its performance must be measured experimentally rather than assuming that the quantum component automatically improves accuracy.

---

## 3. Project Objectives

1. Understand the basic workflow of quantum machine learning.
2. Build a classical neural-network component using PyTorch.
3. Implement a parameterized quantum circuit using PennyLane.
4. Connect classical features to a quantum circuit through angle encoding.
5. Train the combined model on a small MNIST dataset subset.
6. Measure test performance using classification accuracy and loss.
7. Establish a foundation for comparison with a classical-only model.
8. Explore noisy simulation and IBM Quantum execution as future extensions.

---

## 4. How It Works

The starter model follows this pipeline:

```text
MNIST image (28 × 28 pixels)
             |
             v
     Flatten image
       784 values
             |
             v
  Classical feature layer
        4 features
             |
             v
   Angle encoding (RY)
        4 qubits
             |
             v
 Trainable quantum circuit
  RY/RZ rotations + CNOTs
             |
             v
   Quantum measurements
    4 expectation values
             |
             v
   Classical output layer
        2 output scores
             |
             v
      Predicted digit
          0 or 1
```

### In simple terms

- **Input:** A grayscale image of a handwritten zero or one.
- **Classical feature layer:** Converts 784 pixel values into four learned features.
- **Quantum encoding:** Uses the four features as rotation angles for four qubits.
- **Quantum processing:** Trainable rotations and CNOT gates transform the quantum state.
- **Measurement:** The circuit returns four expectation values.
- **Classification:** A classical output layer uses these values to predict whether the image is a zero or a one.
- **Training:** The loss is calculated and an optimizer updates the trainable classical and quantum parameters.

---

## 5. Model Architecture

| Component | Purpose |
|---|---|
| Input image | A 28 × 28 grayscale MNIST image |
| Flatten layer | Converts the image into a vector of 784 pixel values |
| Classical linear layer | Maps 784 values to four learned features |
| Tanh activation | Keeps features in a bounded range |
| Angle encoding | Encodes features as Y-axis rotations on four qubits |
| Variational quantum circuit | Applies trainable RY/RZ rotations and CNOT entanglement |
| Measurement layer | Returns the expectation value of Pauli-Z for each qubit |
| Classical output layer | Maps four measurements to two class scores |
| Loss function | Cross-entropy loss for the two classes |
| Optimizer | Adam updates the model parameters |

### Initial experiment settings

The starter code uses these settings:

- **Classification task:** Digit 0 vs. digit 1
- **Input size:** 28 × 28 pixels
- **Qubits:** 4
- **Quantum circuit layers:** 2
- **Training subset:** Up to 400 images
- **Test subset:** Up to 100 images
- **Batch size:** 16
- **Training epochs:** 3
- **Quantum device:** PennyLane `default.qubit` simulator
- **Optimizer:** Adam
- **Loss function:** Cross-entropy

These are intentionally small starting settings. They can be increased after the first run works reliably.

---

## 6. Technologies Used

| Technology | Role |
|---|---|
| Python | Main programming language |
| PyTorch | Classical neural-network layers, loss, optimization and training |
| PennyLane | Quantum circuit definition and integration with PyTorch |
| MNIST | Handwritten-digit image dataset |
| NumPy | Numerical utilities and reproducibility |
| Local quantum simulator | Simulates quantum-circuit execution on a conventional computer |

The current starter project does not require IBM Quantum credentials. IBM Quantum hardware execution is a possible future extension and is not yet implemented in this starter script.

---

## 7. Dataset

The project uses the **MNIST handwritten-digit dataset**, which contains 28 × 28 grayscale images of handwritten digits from 0 through 9.

For the first experiment, only digits **0 and 1** are selected. This makes the first classification task smaller and easier to debug than a ten-class problem.

The script downloads MNIST automatically through `torchvision` the first time it runs. An internet connection is therefore required for the initial download.

The starter script selects up to:
- 200 training examples of digit 0
- 200 training examples of digit 1
- 50 test examples of digit 0
- 50 test examples of digit 1

The exact number available depends on the dataset. The script prints the actual subset sizes before training.

**Important:** The starter subset is for implementation and learning. A final evaluation should use a clearly documented data split, larger datasets where feasible, and repeated or controlled experiments.

---

## 8. Project Structure

```text
qclassicalnn_starter/
├── train.py
├── requirements.txt
├── README.md
└── data/                         # Created when MNIST is downloaded
```

After training, the script also creates:

```text
qclassicalnn_mnist_0_vs_1.pt       # Saved trained model parameters
```

### File descriptions

- `train.py` — loads MNIST, defines the hybrid model, trains it, evaluates it, and prints sample predictions.
- `requirements.txt` — lists the Python packages needed to run the starter.
- `README.md` — project documentation and setup instructions.
- `data/` — local dataset cache created by `torchvision`.
- `qclassicalnn_mnist_0_vs_1.pt` — saved model state created after successful training.

---

## 9. Requirements

### Software

- Python 3.10 or 3.11 recommended for a beginner-friendly setup
- pip
- Internet connection for the initial MNIST download
- Terminal on macOS/Linux or PowerShell on Windows

### Hardware

A normal laptop can run this small simulation. The first experiment uses only four qubits, but quantum simulation cost can grow rapidly as the number of qubits increases.

A physical quantum computer is **not required** to run the starter version.

---

## 10. Installation

### macOS / Linux

Open Terminal, navigate to the extracted project folder, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

Open PowerShell in the extracted project folder and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 11. Run the Model

Make sure the virtual environment is active and that the terminal is in the project directory.

Run:

```bash
python train.py
```

On macOS/Linux, if `python` points to a different installation, try:

```bash
python3 train.py
```

The first run downloads the MNIST dataset. Training then begins, and the program prints the training loss, test loss, and test accuracy for each epoch.

When training completes successfully, the model parameters are saved to:

```text
qclassicalnn_mnist_0_vs_1.pt
```

---

## 12. Expected Output

The output format will look similar to this:

```text
Training on 400 images; testing on 100 images.
The first run downloads MNIST and may take a few minutes.
Epoch 1/3 | train loss ... | test loss ... | test accuracy ...%
Epoch 2/3 | train loss ... | test loss ... | test accuracy ...%
Epoch 3/3 | train loss ... | test loss ... | test accuracy ...%
Saved trained model to qclassicalnn_mnist_0_vs_1.pt
Sample predictions (actual -> predicted):
0 -> 0
1 -> 1
...
```

The numbers above are placeholders for the output format, **not reported experimental results**. Actual accuracy and loss must be taken from a successful run on your machine. Results can vary with package versions, initialization, hardware and training settings.

---

## 13. Understanding the Quantum Circuit

The quantum circuit has four qubits.

### A. Encoding

Each of the four classical features is used as a rotation angle for an `RY` operation. This is called **angle encoding**.

### B. Trainable rotations

The circuit applies trainable `RY` and `RZ` gates. Their parameters are updated during model training.

### C. Entanglement

CNOT gates connect neighboring qubits, including a connection between the last and first qubits. These operations allow the circuit to represent correlations between qubits.

### D. Measurement

The circuit measures the expectation value of the Pauli-Z observable on each qubit. These four numerical outputs are passed back to the classical output layer.

### Why is the model called hybrid?

The classical neural network and quantum circuit are components of one learning pipeline. The classical layers process image pixels and produce the final class scores; the quantum layer transforms compact features and returns measurements.

---

## 14. Evaluation Plan

The first successful run is a proof of implementation, not evidence of quantum advantage.

For a meaningful project evaluation, compare the hybrid model against a classical-only baseline using the same data split and comparable training conditions.

Recommended experiments:

| Experiment | Purpose |
|---|---|
| Classical-only neural network | Establish a baseline |
| Q-ClassicalNN with 4 qubits | Test the initial hybrid architecture |
| Q-ClassicalNN with different circuit depths | Study the effect of circuit depth |
| Multiple random seeds | Check whether results are stable |
| Noisy simulation | Explore sensitivity to noise |
| IBM Quantum execution, if available | Explore execution on real quantum hardware |

Recommended metrics:
- Test accuracy
- Precision, recall and F1-score
- Confusion matrix
- Training time and inference time
- Training and test loss
- Circuit depth and number of circuit executions, where relevant

Record the dataset split, random seed, software versions, model settings and actual results. Do not claim that the quantum model is faster or more accurate unless the experiments support that conclusion.

---

## 15. Current Scope and Limitations

This starter version has a deliberately limited scope:

- It classifies only digits 0 and 1.
- It uses a small subset of MNIST to help the first run finish sooner.
- It runs the quantum circuit on a local simulator.
- It does not yet contain a classical-only comparison model.
- It does not yet include a graphical user interface.
- It does not yet execute circuits on IBM Quantum hardware.
- Its initial accuracy is not guaranteed and must be measured.
- A small simulation does not establish quantum advantage over classical machine learning.

These limitations define the next development tasks rather than indicating that the project is complete.

---

## 16. Future Enhancements

Planned improvements may include:

1. Add a classical-only neural network baseline.
2. Plot training loss and test accuracy.
3. Increase the dataset size and run more epochs.
4. Experiment with circuit depth and number of qubits.
5. Extend classification from two digits to all ten MNIST classes, if feasible.
6. Test noise models and compare noisy versus ideal simulation.
7. Explore execution of small circuits on IBM Quantum hardware using compatible Qiskit/PennyLane workflows.
8. Build a Streamlit web interface for uploading an image and viewing the prediction.
9. Prepare a reproducible experiment report with measured results and limitations.

---

## 17. Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

### MNIST download fails

Check your internet connection and run `python train.py` again. The first run needs to download the dataset.

### Python version or package installation errors

Check your version:

```bash
python --version
```

Python 3.10 or 3.11 is recommended for this starter setup. If package installation fails, create a fresh virtual environment with a supported Python version and retry.

### Training is slow

The quantum layer is simulated on a classical computer and can be computationally expensive. Keep the small starter settings initially. Avoid increasing the number of qubits until the first experiment completes.

### Test accuracy is low or unstable

Three epochs and a small dataset are only a starting point. Check that training runs without errors, then try more data, more epochs and multiple random seeds. Compare results with a classical baseline before drawing conclusions.

### Saved model file is missing

The file is written only after training completes successfully. Check the terminal for an exception and confirm that you are running the script from the project directory.

---

## 18. Team

**Department:** Artificial Intelligence & Data Science (AIDS)  
**Academic Year:** 2026–2027

**Presented by:**
- Paksha Bhamare
- Sakshi Chaudhary
- Nidhi Gangurde
- Sakshi Singh

---

## 19. Academic Integrity and Reporting

This repository is an initial implementation and learning scaffold. Before submitting the project:

- Run the code and preserve actual logs.
- Verify all reported metrics experimentally.
- Document any code changes and package versions.
- Explain the circuit and model architecture in your own words.
- Distinguish simulator results from physical-hardware results.
- Clearly report limitations and avoid unsupported claims of quantum advantage.

---

## 20. References

- **MNIST database:** http://yann.lecun.com/exdb/mnist/
- **PyTorch documentation:** https://pytorch.org/docs/
- **Torchvision datasets documentation:** https://pytorch.org/vision/stable/datasets.html
- **PennyLane documentation:** https://docs.pennylane.ai/
- **IBM Quantum documentation:** https://quantum.ibm.com/

---
