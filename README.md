# Quantum-Classical Neural Network (Q-ClassicalNN)

**Tagline:** Teaching Machines to See Through Quantum Intelligence

## Overview

Q-ClassicalNN is a hybrid machine-learning project that combines
classical neural-network layers with a trainable quantum circuit.

The initial implementation classifies handwritten digits 0 and 1
from the MNIST dataset.

## Architecture

MNIST Image (28×28)
        ↓
Classical Feature Extraction
        ↓
Four Numerical Features
        ↓
Four-Qubit Quantum Circuit
        ↓
Quantum Measurements
        ↓
Classical Classifier
        ↓
Prediction: 0 or 1

## Technology Stack

- Python
- PyTorch
- PennyLane
- NumPy
- MNIST

## Installation

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

## Training

Run from the project root:

    python -m qclassicalnn.train

## Evaluation

Run:

    python -m qclassicalnn.evaluate

## Initial Configuration

- Classification: Digit 0 vs Digit 1
- Image size: 28×28
- Qubits: 4
- Quantum circuit layers: 2
- Training examples: Up to 400
- Test examples: Up to 100

## Current Status

The model runs on PennyLane's local quantum simulator.
IBM Quantum hardware execution is planned as a future extension.

## Limitations

The initial experiment uses a small dataset subset. Its results
must not be interpreted as proof of quantum advantage.

A classical-only baseline and larger controlled evaluations
are required for meaningful performance comparisons.

## Team

Department: Artificial Intelligence & Data Science (AIDS)

Academic Year: 2026–2027

- Paksha Bhamare
- Sakshi Chaudhary
- Nidhi Gangurde
- Sakshi Singh