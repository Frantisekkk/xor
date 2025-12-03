# Neural Network from Scratch: XOR Problem

This is a learning project focused on understanding the core concepts of neural networks rather than using pre-built training methods from frameworks like TensorFlow or PyTorch.


## 🔧 Implementation Details

**XOR Truth Table:**
```
Input (x1, x2) | Output
---------------|-------
(0, 0)         | 0
(0, 1)         | 1
(1, 0)         | 1
(1, 1)         | 0
```

## 🏗️ Architecture

- **Input Layer:** 2 neurons (x1, x2)
- **Hidden Layer:** 2 neurons with sigmoid activation
- **Output Layer:** 1 neuron with sigmoid activation

## ✨ Features

- ✅ Forward propagation
- ✅ Backpropagation algorithm
- ✅ Binary cross-entropy loss function
- ✅ Manual weight and bias updates
- ✅ No high-level ML frameworks (only NumPy for basic operations)

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- NumPy


### Run

```bash
python3 xor.py
```

## 📊 Example Output

```
Testing different weight initializations...
✓ Seed 456 achieves 100% accuracy

Using seed 456 for full training

[Epoch     0] ✗ Accuracy: 2/4 (50%) Avg Loss: 0.7298
[Epoch  1000] ✓ Accuracy: 4/4 (100%) Avg Loss: 0.0108

✓ Perfect accuracy achieved at epoch 1000!

============================================================
TRAINING COMPLETE - Final Parameters:
============================================================
Hidden Layer 1: w11= -5.060 w12= -5.059 b1=  7.506
Hidden Layer 2: w21= -7.023 w22= -7.137 b2=  2.930
Output Layer:   v1= 10.543 v2=-10.949 b3= -4.902
============================================================

============================================================
TEST RESULTS:
============================================================
✓ XOR(0,0) = 0 → Predicted: 0 (prob: 0.0085)
✓ XOR(0,1) = 1 → Predicted: 1 (prob: 0.9904)
✓ XOR(1,0) = 1 → Predicted: 1 (prob: 0.9902)
✓ XOR(1,1) = 0 → Predicted: 0 (prob: 0.0150)
============================================================
Accuracy: 4/4 (100%)
============================================================
```

## 📚 Learning Objectives

This project was created to understand neural networks at the most fundamental level by:

- Implementing perceptrons and multi-layer networks manually
- Understanding forward and backward propagation
- Learning gradient descent and weight updates
- Avoiding high-level abstractions to grasp the underlying mathematics

## 🔧 Implementation Details

- **Activation Function:** Sigmoid
- **Loss Function:** Binary Cross-Entropy
- **Optimization:** Gradient Descent
- **Learning Rate:** 0.5
- **Epochs:** 30,000 (with early stopping)
- **Initialization:** Tests multiple random seeds to find optimal starting weights
