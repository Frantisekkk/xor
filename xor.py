import numpy as np
import random as rn

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hidden_layer(x1, x2):
    z1 = w11 * x1 + w12 * x2 + b1
    z2 = w21 * x1 + w22 * x2 + b2
    a1 = sigmoid(z1)
    a2 = sigmoid(z2)
    return a1, a2

def output_layer(a1, a2):
    z3 = v1 * a1 + v2 * a2 + b3
    output = sigmoid(z3)
    return output

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-10
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def error_calc(y,y_pred):
    return binary_cross_entropy(y, y_pred)

def Backpropagation(a1,w,dlt):
    return a1 * (1-a1) * w * dlt

def gradients(input, dlt):
    return input * dlt

def adjust_weight(weight, learning_rate, gradient):
    return weight - (learning_rate * gradient)

def adjust_bias(bias, learning_rate, dlt):
    return bias - learning_rate * dlt

inputs = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0)
]

learning_rate = 0.5
epochs = 30000

# Try different random seeds to find one that converges well
# Some initializations get stuck in local minima
seeds_to_try = [42, 123, 456, 789, 999, 1337, 2024, 0]
best_seed = 0
best_acc = 0

print("Testing different weight initializations...")
for seed in seeds_to_try:
    np.random.seed(seed)
    rn.seed(seed)
    
    w11, w12, b1 = rn.uniform(-1,1), rn.uniform(-1,1), rn.uniform(-1,1)
    w21, w22, b2 = rn.uniform(-1,1), rn.uniform(-1,1), rn.uniform(-1,1)
    v1, v2, b3   = rn.uniform(-1,1), rn.uniform(-1,1), rn.uniform(-1,1)
    
    # Quick test training
    for _ in range(3000):
        for x1, x2, y in inputs:
            a1, a2 = hidden_layer(x1, x2)
            y_pred = output_layer(a1, a2)
            dlt = y_pred - y
            dlt1 = Backpropagation(a1, v1, dlt)
            dlt2 = Backpropagation(a2, v2, dlt)
            gv1, gv2, gb3 = a1 * dlt, a2 * dlt, dlt
            gw11, gw12, gb1 = x1 * dlt1, x2 * dlt1, dlt1
            gw21, gw22, gb2 = x1 * dlt2, x2 * dlt2, dlt2
            v1 = adjust_weight(v1, learning_rate, gv1)
            v2 = adjust_weight(v2, learning_rate, gv2)
            b3 = adjust_bias(b3, learning_rate, gb3)
            w11 = adjust_weight(w11, learning_rate, gw11)
            w12 = adjust_weight(w12, learning_rate, gw12)
            b1  = adjust_bias(b1, learning_rate, gb1)
            w21 = adjust_weight(w21, learning_rate, gw21)
            w22 = adjust_weight(w22, learning_rate, gw22)
            b2  = adjust_bias(b2, learning_rate, gb2)
    
    # Check accuracy
    correct = 0
    for x1, x2, y in inputs:
        a1, a2 = hidden_layer(x1, x2)
        y_pred = output_layer(a1, a2)
        pred = 0 if y_pred < 0.5 else 1
        if pred == y:
            correct += 1
    
    if correct > best_acc:
        best_acc = correct
        best_seed = seed
    if correct == len(inputs):
        print(f"✓ Seed {seed} achieves 100% accuracy\n")
        break

# Initialize with best seed
np.random.seed(best_seed)
rn.seed(best_seed)
w11, w12, b1 = rn.uniform(-1,1), rn.uniform(-1,1), rn.uniform(-1,1)
w21, w22, b2 = rn.uniform(-1,1), rn.uniform(-1,1), rn.uniform(-1,1)
v1, v2, b3   = rn.uniform(-1,1), rn.uniform(-1,1), rn.uniform(-1,1)
print(f"Using seed {best_seed} for full training\n")

for i in range(epochs):
    # Train on all samples each epoch (batch training)
    for x1, x2, y in inputs:
        # Forward pass
        a1, a2 = hidden_layer(x1, x2)
        y_pred = output_layer(a1, a2)
        loss = error_calc(y, y_pred)

        # Backprop
        dlt = y_pred - y
        dlt1 = Backpropagation(a1, v1, dlt)
        dlt2 = Backpropagation(a2, v2, dlt)

        # Gradients
        gv1, gv2, gb3 = a1 * dlt, a2 * dlt, dlt
        gw11, gw12, gb1 = x1 * dlt1, x2 * dlt1, dlt1
        gw21, gw22, gb2 = x1 * dlt2, x2 * dlt2, dlt2

        # Updates
        v1 = adjust_weight(v1, learning_rate, gv1)
        v2 = adjust_weight(v2, learning_rate, gv2)
        b3 = adjust_bias(b3, learning_rate, gb3)

        w11 = adjust_weight(w11, learning_rate, gw11)
        w12 = adjust_weight(w12, learning_rate, gw12)
        b1  = adjust_bias(b1, learning_rate, gb1)

        w21 = adjust_weight(w21, learning_rate, gw21)
        w22 = adjust_weight(w22, learning_rate, gw22)
        b2  = adjust_bias(b2, learning_rate, gb2)
    
    # Check accuracy every 1000 epochs
    if i % 1000 == 0:
        correct = 0
        total_loss = 0
        for x1, x2, y in inputs:
            a1, a2 = hidden_layer(x1, x2)
            y_pred = output_layer(a1, a2)
            pred = 0 if y_pred < 0.5 else 1
            if pred == y:
                correct += 1
            total_loss += error_calc(y, y_pred)
        avg_loss = total_loss / len(inputs)
        accuracy = correct / len(inputs)
        status = "✓" if accuracy == 1.0 else "✗"
        print(f"[Epoch {i:5d}] {status} Accuracy: {correct}/{len(inputs)} ({accuracy:.0%}) Avg Loss: {avg_loss:.4f}")
        
        # Early stopping if perfect accuracy
        if accuracy == 1.0:
            print(f"\n✓ Perfect accuracy achieved at epoch {i}!")
            break

print("\n" + "="*60)
print("TRAINING COMPLETE - Final Parameters:")
print("="*60)
print(f"Hidden Layer 1: w11={w11:7.3f} w12={w12:7.3f} b1={b1:7.3f}")
print(f"Hidden Layer 2: w21={w21:7.3f} w22={w22:7.3f} b2={b2:7.3f}")
print(f"Output Layer:   v1={v1:7.3f} v2={v2:7.3f} b3={b3:7.3f}")
print("="*60)

# Testing after training
print("\n" + "="*60)
print("TEST RESULTS:")
print("="*60)
correct = 0
for x1, x2, y in inputs:
    a1, a2 = hidden_layer(x1, x2)
    y_pred = output_layer(a1, a2)
    pred = 0 if y_pred < 0.5 else 1
    match = "✓" if pred == y else "✗"
    if pred == y:
        correct += 1
    print(f"{match} XOR({x1},{x2}) = {y} → Predicted: {pred} (prob: {y_pred:.4f})")
print("="*60)
print(f"Accuracy: {correct}/{len(inputs)} ({100*correct/len(inputs):.0f}%)")
print("="*60)

