import numpy as np
import random as rn

np.random.seed(0)
rn.seed(0)

w11, w12, b1 = rn.uniform(-1,1), rn.uniform(-1,1), rn.uniform(-1,1)
w21, w22, b2 = rn.uniform(-1,1), rn.uniform(-1,1), rn.uniform(-1,1)
v1, v2, b3   = rn.uniform(-1,1), rn.uniform(-1,1), rn.uniform(-1,1)

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
epochs = 10000

for i in range(epochs):
    x1, x2, y = inputs[i % len(inputs)]

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

    # Print progress every 1000 epochs
    if i % 1000 == 0:
        print(f"Epoch {i}")
        print(f" Input=({x1},{x2}), Target={y}, Pred={y_pred:.4f}, Loss={loss:.4f}")
        print(f" Weights: w11={w11:.3f}, w12={w12:.3f}, w21={w21:.3f}, w22={w22:.3f}")
        print(f" Biases: b1={b1:.3f}, b2={b2:.3f}, b3={b3:.3f}")
        print("-"*50)

print("\nFinal weights and biases:")
print(f"w11={w11:.3f}, w12={w12:.3f}, b1={b1:.3f}")
print(f"w21={w21:.3f}, w22={w22:.3f}, b2={b2:.3f}")
print(f"v1={v1:.3f}, v2={v2:.3f}, b3={b3:.3f}")

# Testing after training
print("\nTesting XOR after training:")
for x1, x2, y in inputs:
    a1, a2 = hidden_layer(x1, x2)
    y_pred = output_layer(a1, a2)
    pred = 0 if y_pred < 0.5 else 1
    print(f"x1={x1}, x2={x2} | y_true={y} -> y_pred={y_pred:.4f}, pred={pred}")

