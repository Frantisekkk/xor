This project was created so i can learn neural networks on most basic and low levels projects. The state it is in is the most advanced version for now that i pushed it to. 

Great project to understand perceptrons by coding them radther then using training methods from python on mnist datasets.

to run: 
    python3 xor.py

output of a run:

Epoch 0
 Input=(0,0), Target=0, Pred=0.5091, Loss=0.7115
 Weights: w11=0.689, w12=0.516, w21=-0.482, w22=0.023
 Biases: b1=-0.195, b2=-0.165, b3=-0.301
--------------------------------------------------
Epoch 1000
 Input=(0,0), Target=0, Pred=0.0833, Loss=0.0870
 Weights: w11=0.066, w12=-1.434, w21=-4.383, w22=-4.608
 Biases: b1=-1.046, b2=0.712, b3=0.277
--------------------------------------------------
Epoch 2000
 Input=(0,0), Target=0, Pred=0.0177, Loss=0.0178
 Weights: w11=0.748, w12=-5.473, w21=-5.822, w22=-7.471
 Biases: b1=0.138, b2=1.666, b3=-0.134
--------------------------------------------------
Epoch 3000
 Input=(0,0), Target=0, Pred=0.0067, Loss=0.0067
 Weights: w11=1.361, w12=-6.815, w21=-6.116, w22=-8.486
 Biases: b1=-0.262, b2=1.764, b3=-0.130
--------------------------------------------------
Epoch 4000
 Input=(0,0), Target=0, Pred=0.0037, Loss=0.0037
 Weights: w11=1.761, w12=-7.577, w21=-6.246, w22=-9.026
 Biases: b1=-0.548, b2=1.773, b3=-0.131
--------------------------------------------------
Epoch 5000
 Input=(0,0), Target=0, Pred=0.0025, Loss=0.0025
 Weights: w11=2.031, w12=-8.089, w21=-6.322, w22=-9.387
 Biases: b1=-0.744, b2=1.772, b3=-0.131
--------------------------------------------------
Epoch 6000
 Input=(0,0), Target=0, Pred=0.0019, Loss=0.0019
 Weights: w11=2.229, w12=-8.470, w21=-6.374, w22=-9.658
 Biases: b1=-0.890, b2=1.769, b3=-0.132
--------------------------------------------------
Epoch 7000
 Input=(0,0), Target=0, Pred=0.0015, Loss=0.0015
 Weights: w11=2.384, w12=-8.771, w21=-6.413, w22=-9.873
 Biases: b1=-1.003, b2=1.768, b3=-0.132
--------------------------------------------------
Epoch 8000
 Input=(0,0), Target=0, Pred=0.0013, Loss=0.0013
 Weights: w11=2.509, w12=-9.018, w21=-6.444, w22=-10.053
 Biases: b1=-1.095, b2=1.767, b3=-0.132
--------------------------------------------------
Epoch 9000
 Input=(0,0), Target=0, Pred=0.0011, Loss=0.0011
 Weights: w11=2.614, w12=-9.229, w21=-6.469, w22=-10.206
 Biases: b1=-1.172, b2=1.767, b3=-0.132
--------------------------------------------------

Final weights and biases:
w11=2.704, w12=-9.411, b1=-1.238
w21=-6.490, w22=-10.341, b2=1.766
v1=7.711, v2=-10.038, b3=-0.132

Testing XOR after training:
x1=0, x2=0 | y_true=0 -> y_pred=0.0009, pred=0
x1=0, x2=1 | y_true=1 -> y_pred=0.4666, pred=0
x1=1, x2=0 | y_true=1 -> y_pred=0.9976, pred=1
x1=1, x2=1 | y_true=0 -> y_pred=0.4677, pred=0