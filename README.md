# simple-based-neuralnetwork
Neural networks were originally much simpler, but today them have become too complex.
This repository contains the classic but simplest regression analysis neural networks. It is available in three languages: C, Node.js and Python. This neural network uses the least-squares method for the loss function.

The three files are independent of each other and can be executed immediately without additional modules.

# C language  
### build
`gcc -Wall -o "cdevice" "cdevice.c" -lm`
### excute
`./cdevice`
  
# Node.js
### excute
`node nodevice.js`

# Python
### excute
`python3 pydevice.py`

# Validation
XOR and cellular automaton 30 and 90 truth tables for comparison and verification of neural networks. The files are stored in the csv folder and json folder.  
C language refers to the csv folder and node.js pyhton refers to the json folder.

# Activation Function
The activation function implements a sigmoid function or ReLU. It is possible to switch between them by the variable value (0 or 1) of each file.

# ReLU
ReLU is up to 10 times faster than the sigmoid function, but it is prone to local minimum. It is unstable. The cell automaton 90 was unlearnable.
ReLU and sigmoid functions differ in the activation function and bias and derivative of the hidden layer. Therefore, these three points are separated by an if statement.

# Training
Training data and test data are same. If you want to separate them, you need to save weight (v, w).
