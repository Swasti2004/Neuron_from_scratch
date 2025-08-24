# Problem-Statement
Build a single neuron (a Perceptron) from scratch and investigate its ability to learn basic logic gates.
Your task is to implement the neuron's core structure (inputs, weights, bias), its activation function,
and its learning mechanism (e.g., the Perceptron learning rule). You will then test your neuron by
training it on different datasets representing the AND , OR , NAND , and XOR logic gates to analyze and
compare its performance on each.



# 🔌 Neural Networks & Logic Gates

This project demonstrates how **Neurons**, **Perceptrons**, and **Multi-Layer Perceptrons (MLPs)** can be used to implement **logic gates** such as AND, OR, NAND, and XOR.

---

 🧩 What is a Neuron?

A **neuron** is the fundamental unit of a neural network.

* It takes **inputs** ($x_1, x_2, \dots, x_n$)
* Each input is multiplied by a **weight** ($w_i$)
* A **bias** ($b$) is added
* The sum is passed through an **activation function** to produce the output.

Mathematically:



🧠 What is a Neural Network?

A neural network is a collection of interconnected neurons arranged in layers:

* Input layer: Receives raw data.
* Hidden layers: Extract features and model complex patterns.
* Output layer: Produces the final prediction.

When multiple neurons and layers are combined, they can approximate complex non-linear functions.

---

## ⚙️ What is a Perceptron?

A Perceptron is the simplest type of neural network (a single neuron).

* It is a linear classifier that separates data using a straight line (or hyperplane in higher dimensions).
* It applies a step activation function:



👉 Perceptrons can solve **linearly separable problems** like **AND, OR, NAND**, but fail on **XOR** because XOR is not linearly separable.

---

## 🏗️ Neuron Architecture

* Inputs: Features or signals entering the neuron.
* Weights  Parameters that determine the influence of each input.
* Bias (b): Shifts the decision boundary for more flexible learning.
* Summation: Computes z=w*x +b.
* Activation Function: Applies a transformation (Step, Sigmoid, ReLU, etc.) to decide the output.



 📐 Perceptron Convergence Theorem

The Perceptron Convergence Theorem states:

* If the dataset is **linearly separable**, the perceptron learning rule is guaranteed to find a solution (weights & bias) in a finite number of steps.
* For non-linearly separable datasets (like XOR), the algorithm will never converge.

---

## ✅ Logic Gate Results

* AND Gate: Learned perfectly with perceptron.
* OR Gate: Learned perfectly with perceptron.
* NAND Gate: Learned perfectly with perceptron.
* XOR Gate: Failed with single perceptron (\~50% accuracy).



## 🔄 Multi-Layer Perceptron (MLP) for XOR

* Introduced a hidden layer with 4 neurons.
* Used sigmoid activation and backpropagation.
* Achieved 100% accuracy  on XOR.



---

## ⚡ Challenges Faced

* Choosing the learning rate:

  * Too small → slow convergence.
  * Too large → oscillations, no convergence.
* Single Perceptron Limitation: Cannot solve XOR.
* Need for hidden layers + non-linear activations for complex problems.
* Training stability and number of epochs required for convergence.

---

📝 Key Learnings

* Neurons are the building blocks of neural networks.
* A single-layer perceptron solves only linearly separable problems.
* The perceptron Convergence Theorem guarantees convergence on separable data.
* Multi-Layer Perceptrons (MLPs) extend capability to non-linear problems like XOR.
* Deep learning’s strength comes from  stacking layers and activations to model complex functions.

---

 📌 References

* Perceptron Learning Rule & Activation Functions
* [Medium Article on Perceptron](https://medium.com/codex/single-layer-perceptron-and-activation-function-b6b74b4aae66)





