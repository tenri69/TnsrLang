# Core Concepts & Tensor Mathematics

Welcome to **TnsrLang**! If you are new to machine learning, neural networks, or the inner workings of an AutoGrad engine, this guide will introduce you to the core mathematical concepts and explain how they are implemented under the hood in this language.

---

## 1. What is a Tensor?

In machine learning, a **Tensor** is a generalization of numbers and arrays to an arbitrary number of dimensions.

* **Scalar (0-Dimensional Tensor):** A single number. Example: `5.0`. In TnsrLang, this is represented internally as a 1x1 Matrix.
* **Vector (1-Dimensional Tensor):** An array of numbers. Think of it as a point in space or a list of features. Example: `[1.0, 2.0, 3.0]`. In TnsrLang, this is a column vector (an N x 1 Matrix).
* **Matrix (2-Dimensional Tensor):** A grid of numbers with rows and columns. Matrices are used to hold the "weights" of our neural networks. Example:
  ```
  [ 0.1, 0.2 ]
  [ 0.3, 0.4 ]
  ```
* **Higher-Dimensional Tensors:** Used for things like color images (Width x Height x RGB), though TnsrLang currently focuses on 2D matrices for simplicity.

---

## 2. The Mechanics of a Neural Network

A neural network is essentially a mathematical function that takes an input (like image pixels or housing data) and transforms it into an output (like a classification or a price prediction). It does this through layers of "neurons".

### Forward Pass
The process of generating a prediction is called the **forward pass**. It consists of two main operations:

1. **Dot Product (Matrix Multiplication):** We multiply our `input` vector by a `weights` matrix. The weights represent the network's "knowledge". 
   * In TnsrLang: `let raw_output = DotProduct(weights, input)`
2. **Activation Function:** To allow the network to learn complex, non-linear patterns, we pass the raw output through an activation function.
   * **ReLU (Rectified Linear Unit):** Turns all negative numbers to 0. `ReLU(x)`
   * **Sigmoid:** Squashes all numbers into a range between 0 and 1, acting like a probability. `Sigmoid(x)`

### Measuring the Error (Loss)
Once we have a prediction, we compare it to our target (the correct answer). 
```javascript
let error = prediction - target
let loss = DotProduct(error, Transpose(error)) // Mean Squared Error
```

---

## 3. How Learning Works: AutoGrad

If the network guesses wrong, how does it know how to adjust its weights to get closer next time? This is done using **Calculus** (specifically, the Chain Rule), and is handled automatically by TnsrLang's **AutoGrad** (Automatic Differentiation) engine.

### The Computational Graph
When you run code in TnsrLang, the `MathEngine` does not just calculate the numbers. It silently builds a **Directed Acyclic Graph (DAG)**. 

Every time you perform an operation (e.g., `let c = a + b`), the engine creates a new Tensor `c` that remembers:
* Its parents (`a` and `b`).
* The operation that created it (Addition).

### The Backward Pass
When you call `backward(loss)` in TnsrLang, the engine traverses this graph backwards from the `loss` all the way to the original `weights`. 

For every node, it applies the derivative of the operation that created it, accumulating a value called a **Gradient** (`.grad`). The gradient tells us exactly how much a change in a specific weight will affect the final loss.

### Gradient Descent
Finally, we use the gradients to update our weights, taking a small step (determined by the `learning_rate`) in the direction that minimizes the loss:
```javascript
// In concept:
weights = weights - (weights.grad * learning_rate)
```

---

## 4. TnsrLang Internal Architecture

If you are a contributor looking at the C++ code, here is how the concepts map to the implementation:

1. **`Tensor` & `TensorImpl` (MathEngine.h)**
   * To support the computational graph, `Tensor` is just a lightweight wrapper around a `std::shared_ptr<TensorImpl>`. 
   * `TensorImpl` holds the actual `std::vector<double> data`, the `std::vector<double> grad`, and a list of `std::shared_ptr<TensorImpl>` pointing to its parents.
2. **`backward_op` (MathEngine.cpp)**
   * Every math function (like `DotProduct` or `Sigmoid`) assigns a `backward_op` lambda function to the resulting tensor. This lambda contains the exact calculus derivative for that specific operation.
3. **`backward()`**
   * When triggered, it performs a topological sort of all parent tensors, ensuring gradients are computed in the correct order (from output back to inputs) and calls each `backward_op`.

Welcome to the cutting edge of brutalist neural network programming!
