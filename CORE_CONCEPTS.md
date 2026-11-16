# Core Concepts & Tensor Mathematics

Welcome to **TnsrLang**! If you are new to machine learning, neural networks, or the inner workings of an AutoGrad engine, this guide will introduce you to the core mathematical concepts and explain how they are implemented under the hood in this language.

---

## 1. What is a Tensor?

In machine learning, a **Tensor** is a generalization of numbers and arrays to an arbitrary number of dimensions. TnsrLang's core engine uses a dynamic `shape` vector to represent these.

* **Scalar (0-Dimensional Tensor):** A single number. Example: `5.0`.
* **Vector (1-Dimensional Tensor):** An array of numbers. Think of it as a point in space or a list of features. Example: `[1.0, 2.0, 3.0]`. 
* **Matrix (2-Dimensional Tensor):** A grid of numbers with rows and columns. Matrices are used to hold the "weights" of our neural networks. Example:
  ```
  [ 0.1, 0.2 ]
  [ 0.3, 0.4 ]
  ```
* **Higher-Dimensional Tensors:** Used for things like color images (Width x Height x RGB) or batches of data. Example: a 4D tensor `[32, 256, 256, 3]` represents 32 images of 256x256 pixels with 3 color channels.

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

// In TnsrLang:
weights = SGD_Step(weights, learning_rate)
```

---

## 4. TnsrLang Internal Architecture

If you are a contributor looking at the C++ code, here is how the concepts map to the implementation:

1. **`Tensor` & `TensorImpl` (MathEngine.h)**
   * To support the computational graph, `Tensor` is a lightweight handle wrapping a `std::shared_ptr<TensorImpl>`. 
   * `TensorImpl` holds the actual contiguous `std::vector<double> data` (flat row-major array), dimensions, accumulated `grad`, and a list of parent pointers.
2. **`backward_op` (MathEngine.cpp)**
   * Every math operation attaches a `backward_op` closure to the resulting tensor, computing local derivatives and accumulating upstream gradients via the chain rule. To ensure zero memory leaks, closures capture non-owning raw pointers (`o_ptr = out.impl.get()`) to avoid cyclic `shared_ptr` retention.
3. **The Standard Library (Evaluator.cpp)**
   * TnsrLang's compiler (Lexer/Parser) has no hardcoded AI dependencies. Instead, math operations are registered in the global environment as Native C++ Functions, cleanly decoupling language syntax from low-level execution.

Welcome to the cutting edge of brutalist neural network programming!
