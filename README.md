# TnsrLang 🚀

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)

**TnsrLang** is a custom, dynamically typed Domain-Specific Language (DSL) and interpreter built from scratch in C++. It is designed specifically for tensor mathematics, neural networks, and automatic differentiation (AutoGrad). 

Instead of hiding the math behind massive library abstractions, TnsrLang provides an educational and highly transparent "brutalist" environment where you can build, train, and visualize neural networks directly via code.

## 🌟 Key Features

* **AutoGrad Engine**: Built-in computational graph that tracks operations. Simply call `backward(loss)` and the engine automatically computes gradients for all linked tensors!
* **Matrix & Vector Math**: Native support for high-performance tensor operations including `DotProduct` and `Transpose`.
* **Neural Network Primitives**: Built-in activation functions like `ReLU`, `Sigmoid`, and `Softmax`.
* **Lexical Scoping & Functions**: Create modular layers using custom `fn` blocks with local variable scoping.
* **Integrated Web Terminal**: Comes with an embedded C++ HTTP server (`httplib`) that serves a brutalist, interactive browser terminal for writing code and visualizing network architectures in real-time ASCII.
* **Model Persistence**: Native `Save()` and `Load()` capabilities to store trained weights to disk.

## 🛠️ Architecture

TnsrLang is built completely from scratch without lexer/parser generators like Bison or Yacc.

1. **Lexer**: Tokenizes raw string inputs into meaningful symbols.
2. **Parser**: A recursive descent parser that constructs an Abstract Syntax Tree (AST).
3. **Evaluator**: A Tree-Walk interpreter that manages environments, control flow, and executes the math.
4. **MathEngine**: The core backbone handling all `Tensor` objects (wrapping `TensorImpl` for reference counting). It builds a Directed Acyclic Graph (DAG) during forward passes to support `.backward()` topological sorts.

## 🚀 Getting Started

### Prerequisites
* A modern C++17 compatible compiler (e.g., `clang++` or `g++`).

### Building and Running
To compile the interpreter and start the interactive web playground:

```bash
# Compile the source files
clang++ -std=c++17 src/*.cpp -o tnsrlang

# Start the interactive web server
./tnsrlang --server
```

Then, open your browser and navigate to `http://localhost:8080/`.

## 📚 Documentation

Check out the [Syntax Guide (SYNTAX.md)](SYNTAX.md) for a comprehensive overview of the language, including variables, loops, conditionals, and AutoGrad examples.

## 🤝 Contributing
Contributions are always welcome! Feel free to open an issue or submit a Pull Request if you have ideas for new features (like built-in optimizers, standard loss functions, or UI enhancements).

---
*Created by [tenri69](https://github.com/tenri69)*
