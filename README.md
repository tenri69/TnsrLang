```text
 _______ _   _  _____ _____  _               _   _  _____ 
|__   __| \ | |/ ____|  __ \| |        /\   | \ | |/ ____|
   | |  |  \| | (___ | |__) | |       /  \  |  \| | |  __ 
   | |  | . ` |\___ \|  _  /| |      / /\ \ | . ` | | |_ |
   | |  | |\  |____) | | \ \| |____ / ____ \| |\  | |__| |
   |_|  |_| \_|_____/|_|  \_\______/_/    \_\_| \_|\_____|
```

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)

**TnsrLang** is a custom, dynamically typed Domain-Specific Language (DSL) and interpreter built from scratch in C++. It is designed specifically for tensor mathematics, neural networks, and automatic differentiation (AutoGrad). 

Instead of hiding the math behind massive library abstractions, TnsrLang provides an educational and highly transparent "brutalist" environment where you can build, train, and visualize neural networks directly via code.

## Key Features

* **AutoGrad Engine**: Built-in computational graph that tracks operations. Simply call `backward(loss)` and the engine automatically computes gradients for all linked tensors!
* **High-Level `Sequential` API**: Build neural networks easily using `DenseLayer`, `Conv2DLayer`, `ReLULayer`, `MaxPool2DLayer`, and even `MultiHeadAttentionLayer`.
* **TnsrJIT Compiler**: TnsrLang features a built-in Just-In-Time (JIT) compiler! By calling `JIT(model)`, your network is dynamically transpiled into optimized C++ machine code, compiled via Clang, and loaded directly into RAM for extreme execution speed.
* **Apple Accelerate Integration**: Deeply integrated with macOS Accelerate (`cblas_dgemm`) for lightning-fast matrix multiplications.
* **Transformer Capabilities**: Capable of full Language Modeling and Tokenization workflows, proving that the engine scales from simple math to complex AI architectures like GPT.
* **Integrated Web IDE**: Comes with an embedded C++ HTTP server (`httplib`) that serves a modern browser playground (featuring CodeMirror syntax highlighting) for writing and running code.

## Architecture

TnsrLang is built completely from scratch without lexer/parser generators like Bison or Yacc.

1. **Lexer**: Tokenizes raw string inputs into meaningful symbols.
2. **Parser**: A recursive descent parser that constructs an Abstract Syntax Tree (AST).
3. **Evaluator**: A Tree-Walk interpreter that manages environments, control flow, and executes the math.
4. **MathEngine**: The core backbone handling all `Tensor` objects (wrapping `TensorImpl` for reference counting). It builds a Directed Acyclic Graph (DAG) during forward passes to support `.backward()` topological sorts.
5. **JIT Engine**: Analyzes AST structures and dynamically compiles shared libraries (`.dylib`) using `clang++` to completely bypass the evaluator loop during Neural Network training.

## Getting Started

### Prerequisites
* A modern C++17 compatible compiler (e.g., `clang++` or `g++`).
* macOS (for Accelerate framework).

### Building and Running
To compile the interpreter and start the interactive web playground:

```bash
# Compile the source files with Apple Accelerate framework
clang++ -std=c++17 -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -fPIC src/*.cpp -o tnsrlang -framework Accelerate

# Start the interactive web server
./tnsrlang --server
```

Then, open your browser and navigate to `http://localhost:8080/`.

### Running Demos
TnsrLang comes with several pre-written demonstrations showcasing different AI architectures. You can find these in the `examples/` directory.

```bash
./tnsrlang examples/demo_cnn.tnsr       # Convolutional Neural Network
./tnsrlang examples/demo_transformer.tnsr # Multi-Head Attention
./tnsrlang examples/demo_gpt.tnsr       # Mini-GPT Language Model
```

## Documentation

Check out the [Syntax Guide (SYNTAX.md)](SYNTAX.md) for a comprehensive overview of the language, including variables, loops, conditionals, and AutoGrad examples.

New to machine learning? Read the [Core Concepts & Math Guide (CORE_CONCEPTS.md)](CORE_CONCEPTS.md) to understand how tensors, neural networks, and automatic differentiation work under the hood.

## Contributing
Contributions are always welcome! Feel free to open an issue or submit a Pull Request if you have ideas for new features (like built-in optimizers, standard loss functions, or UI enhancements).

---
*Created by [tenri69](https://github.com/tenri69)*
