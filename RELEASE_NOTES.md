# TnsrLang v3.0 Release Notes

Welcome to the biggest update to TnsrLang yet! This release transitions TnsrLang from a basic educational math parser into a high-performance, State-of-the-Art Deep Learning engine capable of Language Modeling and Image Recognition.

## 🔥 Major Features

### 1. TnsrJIT: Just-In-Time Compiler
TnsrLang is no longer just a slow tree-walking interpreter. By calling `let _j = JIT(model)`, TnsrLang analyzes your Neural Network architecture, transpiles it into optimized C++ source code (`jit_model_X.cpp`), compiles it using Apple's Clang (`-O3`), and loads the machine code dynamically into RAM. This effectively mimics PyTorch 2.0's `torch.compile`, resulting in extreme performance leaps by completely bypassing the evaluator loop.

### 2. Transformer Architecture & Attention Mechanisms
We've successfully implemented `MultiHeadAttentionLayer(embed_dim, heads)` into the `Sequential` API.
To support this, the underlying C++ `MathEngine` was extended with two highly complex Native Primitives (complete with their own topological AutoGrad derivatives):
*   `Concat(a, b, axis)`
*   `LayerNorm(a)`

TnsrLang is now fully capable of building Mini-GPT architectures! (See `examples/demo_gpt.tnsr`).

### 3. Convolutional Neural Networks (CNNs)
TnsrLang now supports Image Recognition! We implemented `im2col` matrix algorithms in raw C++ to build:
*   `Conv2DLayer(in_channels, out_channels, kernel_size, stride)`
*   `MaxPool2DLayer(kernel_size, stride)`
This enables state-of-the-art digit recognition (MNIST).

### 4. CodeMirror Web IDE Integration
The local TnsrLang playground (`http://localhost:8080`) has been upgraded. We integrated CodeMirror, providing real-time syntax highlighting for your TnsrLang scripts, replacing the standard HTML `<textarea>`.

## 🧹 Housekeeping & Project Structure
To keep the codebase clean, we have officially moved all demonstrations into the `examples/` directory and all neural network data (like `.csv`, `.gz`, and `.tnsrdata`) into the `data/` directory.

## 📝 New Native Functions
*   `GetRow(tensor, index)`: Efficiently extracts a single row from a batched dataset.
*   `OneHot(labels, num_classes)`: Converts scalar category labels into one-hot encoded tensors.
*   `Slice(...)`: Supports memory slicing for Attention mechanisms.
*   `CrossEntropyLoss(logits, targets)`: Automatically handles Softmax probabilities + loss calculation for stable backward passes.
