# TnsrLang v3.0 - Syntax Guide

TnsrLang is a custom, dynamically typed language built specifically for tensor mathematics, neural networks, and automatic differentiation (AutoGrad). This guide provides a complete overview of the syntax and the neural network API.

## 1. Variables & Data Types
Variables are declared using the `let` keyword. 

```javascript
// Numbers (Scalars are treated as 1x1 Tensors internally)
let learning_rate = 0.05
let max_epochs = 100

// Vectors (Column vectors)
let input = Vector([1.0, 2.0, 3.0])

// Matrices (Rows, Columns, [Data...])
let weights = Matrix(2, 2, [
    0.1, 0.2, 
    0.3, 0.4
])
```

## 2. Mathematical Operations
Standard arithmetic operators work directly on tensors. Broadcasting rules apply when multiplying matrices by scalars.

```javascript
let a = Vector([1.0, 2.0])
let b = Vector([3.0, 4.0])

let sum = a + b
let diff = a - b
let scaled = a * 0.5
```

## 3. Built-in Math Functions
TnsrLang has several built-in functions optimized for neural networks. All of these support automatic differentiation!

```javascript
// Matrix operations
let prediction = DotProduct(weights, input)
let w_T = Transpose(weights)
let flat = Reshape(prediction, 10, 1)

// Neural Network functions
let activated_relu = ReLU(prediction)         // max(0, x)
let activated_sig  = Sigmoid(prediction)      // 1 / (1 + e^-x)
let activated_soft = Softmax(prediction)      // e^x_i / sum(e^x_j)

// Data manipulations
let row_data = GetRow(input_matrix, 5)        // Fetches the 5th row
let onehot_data = OneHot(row_data, 10)        // OneHot encoding (10 classes)
```

## 4. The `Sequential` Neural Network API
TnsrLang features a powerful, high-level API similar to PyTorch or Keras for building Deep Learning models.

```javascript
let model = Sequential([
    DenseLayer(784, 128),
    ReLULayer(),
    DenseLayer(128, 64),
    ReLULayer(),
    DenseLayer(64, 10)
])

// You can also use advanced layers!
let cnn = Sequential([
    Conv2DLayer(1, 16, 3, 1),   // InChannels, OutChannels, Kernel, Stride
    MaxPool2DLayer(2, 2),       // Kernel, Stride
    DenseLayer(16 * 13 * 13, 10)
])

let gpt = Sequential([
    DenseLayer(34, 64),
    MultiHeadAttentionLayer(64, 4), // embed_dim, num_heads
    DenseLayer(64, 34)
])
```

## 5. Just-In-Time (JIT) Compiler
TnsrLang has an advanced JIT compiler that transpiles your neural network into optimized C++ machine code at runtime for extreme performance.

```javascript
// JIT-compiles the model and dynamically loads the machine code into RAM
let _j = JIT(model)

// Future Forward calls will execute the C++ machine code instead of the interpreter!
let preds = Forward(model, X, 1.0)
```

## 6. Model Training & AutoGrad
Training models in TnsrLang is done seamlessly via the built-in computational graph.

```javascript
let lr = 0.05
for epoch = 1 to 10 {
    // 1. Forward Pass (1.0 = is_training)
    let preds = Forward(model, X, 1.0)
    
    // 2. Loss Calculation
    let loss = CrossEntropyLoss(preds, Y)
    
    // 3. Backward Pass (AutoGrad computes all gradients)
    backward(loss)
    
    // 4. Update Weights
    let _opt = Optimize(model, lr)
}
```

## 7. Control Flow & Loops
You can use `if`, `else`, and `for` loops.

```javascript
for i = 1 to 5 {
    if i > 3 {
        Print("Almost done!")
    } else {
        Print("Keep going...")
    }
}
```

## 8. Functions (`fn`)
You can encapsulate logic using custom functions with the `fn` keyword. Functions support lexical scoping and can return values.

```javascript
fn forward_pass(w, x) {
    let raw = DotProduct(w, x)
    return Sigmoid(raw)
}
```

## 9. I/O & Persistence
You can load datasets and save/load model weights.

```javascript
// Load a CSV file into a Matrix Tensor
let dataset = LoadCSV("data/mnist_train.csv")

// Save and Load weights
Save(model, "weights.tnsrdata")
let loaded_weights = Load("weights.tnsrdata")
```

## 10. Comments
Single line comments are supported using `//`.
