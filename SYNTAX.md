# TnsrLang v2.0 - Syntax Guide

TnsrLang is a custom, dynamically typed language built specifically for tensor mathematics, neural networks, and automatic differentiation (AutoGrad). Here is a complete overview of the syntax you can use.

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
// Matrix multiplication (Dot Product)
let prediction = DotProduct(weights, input)

// Transpose a matrix or vector
let w_T = Transpose(weights)

// Activation Functions
let activated_relu = ReLU(prediction)         // max(0, x)
let activated_sig  = Sigmoid(prediction)      // 1 / (1 + e^-x)
let activated_soft = Softmax(prediction)      // e^x_i / sum(e^x_j)
```

## 4. Control Flow (If / Else)
You can use `if` and `else` for conditional logic. Currently, `<` and `>` operators are supported.

```javascript
let error = 0.005
if error < 0.01 {
    Print("Target reached!")
} else {
    Print("Keep training...")
}
```

## 5. Loops (`for`)
TnsrLang supports `for` loops with a start and end value (inclusive).

```javascript
for epoch = 1 to 10 {
    Print("Training epoch:")
    Print(epoch)
}
```

## 6. Functions (`fn`)
You can encapsulate logic using custom functions with the `fn` keyword. Functions support lexical scoping and can return values using `return`.

```javascript
fn forward_pass(w, x) {
    let raw = DotProduct(w, x)
    return Sigmoid(raw)
}

let result = forward_pass(weights, input)
```

## 7. Automatic Differentiation (AutoGrad)
TnsrLang builds a computational graph behind the scenes. You can compute the gradient of any scalar loss with respect to all its parent variables using the `backward()` function.

```javascript
let loss = DotProduct(error, Transpose(error)) // Example loss calculation
backward(loss) // Computes gradients for all tensors leading to 'loss'

// Note: In the current version, .grad values are used internally.
```

## 8. I/O (Printing & Saving)
You can print strings and tensors to the standard output. You can also save and load tensors to the disk!

```javascript
// Printing
Print("This is a string literal")
Print(weights) // Prints the tensor dimensions and data

// Persistence
Save(weights, "model_weights.bin")
let loaded_weights = Load("model_weights.bin")
```

## 9. Comments
Single line comments are supported using `//`.

```javascript
// This is a comment and will be ignored by the Lexer
let x = 5 // This is also valid
```
