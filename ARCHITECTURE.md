# TnsrLang Architecture & Deep Dive

This document provides a comprehensive technical overview of **TnsrLang**'s architecture, compiler design, evaluation pipeline, and the mathematical AutoGrad graph.

---

## 1. High-Level System Architecture

TnsrLang is designed as an educational, transparent, and modular domain-specific language (DSL) written from scratch in C++17. The system follows a clean pipeline:

```mermaid
flowchart TD
    subgraph Frontend["Language Frontend"]
        Source["Source Code (.tnsr)"] --> Lexer["Lexer (Lexical Analysis)"]
        Lexer -->|"Tokens"| Parser["Parser (Recursive Descent)"]
        Parser -->|"Abstract Syntax Tree"| AST["AST (Statements & Expressions)"]
    end

    subgraph Runtime["Runtime & Execution"]
        AST --> Evaluator["Evaluator (Tree-Walk Interpreter)"]
        Env["Environment (Lexical Scopes)"] <--> Evaluator
        Evaluator --> NativeLib["Native Standard Library"]
    end

    subgraph MathCore["Math & AutoGrad Core"]
        NativeLib --> MathEngine["MathEngine"]
        MathEngine --> BLAS["Apple Accelerate (CBLAS)"]
        MathEngine --> DAG["AutoGrad Computational Graph (DAG)"]
        DAG --> Backprop["Topological Sort & Backward Pass"]
    end

    subgraph Interfaces["User Interfaces"]
        Evaluator --> CLI["CLI Runner & Terminal Output"]
        Evaluator --> WebServer["Embedded Web Server (httplib)"]
        WebServer --> UI["Interactive Browser UI / MNIST Canvas"]
    end
```

---

## 2. Execution Pipeline

### Step 1: Lexical Analysis (`Lexer.h`, `Lexer.cpp`)
The `Lexer` scans raw source code character-by-character and produces a linear stream of strongly typed `Token` objects (e.g. `LET`, `IDENTIFIER`, `NUMBER`, `PLUS`, `WHILE`, `FN`). Comments (`//`) and whitespace are stripped automatically.

### Step 2: Parsing & AST Construction (`Parser.h`, `Parser.cpp`, `AST.h`, `AST.cpp`)
The `Parser` implements a **Recursive Descent Parser** that reads the token stream and verifies syntactic grammar rules. It constructs an **Abstract Syntax Tree (AST)** using modern C++ smart pointers (`std::unique_ptr<Stmt>` and `std::unique_ptr<Expr>`).

* **Statements (`Stmt`):** Variable declarations (`VarDecl`), assignments (`Assignment`), loops (`WhileStmt`, `ForStmt`), conditionals (`IfStmt`), function declarations (`FunctionStmt`), return statements (`ReturnStmt`), and expressions as statements (`ExprStmt`).
* **Expressions (`Expr`):** Binary operators (`BinaryExpr`), variables (`VariableExpr`), literals (`LiteralExpr`), function calls (`CallExpr`), arrays (`ArrayExpr`), and logical comparisons.

```mermaid
graph TD
    Program["Program AST"]
    Program --> S1["VarDecl: let W1 = RandomMatrix(784, 64)"]
    Program --> S2["WhileStmt: while epoch < 100"]
    S2 --> Body["Block: Loop Body"]
    Body --> B1["Assignment: let hidden = ReLU(DotProduct(X, W1))"]
    Body --> B2["CallExpr: backward(loss)"]
    Body --> B3["Assignment: W1 = SGD_Step(W1, lr)"]
```

### Step 3: Evaluation & Environments (`Evaluator.h`, `Evaluator.cpp`)
The `Evaluator` is a **Tree-Walk Interpreter**. It traverses the AST recursively and evaluates statements in order:
* **Lexical Scoping (`Environment`):** Variables are stored in an `Environment` chain. Entering a function or block creates a child environment referencing its enclosing parent. When a scope exits, all local allocations are safely deallocated.
* **Native Function Decoupling:** Rather than hardcoding AI operations into grammar tokens, functions like `DotProduct`, `ReLU`, `Sigmoid`, `CrossEntropyLoss`, `LoadCSV`, and `SGD_Step` are registered as native C++ lambda bindings in the root environment.

---

## 3. The MathEngine & AutoGrad Engine

The mathematical foundation of TnsrLang is contained in `src/MathEngine.h` and `src/MathEngine.cpp`.

### Tensor & TensorImpl Design
To balance lightweight copying with shared ownership in computational graphs, tensors utilize a **Handle-Body Pattern**:

```mermaid
classDiagram
    class Tensor {
        +shared_ptr~TensorImpl~ impl
        +rows() size_t
        +cols() size_t
        +operator+(other)
        +operator-(other)
        +operator*(other)
        +get(r, c)
        +set(r, c, v)
    }

    class TensorImpl {
        +vector~size_t~ shape
        +vector~double~ data
        +vector~double~ grad
        +vector~shared_ptr~TensorImpl~~ parents
        +function~void()~ backward_op
        +zero_grad()
    }

    Tensor --> TensorImpl : wraps
    TensorImpl --> TensorImpl : parents DAG
```

* `Tensor`: Value-type wrapper copied cheaply across the interpreter.
* `TensorImpl`: The heap-allocated object holding:
  1. `data`: Continuous flat `std::vector<double>`.
  2. `grad`: Accumulated gradient values for backpropagation.
  3. `parents`: Direct dependency links to the parent tensors that created it.
  4. `backward_op`: A lambda function storing the exact local derivative.

### Hardware Acceleration (Apple Accelerate BLAS)
For matrix-matrix multiplication (`DotProduct`), TnsrLang invokes Apple's native `cblas_dgemm` routine:
$$\mathbf{C} = \alpha \mathbf{A} \mathbf{B} + \beta \mathbf{C}$$
This leverages vector registers (SIMD) on Apple Silicon, delivering hardware-level performance while keeping the core C++ codebase clean.

---

## 4. Automatic Differentiation (The Backpropagation DAG)

### Building the Computational Graph (Forward Pass)
Whenever an arithmetic operator or math function is called, the result tensor registers its inputs in `parents` and attaches a `backward_op` closure:

```mermaid
flowchart LR
    X["X (Input Data)"] -->|"DotProduct"| Z1["Z1 = X * W1"]
    W1["W1 (Weights)"] -->|"DotProduct"| Z1
    Z1 -->|"ReLU"| A1["A1 = ReLU(Z1)"]
    A1 -->|"DotProduct"| Z2["Z2 = A1 * W2"]
    W2["W2 (Weights)"] -->|"DotProduct"| Z2
    Z2 -->|"CrossEntropyLoss"| Loss["Loss (Scalar)"]
    Y["Y (Ground Truth)"] -->|"CrossEntropyLoss"| Loss

    style Loss fill:#f96,stroke:#333,stroke-width:2px
    style W1 fill:#9cf,stroke:#333,stroke-width:2px
    style W2 fill:#9cf,stroke:#333,stroke-width:2px
```

### Topological Sort & Reverse Traversal (Backward Pass)
When `backward(loss)` is executed:
1. **Topological Sort (`build_topo`):** A Depth-First Search (DFS) traverses all reachable parent nodes in the DAG and sorts them in dependency order.
2. **Seed Gradient:** The loss tensor's gradient is initialized to `1.0`:
   $$\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1.0$$
3. **Chain Rule Propagation:** The engine visits each node in reverse topological order, invoking `node->backward_op()` to compute upstream gradients and accumulate them into `parent->grad`:

```mermaid
sequenceDiagram
    autonumber
    actor User as TnsrLang Script
    participant Eval as Evaluator
    participant Math as MathEngine
    participant DAG as DAG Nodes
    
    User->>Eval: backward(loss)
    Eval->>Math: MathEngine::backward(loss_tensor)
    Math->>Math: Topological Sort (DFS)
    Math->>DAG: loss_impl->grad[0] = 1.0
    loop Reverse Topological Order
        Math->>DAG: node->backward_op()
        DAG->>DAG: Accumulate gradients via Chain Rule (parent->grad += ...)
    end
    Math-->>Eval: Complete
    Eval-->>User: Gradients ready on all weights
```

### Memory Safety & Preventing Circular References
Because `TensorImpl` stores a `std::function<void()> backward_op`, capturing `shared_ptr<TensorImpl>` by value would form a circular reference (`TensorImpl -> backward_op -> TensorImpl`), preventing C++'s reference counter from reaching zero and causing a RAM leak.

**Solution:** Closures capture only non-owning raw pointers to the output node (`o_ptr = o_impl.get()`) while retaining `shared_ptr` to the parents (`a_impl`, `b_impl`). This guarantees that after `backward()` or when a variable goes out of scope, all intermediate memory is immediately reclaimed by the OS.

---

## 5. Web Playground & MNIST Visualizer Architecture

TnsrLang includes an embedded HTTP server based on `httplib.h`:

```mermaid
sequenceDiagram
    autonumber
    actor Browser as Web Browser (User)
    participant Server as TnsrLang Server (:8080)
    participant Engine as Evaluator & MathEngine
    
    Browser->>Server: GET /
    Server-->>Browser: index.html (MNIST Canvas UI)
    Browser->>Browser: User draws digit '7' on 28x28 Canvas
    Browser->>Server: POST /run (TnsrLang script with pixel data)
    Server->>Engine: executeTnsrCode(source)
    Engine->>Engine: Load W1/W2, Forward Pass, Softmax
    Engine-->>Server: Output probabilities [0..9]
    Server-->>Browser: JSON/Text Response
    Browser->>Browser: Update UI prediction bars & ASCII display
```

The web server allows live experimentation without external dependencies like Node.js or Python.
