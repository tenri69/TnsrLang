import sys

def inject():
    with open('src/MathEngine.cpp', 'r') as f:
        content = f.read()

    new_code = """
Tensor Slice(const Tensor& a, size_t row_start, size_t row_end, size_t col_start, size_t col_end) {
    if (row_start >= row_end || col_start >= col_end || row_end > a.rows() || col_end > a.cols()) {
        throw std::runtime_error("Invalid slice dimensions");
    }
    size_t new_rows = row_end - row_start;
    size_t new_cols = col_end - col_start;
    Tensor out(new_rows, new_cols);
    for (size_t r = 0; r < new_rows; ++r) {
        for (size_t c = 0; c < new_cols; ++c) {
            out.set(r, c, a.get(row_start + r, col_start + c));
        }
    }
    return out;
}

Tensor OneHot(const Tensor& labels, size_t num_classes) {
    size_t N = labels.rows() > labels.cols() ? labels.rows() : labels.cols();
    Tensor out(N, num_classes);
    for (size_t i = 0; i < N; ++i) {
        int label = (int)labels.get(labels.rows() > labels.cols() ? i : 0, labels.cols() > labels.rows() ? i : 0);
        if (label >= 0 && label < num_classes) {
            out.set(i, label, 1.0);
        }
    }
    return out;
}

Tensor CrossEntropyLoss(const Tensor& logits, const Tensor& targets) {
    if (logits.rows() != targets.rows() || logits.cols() != targets.cols()) {
        throw std::runtime_error("Logits and targets must have same dimensions for CrossEntropyLoss");
    }
    
    size_t N = logits.rows();
    size_t C = logits.cols();
    
    Tensor probs(N, C);
    double total_loss = 0.0;
    
    for (size_t i = 0; i < N; ++i) {
        double max_val = logits.get(i, 0);
        for (size_t j = 1; j < C; ++j) {
            if (logits.get(i, j) > max_val) max_val = logits.get(i, j);
        }
        
        double sum_exp = 0.0;
        for (size_t j = 0; j < C; ++j) {
            double e = std::exp(logits.get(i, j) - max_val);
            probs.set(i, j, e);
            sum_exp += e;
        }
        
        for (size_t j = 0; j < C; ++j) {
            double p = probs.get(i, j) / sum_exp;
            probs.set(i, j, p);
            if (targets.get(i, j) == 1.0) {
                total_loss += -std::log(p + 1e-9);
            }
        }
    }
    
    Tensor loss_tensor(1, 1);
    loss_tensor.set(0, 0, total_loss / N);
    
    auto logits_impl = logits.impl;
    auto loss_impl = loss_tensor.impl;
    
    loss_impl->parents = {logits_impl};
    loss_impl->backward_op = [logits_impl, probs, targets, N, C, loss_impl]() {
        double upstream_grad = loss_impl->grad[0];
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < C; ++j) {
                double diff = probs.get(i, j) - targets.get(i, j);
                logits_impl->grad[i * C + j] += upstream_grad * diff / N;
            }
        }
    };
    
    return loss_tensor;
}

Tensor RandomMatrix(size_t rows, size_t cols) {"""

    content = content.replace("Tensor RandomMatrix(size_t rows, size_t cols) {", new_code)
    
    with open('src/MathEngine.cpp', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    inject()
