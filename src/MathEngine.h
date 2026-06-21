/*
 * =====================================================================================
 *
 *       Filename:  MathEngine.h
 *
 *    Description:  TnsrLang Interpreter Engine Component
 *
 *        Version:  1.0
 *        Created:  2026-06-20
 *       Revision:  none
 *       Compiler:  clang++
 *
 *         Author:  tenri69
 *   Organization:  TnsrLang Open Source
 *
 * =====================================================================================
 */

#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <memory>
#include <functional>

namespace tnsr {

/**
 * @brief TensorImpl is the core data structure holding N-Dimensional tensor data.
 * It manages the forward data, the gradients for backpropagation, and the computation graph.
 */
struct TensorImpl {
    std::vector<size_t> shape;
    std::vector<double> data;
    std::vector<double> grad;
    
    size_t rows() const { return shape.size() > 0 ? shape[0] : 1; }
    size_t cols() const { return shape.size() > 1 ? shape[1] : 1; }
    
    // AutoGrad
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void()> backward_op;

    TensorImpl(const std::vector<size_t>& s) : shape(s) {
        size_t num = 1;
        for (auto d : shape) num *= d;
        data.assign(num, 0.0);
        grad.assign(num, 0.0);
    }
    
    TensorImpl(const std::vector<size_t>& s, const std::vector<double>& vals) : shape(s), data(vals) {
        size_t num = 1;
        for (auto d : shape) num *= d;
        if (vals.size() != num) {
            throw std::invalid_argument("Size of values does not match shape dimensions");
        }
        grad.assign(num, 0.0);
    }
};

/**
 * @brief Tensor is a lightweight wrapper around std::shared_ptr<TensorImpl>.
 * It acts as the primary interface for all mathematical operations and AutoGrad.
 * Using a shared_ptr ensures that the computation graph stays alive as long as 
 * any tensor references it.
 */
class Tensor {
public:
    std::shared_ptr<TensorImpl> impl;

    Tensor() : impl(nullptr) {}
    Tensor(const std::vector<size_t>& shape) : impl(std::make_shared<TensorImpl>(shape)) {}
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& vals) : impl(std::make_shared<TensorImpl>(shape, vals)) {}
    Tensor(size_t r, size_t c) : impl(std::make_shared<TensorImpl>(std::vector<size_t>{r, c})) {}
    Tensor(size_t r, size_t c, const std::vector<double>& vals) : impl(std::make_shared<TensorImpl>(std::vector<size_t>{r, c}, vals)) {}
    Tensor(std::shared_ptr<TensorImpl> impl) : impl(impl) {}

    size_t rows() const { return impl->shape.size() > 0 ? impl->shape[0] : 1; }
    size_t cols() const { return impl->shape.size() > 1 ? impl->shape[1] : 1; }

    double get(size_t r, size_t c) const { return impl->data[r * cols() + c]; }
    void set(size_t r, size_t c, double val) { impl->data[r * cols() + c] = val; }
    
    double get_grad(size_t r, size_t c) const { return impl->grad[r * cols() + c]; }
    void set_grad(size_t r, size_t c, double val) { impl->grad[r * cols() + c] = val; }

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const; // Element-wise multiplication
    Tensor operator*(double scalar) const;

    Tensor get_row(size_t index) const;

    void print(std::ostream& out = std::cout) const;
    void print_grad(std::ostream& out = std::cout) const;
    void backward();
};

/**
 * @brief Performs Matrix Multiplication (or batched dot product) between two tensors.
 */
Tensor DotProduct(const Tensor& a, const Tensor& b);

/**
 * @brief Rectified Linear Unit activation function. f(x) = max(0, x)
 */
Tensor ReLU(const Tensor& a);

/**
 * @brief Sigmoid activation function. f(x) = 1 / (1 + e^-x)
 */
Tensor Sigmoid(const Tensor& a);
Tensor Softmax(const Tensor& a);
Tensor Transpose(const Tensor& a);

void SaveTensor(const Tensor& t, const std::string& filename);
Tensor LoadTensor(const std::string& filename);
Tensor LoadCSV(const std::string& filename);
Tensor RandomMatrix(size_t rows, size_t cols);

/**
 * @brief Slices a tensor to extract a submatrix.
 */
Tensor Slice(const Tensor& a, size_t row_start, size_t row_end, size_t col_start, size_t col_end);

/**
 * @brief Converts a 1D tensor of class labels into a 2D one-hot encoded matrix.
 */
Tensor OneHot(const Tensor& labels, size_t num_classes);

/**
 * @brief Computes Softmax probabilities and Cross-Entropy Loss in a single, numerically stable step.
 * Returns a 1x1 scalar tensor containing the average loss.
 */
Tensor CrossEntropyLoss(const Tensor& logits, const Tensor& targets);

} // namespace tnsr
