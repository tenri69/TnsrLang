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

struct TensorImpl {
    size_t rows;
    size_t cols;
    std::vector<double> data;
    std::vector<double> grad;
    
    // AutoGrad
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void()> backward_op;

    TensorImpl(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0), grad(r * c, 0.0) {}
    TensorImpl(size_t r, size_t c, const std::vector<double>& vals) : rows(r), cols(c), data(vals), grad(r * c, 0.0) {
        if (vals.size() != r * c) {
            throw std::invalid_argument("Size of values does not match rows * cols");
        }
    }
};

class Tensor {
public:
    std::shared_ptr<TensorImpl> impl;

    Tensor() : impl(nullptr) {}
    Tensor(size_t r, size_t c) : impl(std::make_shared<TensorImpl>(r, c)) {}
    Tensor(size_t r, size_t c, const std::vector<double>& vals) : impl(std::make_shared<TensorImpl>(r, c, vals)) {}
    Tensor(std::shared_ptr<TensorImpl> impl) : impl(impl) {}

    double get(size_t r, size_t c) const { return impl->data[r * impl->cols + c]; }
    void set(size_t r, size_t c, double val) { impl->data[r * impl->cols + c] = val; }
    
    double get_grad(size_t r, size_t c) const { return impl->grad[r * impl->cols + c]; }
    void set_grad(size_t r, size_t c, double val) { impl->grad[r * impl->cols + c] = val; }

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    void print(std::ostream& out = std::cout) const;
    void print_grad(std::ostream& out = std::cout) const;
    void backward();
};

Tensor DotProduct(const Tensor& a, const Tensor& b);
Tensor ReLU(const Tensor& a);
Tensor Sigmoid(const Tensor& a);
Tensor Softmax(const Tensor& a);
Tensor Transpose(const Tensor& a);

void SaveTensor(const Tensor& t, const std::string& filename);
Tensor LoadTensor(const std::string& filename);

} // namespace tnsr
