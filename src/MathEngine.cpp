/*
 * =====================================================================================
 *
 *       Filename:  MathEngine.cpp
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

#include "MathEngine.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <set>
#include <random>

namespace tnsr {

// --- AutoGrad: The Backward Pass ---
// Performs a topological sort of the computational graph and applies the chain rule
// by calling `backward_op` on each tensor in reverse topological order.
void Tensor::backward() {
    std::vector<std::shared_ptr<TensorImpl>> topo;
    std::set<TensorImpl*> visited;
    
    std::function<void(std::shared_ptr<TensorImpl>)> build_topo = [&](std::shared_ptr<TensorImpl> v) {
        if (visited.find(v.get()) == visited.end()) {
            visited.insert(v.get());
            for (auto child : v->parents) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };
    
    build_topo(impl);
    
    std::fill(impl->grad.begin(), impl->grad.end(), 1.0);
    
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_op) {
            (*it)->backward_op();
        }
    }
}

// --- Operator Overloading & Broadcasting ---
// Implements addition with support for "Broadcasting". If one tensor is a column vector
// or row vector, it will automatically broadcast its values to match the dimensions 
// of the larger matrix.
Tensor Tensor::operator+(const Tensor& other) const {
    if (impl->rows() == other.impl->rows() && impl->cols() == other.impl->cols()) {
        Tensor out(impl->rows(), impl->cols());
        for (size_t i = 0; i < impl->data.size(); ++i) {
            out.impl->data[i] = impl->data[i] + other.impl->data[i];
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o_ptr = o.get()]() {
            for (size_t i = 0; i < o_ptr->grad.size(); ++i) {
                a->grad[i] += o_ptr->grad[i];
                b->grad[i] += o_ptr->grad[i];
            }
        };
        return out;
    } else if (impl->cols() == other.impl->cols() && other.impl->rows() == 1) {
        // Broadcast across rows
        Tensor out(impl->rows(), impl->cols());
        for (size_t r = 0; r < impl->rows(); ++r) {
            for (size_t c = 0; c < impl->cols(); ++c) {
                out.impl->data[r * impl->cols() + c] = impl->data[r * impl->cols() + c] + other.impl->data[c];
            }
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o_ptr = o.get()]() {
            for (size_t r = 0; r < o_ptr->rows(); ++r) {
                for (size_t c = 0; c < o_ptr->cols(); ++c) {
                    a->grad[r * o_ptr->cols() + c] += o_ptr->grad[r * o_ptr->cols() + c];
                    b->grad[c] += o_ptr->grad[r * o_ptr->cols() + c];
                }
            }
        };
        return out;
    } else if (impl->rows() == other.impl->rows() && other.impl->cols() == 1) {
        // Broadcast across columns
        Tensor out(impl->rows(), impl->cols());
        for (size_t r = 0; r < impl->rows(); ++r) {
            for (size_t c = 0; c < impl->cols(); ++c) {
                out.impl->data[r * impl->cols() + c] = impl->data[r * impl->cols() + c] + other.impl->data[r];
            }
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o_ptr = o.get()]() {
            for (size_t r = 0; r < o_ptr->rows(); ++r) {
                for (size_t c = 0; c < o_ptr->cols(); ++c) {
                    a->grad[r * o_ptr->cols() + c] += o_ptr->grad[r * o_ptr->cols() + c];
                    b->grad[r] += o_ptr->grad[r * o_ptr->cols() + c];
                }
            }
        };
        return out;
    } else if (other.impl->rows() == 1 && other.impl->cols() == 1) {
        // Scalar broadcasting
        Tensor out(impl->rows(), impl->cols());
        for (size_t i = 0; i < impl->data.size(); ++i) {
            out.impl->data[i] = impl->data[i] + other.impl->data[0];
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o_ptr = o.get()]() {
            for (size_t i = 0; i < o_ptr->grad.size(); ++i) {
                a->grad[i] += o_ptr->grad[i];
                b->grad[0] += o_ptr->grad[i];
            }
        };
        return out;
    }
    
    throw std::invalid_argument("Dimensions must match or be broadcastable for addition");
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (impl->rows() == other.impl->rows() && impl->cols() == other.impl->cols()) {
        Tensor out(impl->rows(), impl->cols());
        for (size_t i = 0; i < impl->data.size(); ++i) {
            out.impl->data[i] = impl->data[i] - other.impl->data[i];
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o_ptr = o.get()]() {
            for (size_t i = 0; i < o_ptr->grad.size(); ++i) {
                a->grad[i] += o_ptr->grad[i];
                b->grad[i] -= o_ptr->grad[i];
            }
        };
        return out;
    } else if (impl->cols() == other.impl->cols() && other.impl->rows() == 1) {
        // Broadcast across rows
        Tensor out(impl->rows(), impl->cols());
        for (size_t r = 0; r < impl->rows(); ++r) {
            for (size_t c = 0; c < impl->cols(); ++c) {
                out.impl->data[r * impl->cols() + c] = impl->data[r * impl->cols() + c] - other.impl->data[c];
            }
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o_ptr = o.get()]() {
            for (size_t r = 0; r < o_ptr->rows(); ++r) {
                for (size_t c = 0; c < o_ptr->cols(); ++c) {
                    a->grad[r * o_ptr->cols() + c] += o_ptr->grad[r * o_ptr->cols() + c];
                    b->grad[c] -= o_ptr->grad[r * o_ptr->cols() + c];
                }
            }
        };
        return out;
    } else if (impl->rows() == other.impl->rows() && other.impl->cols() == 1) {
        // Broadcast across columns
        Tensor out(impl->rows(), impl->cols());
        for (size_t r = 0; r < impl->rows(); ++r) {
            for (size_t c = 0; c < impl->cols(); ++c) {
                out.impl->data[r * impl->cols() + c] = impl->data[r * impl->cols() + c] - other.impl->data[r];
            }
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o_ptr = o.get()]() {
            for (size_t r = 0; r < o_ptr->rows(); ++r) {
                for (size_t c = 0; c < o_ptr->cols(); ++c) {
                    a->grad[r * o_ptr->cols() + c] += o_ptr->grad[r * o_ptr->cols() + c];
                    b->grad[r] -= o_ptr->grad[r * o_ptr->cols() + c];
                }
            }
        };
        return out;
    } else if (other.impl->rows() == 1 && other.impl->cols() == 1) {
        // Scalar broadcasting
        Tensor out(impl->rows(), impl->cols());
        for (size_t i = 0; i < impl->data.size(); ++i) {
            out.impl->data[i] = impl->data[i] - other.impl->data[0];
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o_ptr = o.get()]() {
            for (size_t i = 0; i < o_ptr->grad.size(); ++i) {
                a->grad[i] += o_ptr->grad[i];
                b->grad[0] -= o_ptr->grad[i];
            }
        };
        return out;
    }
    
    throw std::invalid_argument("Dimensions must match or be broadcastable for subtraction");
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (impl->rows() != other.impl->rows() || impl->cols() != other.impl->cols()) {
        throw std::invalid_argument("Dimensions must match for element-wise multiplication");
    }
    Tensor out(impl->rows(), impl->cols());
    for (size_t i = 0; i < impl->data.size(); ++i) {
        out.impl->data[i] = impl->data[i] * other.impl->data[i];
    }
    
    auto a = impl;
    auto b = other.impl;
    auto o = out.impl;
    
    o->parents = {a, b};
    o->backward_op = [a, b, o_ptr = o.get()]() {
        for (size_t i = 0; i < o_ptr->grad.size(); ++i) {
            a->grad[i] += b->data[i] * o_ptr->grad[i];
            b->grad[i] += a->data[i] * o_ptr->grad[i];
        }
    };
    return out;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor out(impl->rows(), impl->cols());
    for (size_t i = 0; i < impl->data.size(); ++i) {
        out.impl->data[i] = impl->data[i] * scalar;
    }
    
    auto a = impl;
    auto o = out.impl;
    
    o->parents = {a};
    o->backward_op = [a, o_ptr = o.get(), scalar]() {
        for (size_t i = 0; i < o_ptr->grad.size(); ++i) {
            a->grad[i] += scalar * o_ptr->grad[i];
        }
    };
    return out;
}

Tensor Tensor::get_row(size_t index) const {
    if (index >= impl->rows()) {
        throw std::out_of_range("Row index out of bounds");
    }
    Tensor out(1, impl->cols());
    for (size_t c = 0; c < impl->cols(); ++c) {
        out.impl->data[c] = get(index, c);
    }
    
    auto a = impl;
    auto o = out.impl;
    
    o->parents = {a};
    o->backward_op = [a, o_ptr = o.get(), index]() {
        for (size_t c = 0; c < o_ptr->cols(); ++c) {
            a->grad[index * a->cols() + c] += o_ptr->grad[c];
        }
    };
    return out;
}

void Tensor::print(std::ostream& out) const {
    if (!impl) {
        out << "[null tensor]\n";
        return;
    }
    out << "[\n";
    for (size_t r = 0; r < impl->rows(); ++r) {
        out << "  [";
        for (size_t c = 0; c < impl->cols(); ++c) {
            out << std::setw(8) << std::setprecision(4) << std::fixed << get(r, c);
            if (c < impl->cols() - 1) out << ", ";
        }
        out << " ]\n";
    }
    out << "]\n";
}

void Tensor::print_grad(std::ostream& out) const {
    if (!impl) {
        out << "[null tensor]\n";
        return;
    }
    out << "[\n";
    for (size_t r = 0; r < impl->rows(); ++r) {
        out << "  [";
        for (size_t c = 0; c < impl->cols(); ++c) {
            out << std::setw(8) << std::setprecision(4) << std::fixed << get_grad(r, c);
            if (c < impl->cols() - 1) out << ", ";
        }
    }
    out << "]\n";
}

// --- Native Math Operations ---
// These functions are injected into the language as Native Functions.
// Implemented directly in standard C++ without external dependencies.

Tensor DotProduct(const Tensor& a, const Tensor& b) {
    if (a.cols() != b.rows()) {
        throw std::runtime_error("Matrix dimensions must agree for DotProduct");
    }
    
    Tensor out(a.rows(), b.cols());
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t k = 0; k < a.cols(); ++k) {
            double a_val = a.impl->data[i * a.cols() + k];
            for (size_t j = 0; j < b.cols(); ++j) {
                out.impl->data[i * b.cols() + j] += a_val * b.impl->data[k * b.cols() + j];
            }
        }
    }
    
    auto a_impl = a.impl;
    auto b_impl = b.impl;
    auto o_impl = out.impl;
    
    o_impl->parents = {a_impl, b_impl};
    o_impl->backward_op = [a_impl, b_impl, o_ptr = o_impl.get()]() {
        for (size_t i = 0; i < a_impl->rows(); ++i) {
            for (size_t j = 0; j < b_impl->cols(); ++j) {
                double g = o_ptr->grad[i * o_ptr->cols() + j];
                for (size_t k = 0; k < a_impl->cols(); ++k) {
                    a_impl->grad[i * a_impl->cols() + k] += g * b_impl->data[k * b_impl->cols() + j];
                    b_impl->grad[k * b_impl->cols() + j] += a_impl->data[i * a_impl->cols() + k] * g;
                }
            }
        }
    };
    
    return out;
}

Tensor ReLU(const Tensor& a) {
    Tensor out(a.impl->rows(), a.impl->cols());
    for (size_t i = 0; i < a.impl->data.size(); ++i) {
        out.impl->data[i] = std::max(0.0, a.impl->data[i]);
    }
    
    auto a_impl = a.impl;
    auto o_impl = out.impl;
    
    o_impl->parents = {a_impl};
    o_impl->backward_op = [a_impl, o_ptr = o_impl.get()]() {
        for (size_t i = 0; i < o_ptr->grad.size(); ++i) {
            a_impl->grad[i] += (a_impl->data[i] > 0 ? 1.0 : 0.0) * o_ptr->grad[i];
        }
    };
    return out;
}

Tensor Sigmoid(const Tensor& a) {
    Tensor out(a.impl->rows(), a.impl->cols());
    for (size_t i = 0; i < a.impl->data.size(); ++i) {
        out.impl->data[i] = 1.0 / (1.0 + std::exp(-a.impl->data[i]));
    }
    
    auto a_impl = a.impl;
    auto o_impl = out.impl;
    
    o_impl->parents = {a_impl};
    o_impl->backward_op = [a_impl, o_ptr = o_impl.get()]() {
        for (size_t i = 0; i < o_ptr->grad.size(); ++i) {
            double s = o_ptr->data[i];
            a_impl->grad[i] += s * (1.0 - s) * o_ptr->grad[i];
        }
    };
    return out;
}

Tensor Softmax(const Tensor& a) {
    Tensor out(a.impl->rows(), a.impl->cols());
    for (size_t r = 0; r < a.impl->rows(); ++r) {
        double max_val = a.get(r, 0);
        for (size_t c = 1; c < a.impl->cols(); ++c) {
            max_val = std::max(max_val, a.get(r, c));
        }
        double sum = 0.0;
        for (size_t c = 0; c < a.impl->cols(); ++c) {
            double e = std::exp(a.get(r, c) - max_val);
            out.set(r, c, e);
            sum += e;
        }
        for (size_t c = 0; c < a.impl->cols(); ++c) {
            out.set(r, c, out.get(r, c) / sum);
        }
    }
    
    auto a_impl = a.impl;
    auto o_impl = out.impl;
    
    o_impl->parents = {a_impl};
    o_impl->backward_op = [a_impl, o_ptr = o_impl.get()]() {
        for (size_t r = 0; r < o_ptr->rows(); ++r) {
            for (size_t i = 0; i < o_ptr->cols(); ++i) {
                double out_i = o_ptr->data[r * o_ptr->cols() + i];
                for (size_t j = 0; j < o_ptr->cols(); ++j) {
                    double out_j = o_ptr->data[r * o_ptr->cols() + j];
                    double grad_j = o_ptr->grad[r * o_ptr->cols() + j];
                    if (i == j) {
                        a_impl->grad[r * a_impl->cols() + i] += out_i * (1.0 - out_i) * grad_j;
                    } else {
                        a_impl->grad[r * a_impl->cols() + i] += -out_j * out_i * grad_j;
                    }
                }
            }
        }
    };
    return out;
}

Tensor Transpose(const Tensor& a) {
    Tensor out(a.impl->cols(), a.impl->rows());
    for (size_t i = 0; i < a.impl->rows(); ++i) {
        for (size_t j = 0; j < a.impl->cols(); ++j) {
            out.set(j, i, a.get(i, j));
        }
    }
    
    auto a_impl = a.impl;
    auto o_impl = out.impl;
    o_impl->parents = {a_impl};
    o_impl->backward_op = [a_impl, o_ptr = o_impl.get()]() {
        for (size_t i = 0; i < a_impl->rows(); ++i) {
            for (size_t j = 0; j < a_impl->cols(); ++j) {
                a_impl->grad[i * a_impl->cols() + j] += o_ptr->grad[j * o_ptr->cols() + i];
            }
        }
    };
    return out;
}

void SaveTensor(const Tensor& t, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("Could not open file for saving");
    file << t.impl->rows() << " " << t.impl->cols() << "\n";
    for (double val : t.impl->data) {
        file << val << " ";
    }
}

Tensor LoadTensor(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Could not open file for loading");
    size_t r, c;
    file >> r >> c;
    std::vector<double> vals(r * c);
    for (size_t i = 0; i < r * c; ++i) {
        file >> vals[i];
    }
    return Tensor(r, c, vals);
}

Tensor LoadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + filename);
    }
    
    std::vector<double> values;
    std::string line;
    size_t rows = 0;
    size_t cols = 0;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        size_t current_cols = 0;
        
        while (std::getline(ss, cell, ',')) {
            // Trim whitespace
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            if (!cell.empty()) {
                values.push_back(std::stod(cell));
                current_cols++;
            }
        }
        
        if (current_cols > 0) {
            if (rows == 0) cols = current_cols;
            else if (current_cols != cols) throw std::runtime_error("CSV rows have inconsistent column counts");
            rows++;
        }
    }
    
    return Tensor(rows, cols, values);
}


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
    loss_impl->backward_op = [logits_impl, probs, targets, N, C, loss_ptr = loss_impl.get()]() {
        double upstream_grad = loss_ptr->grad[0];
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < C; ++j) {
                double diff = probs.get(i, j) - targets.get(i, j);
                logits_impl->grad[i * C + j] += upstream_grad * diff / N;
            }
        }
    };
    
    return loss_tensor;
}

Tensor RandomMatrix(size_t rows, size_t cols) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 0.1);
    
    std::vector<double> vals(rows * cols);
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = d(gen);
    }
    return Tensor(rows, cols, vals);
}

} // namespace tnsr
