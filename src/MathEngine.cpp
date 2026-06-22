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
#include "MetalEngine.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <set>
#include <random>
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>

namespace tnsr {

// --- AutoGrad: The Backward Pass ---
// Performs a topological sort of the computational graph and applies the chain rule
// by calling `backward_op` on each tensor in reverse topological order.
void Tensor::backward() {
    std::vector<std::shared_ptr<TensorImpl>> topo;
    std::set<TensorImpl*> visited;
    
    std::vector<std::pair<std::shared_ptr<TensorImpl>, bool>> stack;
    std::set<TensorImpl*> added;
    stack.push_back({impl, false});
    
    while (!stack.empty()) {
        auto current_pair = stack.back();
        auto v = current_pair.first;
        bool children_processed = current_pair.second;
        stack.pop_back();
        
        if (children_processed) {
            if (added.find(v.get()) == added.end()) {
                added.insert(v.get());
                topo.push_back(v);
            }
        } else {
            if (visited.find(v.get()) == visited.end()) {
                visited.insert(v.get());
                stack.push_back({v, true});
                for (auto child : v->parents) {
                    if (visited.find(child.get()) == visited.end()) {
                        stack.push_back({child, false});
                    }
                }
            }
        }
    }
    
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
        Tensor out(impl->shape);
        double* out_data = out.impl->data.data();
        const double* a_data = impl->data.data();
        const double* b_data = other.impl->data.data();
        dispatch_apply(impl->data.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            out_data[i] = a_data[i] + b_data[i];
        });
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o]() {
            double* a_grad = a->grad.data();
            double* b_grad = b->grad.data();
            const double* o_grad = o->grad.data();
            dispatch_apply(o->grad.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
                a_grad[i] += o_grad[i];
                b_grad[i] += o_grad[i];
            });
        };
        return out;
    } else if (impl->cols() == other.impl->cols() && other.impl->rows() == 1) {
        // Broadcast across rows
        Tensor out(impl->shape);
        double* out_data = out.impl->data.data();
        const double* a_data = impl->data.data();
        const double* b_data = other.impl->data.data();
        size_t cols = impl->cols();
        dispatch_apply(impl->rows(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
            for (size_t c = 0; c < cols; ++c) {
                out_data[r * cols + c] = a_data[r * cols + c] + b_data[c];
            }
        });
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o]() {
            // Because b_grad is accumulated across rows, we can't easily parallelize b_grad update without atomics/locks.
            // But we can parallelize a_grad, and do b_grad sequentially or with a separate loop.
            double* a_grad = a->grad.data();
            double* b_grad = b->grad.data();
            const double* o_grad = o->grad.data();
            size_t o_cols = o->cols();
            size_t o_rows = o->rows();
            
            // Parallelize a_grad
            dispatch_apply(o_rows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
                for (size_t c = 0; c < o_cols; ++c) {
                    a_grad[r * o_cols + c] += o_grad[r * o_cols + c];
                }
            });
            // Sequential b_grad to avoid race conditions
            for (size_t r = 0; r < o_rows; ++r) {
                for (size_t c = 0; c < o_cols; ++c) {
                    b_grad[c] += o_grad[r * o_cols + c];
                }
            }
        };
        return out;
    } else if (impl->rows() == other.impl->rows() && other.impl->cols() == 1) {
        // Broadcast across columns
        Tensor out(impl->shape);
        double* out_data = out.impl->data.data();
        const double* a_data = impl->data.data();
        const double* b_data = other.impl->data.data();
        size_t cols = impl->cols();
        dispatch_apply(impl->rows(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
            for (size_t c = 0; c < cols; ++c) {
                out_data[r * cols + c] = a_data[r * cols + c] + b_data[r];
            }
        });
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o]() {
            double* a_grad = a->grad.data();
            double* b_grad = b->grad.data();
            const double* o_grad = o->grad.data();
            size_t o_cols = o->cols();
            size_t o_rows = o->rows();
            
            dispatch_apply(o_rows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
                double row_sum = 0.0;
                for (size_t c = 0; c < o_cols; ++c) {
                    a_grad[r * o_cols + c] += o_grad[r * o_cols + c];
                    row_sum += o_grad[r * o_cols + c];
                }
                // b_grad can be updated in parallel because each row 'r' uniquely updates b_grad[r]
                b_grad[r] += row_sum;
            });
        };
        return out;
    } else if (other.impl->rows() == 1 && other.impl->cols() == 1) {
        // Scalar broadcasting
        Tensor out(impl->shape);
        double* out_data = out.impl->data.data();
        const double* a_data = impl->data.data();
        double b_val = other.impl->data[0];
        dispatch_apply(impl->data.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            out_data[i] = a_data[i] + b_val;
        });
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o]() {
            double* a_grad = a->grad.data();
            double* b_grad = b->grad.data();
            const double* o_grad = o->grad.data();
            
            dispatch_apply(o->grad.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
                a_grad[i] += o_grad[i];
            });
            // Sequential sum for scalar b
            double sum = 0.0;
            for (size_t i = 0; i < o->grad.size(); ++i) {
                sum += o_grad[i];
            }
            b_grad[0] += sum;
        };
        return out;
    }
    
    throw std::invalid_argument("Dimensions must match or be broadcastable for addition");
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (impl->rows() == other.impl->rows() && impl->cols() == other.impl->cols()) {
        Tensor out(impl->shape);
        for (size_t i = 0; i < impl->data.size(); ++i) {
            out.impl->data[i] = impl->data[i] - other.impl->data[i];
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o]() {
            for (size_t i = 0; i < o->grad.size(); ++i) {
                a->grad[i] += o->grad[i];
                b->grad[i] -= o->grad[i];
            }
        };
        return out;
    } else if (impl->cols() == other.impl->cols() && other.impl->rows() == 1) {
        // Broadcast across rows
        Tensor out(impl->shape);
        for (size_t r = 0; r < impl->rows(); ++r) {
            for (size_t c = 0; c < impl->cols(); ++c) {
                out.impl->data[r * impl->cols() + c] = impl->data[r * impl->cols() + c] - other.impl->data[c];
            }
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o]() {
            for (size_t r = 0; r < o->rows(); ++r) {
                for (size_t c = 0; c < o->cols(); ++c) {
                    a->grad[r * o->cols() + c] += o->grad[r * o->cols() + c];
                    b->grad[c] -= o->grad[r * o->cols() + c];
                }
            }
        };
        return out;
    } else if (impl->rows() == other.impl->rows() && other.impl->cols() == 1) {
        // Broadcast across columns
        Tensor out(impl->shape);
        for (size_t r = 0; r < impl->rows(); ++r) {
            for (size_t c = 0; c < impl->cols(); ++c) {
                out.impl->data[r * impl->cols() + c] = impl->data[r * impl->cols() + c] - other.impl->data[r];
            }
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o]() {
            for (size_t r = 0; r < o->rows(); ++r) {
                for (size_t c = 0; c < o->cols(); ++c) {
                    a->grad[r * o->cols() + c] += o->grad[r * o->cols() + c];
                    b->grad[r] -= o->grad[r * o->cols() + c];
                }
            }
        };
        return out;
    } else if (other.impl->rows() == 1 && other.impl->cols() == 1) {
        // Scalar broadcasting
        Tensor out(impl->shape);
        for (size_t i = 0; i < impl->data.size(); ++i) {
            out.impl->data[i] = impl->data[i] - other.impl->data[0];
        }
        auto a = impl; auto b = other.impl; auto o = out.impl;
        o->parents = {a, b};
        o->backward_op = [a, b, o]() {
            for (size_t i = 0; i < o->grad.size(); ++i) {
                a->grad[i] += o->grad[i];
                b->grad[0] -= o->grad[i];
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
    Tensor out(impl->shape);
    for (size_t i = 0; i < impl->data.size(); ++i) {
        out.impl->data[i] = impl->data[i] * other.impl->data[i];
    }
    
    auto a = impl;
    auto b = other.impl;
    auto o = out.impl;
    
    o->parents = {a, b};
    o->backward_op = [a, b, o]() {
        for (size_t i = 0; i < o->grad.size(); ++i) {
            a->grad[i] += b->data[i] * o->grad[i];
            b->grad[i] += a->data[i] * o->grad[i];
        }
    };
    return out;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor out(impl->shape);
    for (size_t i = 0; i < impl->data.size(); ++i) {
        out.impl->data[i] = impl->data[i] * scalar;
    }
    
    auto a = impl;
    auto o = out.impl;
    
    o->parents = {a};
    o->backward_op = [a, o, scalar]() {
        for (size_t i = 0; i < o->grad.size(); ++i) {
            a->grad[i] += scalar * o->grad[i];
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
    o->backward_op = [a, o, index]() {
        for (size_t c = 0; c < o->cols(); ++c) {
            a->grad[index * a->cols() + c] += o->grad[c];
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
// They use Apple Accelerate (BLAS) for extreme performance when applicable.

Tensor DotProduct(const Tensor& a, const Tensor& b) {
    if (a.cols() != b.rows()) {
        throw std::runtime_error("Matrix dimensions must agree for DotProduct");
    }
    
    Tensor out(a.rows(), b.cols());
    
    bool metal_used = false;
    if (a.rows() * b.cols() * a.cols() > 100) { // Lowered for testing
        if (!IsMetalAvailable()) MetalInitialize();
        if (IsMetalAvailable()) {
            metal_used = MetalComputeDotProduct(a.impl->data, b.impl->data, out.impl->data, a.rows(), b.cols(), a.cols(), false, false);
        }
    }
    
    if (!metal_used) {
        // Fallback to Apple Accelerate (BLAS) CPU
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    a.rows(), b.cols(), a.cols(), 
                    1.0, 
                    a.impl->data.data(), a.cols(), 
                    b.impl->data.data(), b.cols(), 
                    0.0, 
                    out.impl->data.data(), out.cols());
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
    Tensor out(a.impl->shape);
    double* out_data = out.impl->data.data();
    const double* a_data = a.impl->data.data();
    dispatch_apply(a.impl->data.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        out_data[i] = std::max(0.0, a_data[i]);
    });
    
    auto a_impl = a.impl;
    auto o_impl = out.impl;
    
    o_impl->parents = {a_impl};
    o_impl->backward_op = [a_impl, o_ptr = o_impl.get()]() {
        double* a_grad = a_impl->grad.data();
        const double* a_data_ptr = a_impl->data.data();
        const double* o_grad = o_ptr->grad.data();
        dispatch_apply(o_ptr->grad.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            a_grad[i] += (a_data_ptr[i] > 0 ? 1.0 : 0.0) * o_grad[i];
        });
    };
    return out;
}

Tensor Sigmoid(const Tensor& a) {
    Tensor out(a.impl->shape);
    double* out_data = out.impl->data.data();
    const double* a_data = a.impl->data.data();
    dispatch_apply(a.impl->data.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        out_data[i] = 1.0 / (1.0 + std::exp(-a_data[i]));
    });
    
    auto a_impl = a.impl;
    auto o_impl = out.impl;
    
    o_impl->parents = {a_impl};
    o_impl->backward_op = [a_impl, o_ptr = o_impl.get()]() {
        double* a_grad = a_impl->grad.data();
        const double* o_data = o_ptr->data.data();
        const double* o_grad = o_ptr->grad.data();
        dispatch_apply(o_ptr->grad.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            double s = o_data[i];
            a_grad[i] += s * (1.0 - s) * o_grad[i];
        });
    };
    return out;
}

Tensor Tanh(const Tensor& a) {
    Tensor out(a.impl->shape);
    double* out_data = out.impl->data.data();
    const double* a_data = a.impl->data.data();
    dispatch_apply(a.impl->data.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        out_data[i] = std::tanh(a_data[i]);
    });
    
    auto a_impl = a.impl;
    auto o_impl = out.impl;
    
    o_impl->parents = {a_impl};
    o_impl->backward_op = [a_impl, o_ptr = o_impl.get()]() {
        double* a_grad = a_impl->grad.data();
        const double* o_data = o_ptr->data.data();
        const double* o_grad = o_ptr->grad.data();
        dispatch_apply(o_ptr->grad.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            double t = o_data[i];
            // To avoid race conditions, we assume single assignment or we use atomic adds if multiple parents.
            // Since backward might accumulate gradients, and we do +=, we must be careful.
            // Wait, a_grad[i] += ... could be a race condition if multiple things dispatch to the same i.
            // But here, one thread processes exactly one 'i', so a_grad[i] is updated by only one thread per operation!
            // BUT what if multiple backward_ops run concurrently? Topo sort runs sequentially, so backward_ops are sequential.
            // Inside ONE backward_op, 'i' is unique per thread, so NO RACE CONDITION!
            a_grad[i] += (1.0 - t * t) * o_grad[i];
        });
    };
    return out;
}

Tensor Softmax(const Tensor& a) {
    Tensor out(a.impl->shape);
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

Tensor LayerNorm(const Tensor& a) {
    Tensor out(a.impl->shape);
    const double epsilon = 1e-5;
    
    // To support backward pass, we save mean and variance
    std::vector<double> means(a.rows());
    std::vector<double> vars(a.rows());
    std::vector<double> inv_std(a.rows());
    
    for (size_t r = 0; r < a.rows(); ++r) {
        double sum = 0.0;
        for (size_t c = 0; c < a.cols(); ++c) sum += a.get(r, c);
        double mean = sum / a.cols();
        means[r] = mean;
        
        double var_sum = 0.0;
        for (size_t c = 0; c < a.cols(); ++c) {
            double diff = a.get(r, c) - mean;
            var_sum += diff * diff;
        }
        double var = var_sum / a.cols();
        vars[r] = var;
        inv_std[r] = 1.0 / std::sqrt(var + epsilon);
        
        for (size_t c = 0; c < a.cols(); ++c) {
            out.set(r, c, (a.get(r, c) - mean) * inv_std[r]);
        }
    }
    
    auto a_impl = a.impl;
    auto o_impl = out.impl;
    
    o_impl->parents = {a_impl};
    o_impl->backward_op = [a_impl, o_ptr = o_impl.get(), means, inv_std]() {
        size_t N = a_impl->cols();
        for (size_t r = 0; r < a_impl->rows(); ++r) {
            double sum_g = 0.0;
            double sum_g_x_hat = 0.0;
            for (size_t c = 0; c < N; ++c) {
                double g = o_ptr->grad[r * N + c];
                double x_hat = (a_impl->data[r * N + c] - means[r]) * inv_std[r];
                sum_g += g;
                sum_g_x_hat += g * x_hat;
            }
            for (size_t c = 0; c < N; ++c) {
                double x_hat = (a_impl->data[r * N + c] - means[r]) * inv_std[r];
                double g = o_ptr->grad[r * N + c];
                a_impl->grad[r * N + c] += (inv_std[r] / N) * (N * g - sum_g - x_hat * sum_g_x_hat);
            }
        }
    };
    return out;
}

Tensor Concat(const Tensor& a, const Tensor& b, int axis) {
    if (axis == 0) {
        if (a.cols() != b.cols()) throw std::invalid_argument("Concat axis 0 requires matching columns");
        Tensor out(a.rows() + b.rows(), a.cols());
        for (size_t r = 0; r < a.rows(); ++r) {
            for (size_t c = 0; c < a.cols(); ++c) out.set(r, c, a.get(r, c));
        }
        for (size_t r = 0; r < b.rows(); ++r) {
            for (size_t c = 0; c < b.cols(); ++c) out.set(r + a.rows(), c, b.get(r, c));
        }
        auto a_impl = a.impl; auto b_impl = b.impl; auto o_impl = out.impl;
        o_impl->parents = {a_impl, b_impl};
        o_impl->backward_op = [a_impl, b_impl, o_ptr = o_impl.get()]() {
            for (size_t r = 0; r < a_impl->rows(); ++r) {
                for (size_t c = 0; c < a_impl->cols(); ++c) a_impl->grad[r * a_impl->cols() + c] += o_ptr->grad[r * o_ptr->cols() + c];
            }
            for (size_t r = 0; r < b_impl->rows(); ++r) {
                for (size_t c = 0; c < b_impl->cols(); ++c) b_impl->grad[r * b_impl->cols() + c] += o_ptr->grad[(r + a_impl->rows()) * o_ptr->cols() + c];
            }
        };
        return out;
    } else if (axis == 1) {
        if (a.rows() != b.rows()) throw std::invalid_argument("Concat axis 1 requires matching rows");
        Tensor out(a.rows(), a.cols() + b.cols());
        for (size_t r = 0; r < a.rows(); ++r) {
            for (size_t c = 0; c < a.cols(); ++c) out.set(r, c, a.get(r, c));
            for (size_t c = 0; c < b.cols(); ++c) out.set(r, c + a.cols(), b.get(r, c));
        }
        auto a_impl = a.impl; auto b_impl = b.impl; auto o_impl = out.impl;
        o_impl->parents = {a_impl, b_impl};
        o_impl->backward_op = [a_impl, b_impl, o_ptr = o_impl.get()]() {
            for (size_t r = 0; r < a_impl->rows(); ++r) {
                for (size_t c = 0; c < a_impl->cols(); ++c) a_impl->grad[r * a_impl->cols() + c] += o_ptr->grad[r * o_ptr->cols() + c];
                for (size_t c = 0; c < b_impl->cols(); ++c) b_impl->grad[r * b_impl->cols() + c] += o_ptr->grad[r * o_ptr->cols() + (c + a_impl->cols())];
            }
        };
        return out;
    }
    throw std::invalid_argument("Invalid axis for concat");
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
    file << t.impl->shape.size() << "\n";
    for (size_t d : t.impl->shape) file << d << " ";
    file << "\n";
    for (double val : t.impl->data) {
        file << val << " ";
    }
}

Tensor LoadTensor(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Could not open file for loading");
    
    std::string first_line;
    std::getline(file, first_line);
    std::stringstream ss(first_line);
    std::vector<size_t> header;
    size_t h_val;
    while (ss >> h_val) header.push_back(h_val);
    
    if (header.size() == 1) {
        // New format
        size_t dims = header[0];
        std::vector<size_t> shape(dims);
        size_t total_size = 1;
        for (size_t i = 0; i < dims; ++i) {
            file >> shape[i];
            total_size *= shape[i];
        }
        std::vector<double> vals(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            file >> vals[i];
        }
        return Tensor(shape, vals);
    } else if (header.size() == 2) {
        // Old 2D format
        size_t r = header[0], c = header[1];
        std::vector<double> vals(r * c);
        for (size_t i = 0; i < r * c; ++i) {
            file >> vals[i];
        }
        return Tensor(r, c, vals);
    } else {
        throw std::runtime_error("Invalid tensor file format");
    }
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
    
    auto a_impl = a.impl;
    auto o_impl = out.impl;
    
    o_impl->parents = {a_impl};
    o_impl->backward_op = [a_impl, o_ptr = o_impl.get(), row_start, new_rows, col_start, new_cols]() {
        size_t a_cols = a_impl->cols();
        for (size_t r = 0; r < new_rows; ++r) {
            for (size_t c = 0; c < new_cols; ++c) {
                a_impl->grad[(row_start + r) * a_cols + (col_start + c)] += o_ptr->grad[r * new_cols + c];
            }
        }
    };
    
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


// ==================== CNN HELPERS ====================
void im2col_cpu(const double* data_im, int channels, int height, int width,
                int ksize, int stride, int pad, double* data_col) {
    int out_h = (height + 2 * pad - ksize) / stride + 1;
    int out_w = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                int im_row = h_offset + h * stride - pad;
                int im_col = w_offset + w * stride - pad;
                int col_index = (c * out_h + h) * out_w + w;
                if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                    data_col[col_index] = data_im[(c_im * height + im_row) * width + im_col];
                } else {
                    data_col[col_index] = 0.0;
                }
            }
        }
    }
}

void col2im_cpu(const double* data_col, int channels, int height, int width,
                int ksize, int stride, int pad, double* data_im) {
    int out_h = (height + 2 * pad - ksize) / stride + 1;
    int out_w = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for (int i = 0; i < channels * height * width; ++i) data_im[i] = 0.0;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                int im_row = h_offset + h * stride - pad;
                int im_col = w_offset + w * stride - pad;
                int col_index = (c * out_h + h) * out_w + w;
                if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                    data_im[(c_im * height + im_row) * width + im_col] += data_col[col_index];
                }
            }
        }
    }
}

// ==================== Conv2D ====================
Tensor Conv2D(const Tensor& X, const Tensor& W, int stride, int pad) {
    if (X.impl->shape.size() != 4 || W.impl->shape.size() != 4) {
        throw std::runtime_error("Conv2D requires 4D tensors (X:[N,C,H,W], W:[K,C,kH,kW])");
    }
    
    int N = X.impl->shape[0];
    int C = X.impl->shape[1];
    int H = X.impl->shape[2];
    int W_dim = X.impl->shape[3];
    
    int K = W.impl->shape[0];
    int C_w = W.impl->shape[1];
    int kH = W.impl->shape[2];
    int kW = W.impl->shape[3];
    
    if (C != C_w) throw std::runtime_error("Conv2D: Channels mismatch");
    
    int outH = (H + 2 * pad - kH) / stride + 1;
    int outW = (W_dim + 2 * pad - kW) / stride + 1;
    
    Tensor out(std::vector<size_t>{(size_t)N, (size_t)K, (size_t)outH, (size_t)outW});
    
    // We will do im2col per image
    std::vector<double> col_data(C * kH * kW * outH * outW);
    
    for (int n = 0; n < N; ++n) {
        const double* im_ptr = X.impl->data.data() + n * (C * H * W_dim);
        im2col_cpu(im_ptr, C, H, W_dim, kH, stride, pad, col_data.data());
        
        // Multiply W [K, C*kH*kW] by col [C*kH*kW, outH*outW] -> [K, outH*outW]
        double* out_ptr = out.impl->data.data() + n * (K * outH * outW);
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    K, outH * outW, C * kH * kW,
                    1.0, W.impl->data.data(), C * kH * kW,
                    col_data.data(), outH * outW,
                    0.0, out_ptr, outH * outW);
    }
    
    std::shared_ptr<TensorImpl> x_impl = X.impl;
    std::shared_ptr<TensorImpl> w_impl = W.impl;
    std::shared_ptr<TensorImpl> o_impl = out.impl;
    
    out.impl->parents = {X.impl, W.impl};
    out.impl->backward_op = [x_impl, w_impl, o_ptr = o_impl.get(), N, C, H, W_dim, K, kH, kW, outH, outW, stride, pad]() {
        std::vector<double> col_data(C * kH * kW * outH * outW);
        std::vector<double> dcol(C * kH * kW * outH * outW);
        
        for (int n = 0; n < N; ++n) {
            // Forward im2col (needed for W_grad)
            const double* im_ptr = x_impl->data.data() + n * (C * H * W_dim);
            im2col_cpu(im_ptr, C, H, W_dim, kH, stride, pad, col_data.data());
            
            // Output gradient for this image: [K, outH * outW]
            const double* grad_out_ptr = o_ptr->grad.data() + n * (K * outH * outW);
            
            // W_grad = grad_out * col^T -> [K, outH*outW] * [outH*outW, C*kH*kW] -> [K, C*kH*kW]
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        K, C * kH * kW, outH * outW,
                        1.0, grad_out_ptr, outH * outW,
                        col_data.data(), outH * outW,
                        1.0, w_impl->grad.data(), C * kH * kW);
            
            // X_col_grad = W^T * grad_out -> [C*kH*kW, K] * [K, outH*outW] -> [C*kH*kW, outH*outW]
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        C * kH * kW, outH * outW, K,
                        1.0, w_impl->data.data(), C * kH * kW,
                        grad_out_ptr, outH * outW,
                        0.0, dcol.data(), outH * outW);
            
            // X_grad = col2im(X_col_grad)
            double* grad_im_ptr = x_impl->grad.data() + n * (C * H * W_dim);
            col2im_cpu(dcol.data(), C, H, W_dim, kH, stride, pad, grad_im_ptr);
        }
    };
    
    return out;
}

// ==================== MaxPool2D ====================
Tensor MaxPool2D(const Tensor& X, int pool_size, int stride) {
    if (X.impl->shape.size() != 4) {
        throw std::runtime_error("MaxPool2D requires 4D tensor [N, C, H, W]");
    }
    
    int N = X.impl->shape[0];
    int C = X.impl->shape[1];
    int H = X.impl->shape[2];
    int W_dim = X.impl->shape[3];
    
    int outH = (H - pool_size) / stride + 1;
    int outW = (W_dim - pool_size) / stride + 1;
    
    Tensor out(std::vector<size_t>{(size_t)N, (size_t)C, (size_t)outH, (size_t)outW});
    
    // Store max indices for backward pass
    std::shared_ptr<std::vector<int>> max_indices = std::make_shared<std::vector<int>>(N * C * outH * outW);
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < outH; ++oh) {
                for (int ow = 0; ow < outW; ++ow) {
                    double max_val = -1e9;
                    int max_idx = -1;
                    
                    int h_start = oh * stride;
                    int w_start = ow * stride;
                    
                    for (int ph = 0; ph < pool_size; ++ph) {
                        for (int pw = 0; pw < pool_size; ++pw) {
                            int h = h_start + ph;
                            int w = w_start + pw;
                            int idx = ((n * C + c) * H + h) * W_dim + w;
                            double val = X.impl->data[idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = idx;
                            }
                        }
                    }
                    int out_idx = ((n * C + c) * outH + oh) * outW + ow;
                    out.impl->data[out_idx] = max_val;
                    (*max_indices)[out_idx] = max_idx;
                }
            }
        }
    }
    
    std::shared_ptr<TensorImpl> x_impl = X.impl;
    std::shared_ptr<TensorImpl> o_impl = out.impl;
    
    out.impl->parents = {X.impl};
    out.impl->backward_op = [x_impl, o_ptr = o_impl.get(), max_indices, N, C, outH, outW]() {
        for (int i = 0; i < N * C * outH * outW; ++i) {
            int max_idx = (*max_indices)[i];
            if (max_idx != -1) {
                x_impl->grad[max_idx] += o_ptr->grad[i];
            }
        }
    };
    
    return out;
}

// ==================== Dropout ====================
Tensor Dropout(const Tensor& X, double rate, double is_training) {
    if (!X.impl) return X;
    
    Tensor out(X.impl->shape);
    std::shared_ptr<std::vector<double>> mask = std::make_shared<std::vector<double>>(X.impl->data.size(), 1.0);
    
    if (is_training != 0.0) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::bernoulli_distribution d(1.0 - rate);
        double scale = 1.0 / (1.0 - rate);
        
        for (size_t i = 0; i < X.impl->data.size(); ++i) {
            if (d(gen)) {
                (*mask)[i] = scale;
                out.impl->data[i] = X.impl->data[i] * scale;
            } else {
                (*mask)[i] = 0.0;
                out.impl->data[i] = 0.0;
            }
        }
    } else {
        out.impl->data = X.impl->data;
    }
    
    std::shared_ptr<TensorImpl> x_impl = X.impl;
    std::shared_ptr<TensorImpl> o_impl = out.impl;
    
    out.impl->parents = {X.impl};
    out.impl->backward_op = [x_impl, o_ptr = o_impl.get(), mask]() {
        for (size_t i = 0; i < x_impl->grad.size(); ++i) {
            x_impl->grad[i] += o_ptr->grad[i] * (*mask)[i];
        }
    };
    
    return out;
}

// ==================== BatchNorm (2D) ====================
Tensor BatchNorm(const Tensor& X, const Tensor& gamma, const Tensor& beta, const Tensor& running_mean, const Tensor& running_var, double momentum, double is_training) {
    if (X.impl->shape.size() != 2) throw std::runtime_error("BatchNorm currently only supports 2D tensors [N, D]");
    
    int N = X.impl->shape[0];
    int D = X.impl->shape[1];
    
    Tensor out(X.impl->shape);
    double eps = 1e-5;
    
    std::shared_ptr<std::vector<double>> batch_mean = std::make_shared<std::vector<double>>(D, 0.0);
    std::shared_ptr<std::vector<double>> batch_var = std::make_shared<std::vector<double>>(D, 0.0);
    std::shared_ptr<std::vector<double>> x_norm = std::make_shared<std::vector<double>>(N * D, 0.0);
    
    if (is_training != 0.0) {
        // Compute mean
        for (int n = 0; n < N; ++n) {
            for (int d = 0; d < D; ++d) {
                (*batch_mean)[d] += X.impl->data[n * D + d];
            }
        }
        for (int d = 0; d < D; ++d) (*batch_mean)[d] /= N;
        
        // Compute variance
        for (int n = 0; n < N; ++n) {
            for (int d = 0; d < D; ++d) {
                double diff = X.impl->data[n * D + d] - (*batch_mean)[d];
                (*batch_var)[d] += diff * diff;
            }
        }
        for (int d = 0; d < D; ++d) (*batch_var)[d] /= N;
        
        // Update running stats
        for (int d = 0; d < D; ++d) {
            running_mean.impl->data[d] = momentum * running_mean.impl->data[d] + (1.0 - momentum) * (*batch_mean)[d];
            running_var.impl->data[d] = momentum * running_var.impl->data[d] + (1.0 - momentum) * (*batch_var)[d];
        }
    } else {
        // Use running stats
        for (int d = 0; d < D; ++d) {
            (*batch_mean)[d] = running_mean.impl->data[d];
            (*batch_var)[d] = running_var.impl->data[d];
        }
    }
    
    // Normalize and scale
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            double diff = X.impl->data[n * D + d] - (*batch_mean)[d];
            double inv_std = 1.0 / std::sqrt((*batch_var)[d] + eps);
            double xn = diff * inv_std;
            (*x_norm)[n * D + d] = xn;
            out.impl->data[n * D + d] = gamma.impl->data[d] * xn + beta.impl->data[d];
        }
    }
    
    std::shared_ptr<TensorImpl> x_impl = X.impl;
    std::shared_ptr<TensorImpl> g_impl = gamma.impl;
    std::shared_ptr<TensorImpl> b_impl = beta.impl;
    std::shared_ptr<TensorImpl> o_impl = out.impl;
    
    out.impl->parents = {X.impl, gamma.impl, beta.impl};
    out.impl->backward_op = [x_impl, g_impl, b_impl, o_ptr = o_impl.get(), batch_mean, batch_var, x_norm, N, D, eps]() {
        for (int d = 0; d < D; ++d) {
            double dgamma = 0.0;
            double dbeta = 0.0;
            double inv_std = 1.0 / std::sqrt((*batch_var)[d] + eps);
            
            for (int n = 0; n < N; ++n) {
                int idx = n * D + d;
                dgamma += o_ptr->grad[idx] * (*x_norm)[idx];
                dbeta += o_ptr->grad[idx];
            }
            
            g_impl->grad[d] += dgamma;
            b_impl->grad[d] += dbeta;
            
            double dx_norm_sum = 0.0;
            double dx_norm_x_sum = 0.0;
            
            for (int n = 0; n < N; ++n) {
                int idx = n * D + d;
                double dx_norm = o_ptr->grad[idx] * g_impl->data[d];
                dx_norm_sum += dx_norm;
                dx_norm_x_sum += dx_norm * (*x_norm)[idx];
            }
            
            for (int n = 0; n < N; ++n) {
                int idx = n * D + d;
                double dx_norm = o_ptr->grad[idx] * g_impl->data[d];
                x_impl->grad[idx] += (inv_std / N) * (N * dx_norm - dx_norm_sum - (*x_norm)[idx] * dx_norm_x_sum);
            }
        }
    };
    
    return out;
}

Tensor Adam_Step(const Tensor& w, double lr, double beta1, double beta2, double t) {
    if (!w.impl) return w;
    size_t size = w.impl->data.size();
    
    if (w.impl->m.size() != size) w.impl->m.assign(size, 0.0);
    if (w.impl->v.size() != size) w.impl->v.assign(size, 0.0);

    Tensor out(w.impl->shape);
    double epsilon = 1e-8;

    for (size_t i = 0; i < size; ++i) {
        double g = w.impl->grad[i];
        
        w.impl->m[i] = beta1 * w.impl->m[i] + (1.0 - beta1) * g;
        w.impl->v[i] = beta2 * w.impl->v[i] + (1.0 - beta2) * (g * g);

        double m_hat = w.impl->m[i] / (1.0 - std::pow(beta1, t));
        double v_hat = w.impl->v[i] / (1.0 - std::pow(beta2, t));

        out.impl->data[i] = w.impl->data[i] - lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
    
    out.impl->m = w.impl->m;
    out.impl->v = w.impl->v;

    return out;
}

Tensor Reshape(const Tensor& t, const std::vector<size_t>& new_shape) {
    if (!t.impl) throw std::runtime_error("Reshape on null tensor");
    size_t old_size = 1, new_size = 1;
    for (auto d : t.impl->shape) old_size *= d;
    for (auto d : new_shape) new_size *= d;
    if (old_size != new_size) throw std::runtime_error("Reshape: dimensions mismatch");
    
    Tensor out(new_shape);
    out.impl->data = t.impl->data;
    
    std::shared_ptr<TensorImpl> t_impl = t.impl;
    std::shared_ptr<TensorImpl> o_impl = out.impl;
    
    out.impl->parents = {t.impl};
    out.impl->backward_op = [t_impl, o_ptr = o_impl.get(), old_size]() {
        for (size_t i = 0; i < old_size; ++i) {
            t_impl->grad[i] += o_ptr->grad[i];
        }
    };
    return out;
}

} // namespace tnsr
