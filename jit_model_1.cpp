#include "src/MathEngine.h"
#include <cmath>
using namespace tnsr;
extern "C" Tensor jit_forward_1(Tensor X, std::vector<Tensor>& weights, double is_training) {
    int w_idx = 0;
    X = DotProduct(X, weights[w_idx++]);
    {
        Tensor gamma = weights[w_idx++];
        Tensor beta = weights[w_idx++];
        Tensor rm = weights[w_idx++];
        Tensor rv = weights[w_idx++];
        X = BatchNorm(X, gamma, beta, rm, rv, 0.1, is_training);
    }
    X = ReLU(X);
    X = Dropout(X, 0.2, is_training);
    X = DotProduct(X, weights[w_idx++]);
    {
        Tensor gamma = weights[w_idx++];
        Tensor beta = weights[w_idx++];
        Tensor rm = weights[w_idx++];
        Tensor rv = weights[w_idx++];
        X = BatchNorm(X, gamma, beta, rm, rv, 0.1, is_training);
    }
    X = ReLU(X);
    X = Dropout(X, 0.2, is_training);
    X = DotProduct(X, weights[w_idx++]);
    return X;
}
