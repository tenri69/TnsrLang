#include "src/MathEngine.h"
#include <cmath>
using namespace tnsr;
extern "C" Tensor jit_forward_1(Tensor X, std::vector<Tensor>& weights, double is_training) {
    int w_idx = 0;
    X = DotProduct(X, weights[w_idx++]);
    {
        Tensor W_q = weights[w_idx++];
        Tensor W_k = weights[w_idx++];
        Tensor W_v = weights[w_idx++];
        Tensor W_o = weights[w_idx++];
        Tensor Q = DotProduct(X, W_q);
        Tensor K = DotProduct(X, W_k);
        Tensor V = DotProduct(X, W_v);
        double scale = 1.0 / std::sqrt(16.0);
        std::vector<Tensor> heads;
        {
            Tensor q_h = Slice(Q, 0, Q.rows(), 0, 16);
            Tensor k_h = Slice(K, 0, K.rows(), 0, 16);
            Tensor v_h = Slice(V, 0, V.rows(), 0, 16);
            heads.push_back(DotProduct(Softmax(DotProduct(q_h, Transpose(k_h)) * scale), v_h));
        }
        {
            Tensor q_h = Slice(Q, 0, Q.rows(), 16, 32);
            Tensor k_h = Slice(K, 0, K.rows(), 16, 32);
            Tensor v_h = Slice(V, 0, V.rows(), 16, 32);
            heads.push_back(DotProduct(Softmax(DotProduct(q_h, Transpose(k_h)) * scale), v_h));
        }
        {
            Tensor q_h = Slice(Q, 0, Q.rows(), 32, 48);
            Tensor k_h = Slice(K, 0, K.rows(), 32, 48);
            Tensor v_h = Slice(V, 0, V.rows(), 32, 48);
            heads.push_back(DotProduct(Softmax(DotProduct(q_h, Transpose(k_h)) * scale), v_h));
        }
        {
            Tensor q_h = Slice(Q, 0, Q.rows(), 48, 64);
            Tensor k_h = Slice(K, 0, K.rows(), 48, 64);
            Tensor v_h = Slice(V, 0, V.rows(), 48, 64);
            heads.push_back(DotProduct(Softmax(DotProduct(q_h, Transpose(k_h)) * scale), v_h));
        }
        Tensor concat_heads = heads[0];
        concat_heads = Concat(concat_heads, heads[1], 1);
        concat_heads = Concat(concat_heads, heads[2], 1);
        concat_heads = Concat(concat_heads, heads[3], 1);
        X = LayerNorm(X + DotProduct(concat_heads, W_o));
    }
    X = DotProduct(X, weights[w_idx++]);
    X = ReLU(X);
    X = DotProduct(X, weights[w_idx++]);
    X = DotProduct(X, weights[w_idx++]);
    return X;
}
