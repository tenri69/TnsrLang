/*
 * =====================================================================================
 *
 *       Filename:  Evaluator.cpp
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

#include "Evaluator.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <cstdlib>
#include <cmath>

namespace tnsr {

// The Evaluator executes the AST. It walks the tree (Tree-Walk Interpreter) and 
// evaluates each node. It also manages variables and scope via the Environment.

Evaluator::Evaluator(std::ostream& out) : out(out), environment(&globalEnv), lastValue(0, 0) {
    // --- Standard Library (Native Functions) ---
    // Here we inject C++ functions into the TnsrLang global environment.
    // This allows the scripting language to call fast C++ code natively.
    globalEnv.defineNative("MSELoss", [](const std::vector<Tensor>& args) {
        if (args.size() != 2) throw std::runtime_error("MSELoss expects 2 arguments");
        Tensor pred = args[0];
        Tensor target = args[1];
        Tensor error = pred - target;
        Tensor loss = DotProduct(Transpose(error), error);
        return loss * (1.0 / (pred.impl->rows() * pred.impl->cols()));
    });

    globalEnv.defineNative("CrossEntropyLoss", [](const std::vector<Tensor>& args) {
        if (args.size() != 2) throw std::runtime_error("CrossEntropyLoss expects 2 arguments: (logits, targets)");
        return CrossEntropyLoss(args[0], args[1]);
    });

    globalEnv.defineNative("Slice", [](const std::vector<Tensor>& args) {
        if (args.size() != 5) throw std::runtime_error("Slice expects 5 arguments: (tensor, r_start, r_end, c_start, c_end)");
        return Slice(args[0], (size_t)args[1].get(0,0), (size_t)args[2].get(0,0), (size_t)args[3].get(0,0), (size_t)args[4].get(0,0));
    });

    globalEnv.defineNative("OneHot", [](const std::vector<Tensor>& args) {
        if (args.size() != 2) throw std::runtime_error("OneHot expects 2 arguments: (labels, num_classes)");
        return OneHot(args[0], (size_t)args[1].get(0,0));
    });

    globalEnv.defineNative("SGD_Step", [](const std::vector<Tensor>& args) {
        if (args.size() != 2) throw std::runtime_error("SGD_Step expects 2 arguments: (weights, learning_rate)");
        Tensor w = args[0];
        double lr = args[1].get(0, 0);
        Tensor out(w.impl->rows(), w.impl->cols());
        for (size_t i = 0; i < out.impl->data.size(); ++i) {
            out.impl->data[i] = w.impl->data[i] - w.impl->grad[i] * lr;
        }
        return out;
    });

    
    globalEnv.defineNative("Conv2D", [](const std::vector<Tensor>& args) {
        if (args.size() < 2) throw std::runtime_error("Conv2D expects 2 arguments: (X, W)");
        return Conv2D(args[0], args[1]);
    });

    globalEnv.defineNative("MaxPool2D", [](const std::vector<Tensor>& args) {
        if (args.size() < 1) throw std::runtime_error("MaxPool2D expects 1 argument: (X)");
        return MaxPool2D(args[0]);
    });

    globalEnv.defineNative("Reshape", [](const std::vector<Tensor>& args) {
        if (args.size() != 2) throw std::runtime_error("Reshape expects 2 args: (tensor, shape_vector)");
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < args[1].impl->data.size(); ++i) {
            new_shape.push_back((size_t)args[1].impl->data[i]);
        }
        return Reshape(args[0], new_shape);
    });

    globalEnv.defineNative("Dropout", [](const std::vector<Tensor>& args) {
        if (args.size() != 3) throw std::runtime_error("Dropout expects 3 arguments: (tensor, rate, is_training)");
        return Dropout(args[0], args[1].get(0, 0), args[2].get(0, 0));
    });

    globalEnv.defineNative("BatchNorm", [](const std::vector<Tensor>& args) {
        if (args.size() != 7) throw std::runtime_error("BatchNorm expects 7 arguments: (X, gamma, beta, running_mean, running_var, momentum, is_training)");
        return BatchNorm(args[0], args[1], args[2], args[3], args[4], args[5].get(0, 0), args[6].get(0, 0));
    });

    globalEnv.defineNative("Adam_Step", [](const std::vector<Tensor>& args) {
        if (args.size() != 5) throw std::runtime_error("Adam_Step expects 5 arguments: (weights, lr, beta1, beta2, t)");
        Tensor w = args[0];
        double lr = args[1].get(0, 0);
        double b1 = args[2].get(0, 0);
        double b2 = args[3].get(0, 0);
        double t = args[4].get(0, 0);
        return Adam_Step(w, lr, b1, b2, t);
    });

    globalEnv.defineNative("DotProduct", [](const std::vector<Tensor>& args) {
        if (args.size() != 2) throw std::runtime_error("DotProduct expects 2 arguments");
        return DotProduct(args[0], args[1]);
    });

    globalEnv.defineNative("Sigmoid", [](const std::vector<Tensor>& args) {
        if (args.size() != 1) throw std::runtime_error("Sigmoid expects 1 argument");
        return Sigmoid(args[0]);
    });

    globalEnv.defineNative("ReLU", [](const std::vector<Tensor>& args) {
        if (args.size() != 1) throw std::runtime_error("ReLU expects 1 argument");
        return ReLU(args[0]);
    });

    globalEnv.defineNative("Transpose", [](const std::vector<Tensor>& args) {
        if (args.size() != 1) throw std::runtime_error("Transpose expects 1 argument");
        return Transpose(args[0]);
    });

    globalEnv.defineNative("Softmax", [](const std::vector<Tensor>& args) {
        if (args.size() != 1) throw std::runtime_error("Softmax expects 1 argument");
        return Softmax(args[0]);
    });

    globalEnv.defineNative("RandomMatrix", [](const std::vector<Tensor>& args) {
        if (args.size() != 2) throw std::runtime_error("RandomMatrix expects 2 arguments (rows, cols)");
        return RandomMatrix((size_t)args[0].get(0, 0), (size_t)args[1].get(0, 0));
    });
    
    globalEnv.defineNative("GetRow", [](const std::vector<Tensor>& args) {
        if (args.size() != 2) throw std::runtime_error("GetRow expects 2 arguments (tensor, index)");
        return args[0].get_row((size_t)args[1].get(0, 0));
    });
    
    globalEnv.defineNative("OneHot", [](const std::vector<Tensor>& args) {
        if (args.size() != 2) throw std::runtime_error("OneHot expects 2 arguments (tensor, num_classes)");
        return OneHot(args[0], (size_t)args[1].get(0, 0));
    });

    globalEnv.defineNative("DenseLayer", [](const std::vector<Tensor>& args) {
        return Tensor(1, 3, {1.0, args[0].get(0,0), args[1].get(0,0)});
    });
    globalEnv.defineNative("Conv2DLayer", [](const std::vector<Tensor>& args) {
        return Tensor(1, 5, {2.0, args[0].get(0,0), args[1].get(0,0), args[2].get(0,0), args[3].get(0,0)});
    });
    globalEnv.defineNative("ReLULayer", [](const std::vector<Tensor>& args) {
        return Tensor(1, 1, {3.0});
    });
    globalEnv.defineNative("MaxPool2DLayer", [](const std::vector<Tensor>& args) {
        return Tensor(1, 1, {4.0});
    });
    globalEnv.defineNative("FlattenLayer", [](const std::vector<Tensor>& args) {
        return Tensor(1, 1, {5.0});
    });
    globalEnv.defineNative("DropoutLayer", [](const std::vector<Tensor>& args) {
        return Tensor(1, 2, {6.0, args[0].get(0,0)});
    });
    globalEnv.defineNative("RNNLayer", [](const std::vector<Tensor>& args) {
        // [7.0, in_features, hidden_units]
        return Tensor(1, 3, {7.0, args[0].get(0,0), args[1].get(0,0)});
    });
    globalEnv.defineNative("LSTMLayer", [](const std::vector<Tensor>& args) {
        // [8.0, in_features, hidden_units]
        return Tensor(1, 3, {8.0, args[0].get(0,0), args[1].get(0,0)});
    });
    globalEnv.defineNative("MultiHeadAttentionLayer", [](const std::vector<Tensor>& args) {
        // [9.0, embed_dim, num_heads]
        return Tensor(1, 3, {9.0, args[0].get(0,0), args[1].get(0,0)});
    });
    globalEnv.defineNative("Tanh", [](const std::vector<Tensor>& args) {
        if (args.size() != 1) throw std::runtime_error("Tanh expects 1 argument");
        return Tanh(args[0]);
    });

    static std::unordered_map<int, std::vector<Tensor>> model_weights;
    static std::unordered_map<int, Tensor> model_configs;
    static int model_counter = 0;
    
    typedef Tensor (*JitForwardFunc)(Tensor, std::vector<Tensor>&, double);
    static std::unordered_map<int, JitForwardFunc> jit_functions;

    globalEnv.defineNative("JIT", [](const std::vector<Tensor>& args) {
        int id = args[0].get(0, 0);
        Tensor config = model_configs[id];
        
        std::string filename = "jit_model_" + std::to_string(id) + ".cpp";
        std::ofstream out(filename);
        if (!out) throw std::runtime_error("Could not create JIT file");
        
        out << "#include \"src/MathEngine.h\"\n";
        out << "#include <cmath>\n";
        out << "using namespace tnsr;\n";
        out << "extern \"C\" Tensor jit_forward_" << id << "(Tensor X, std::vector<Tensor>& weights, double is_training) {\n";
        out << "    int w_idx = 0;\n";
        
        for (size_t i = 0; i < config.impl->rows(); ++i) {
            double type = config.get(i, 0);
            if (type == 1.0) { // Dense
                out << "    X = DotProduct(X, weights[w_idx++]);\n";
            } else if (type == 3.0) { // ReLU
                out << "    X = ReLU(X);\n";
            } else if (type == 9.0) { // MultiHeadAttention
                size_t num_heads = config.get(i, 2);
                size_t embed_dim = config.get(i, 1);
                size_t d_k = embed_dim / num_heads;
                out << "    {\n";
                out << "        Tensor W_q = weights[w_idx++];\n";
                out << "        Tensor W_k = weights[w_idx++];\n";
                out << "        Tensor W_v = weights[w_idx++];\n";
                out << "        Tensor W_o = weights[w_idx++];\n";
                out << "        Tensor Q = DotProduct(X, W_q);\n";
                out << "        Tensor K = DotProduct(X, W_k);\n";
                out << "        Tensor V = DotProduct(X, W_v);\n";
                out << "        double scale = 1.0 / std::sqrt(" << d_k << ".0);\n";
                out << "        std::vector<Tensor> heads;\n";
                for (size_t h = 0; h < num_heads; ++h) {
                    size_t start = h * d_k;
                    size_t end = (h + 1) * d_k;
                    out << "        {\n";
                    out << "            Tensor q_h = Slice(Q, 0, Q.rows(), " << start << ", " << end << ");\n";
                    out << "            Tensor k_h = Slice(K, 0, K.rows(), " << start << ", " << end << ");\n";
                    out << "            Tensor v_h = Slice(V, 0, V.rows(), " << start << ", " << end << ");\n";
                    out << "            heads.push_back(DotProduct(Softmax(DotProduct(q_h, Transpose(k_h)) * scale), v_h));\n";
                    out << "        }\n";
                }
                out << "        Tensor concat_heads = heads[0];\n";
                for (size_t h = 1; h < num_heads; ++h) {
                    out << "        concat_heads = Concat(concat_heads, heads[" << h << "], 1);\n";
                }
                out << "        X = LayerNorm(X + DotProduct(concat_heads, W_o));\n";
                out << "    }\n";
            }
        }
        out << "    return X;\n";
        out << "}\n";
        out.close();
        
        std::string libname = "./jit_model_" + std::to_string(id) + ".dylib";
        std::string cmd = "clang++ -shared -fPIC -std=c++17 -O3 " + filename + " src/MathEngine.cpp -o " + libname + " -framework Accelerate";
        std::cout << "[JIT] Compiling C++ model to machine code... (" << cmd << ")\n";
        int ret = system(cmd.c_str());
        if (ret != 0) throw std::runtime_error("JIT Compilation failed!");
        
        void* handle = dlopen(libname.c_str(), RTLD_NOW);
        if (!handle) throw std::runtime_error(std::string("Failed to load JIT dylib: ") + dlerror());
        
        std::string func_name = "jit_forward_" + std::to_string(id);
        JitForwardFunc func = (JitForwardFunc)dlsym(handle, func_name.c_str());
        if (!func) throw std::runtime_error(std::string("Failed to locate JIT function: ") + dlerror());
        
        jit_functions[id] = func;
        std::cout << "[JIT] Model dynamically loaded and bound!\n";
        
        return args[0];
    });

    globalEnv.defineNative("Sequential", [](const std::vector<Tensor>& args) {
        int id = ++model_counter;
        model_configs[id] = args[0];
        std::vector<Tensor> weights;
        Tensor config = args[0];
        
        for (size_t i = 0; i < config.impl->rows(); ++i) {
             double type = config.get(i, 0);
             if (type == 1.0) { // Dense
                 double scale = std::sqrt(2.0 / config.get(i, 1)); // He init
                 Tensor w = RandomMatrix(config.get(i, 1), config.get(i, 2));
                 for (double& val : w.impl->data) val = (val - 0.5) * scale;
                 weights.push_back(w);
             } else if (type == 2.0) { // Conv2D
                 size_t in_c = config.get(i, 1);
                 size_t out_c = config.get(i, 2);
                 size_t kH = config.get(i, 3);
                 size_t kW = config.get(i, 4);
                 double scale = std::sqrt(2.0 / (in_c * kH * kW));
                 Tensor w = RandomMatrix(out_c, in_c * kH * kW);
                 for (double& val : w.impl->data) val = (val - 0.5) * scale;
                 weights.push_back(Reshape(w, {out_c, in_c, kH, kW}));
             } else if (type == 7.0) { // RNN
                 size_t in_f = config.get(i, 1);
                 size_t hidden = config.get(i, 2);
                 double scale_x = std::sqrt(2.0 / in_f);
                 Tensor W_x = RandomMatrix(in_f, hidden);
                 for (double& val : W_x.impl->data) val = (val - 0.5) * scale_x;
                 weights.push_back(W_x);
                 
                 double scale_h = std::sqrt(2.0 / hidden);
                 Tensor W_h = RandomMatrix(hidden, hidden);
                 for (double& val : W_h.impl->data) val = (val - 0.5) * scale_h;
                 weights.push_back(W_h);
                 
                 Tensor b_h = Tensor(1, hidden, std::vector<double>(hidden, 0.0));
                 weights.push_back(b_h);
             } else if (type == 8.0) { // LSTM
                 size_t in_f = config.get(i, 1);
                 size_t hidden = config.get(i, 2);
                 double scale_x = std::sqrt(2.0 / in_f);
                 Tensor W_x = RandomMatrix(in_f, hidden * 4);
                 for (double& val : W_x.impl->data) val = (val - 0.5) * scale_x;
                 weights.push_back(W_x);
                 
                 double scale_h = std::sqrt(2.0 / hidden);
                 Tensor W_h = RandomMatrix(hidden, hidden * 4);
                 for (double& val : W_h.impl->data) val = (val - 0.5) * scale_h;
                 weights.push_back(W_h);
                 
                 Tensor b_h = Tensor(1, hidden * 4, std::vector<double>(hidden * 4, 0.0));
                 for (size_t k = hidden; k < hidden * 2; ++k) b_h.impl->data[k] = 1.0;
                 weights.push_back(b_h);
             } else if (type == 9.0) { // MultiHeadAttention
                 size_t embed_dim = config.get(i, 1);
                 double scale = std::sqrt(2.0 / embed_dim);
                 
                 Tensor W_q = RandomMatrix(embed_dim, embed_dim);
                 for (double& val : W_q.impl->data) val = (val - 0.5) * scale;
                 weights.push_back(W_q);
                 
                 Tensor W_k = RandomMatrix(embed_dim, embed_dim);
                 for (double& val : W_k.impl->data) val = (val - 0.5) * scale;
                 weights.push_back(W_k);
                 
                 Tensor W_v = RandomMatrix(embed_dim, embed_dim);
                 for (double& val : W_v.impl->data) val = (val - 0.5) * scale;
                 weights.push_back(W_v);
                 
                 Tensor W_o = RandomMatrix(embed_dim, embed_dim);
                 for (double& val : W_o.impl->data) val = (val - 0.5) * scale;
                 weights.push_back(W_o);
             }
        }
        model_weights[id] = weights;
        return Tensor(1, 1, {(double)id});
    });

    globalEnv.defineNative("Forward", [](const std::vector<Tensor>& args) {
        int id = args[0].get(0, 0);
        Tensor X = args[1];
        double is_training = args[2].get(0, 0);
        
        std::vector<Tensor>& weights = model_weights[id];
        
        if (jit_functions.find(id) != jit_functions.end()) {
            return jit_functions[id](X, weights, is_training);
        }
        
        Tensor config = model_configs[id];
        int w_idx = 0;
        
        for (size_t i = 0; i < config.impl->rows(); ++i) {
            double type = config.get(i, 0);
            if (type == 1.0) {
                X = DotProduct(X, weights[w_idx++]);
            } else if (type == 2.0) {
                X = Conv2D(X, weights[w_idx++]);
            } else if (type == 3.0) {
                X = ReLU(X);
            } else if (type == 4.0) {
                X = MaxPool2D(X);
            } else if (type == 5.0) {
                size_t N = X.impl->shape[0];
                size_t flat_size = X.impl->data.size() / N;
                X = Reshape(X, {N, flat_size});
            } else if (type == 6.0) {
                X = Dropout(X, config.get(i, 1), is_training);
            } else if (type == 7.0) { // RNN
                size_t in_f = config.get(i, 1);
                size_t hidden = config.get(i, 2);
                size_t N = X.impl->rows();
                size_t T = X.impl->cols() / in_f;
                
                Tensor W_x = weights[w_idx++];
                Tensor W_h = weights[w_idx++];
                Tensor b_h = weights[w_idx++];
                
                Tensor h = Tensor(N, hidden, std::vector<double>(N * hidden, 0.0));
                
                for (size_t t = 0; t < T; ++t) {
                    Tensor x_t = Slice(X, 0, N, t * in_f, (t + 1) * in_f);
                    h = Tanh(DotProduct(x_t, W_x) + DotProduct(h, W_h) + b_h);
                }
                X = h;
            } else if (type == 8.0) { // LSTM
                size_t in_f = config.get(i, 1);
                size_t hidden = config.get(i, 2);
                size_t N = X.impl->rows();
                size_t T = X.impl->cols() / in_f;
                
                Tensor W_x = weights[w_idx++];
                Tensor W_h = weights[w_idx++];
                Tensor b_h = weights[w_idx++];
                
                Tensor h = Tensor(N, hidden, std::vector<double>(N * hidden, 0.0));
                Tensor c = Tensor(N, hidden, std::vector<double>(N * hidden, 0.0));
                
                for (size_t t = 0; t < T; ++t) {
                    Tensor x_t = Slice(X, 0, N, t * in_f, (t + 1) * in_f);
                    Tensor gates = DotProduct(x_t, W_x) + DotProduct(h, W_h) + b_h;
                    
                    Tensor i_gate = Sigmoid(Slice(gates, 0, N, 0, hidden));
                    Tensor f_gate = Sigmoid(Slice(gates, 0, N, hidden, hidden * 2));
                    Tensor c_gate = Tanh(Slice(gates, 0, N, hidden * 2, hidden * 3));
                    Tensor o_gate = Sigmoid(Slice(gates, 0, N, hidden * 3, hidden * 4));
                    
                    c = (f_gate * c) + (i_gate * c_gate);
                    h = o_gate * Tanh(c);
                }
                X = h;
            } else if (type == 9.0) { // MultiHeadAttention
                size_t embed_dim = config.get(i, 1);
                size_t num_heads = config.get(i, 2);
                size_t d_k = embed_dim / num_heads;
                
                Tensor W_q = weights[w_idx++];
                Tensor W_k = weights[w_idx++];
                Tensor W_v = weights[w_idx++];
                Tensor W_o = weights[w_idx++];
                
                Tensor Q = DotProduct(X, W_q);
                Tensor K = DotProduct(X, W_k);
                Tensor V = DotProduct(X, W_v);
                
                std::vector<Tensor> heads;
                double scale = 1.0 / std::sqrt(d_k);
                
                for (size_t h = 0; h < num_heads; ++h) {
                    Tensor q_h = Slice(Q, 0, Q.rows(), h * d_k, (h + 1) * d_k);
                    Tensor k_h = Slice(K, 0, K.rows(), h * d_k, (h + 1) * d_k);
                    Tensor v_h = Slice(V, 0, V.rows(), h * d_k, (h + 1) * d_k);
                    
                    Tensor scores = DotProduct(q_h, Transpose(k_h)) * scale;
                    Tensor attention_weights = Softmax(scores);
                    Tensor attention_out = DotProduct(attention_weights, v_h);
                    heads.push_back(attention_out);
                }
                
                Tensor concat_heads = heads[0];
                for (size_t h = 1; h < num_heads; ++h) {
                    concat_heads = Concat(concat_heads, heads[h], 1);
                }
                
                Tensor output = DotProduct(concat_heads, W_o);
                
                // Residual connection + LayerNorm
                X = LayerNorm(X + output);
            }
        }
        return X;
    });
    
    globalEnv.defineNative("GenerateText", [](const std::vector<Tensor>& args) {
        if (args.size() != 3) throw std::runtime_error("GenerateText expects 3 arguments (model_id, seed_tensor, length)");
        
        int model_id = args[0].get(0, 0);
        Tensor seed = args[1]; // Should be [1, seq_len] with integer token IDs
        int length = args[2].get(0, 0);
        
        std::vector<std::string> vocab;
        std::ifstream vocab_file("data/vocab.txt");
        if (vocab_file.is_open()) {
            std::string line;
            while (std::getline(vocab_file, line)) {
                if (line == "\\n") vocab.push_back("\n");
                else if (line == "<space>") vocab.push_back(" ");
                else vocab.push_back(line);
            }
        } else {
            std::cout << "[Warning] Could not open data/vocab.txt\n";
        }
        
        size_t vocab_size = vocab.size() > 0 ? vocab.size() : 34;
        size_t seq_len = seed.cols();
        
        std::vector<double> current_seq = seed.impl->data;
        std::string generated_text = "";
        
        std::cout << "\nAI: \"";
        
        for (int i = 0; i < length; ++i) {
            Tensor input_seq(1, seq_len, current_seq);
            Tensor onehot = OneHot(input_seq, vocab_size);
            
            Tensor preds;
            if (jit_functions.find(model_id) != jit_functions.end()) {
                preds = jit_functions[model_id](onehot, model_weights[model_id], 0.0);
            } else {
                throw std::runtime_error("GenerateText currently requires JIT compiled models. Please call JIT(model) first.");
            }
            
            size_t last_row = preds.rows() - 1;
            double max_val = preds.get(last_row, 0);
            int max_idx = 0;
            for (size_t c = 1; c < vocab_size; ++c) {
                double val = preds.get(last_row, c);
                if (val > max_val) {
                    max_val = val;
                    max_idx = c;
                }
            }
            
            if (max_idx < vocab.size()) {
                std::cout << vocab[max_idx];
                std::cout.flush();
                generated_text += vocab[max_idx];
            } else {
                std::cout << "?";
            }
            
            current_seq.erase(current_seq.begin());
            current_seq.push_back(max_idx);
        }
        
        std::cout << "\"\n\n";
        return Tensor(1, 1, {0.0});
    });

    globalEnv.defineNative("Optimize", [](const std::vector<Tensor>& args) {
        int id = args[0].get(0, 0);
        double lr = args[1].get(0, 0);
        std::vector<Tensor>& weights = model_weights[id];
        
        for (auto& w : weights) {
            for (size_t j = 0; j < w.impl->data.size(); ++j) {
                w.impl->data[j] -= w.impl->grad[j] * lr;
            }
            std::fill(w.impl->grad.begin(), w.impl->grad.end(), 0.0);
        }
        return Tensor(1, 1, {1.0});
    });

    globalEnv.defineNative("ZeroGrad", [](const std::vector<Tensor>& args) {
        if (args.size() != 1) throw std::runtime_error("ZeroGrad expects 1 argument");
        Tensor w = args[0];
        std::fill(w.impl->grad.begin(), w.impl->grad.end(), 0.0);
        return w;
    });
}

void Evaluator::evaluate(const std::vector<std::unique_ptr<Stmt>>& statements) {
    for (const auto& stmt : statements) {
        evaluate(*stmt);
    }
}

void Evaluator::executeBlock(const std::vector<std::unique_ptr<Stmt>>& statements, Environment* env) {
    Environment* previous = this->environment;
    this->environment = env;
    try {
        for (const auto& stmt : statements) {
            evaluate(*stmt);
        }
    } catch (...) {
        this->environment = previous;
        throw;
    }
    this->environment = previous;
}

void Evaluator::evaluate(Stmt& stmt) {
    stmt.accept(*this);
}

Tensor Evaluator::evaluate(Expr& expr) {
    expr.accept(*this);
    return lastValue;
}

void Evaluator::visit(NumberExpr& expr) {
    lastValue = Tensor(1, 1, {expr.value});
}

void Evaluator::visit(VariableExpr& expr) {
    lastValue = environment->get(expr.name);
}

void Evaluator::visit(VectorExpr& expr) {
    lastValue = Tensor(expr.values.size(), 1, expr.values);
}

void Evaluator::visit(MatrixExpr& expr) {
    lastValue = Tensor(expr.rows, expr.cols, expr.values);
}

void Evaluator::visit(ArrayExpr& expr) {
    if (expr.elements.empty()) {
        lastValue = Tensor(0, 0, {});
        return;
    }
    
    std::vector<Tensor> evaluated;
    for (auto& el : expr.elements) evaluated.push_back(evaluate(*el));
    
    bool all_scalars = true;
    for (auto& t : evaluated) {
        if (t.impl->shape.size() != 2 || t.impl->shape[0] != 1 || t.impl->shape[1] != 1) {
            all_scalars = false; break;
        }
    }
    
    if (all_scalars) {
        std::vector<double> data;
        for (auto& t : evaluated) data.push_back(t.impl->data[0]);
        lastValue = Tensor(data.size(), 1, data);
        return;
    }
    
    size_t cols = 0;
    size_t rows = evaluated.size();
    for (auto& t : evaluated) {
        size_t c = t.impl->data.size();
        if (c > cols) cols = c;
    }
    
    std::vector<double> data(rows * cols, 0.0);
    for (size_t i = 0; i < rows; ++i) {
        Tensor& t = evaluated[i];
        size_t c = t.impl->data.size();
        for (size_t j = 0; j < c; ++j) {
            data[i * cols + j] = t.impl->data[j];
        }
    }
    
    lastValue = Tensor(rows, cols, data);
}
void Evaluator::visit(LoadExpr& expr) {
    lastValue = LoadTensor(expr.filename);
}

void Evaluator::visit(BinaryExpr& expr) {
    Tensor left = evaluate(*expr.left);
    Tensor right = evaluate(*expr.right);
    
    if (expr.op == TokenType::PLUS) {
        lastValue = left + right;
    } else if (expr.op == TokenType::MINUS) {
        lastValue = left - right;
    } else if (expr.op == TokenType::STAR) {
        if (left.impl->rows() == 1 && left.impl->cols() == 1) {
            lastValue = right * left.get(0, 0);
        } else if (right.impl->rows() == 1 && right.impl->cols() == 1) {
            lastValue = left * right.get(0, 0);
        } else {
            lastValue = left * right; // Element-wise multiplication
        }
    } else if (expr.op == TokenType::LESS) {
        lastValue = Tensor(1, 1, {left.get(0, 0) < right.get(0, 0) ? 1.0 : 0.0});
    } else if (expr.op == TokenType::GREATER) {
        lastValue = Tensor(1, 1, {left.get(0, 0) > right.get(0, 0) ? 1.0 : 0.0});
    } else {
        throw std::runtime_error("Unsupported binary operation");
    }
}

void Evaluator::visit(CallExpr& expr) {
    std::vector<Tensor> evaluatedArgs;
    for (const auto& arg : expr.arguments) {
        evaluatedArgs.push_back(evaluate(*arg));
    }

    if (environment->hasNative(expr.callee)) {
        lastValue = environment->getNative(expr.callee)(evaluatedArgs);
        return;
    }

    FunctionStmt* func = environment->getFunc(expr.callee);
    if (expr.arguments.size() != func->params.size()) {
        throw std::runtime_error("Expected " + std::to_string(func->params.size()) + " arguments but got " + std::to_string(expr.arguments.size()) + ".");
    }
    
    Environment localEnv(environment);
    for (size_t i = 0; i < evaluatedArgs.size(); ++i) {
        localEnv.define(func->params[i], evaluatedArgs[i]);
    }
    
    try {
        executeBlock(func->body, &localEnv);
    } catch (ReturnException& returnValue) {
        lastValue = returnValue.value;
        return;
    }
    lastValue = Tensor(1, 1, {0.0});
}


void Evaluator::visit(LoadCSVExpr& expr) {
    lastValue = LoadCSV(expr.filename);
}

void Evaluator::visit(PropertyGetExpr& expr) {
    Tensor obj = evaluate(*expr.object);
    if (expr.name == "grad") {
        lastValue = Tensor(obj.impl->rows(), obj.impl->cols(), obj.impl->grad);
    } else {
        throw std::runtime_error("Unknown property: " + expr.name);
    }
}

void Evaluator::visit(IndexExpr& expr) {
    Tensor obj = evaluate(*expr.object);
    Tensor index = evaluate(*expr.index);
    lastValue = obj.get_row((size_t)index.get(0,0));
}

void Evaluator::visit(LetStmt& stmt) {
    Tensor value = evaluate(*stmt.initializer);
    environment->define(stmt.name, value);
}

void Evaluator::visit(AssignStmt& stmt) {
    Tensor value = evaluate(*stmt.value);
    environment->define(stmt.name, value);
}

void Evaluator::visit(PrintStmt& stmt) {
    if (stmt.isString) {
        out << stmt.stringLiteral << "\n";
    } else {
        Tensor val = evaluate(*stmt.expr);
        val.print(out);
    }
}

void Evaluator::visit(ForStmt& stmt) {
    for (int i = stmt.startVal; i <= stmt.endVal; ++i) {
        Environment localEnv(environment);
        localEnv.define(stmt.loopVar, Tensor(1, 1, {(double)i}));
        executeBlock(stmt.body, &localEnv);
    }
}

void Evaluator::visit(WhileStmt& stmt) {
    while (evaluate(*stmt.condition).get(0, 0) != 0.0) {
        Environment localEnv(environment);
        executeBlock(stmt.body, &localEnv);
    }
}

void Evaluator::visit(IfStmt& stmt) {
    Tensor cond = evaluate(*stmt.condition);
    if (cond.get(0, 0) != 0.0) {
        Environment localEnv(environment);
        executeBlock(stmt.thenBranch, &localEnv);
    } else if (!stmt.elseBranch.empty()) {
        Environment localEnv(environment);
        executeBlock(stmt.elseBranch, &localEnv);
    }
}

void Evaluator::visit(FunctionStmt& stmt) {
    environment->defineFunc(stmt.name, &stmt);
}

void Evaluator::visit(ReturnStmt& stmt) {
    Tensor value = evaluate(*stmt.value);
    throw ReturnException(value);
}

void Evaluator::visit(SaveStmt& stmt) {
    Tensor t = evaluate(*stmt.expr);
    SaveTensor(t, stmt.filename);
}

void Evaluator::visit(BackwardStmt& stmt) {
    Tensor t = evaluate(*stmt.expr);
    t.backward();
}

} // namespace tnsr
