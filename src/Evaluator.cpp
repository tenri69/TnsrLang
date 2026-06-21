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

    globalEnv.defineNative("RandomMatrix", [](const std::vector<Tensor>& args) {
        if (args.size() != 2) throw std::runtime_error("RandomMatrix expects 2 arguments (rows, cols)");
        return RandomMatrix((size_t)args[0].get(0, 0), (size_t)args[1].get(0, 0));
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
