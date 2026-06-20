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

Evaluator::Evaluator(std::ostream& out) : out(out), environment(&globalEnv), lastValue(0, 0) {}

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

void Evaluator::visit(DotProductExpr& expr) {
    Tensor left = evaluate(*expr.left);
    Tensor right = evaluate(*expr.right);
    lastValue = DotProduct(left, right);
}

void Evaluator::visit(ReLUExpr& expr) {
    Tensor inner = evaluate(*expr.expr);
    lastValue = ReLU(inner);
}

void Evaluator::visit(SigmoidExpr& expr) {
    Tensor inner = evaluate(*expr.expr);
    lastValue = Sigmoid(inner);
}

void Evaluator::visit(SoftmaxExpr& expr) {
    Tensor inner = evaluate(*expr.expr);
    lastValue = Softmax(inner);
}

void Evaluator::visit(TransposeExpr& expr) {
    Tensor inner = evaluate(*expr.expr);
    lastValue = Transpose(inner);
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
        if (left.impl->rows == 1 && left.impl->cols == 1) {
            lastValue = right * left.get(0, 0);
        } else if (right.impl->rows == 1 && right.impl->cols == 1) {
            lastValue = left * right.get(0, 0);
        } else {
            throw std::runtime_error("Scalar multiplication requires one operand to be a scalar (1x1 Tensor)");
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
    FunctionStmt* func = environment->getFunc(expr.callee);
    if (expr.arguments.size() != func->params.size()) {
        throw std::runtime_error("Expected " + std::to_string(func->params.size()) + " arguments but got " + std::to_string(expr.arguments.size()) + ".");
    }
    
    Environment localEnv(environment);
    for (size_t i = 0; i < expr.arguments.size(); ++i) {
        localEnv.define(func->params[i], evaluate(*expr.arguments[i]));
    }
    
    try {
        executeBlock(func->body, &localEnv);
    } catch (ReturnException& returnValue) {
        lastValue = returnValue.value;
        return;
    }
    lastValue = Tensor(1, 1, {0.0});
}

void Evaluator::visit(LetStmt& stmt) {
    Tensor value = evaluate(*stmt.initializer);
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
