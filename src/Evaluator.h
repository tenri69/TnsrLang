/*
 * =====================================================================================
 *
 *       Filename:  Evaluator.h
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

#include "AST.h"
#include "MathEngine.h"
#include <unordered_map>
#include <string>
#include <vector>

namespace tnsr {

/**
 * @brief Exception thrown when a 'return' statement is executed inside a function.
 * This acts as a control-flow jump to exit the function block early.
 */
class ReturnException : public std::exception {
public:
    Tensor value;
    ReturnException(Tensor v) : value(v) {}
};

/**
 * @brief Environment holds the variables, functions, and native C++ functions for a given scope.
 * Scopes can be nested (e.g., inside an 'if' or 'fn' block), creating a chain of enclosing environments.
 */
class Environment {
public:
    std::unordered_map<std::string, Tensor> values;
    std::unordered_map<std::string, FunctionStmt*> functions;
    std::unordered_map<std::string, std::function<Tensor(const std::vector<Tensor>&)>> nativeFunctions;
    Environment* enclosing;

    Environment() : enclosing(nullptr) {}
    Environment(Environment* enclosing) : enclosing(enclosing) {}

    void define(const std::string& name, Tensor value) {
        Environment* curr = this;
        while (curr) {
            if (curr->values.count(name)) {
                curr->values[name] = value;
                return;
            }
            curr = curr->enclosing;
        }
        values[name] = value;
    }
    
    void defineFunc(const std::string& name, FunctionStmt* func) {
        functions[name] = func;
    }

    Tensor get(const std::string& name) {
        if (values.count(name)) return values[name];
        if (enclosing) return enclosing->get(name);
        throw std::runtime_error("Undefined variable: " + name);
    }
    
    FunctionStmt* getFunc(const std::string& name) {
        if (functions.count(name)) return functions[name];
        if (enclosing) return enclosing->getFunc(name);
        throw std::runtime_error("Undefined function: " + name);
    }
    
    void defineNative(const std::string& name, std::function<Tensor(const std::vector<Tensor>&)> func) {
        nativeFunctions[name] = func;
    }
    
    bool hasNative(const std::string& name) {
        if (nativeFunctions.count(name)) return true;
        if (enclosing) return enclosing->hasNative(name);
        return false;
    }
    
    std::function<Tensor(const std::vector<Tensor>&)> getNative(const std::string& name) {
        if (nativeFunctions.count(name)) return nativeFunctions[name];
        if (enclosing) return enclosing->getNative(name);
        throw std::runtime_error("Undefined native function: " + name);
    }
};

/**
 * @brief The Evaluator traverses the AST and executes the code.
 * It manages the global Environment and acts as the bridge between the TnsrLang syntax and the C++ MathEngine.
 */
class Evaluator : public ASTVisitor {
public:
    Evaluator(std::ostream& out = std::cout);

    void evaluate(const std::vector<std::unique_ptr<Stmt>>& statements);
    void evaluate(Stmt& stmt);
    Tensor evaluate(Expr& expr);

    // ASTVisitor interface
    void visit(NumberExpr& expr) override;
    void visit(VariableExpr& expr) override;
    void visit(VectorExpr& expr) override;
    void visit(MatrixExpr& expr) override;
    void visit(ArrayExpr& expr) override;
    void visit(CallExpr& expr) override;
    void visit(LoadExpr& expr) override;
    void visit(LoadCSVExpr& expr) override;
    void visit(PropertyGetExpr& expr) override;
    void visit(IndexExpr& expr) override;
    void visit(BinaryExpr& expr) override;
    void visit(LetStmt& stmt) override;
    void visit(AssignStmt& stmt) override;
    void visit(PrintStmt& stmt) override;
    void visit(ForStmt& stmt) override;
    void visit(WhileStmt& stmt) override;
    void visit(IfStmt& stmt) override;
    void visit(FunctionStmt& stmt) override;
    void visit(ReturnStmt& stmt) override;
    void visit(SaveStmt& stmt) override;
    void visit(BackwardStmt& stmt) override;

private:
    std::ostream& out;
    Environment* environment;
    Environment globalEnv;
    Tensor lastValue; // Used to return values from expression visits
    
    void executeBlock(const std::vector<std::unique_ptr<Stmt>>& statements, Environment* env);
};

} // namespace tnsr
