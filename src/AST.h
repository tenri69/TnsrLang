/*
 * =====================================================================================
 *
 *       Filename:  AST.h
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

#include "Lexer.h"
#include <memory>
#include <vector>
#include <string>

namespace tnsr {

class ASTVisitor;

class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void accept(ASTVisitor& visitor) = 0;
};

class Expr : public ASTNode {};

class NumberExpr : public Expr {
public:
    double value;
    NumberExpr(double val) : value(val) {}
    void accept(ASTVisitor& visitor) override;
};

class VariableExpr : public Expr {
public:
    std::string name;
    VariableExpr(const std::string& name) : name(name) {}
    void accept(ASTVisitor& visitor) override;
};

class VectorExpr : public Expr {
public:
    std::vector<double> values;
    VectorExpr(const std::vector<double>& values) : values(values) {}
    void accept(ASTVisitor& visitor) override;
};

class MatrixExpr : public Expr {
public:
    size_t rows;
    size_t cols;
    std::vector<double> values;
    MatrixExpr(size_t r, size_t c, const std::vector<double>& vals) : rows(r), cols(c), values(vals) {}
    void accept(ASTVisitor& visitor) override;
};

class DotProductExpr : public Expr {
public:
    std::unique_ptr<Expr> left;
    std::unique_ptr<Expr> right;
    DotProductExpr(std::unique_ptr<Expr> l, std::unique_ptr<Expr> r) : left(std::move(l)), right(std::move(r)) {}
    void accept(ASTVisitor& visitor) override;
};

class ReLUExpr : public Expr {
public:
    std::unique_ptr<Expr> expr;
    ReLUExpr(std::unique_ptr<Expr> e) : expr(std::move(e)) {}
    void accept(ASTVisitor& visitor) override;
};

class SigmoidExpr : public Expr {
public:
    std::unique_ptr<Expr> expr;
    SigmoidExpr(std::unique_ptr<Expr> e) : expr(std::move(e)) {}
    void accept(ASTVisitor& visitor) override;
};

class SoftmaxExpr : public Expr {
public:
    std::unique_ptr<Expr> expr;
    SoftmaxExpr(std::unique_ptr<Expr> e) : expr(std::move(e)) {}
    void accept(ASTVisitor& visitor) override;
};

class TransposeExpr : public Expr {
public:
    std::unique_ptr<Expr> expr;
    TransposeExpr(std::unique_ptr<Expr> e) : expr(std::move(e)) {}
    void accept(ASTVisitor& visitor) override;
};

class CallExpr : public Expr {
public:
    std::string callee;
    std::vector<std::unique_ptr<Expr>> arguments;
    CallExpr(const std::string& callee, std::vector<std::unique_ptr<Expr>> args) : callee(callee), arguments(std::move(args)) {}
    void accept(ASTVisitor& visitor) override;
};

class LoadExpr : public Expr {
public:
    std::string filename;
    LoadExpr(const std::string& filename) : filename(filename) {}
    void accept(ASTVisitor& visitor) override;
};

class BinaryExpr : public Expr {
public:
    std::unique_ptr<Expr> left;
    TokenType op;
    std::unique_ptr<Expr> right;
    BinaryExpr(std::unique_ptr<Expr> l, TokenType op, std::unique_ptr<Expr> r) : left(std::move(l)), op(op), right(std::move(r)) {}
    void accept(ASTVisitor& visitor) override;
};

class Stmt : public ASTNode {};

class LetStmt : public Stmt {
public:
    std::string name;
    std::unique_ptr<Expr> initializer;
    LetStmt(const std::string& name, std::unique_ptr<Expr> init) : name(name), initializer(std::move(init)) {}
    void accept(ASTVisitor& visitor) override;
};

class PrintStmt : public Stmt {
public:
    std::unique_ptr<Expr> expr;
    std::string stringLiteral; // if printing a string
    bool isString;
    
    PrintStmt(std::unique_ptr<Expr> e) : expr(std::move(e)), isString(false) {}
    PrintStmt(const std::string& str) : stringLiteral(str), isString(true) {}
    void accept(ASTVisitor& visitor) override;
};

class ForStmt : public Stmt {
public:
    std::string loopVar;
    int startVal;
    int endVal;
    std::vector<std::unique_ptr<Stmt>> body;
    
    ForStmt(const std::string& var, int start, int end, std::vector<std::unique_ptr<Stmt>> body)
        : loopVar(var), startVal(start), endVal(end), body(std::move(body)) {}
    void accept(ASTVisitor& visitor) override;
};

class IfStmt : public Stmt {
public:
    std::unique_ptr<Expr> condition;
    std::vector<std::unique_ptr<Stmt>> thenBranch;
    std::vector<std::unique_ptr<Stmt>> elseBranch;
    
    IfStmt(std::unique_ptr<Expr> condition, std::vector<std::unique_ptr<Stmt>> thenBranch, std::vector<std::unique_ptr<Stmt>> elseBranch)
        : condition(std::move(condition)), thenBranch(std::move(thenBranch)), elseBranch(std::move(elseBranch)) {}
    void accept(ASTVisitor& visitor) override;
};

class FunctionStmt : public Stmt {
public:
    std::string name;
    std::vector<std::string> params;
    std::vector<std::unique_ptr<Stmt>> body;
    
    FunctionStmt(const std::string& name, const std::vector<std::string>& params, std::vector<std::unique_ptr<Stmt>> body)
        : name(name), params(params), body(std::move(body)) {}
    void accept(ASTVisitor& visitor) override;
};

class ReturnStmt : public Stmt {
public:
    std::unique_ptr<Expr> value;
    ReturnStmt(std::unique_ptr<Expr> value) : value(std::move(value)) {}
    void accept(ASTVisitor& visitor) override;
};

class SaveStmt : public Stmt {
public:
    std::unique_ptr<Expr> expr;
    std::string filename;
    SaveStmt(std::unique_ptr<Expr> expr, const std::string& filename) : expr(std::move(expr)), filename(filename) {}
    void accept(ASTVisitor& visitor) override;
};

class BackwardStmt : public Stmt {
public:
    std::unique_ptr<Expr> expr;
    BackwardStmt(std::unique_ptr<Expr> expr) : expr(std::move(expr)) {}
    void accept(ASTVisitor& visitor) override;
};

class ASTVisitor {
public:
    virtual ~ASTVisitor() = default;
    virtual void visit(NumberExpr& expr) = 0;
    virtual void visit(VariableExpr& expr) = 0;
    virtual void visit(VectorExpr& expr) = 0;
    virtual void visit(MatrixExpr& expr) = 0;
    virtual void visit(DotProductExpr& expr) = 0;
    virtual void visit(ReLUExpr& expr) = 0;
    virtual void visit(SigmoidExpr& expr) = 0;
    virtual void visit(SoftmaxExpr& expr) = 0;
    virtual void visit(TransposeExpr& expr) = 0;
    virtual void visit(CallExpr& expr) = 0;
    virtual void visit(LoadExpr& expr) = 0;
    virtual void visit(BinaryExpr& expr) = 0;
    virtual void visit(LetStmt& stmt) = 0;
    virtual void visit(PrintStmt& stmt) = 0;
    virtual void visit(ForStmt& stmt) = 0;
    virtual void visit(IfStmt& stmt) = 0;
    virtual void visit(FunctionStmt& stmt) = 0;
    virtual void visit(ReturnStmt& stmt) = 0;
    virtual void visit(SaveStmt& stmt) = 0;
    virtual void visit(BackwardStmt& stmt) = 0;
};

} // namespace tnsr
