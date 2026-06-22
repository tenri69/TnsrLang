/*
 * =====================================================================================
 *
 *       Filename:  ASTPrinter.h
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
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace tnsr {

class ASTPrinter : public ASTVisitor {
public:
    ASTPrinter(const std::string& filename);
    ~ASTPrinter();

    void print(const std::vector<std::unique_ptr<Stmt>>& statements);

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
    std::ofstream out;
    int nodeCounter;
    
    int pushNode(const std::string& label);
    void link(int from, int to);
    
    // state for traversal
    int currentId; 
};

} // namespace tnsr
