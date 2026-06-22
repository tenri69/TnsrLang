/*
 * =====================================================================================
 *
 *       Filename:  ASTPrinter.cpp
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

#include "ASTPrinter.h"

namespace tnsr {

ASTPrinter::ASTPrinter(const std::string& filename) : nodeCounter(0), currentId(0) {
    out.open(filename);
    if (!out.is_open()) {
        std::cerr << "Could not open file for AST dump: " << filename << "\n";
    }
    out << "digraph AST {\n";
    out << "  node [shape=box, style=filled, fillcolor=lightblue, fontname=\"Helvetica\"];\n";
}

ASTPrinter::~ASTPrinter() {
    out << "}\n";
    if (out.is_open()) {
        out.close();
    }
}

int ASTPrinter::pushNode(const std::string& label) {
    int id = ++nodeCounter;
    // escape quotes in label
    std::string safeLabel = label;
    size_t pos = 0;
    while((pos = safeLabel.find("\"", pos)) != std::string::npos) {
        safeLabel.replace(pos, 1, "\\\"");
        pos += 2;
    }
    
    out << "  node" << id << " [label=\"" << safeLabel << "\"];\n";
    return id;
}

void ASTPrinter::link(int from, int to) {
    out << "  node" << from << " -> node" << to << ";\n";
}

void ASTPrinter::print(const std::vector<std::unique_ptr<Stmt>>& statements) {
    int rootId = pushNode("Program");
    for (const auto& stmt : statements) {
        stmt->accept(*this);
        link(rootId, currentId);
    }
}

void ASTPrinter::visit(NumberExpr& expr) {
    currentId = pushNode("Number: " + std::to_string(expr.value));
}

void ASTPrinter::visit(VariableExpr& expr) {
    currentId = pushNode("Variable: " + expr.name);
}

void ASTPrinter::visit(VectorExpr& expr) {
    std::stringstream ss;
    ss << "Vector [";
    for (size_t i = 0; i < expr.values.size(); ++i) {
        ss << expr.values[i];
        if (i < expr.values.size() - 1) ss << ", ";
    }
    ss << "]";
    currentId = pushNode(ss.str());
}

void ASTPrinter::visit(MatrixExpr& expr) {
    std::stringstream ss;
    ss << "Matrix (" << expr.rows << "x" << expr.cols << ")";
    currentId = pushNode(ss.str());
}

void ASTPrinter::visit(ArrayExpr& expr) {
    int id = pushNode("Array");
    for (const auto& el : expr.elements) {
        el->accept(*this);
        link(id, currentId);
    }
    currentId = id;
}
void ASTPrinter::visit(CallExpr& expr) {
    currentId = pushNode("Call: " + expr.callee);
}

void ASTPrinter::visit(LoadExpr& expr) {
    currentId = pushNode("Load: " + expr.filename);
}

void ASTPrinter::visit(BinaryExpr& expr) {
    std::string opStr = (expr.op == TokenType::PLUS) ? "+" : ((expr.op == TokenType::MINUS) ? "-" : "*");
    int id = pushNode("Binary: " + opStr);
    
    expr.left->accept(*this);
    link(id, currentId);
    
    expr.right->accept(*this);
    link(id, currentId);
    
    currentId = id;
}

void ASTPrinter::visit(LetStmt& stmt) {
    int id = pushNode("Let: " + stmt.name);
    
    stmt.initializer->accept(*this);
    link(id, currentId);
    
    currentId = id;
}

void ASTPrinter::visit(AssignStmt& stmt) {
    int id = pushNode("Assign: " + stmt.name);
    
    stmt.value->accept(*this);
    link(id, currentId);
    
    currentId = id;
}

void ASTPrinter::visit(PrintStmt& stmt) {
    int id = pushNode("Print");
    
    if (stmt.isString) {
        int strId = pushNode("String: \\\"" + stmt.stringLiteral + "\\\"");
        link(id, strId);
    } else {
        stmt.expr->accept(*this);
        link(id, currentId);
    }
    
    currentId = id;
}

void ASTPrinter::visit(ForStmt& stmt) {
    int id = pushNode("For: " + stmt.loopVar + " = " + std::to_string(stmt.startVal) + " to " + std::to_string(stmt.endVal));
    
    for (const auto& s : stmt.body) {
        s->accept(*this);
        link(id, currentId);
    }
    
    currentId = id;
}

void ASTPrinter::visit(IfStmt& stmt) {
    currentId = pushNode("IfStmt");
}

void ASTPrinter::visit(FunctionStmt& stmt) {
    currentId = pushNode("FunctionStmt: " + stmt.name);
}

void ASTPrinter::visit(ReturnStmt& stmt) {
    currentId = pushNode("ReturnStmt");
}

void ASTPrinter::visit(SaveStmt& stmt) {
    currentId = pushNode("SaveStmt: " + stmt.filename);
}

void ASTPrinter::visit(BackwardStmt& stmt) {
    currentId = pushNode("BackwardStmt");
}


void ASTPrinter::visit(LoadCSVExpr& expr) {
    currentId = pushNode("LoadCSV: " + expr.filename);
}

void ASTPrinter::visit(PropertyGetExpr& expr) {
    currentId = pushNode("Property: ." + expr.name);
}

void ASTPrinter::visit(IndexExpr& expr) {
    currentId = pushNode("Index");
}

void ASTPrinter::visit(WhileStmt& stmt) {
    currentId = pushNode("WhileStmt");
}

} // namespace tnsr
