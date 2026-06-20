/*
 * =====================================================================================
 *
 *       Filename:  AST.cpp
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

#include "AST.h"

namespace tnsr {

void NumberExpr::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void VariableExpr::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void VectorExpr::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void MatrixExpr::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void CallExpr::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void LoadExpr::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void BinaryExpr::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void LetStmt::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void AssignStmt::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void PrintStmt::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void ForStmt::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void IfStmt::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void FunctionStmt::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void ReturnStmt::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void SaveStmt::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void BackwardStmt::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void LoadCSVExpr::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void PropertyGetExpr::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void IndexExpr::accept(ASTVisitor& visitor) { visitor.visit(*this); }
void WhileStmt::accept(ASTVisitor& visitor) { visitor.visit(*this); }
} // namespace tnsr
