/*
 * =====================================================================================
 *
 *       Filename:  Parser.h
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
#include "Lexer.h"
#include <vector>

namespace tnsr {

/**
 * @brief The Parser converts a stream of Tokens into an Abstract Syntax Tree (AST).
 * It uses a Recursive Descent approach, meaning each grammar rule has a corresponding method.
 */
class Parser {
public:
    Parser(const std::vector<Token>& tokens);
    std::vector<std::unique_ptr<Stmt>> parse();

private:
    std::vector<Token> tokens;
    size_t current;

    Token peek() const;
    Token previous() const;
    bool isAtEnd() const;
    Token advance();
    bool check(TokenType type) const;
    bool match(TokenType type);
    Token consume(TokenType type, const std::string& message);

    std::vector<std::unique_ptr<Stmt>> parseBlock();
    
    std::unique_ptr<Stmt> parseStatement();
    std::unique_ptr<LetStmt> parseLetStmt();
    std::unique_ptr<AssignStmt> parseAssignStmt();
    std::unique_ptr<PrintStmt> parsePrintStmt();
    std::unique_ptr<ForStmt> parseForStmt();
    std::unique_ptr<WhileStmt> parseWhileStmt();
    std::unique_ptr<IfStmt> parseIfStmt();
    std::unique_ptr<FunctionStmt> parseFunctionStmt();
    std::unique_ptr<ReturnStmt> parseReturnStmt();
    std::unique_ptr<SaveStmt> parseSaveStmt();
    std::unique_ptr<BackwardStmt> parseBackwardStmt();
    
    std::unique_ptr<Expr> parseExpression();
    std::unique_ptr<Expr> parseComparison();
    std::unique_ptr<Expr> parseTerm();
    std::unique_ptr<Expr> parseFactor();
    std::unique_ptr<Expr> parseCall();
    std::unique_ptr<Expr> parsePrimary();
    
    std::unique_ptr<ArrayExpr> parseArrayExpr();
    std::unique_ptr<VectorExpr> parseVectorExpr();
    std::unique_ptr<MatrixExpr> parseMatrixExpr();
    std::unique_ptr<LoadExpr> parseLoadExpr();
    std::unique_ptr<LoadCSVExpr> parseLoadCSVExpr();
    
    std::vector<double> parseNumberArray();
};

} // namespace tnsr
