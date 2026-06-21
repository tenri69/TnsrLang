/*
 * =====================================================================================
 *
 *       Filename:  Parser.cpp
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

#include "Parser.h"
#include <stdexcept>

namespace tnsr {

// The Parser converts the flat stream of Tokens produced by the Lexer into an 
// Abstract Syntax Tree (AST). It uses Recursive Descent parsing, meaning there is 
// a dedicated C++ method for every grammar rule in the language.

Parser::Parser(const std::vector<Token>& tokens) : tokens(tokens), current(0) {}

Token Parser::peek() const { return tokens[current]; }
Token Parser::previous() const { return tokens[current - 1]; }
bool Parser::isAtEnd() const { return peek().type == TokenType::END_OF_FILE; }

Token Parser::advance() {
    if (!isAtEnd()) current++;
    return previous();
}

bool Parser::check(TokenType type) const {
    if (isAtEnd()) return false;
    return peek().type == type;
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

Token Parser::consume(TokenType type, const std::string& message) {
    if (check(type)) return advance();
    throw std::runtime_error(message + " found: " + peek().text);
}

// Entry point for parsing. Parses statements until EOF is reached.
std::vector<std::unique_ptr<Stmt>> Parser::parse() {
    std::vector<std::unique_ptr<Stmt>> statements;
    while (!isAtEnd()) {
        statements.push_back(parseStatement());
    }
    return statements;
}

std::vector<std::unique_ptr<Stmt>> Parser::parseBlock() {
    consume(TokenType::LBRACE, "Expect '{' before block.");
    std::vector<std::unique_ptr<Stmt>> body;
    while (!check(TokenType::RBRACE) && !isAtEnd()) {
        body.push_back(parseStatement());
    }
    consume(TokenType::RBRACE, "Expect '}' after block.");
    return body;
}

std::unique_ptr<Stmt> Parser::parseStatement() {
    if (match(TokenType::LET)) return parseLetStmt();
    if (match(TokenType::PRINT)) return parsePrintStmt();
    if (match(TokenType::FOR)) return parseForStmt();
    if (match(TokenType::IF)) return parseIfStmt();
    if (match(TokenType::WHILE)) return parseWhileStmt();
    if (match(TokenType::FN)) return parseFunctionStmt();
    if (match(TokenType::RETURN)) return parseReturnStmt();
    if (match(TokenType::SAVE)) return parseSaveStmt();
    if (match(TokenType::BACKWARD)) return parseBackwardStmt();
    if (check(TokenType::IDENTIFIER)) {
        // Need to differentiate between assignment `a = 1` and expression statement (if supported)
        // Since we only support specific statements, let's assume an identifier starting a line is an assignment
        return parseAssignStmt();
    }
    
    // As a fallback, we might have an expression statement (not fully supported by original parser unless it was LET, but let's allow it maybe? No, let's stick to explicitly handled stmts)
    throw std::runtime_error("Unknown statement starting with: " + peek().text);
}

std::unique_ptr<IfStmt> Parser::parseIfStmt() {
    std::unique_ptr<Expr> condition = parseExpression();
    std::vector<std::unique_ptr<Stmt>> thenBranch = parseBlock();
    std::vector<std::unique_ptr<Stmt>> elseBranch;
    if (match(TokenType::ELSE)) {
        if (check(TokenType::IF)) {
            advance();
            elseBranch.push_back(parseIfStmt());
        } else {
            elseBranch = parseBlock();
        }
    }
    return std::make_unique<IfStmt>(std::move(condition), std::move(thenBranch), std::move(elseBranch));
}

std::unique_ptr<FunctionStmt> Parser::parseFunctionStmt() {
    Token name = consume(TokenType::IDENTIFIER, "Expect function name.");
    consume(TokenType::LPAREN, "Expect '(' after function name.");
    std::vector<std::string> params;
    if (!check(TokenType::RPAREN)) {
        do {
            params.push_back(consume(TokenType::IDENTIFIER, "Expect parameter name.").text);
        } while (match(TokenType::COMMA));
    }
    consume(TokenType::RPAREN, "Expect ')' after parameters.");
    std::vector<std::unique_ptr<Stmt>> body = parseBlock();
    return std::make_unique<FunctionStmt>(name.text, params, std::move(body));
}

std::unique_ptr<ReturnStmt> Parser::parseReturnStmt() {
    std::unique_ptr<Expr> value = parseExpression();
    return std::make_unique<ReturnStmt>(std::move(value));
}

std::unique_ptr<SaveStmt> Parser::parseSaveStmt() {
    consume(TokenType::LPAREN, "Expect '(' after Save.");
    std::unique_ptr<Expr> expr = parseExpression();
    consume(TokenType::COMMA, "Expect ',' after expression in Save.");
    Token filename = consume(TokenType::STRING, "Expect filename string in Save.");
    consume(TokenType::RPAREN, "Expect ')' after Save arguments.");
    return std::make_unique<SaveStmt>(std::move(expr), filename.text);
}

std::unique_ptr<BackwardStmt> Parser::parseBackwardStmt() {
    consume(TokenType::LPAREN, "Expect '(' after backward.");
    std::unique_ptr<Expr> expr = parseExpression();
    consume(TokenType::RPAREN, "Expect ')' after backward arguments.");
    return std::make_unique<BackwardStmt>(std::move(expr));
}

std::unique_ptr<ForStmt> Parser::parseForStmt() {
    Token loopVar = consume(TokenType::IDENTIFIER, "Expect loop variable name.");
    consume(TokenType::EQUALS, "Expect '=' after loop variable.");
    Token startVal = consume(TokenType::NUMBER, "Expect start number.");
    consume(TokenType::TO, "Expect 'to' after start number.");
    Token endVal = consume(TokenType::NUMBER, "Expect end number.");
    std::vector<std::unique_ptr<Stmt>> body = parseBlock();
    return std::make_unique<ForStmt>(loopVar.text, (int)startVal.numberValue, (int)endVal.numberValue, std::move(body));
}

std::unique_ptr<WhileStmt> Parser::parseWhileStmt() {
    std::unique_ptr<Expr> condition = parseExpression();
    std::vector<std::unique_ptr<Stmt>> body = parseBlock();
    return std::make_unique<WhileStmt>(std::move(condition), std::move(body));
}

std::unique_ptr<LetStmt> Parser::parseLetStmt() {
    Token name = consume(TokenType::IDENTIFIER, "Expect variable name after 'let'.");
    consume(TokenType::EQUALS, "Expect '=' after variable name.");
    std::unique_ptr<Expr> initializer = parseExpression();
    return std::make_unique<LetStmt>(name.text, std::move(initializer));
}

std::unique_ptr<AssignStmt> Parser::parseAssignStmt() {
    Token name = consume(TokenType::IDENTIFIER, "Expect variable name for assignment.");
    consume(TokenType::EQUALS, "Expect '=' after variable name.");
    std::unique_ptr<Expr> value = parseExpression();
    return std::make_unique<AssignStmt>(name.text, std::move(value));
}

std::unique_ptr<PrintStmt> Parser::parsePrintStmt() {
    consume(TokenType::LPAREN, "Expect '(' after 'Print'.");
    if (match(TokenType::STRING)) {
        std::string str = previous().text;
        consume(TokenType::RPAREN, "Expect ')' after print arguments.");
        return std::make_unique<PrintStmt>(str);
    }
    std::unique_ptr<Expr> expr = parseExpression();
    consume(TokenType::RPAREN, "Expect ')' after print arguments.");
    return std::make_unique<PrintStmt>(std::move(expr));
}

// --- Expression Parsing (Order of Operations) ---
// Expressions are parsed strictly in order of precedence:
// Comparison -> Addition/Subtraction -> Multiplication -> Function Calls -> Primitives

std::unique_ptr<Expr> Parser::parseExpression() {
    return parseComparison();
}

std::unique_ptr<Expr> Parser::parseComparison() {
    std::unique_ptr<Expr> expr = parseTerm();
    while (match(TokenType::LESS) || match(TokenType::GREATER)) {
        TokenType op = previous().type;
        std::unique_ptr<Expr> right = parseTerm();
        expr = std::make_unique<BinaryExpr>(std::move(expr), op, std::move(right));
    }
    return expr;
}

std::unique_ptr<Expr> Parser::parseTerm() {
    std::unique_ptr<Expr> expr = parseFactor();
    while (match(TokenType::PLUS) || match(TokenType::MINUS)) {
        TokenType op = previous().type;
        std::unique_ptr<Expr> right = parseFactor();
        expr = std::make_unique<BinaryExpr>(std::move(expr), op, std::move(right));
    }
    return expr;
}

std::unique_ptr<Expr> Parser::parseFactor() {
    std::unique_ptr<Expr> expr = parseCall();
    while (match(TokenType::STAR)) {
        TokenType op = previous().type;
        std::unique_ptr<Expr> right = parseCall();
        expr = std::make_unique<BinaryExpr>(std::move(expr), op, std::move(right));
    }
    return expr;
}

std::unique_ptr<Expr> Parser::parseCall() {
    std::unique_ptr<Expr> expr = parsePrimary();
    // For function calls like `my_func(a, b)`
    while (true) {
        if (check(TokenType::LPAREN)) {
            if (auto varExpr = dynamic_cast<VariableExpr*>(expr.get())) {
                std::string callee = varExpr->name;
                consume(TokenType::LPAREN, "Expect '(' after function name.");
                std::vector<std::unique_ptr<Expr>> args;
                if (!check(TokenType::RPAREN)) {
                    do {
                        args.push_back(parseExpression());
                    } while (match(TokenType::COMMA));
                }
                consume(TokenType::RPAREN, "Expect ')' after arguments.");
                expr = std::make_unique<CallExpr>(callee, std::move(args));
            } else {
                break;
            }
        } else if (match(TokenType::DOT)) {
            Token name = consume(TokenType::IDENTIFIER, "Expect property name after '.'.");
            expr = std::make_unique<PropertyGetExpr>(std::move(expr), name.text);
        } else if (match(TokenType::LBRACKET)) {
            std::unique_ptr<Expr> index = parseExpression();
            consume(TokenType::RBRACKET, "Expect ']' after index.");
            expr = std::make_unique<IndexExpr>(std::move(expr), std::move(index));
        } else {
            break;
        }
    }
    
    return expr;
}

std::unique_ptr<Expr> Parser::parsePrimary() {
    if (match(TokenType::NUMBER)) return std::make_unique<NumberExpr>(previous().numberValue);
    if (match(TokenType::VECTOR)) return parseVectorExpr();
    if (match(TokenType::MATRIX)) return parseMatrixExpr();
    if (match(TokenType::LOAD)) return parseLoadExpr();
    if (match(TokenType::LOAD_CSV)) return parseLoadCSVExpr();
    if (match(TokenType::IDENTIFIER)) return std::make_unique<VariableExpr>(previous().text);
    if (match(TokenType::LPAREN)) {
        std::unique_ptr<Expr> expr = parseExpression();
        consume(TokenType::RPAREN, "Expect ')' after expression.");
        return expr;
    }
    
    throw std::runtime_error("Unknown expression starting with: " + peek().text);
}

std::vector<double> Parser::parseNumberArray() {
    consume(TokenType::LBRACKET, "Expect '[' to start array.");
    std::vector<double> values;
    if (!check(TokenType::RBRACKET)) {
        do {
            Token num = consume(TokenType::NUMBER, "Expect number in array.");
            values.push_back(num.numberValue);
        } while (match(TokenType::COMMA));
    }
    consume(TokenType::RBRACKET, "Expect ']' to end array.");
    return values;
}

std::unique_ptr<VectorExpr> Parser::parseVectorExpr() {
    consume(TokenType::LPAREN, "Expect '(' after 'Vector'.");
    std::vector<double> values = parseNumberArray();
    consume(TokenType::RPAREN, "Expect ')' after Vector arguments.");
    return std::make_unique<VectorExpr>(values);
}

std::unique_ptr<MatrixExpr> Parser::parseMatrixExpr() {
    consume(TokenType::LPAREN, "Expect '(' after 'Matrix'.");
    Token rowsTok = consume(TokenType::NUMBER, "Expect rows number.");
    consume(TokenType::COMMA, "Expect ',' after rows.");
    Token colsTok = consume(TokenType::NUMBER, "Expect cols number.");
    consume(TokenType::COMMA, "Expect ',' after cols.");
    std::vector<double> values = parseNumberArray();
    consume(TokenType::RPAREN, "Expect ')' after Matrix arguments.");
    return std::make_unique<MatrixExpr>((size_t)rowsTok.numberValue, (size_t)colsTok.numberValue, values);
}

std::unique_ptr<LoadExpr> Parser::parseLoadExpr() {
    consume(TokenType::LPAREN, "Expect '(' after 'Load'.");
    Token filename = consume(TokenType::STRING, "Expect string filename after Load.");
    consume(TokenType::RPAREN, "Expect ')' after Load arguments.");
    return std::make_unique<LoadExpr>(filename.text);
}

std::unique_ptr<LoadCSVExpr> Parser::parseLoadCSVExpr() {
    consume(TokenType::LPAREN, "Expect '(' after 'LoadCSV'.");
    Token filename = consume(TokenType::STRING, "Expect string filename after LoadCSV.");
    consume(TokenType::RPAREN, "Expect ')' after LoadCSV arguments.");
    return std::make_unique<LoadCSVExpr>(filename.text);
}

} // namespace tnsr
