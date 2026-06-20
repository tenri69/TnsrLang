/*
 * =====================================================================================
 *
 *       Filename:  Lexer.h
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

#include <string>
#include <vector>

namespace tnsr {

/**
 * @brief Defines all possible token types in the TnsrLang syntax.
 */
enum class TokenType {
    LET,
    VECTOR,
    MATRIX,
    PRINT,
    FOR,
    TO,
    IF,
    ELSE,
    FN,
    RETURN,
    SAVE,
    LOAD,
    BACKWARD,
    LOAD_CSV,
    WHILE,
    IDENTIFIER,
    NUMBER,
    STRING, // For print statements like Print("Hello")
    EQUALS,
    LPAREN,
    RPAREN,
    LBRACKET,
    RBRACKET,
    LBRACE,
    RBRACE,
    PLUS,
    MINUS,
    STAR,
    LESS,
    GREATER,
    COMMA,
    DOT,
    END_OF_FILE,
    UNKNOWN
};

/**
 * @brief Represents a single lexical token parsed from the source code.
 */
struct Token {
    TokenType type;
    std::string text;
    double numberValue; // Populated only if type is TokenType::NUMBER
};

/**
 * @brief The Lexer takes a raw string of source code and converts it into a stream of Tokens.
 * This is the first stage of the TnsrLang compiler pipeline.
 */
class Lexer {
public:
    Lexer(const std::string& source);
    std::vector<Token> tokenize();

private:
    std::string source;
    size_t current;

    char peek() const;
    char advance();
    bool isAtEnd() const;
    void skipWhitespaceAndComments();
    
    Token number();
    Token identifierOrKeyword();
    Token stringLiteral();
};

} // namespace tnsr
