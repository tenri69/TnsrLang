/*
 * =====================================================================================
 *
 *       Filename:  Lexer.cpp
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

#include "Lexer.h"
#include <cctype>
#include <stdexcept>
#include <iostream>

namespace tnsr {

// The Lexer (or Scanner) takes the raw string of TnsrLang source code
// and breaks it down into a stream of meaningful 'Tokens' (e.g. IDENTIFIER, NUMBER, PLUS).
// This is the first step of the compiler pipeline.

Lexer::Lexer(const std::string& source) : source(source), current(0) {}

char Lexer::peek() const {
    if (isAtEnd()) return '\0';
    return source[current];
}

char Lexer::advance() {
    return source[current++];
}

bool Lexer::isAtEnd() const {
    return current >= source.length();
}

void Lexer::skipWhitespaceAndComments() {
    while (!isAtEnd()) {
        char c = peek();
        if (std::isspace(c)) {
            advance();
        } else if (c == '/' && current + 1 < source.length() && source[current + 1] == '/') {
            // A comment goes until the end of the line.
            while (!isAtEnd() && peek() != '\n') advance();
        } else {
            break;
        }
    }
}

Token Lexer::number() {
    size_t start = current;
    while (!isAtEnd() && (std::isdigit(peek()) || peek() == '.')) {
        advance();
    }
    std::string text = source.substr(start, current - start);
    return Token{TokenType::NUMBER, text, std::stod(text)};
}

Token Lexer::identifierOrKeyword() {
    size_t start = current;
    while (!isAtEnd() && (std::isalnum(peek()) || peek() == '_')) {
        advance();
    }
    std::string text = source.substr(start, current - start);
    
    TokenType type = TokenType::IDENTIFIER;
    if (text == "let") type = TokenType::LET;
    else if (text == "Vector") type = TokenType::VECTOR;
    else if (text == "Matrix") type = TokenType::MATRIX;
    else if (text == "Print") type = TokenType::PRINT;
    else if (text == "for") type = TokenType::FOR;
    else if (text == "to") type = TokenType::TO;
    else if (text == "if") type = TokenType::IF;
    else if (text == "else") type = TokenType::ELSE;
    else if (text == "fn") type = TokenType::FN;
    else if (text == "return") type = TokenType::RETURN;
    else if (text == "Save") type = TokenType::SAVE;
    else if (text == "Load") type = TokenType::LOAD;
    else if (text == "backward") type = TokenType::BACKWARD;
    else if (text == "LoadCSV") type = TokenType::LOAD_CSV;
    else if (text == "while") type = TokenType::WHILE;
    
    return Token{type, text, 0.0};
}

Token Lexer::stringLiteral() {
    // Assume we've already consumed the opening quote
    size_t start = current;
    while (!isAtEnd() && peek() != '"') {
        advance();
    }
    
    if (isAtEnd()) {
        throw std::runtime_error("Unterminated string.");
    }
    
    std::string text = source.substr(start, current - start);
    advance(); // Consume the closing quote
    return Token{TokenType::STRING, text, 0.0};
}

// Main tokenization loop. Reads characters until the source is exhausted.
std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    
    while (!isAtEnd()) {
        skipWhitespaceAndComments();
        if (isAtEnd()) break;
        
        char c = peek();
        
        if (std::isalpha(c) || c == '_') {
            tokens.push_back(identifierOrKeyword());
        } else if (std::isdigit(c)) {
            tokens.push_back(number());
        } else if (c == '"') {
            advance(); // consume opening quote
            tokens.push_back(stringLiteral());
        } else {
            advance();
            switch (c) {
                case '=': tokens.push_back(Token{TokenType::EQUALS, "=", 0}); break;
                case '(': tokens.push_back(Token{TokenType::LPAREN, "(", 0}); break;
                case ')': tokens.push_back(Token{TokenType::RPAREN, ")", 0}); break;
                case '[': tokens.push_back(Token{TokenType::LBRACKET, "[", 0}); break;
                case ']': tokens.push_back(Token{TokenType::RBRACKET, "]", 0}); break;
                case '{': tokens.push_back(Token{TokenType::LBRACE, "{", 0}); break;
                case '}': tokens.push_back(Token{TokenType::RBRACE, "}", 0}); break;
                case '+': tokens.push_back(Token{TokenType::PLUS, "+", 0}); break;
                case '-': tokens.push_back(Token{TokenType::MINUS, "-", 0}); break;
                case '*': tokens.push_back(Token{TokenType::STAR, "*", 0}); break;
                case '<': tokens.push_back(Token{TokenType::LESS, "<", 0}); break;
                case '>': tokens.push_back(Token{TokenType::GREATER, ">", 0}); break;
                case ',': tokens.push_back(Token{TokenType::COMMA, ",", 0}); break;
                case '.': tokens.push_back(Token{TokenType::DOT, ".", 0}); break;
                default:
                    throw std::runtime_error("Unexpected character: " + std::string(1, c));
            }
        }
    }
    
    tokens.push_back(Token{TokenType::END_OF_FILE, "", 0});
    return tokens;
}

} // namespace tnsr
