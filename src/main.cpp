/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
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
#include "Parser.h"
#include "Evaluator.h"
#include "ASTPrinter.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "httplib.h"

using namespace tnsr;

std::string executeTnsrCode(const std::string& source) {
    std::stringstream outputBuffer;
    try {
        Lexer lexer(source);
        std::vector<Token> tokens = lexer.tokenize();

        Parser parser(tokens);
        std::vector<std::unique_ptr<Stmt>> ast = parser.parse();

        Evaluator evaluator(outputBuffer);
        evaluator.evaluate(ast);
    } catch (const std::exception& e) {
        return "Error: " + std::string(e.what());
    }
    return outputBuffer.str();
}



int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.tnsr> [--dump-ast] [--debug-nn] | --server\n";
        return 1;
    }

    std::string filename = "";
    bool dumpAst = false;
    bool debugNN = false;
    bool isServer = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--server") {
            isServer = true;
        } else if (arg == "--dump-ast") {
            dumpAst = true;
        } else if (arg == "--debug-nn") {
            debugNN = true;
        } else if (filename.empty() && arg[0] != '-') {
            filename = arg;
        }
    }

    if (isServer) {
        httplib::Server svr;

        auto ret = svr.set_mount_point("/", "./public");
        if (!ret) {
            std::cerr << "The directory './public' doesn't exist.\n";
        }

        svr.Post("/run", [](const httplib::Request& req, httplib::Response& res) {
            std::string output = executeTnsrCode(req.body);
            res.set_content(output, "text/plain");
        });

        std::cout << "Starting TnsrLang Playground at http://localhost:8080\n";
        svr.listen("0.0.0.0", 8080);
        return 0;
    }

    if (filename.empty()) {
        std::cerr << "Error: No .tnsr source file specified.\n";
        return 1;
    }
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    try {
        // 1. Lexical Analysis
        Lexer lexer(source);
        std::vector<Token> tokens = lexer.tokenize();

        // 2. Parsing
        Parser parser(tokens);
        std::vector<std::unique_ptr<Stmt>> ast = parser.parse();

        // 3. Optional: AST Visualization
        if (dumpAst) {
            ASTPrinter printer("ast.dot");
            printer.print(ast);
            std::cout << "AST dumped to ast.dot\n";
            std::cout << "You can visualize it using: dot -Tpng ast.dot -o ast.png\n";
        }

        // 4. Evaluation
        Evaluator evaluator(std::cout);
        if (debugNN) {
            evaluator.setDebugNN(true);
            std::cout << "[TnsrLang NN Debugger] Active: Monitoring gradients and numerical health.\n";
        }
        evaluator.evaluate(ast);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
