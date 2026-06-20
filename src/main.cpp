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

const std::string HTML_CONTENT = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TNSRLANG TERMINAL</title>
    <style>
        :root {
            --bg: #0c0c0c;
            --fg: #cccccc;
            --dim: #777777;
            --accent: #ffffff;
        }
        body { 
            font-family: 'Consolas', 'Courier New', monospace; 
            background: var(--bg);
            color: var(--fg); 
            display: flex; 
            flex-direction: column; 
            height: 100vh; 
            margin: 0; 
            padding: 10px;
            box-sizing: border-box;
            overflow: hidden;
        }
        header { 
            white-space: pre;
            font-size: 14px;
            color: var(--accent);
            margin-bottom: 20px;
        }
        .container { 
            display: flex; 
            flex: 1; 
            gap: 15px; 
            overflow: hidden; 
        }
        .pane { 
            border: 1px solid var(--dim);
            display: flex; 
            flex-direction: column; 
            overflow: hidden;
        }
        .editor-pane { flex: 1; }
        .right-pane { flex: 1; display: flex; flex-direction: column; gap: 15px; border: none; }
        .vis-pane { flex: 1; border: 1px solid var(--dim); display: flex; flex-direction: column; overflow: hidden; }
        .output-pane { flex: 1; border: 1px solid var(--dim); display: flex; flex-direction: column; overflow: hidden; }
        textarea { 
            flex: 1; 
            background: var(--bg); 
            color: var(--accent); 
            border: none; 
            padding: 10px; 
            font-family: 'Consolas', 'Courier New', monospace; 
            font-size: 14px; 
            line-height: 1.2;
            resize: none; 
            outline: none; 
        }
        .output-content { 
            flex: 1;
            padding: 10px; 
            overflow-y: auto; 
            white-space: pre-wrap; 
            font-size: 14px; 
        }
        .pane-title { 
            border-bottom: 1px solid var(--dim);
            padding: 5px 10px; 
            font-size: 14px; 
            color: var(--dim);
            background: var(--bg);
        }
        .vis-content {
            flex: 1;
            padding: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: auto;
        }
        #ascii-vis {
            font-size: 14px;
            margin: 0;
            color: var(--accent);
        }
        .toolbar {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-bottom: 10px;
        }
        button { 
            background: var(--fg); 
            color: var(--bg); 
            border: none; 
            padding: 5px 15px; 
            cursor: pointer; 
            font-family: 'Consolas', 'Courier New', monospace; 
            font-size: 14px; 
            font-weight: bold;
        }
        button:hover { 
            background: var(--accent); 
        }
        button:active {
            background: var(--dim);
        }
        button:disabled {
            background: var(--dim);
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <header>
=============================================================================
 _______ _   _  _____ _____  _               _   _  _____ 
|__   __| \ | |/ ____|  __ \| |        /\   | \ | |/ ____|
   | |  |  \| | (___ | |__) | |       /  \  |  \| | |  __ 
   | |  | . ` |\___ \|  _  /| |      / /\ \ | . ` | | |_ |
   | |  | |\  |____) | | \ \| |____ / ____ \| |\  | |__| |
   |_|  |_| \_|_____/|_|  \_\______/_/    \_\_| \_|\_____|

 TnsrLang Interactive Terminal v1.0
=============================================================================
    </header>
    
    <div class="toolbar">
        <button id="runBtn" onclick="runCode()">[ RUN_TRAINING.EXE ]</button>
    </div>

    <div class="container">
        <div class="pane editor-pane">
            <div class="pane-title">C:\> type script.tnsr</div>
            <textarea id="code" spellcheck="false">// Advanced Neural Network in TnsrLang v2.0
// Now with Functions, Conditionals, and Sigmoid!

let input = Vector([1.0, 2.0])
let target = Vector([5.0])
let weights = Matrix(1, 2, [0.1, 0.2])
let lr = 0.5 // Higher learning rate for Sigmoid

fn predict(w, x) {
    let raw = DotProduct(w, x)
    return Sigmoid(raw)
}

Print("Initial weights:")
Print(weights)

for i = 1 to 20 {
    let pred = predict(weights, input)
    let error = pred - target
    
    if error < 0.001 {
        // We converged!
    } else {
        let grad = DotProduct(error, Transpose(input))
        let weights = weights - grad * lr
    }
}

Print("Final prediction:")
Print(predict(weights, input))
</textarea>
        </div>
        <div class="right-pane">
            <div class="vis-pane">
                <div class="pane-title">-- LIVE VISUALIZATION --</div>
                <div class="vis-content">
<pre id="ascii-vis">
  [ Input 1 ]           [ Input 2 ]
  (   1.0   )           (   2.0   )
        \                   /
         \                 /
 w: 0.10  \               /  w: 0.20
           \             /
            \           /
             v         v
            [ Prediction ]
            (    0.50    )
</pre>
                </div>
            </div>
            <div class="output-pane">
                <div class="pane-title">STDOUT ></div>
                <div id="output" class="output-content">Waiting for execution...</div>
            </div>
        </div>
    </div>
    <script>
        function updateAsciiVis(w1, w2, pred, isTraining) {
            let predStr = isTraining ? "..." : Number(pred).toFixed(3);
            let w1Str = isTraining ? "..." : Number(w1).toFixed(2);
            let w2Str = isTraining ? "..." : Number(w2).toFixed(2);
            
            // Pad strings to keep alignment
            predStr = predStr.padStart(8, ' ').padEnd(10, ' ');
            w1Str = w1Str.padEnd(5, ' ');
            w2Str = w2Str.padEnd(5, ' ');
            
            let animState = Math.floor(Date.now() / 200) % 3;
            let flowL = isTraining ? (['.', 'o', 'O'][animState]) : '\\';
            let flowR = isTraining ? (['.', 'o', 'O'][(animState+1)%3]) : '/';
            
            const ascii = `
  [ Input 1 ]           [ Input 2 ]
  (   1.0   )           (   2.0   )
        ${flowL}                   ${flowR}
         ${flowL}                 ${flowR}
 w: ${w1Str} ${flowL}               ${flowR}  w: ${w2Str}
           ${flowL}             ${flowR}
            ${flowL}           ${flowR}
             v         v
            [ Prediction ]
            (${predStr})
`;
            const vis = document.getElementById('ascii-vis');
            if (vis) vis.textContent = ascii;
        }

        let animInterval = null;

        async function runCode() {
            try {
                const code = document.getElementById('code').value;
                const outputDiv = document.getElementById('output');
                const runBtn = document.getElementById('runBtn');
                
                if (outputDiv) outputDiv.textContent = 'EXECUTING...\\n';
                if (runBtn) {
                    runBtn.disabled = true;
                    runBtn.textContent = '[ RUNNING... ]';
                }
                
                if (animInterval) clearInterval(animInterval);
                animInterval = setInterval(() => {
                    updateAsciiVis("0.10", "0.20", "0.50", true);
                }, 200);
                
                await new Promise(r => setTimeout(r, 1200));
                
                const response = await fetch('/run', {
                    method: 'POST',
                    body: code
                });
                const result = await response.text();
                if (outputDiv) outputDiv.textContent = result;
                
                if (animInterval) clearInterval(animInterval);
                animInterval = null;
                
                let pred = "0.50";
                let w1 = "0.10";
                let w2 = "0.20";
                
                // Parse output to update visualization
                const predMatch = result.match(/Final prediction:[\s\S]*?\[\s*\[\s*([0-9.-]+)\s*\]\s*\]/);
                if (predMatch && predMatch[1]) {
                    pred = predMatch[1];
                }
                
                const wMatch = result.match(/Initial weights:[\s\S]*?\[\s*\[\s*([0-9.-]+),\s*([0-9.-]+)\s*\]\s*\]/);
                if (wMatch && wMatch[1] && wMatch[2]) {
                    w1 = wMatch[1];
                    w2 = wMatch[2];
                }
                
                updateAsciiVis(w1, w2, pred, false);
                
            } catch (err) {
                if (animInterval) clearInterval(animInterval);
                const outputDiv = document.getElementById('output');
                if (outputDiv) outputDiv.textContent = 'SYSTEM ERROR: ' + err;
            } finally {
                const runBtn = document.getElementById('runBtn');
                if (runBtn) {
                    runBtn.disabled = false;
                    runBtn.textContent = '[ RUN_TRAINING.EXE ]';
                }
            }
        }
        
        // Initial draw
        updateAsciiVis("0.10", "0.20", "0.50", false);
    </script>
</body>
</html>)HTML";

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.tnsr> | --server\n";
        return 1;
    }

    std::string arg1 = argv[1];

    if (arg1 == "--server") {
        httplib::Server svr;

        svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(HTML_CONTENT, "text/html");
        });

        svr.Post("/run", [](const httplib::Request& req, httplib::Response& res) {
            std::string output = executeTnsrCode(req.body);
            res.set_content(output, "text/plain");
        });

        std::cout << "Starting TnsrLang Playground at http://localhost:8080\n";
        svr.listen("0.0.0.0", 8080);
        return 0;
    }

    std::string filename = arg1;
    bool dumpAst = false;

    if (argc >= 3) {
        std::string flag = argv[2];
        if (flag == "--dump-ast") {
            dumpAst = true;
        }
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
        evaluator.evaluate(ast);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
