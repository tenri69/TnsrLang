const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const offscreenCanvas = document.getElementById('offscreenCanvas');
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

const clearBtn = document.getElementById('clearBtn');
const guessBtn = document.getElementById('guessBtn');
const terminalOutput = document.getElementById('terminalOutput');
const predictionResult = document.getElementById('predictionResult');

// Canvas Drawing State
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize canvas with black background
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Drawing Configuration
ctx.strokeStyle = 'white';
ctx.lineWidth = 18; // Thick stroke for 28x28 downscaling
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    lastX = clientX - rect.left;
    lastY = clientY - rect.top;
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault(); // Prevent scrolling on touch devices
    
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    const currentX = clientX - rect.left;
    const currentY = clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();

    lastX = currentX;
    lastY = currentY;
}

function stopDrawing() {
    isDrawing = false;
}

// Event Listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictionResult.innerHTML = `
        <div class="placeholder">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
            <p>Rita till vänster och klicka "Gissa" för att köra en inferens genom TnsrLang-motorn.</p>
        </div>
    `;
    terminalOutput.textContent = "// Redo...";
});

guessBtn.addEventListener('click', async () => {
    // 1. Downscale image to 28x28
    offscreenCtx.fillStyle = 'black';
    offscreenCtx.fillRect(0, 0, 28, 28);
    offscreenCtx.drawImage(canvas, 0, 0, 28, 28);

    // 2. Extract pixel data (Grayscale)
    const imgData = offscreenCtx.getImageData(0, 0, 28, 28).data;
    const pixels = [];
    for (let i = 0; i < imgData.length; i += 4) {
        // imgData is RGBA. Since we draw white on black, R=G=B. We take R.
        pixels.push(imgData[i]);
    }

    // 3. Generate TnsrLang Script
    const script = `
// Inferens-skript autogenererat av webb-UI
Print("Laddar vikter...")
let W1 = Load("W1.tnsrdata")
let W2 = Load("W2.tnsrdata")

Print("Förbereder input tensor (1x784)...")
let input = Matrix(1, 784, [${pixels.join(',')}])
let scale = Vector([0.003921568])
let X_norm = input * scale

Print("Kör framåtpass...")
let hidden = ReLU(DotProduct(X_norm, W1))
let logits = DotProduct(hidden, W2)
let probs = Softmax(logits)

Print("=== PROBABILITIES ===")
Print(probs)
`;

    terminalOutput.textContent = "// Skickar inferens till C++ motorn...\n" + script.substring(0, 200) + "...\n\n";
    guessBtn.disabled = true;
    guessBtn.textContent = "Beräknar...";

    try {
        const response = await fetch('/run', {
            method: 'POST',
            body: script
        });
        const output = await response.text();
        terminalOutput.textContent += output;

        // Parse Output to find the probabilities
        const parts = output.split("=== PROBABILITIES ===");
        if (parts.length > 1) {
            const probStr = parts[1].trim();
            // Expected format:
            // [
            //   [ p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 ]
            // ]
            // Extract the numbers using regex
            const numMatches = probStr.match(/\d+\.\d+(?:e[-+]\d+)?/g);
            if (numMatches && numMatches.length === 10) {
                const probabilities = numMatches.map(Number);
                let maxIdx = 0;
                let maxVal = probabilities[0];
                for (let i = 1; i < 10; i++) {
                    if (probabilities[i] > maxVal) {
                        maxVal = probabilities[i];
                        maxIdx = i;
                    }
                }
                
                predictionResult.innerHTML = `
                    <div class="prediction-result">
                        <h3>Motorns gissning:</h3>
                        <p class="number">${maxIdx}</p>
                        <p class="confidence">${(maxVal * 100).toFixed(1)}% säkerhet</p>
                    </div>
                `;
            } else {
                predictionResult.innerHTML = `<p style="color:red">Kunde inte tolka nätverkets svar.</p>`;
            }
        }
    } catch (err) {
        terminalOutput.textContent += "\nError: " + err.message;
    } finally {
        guessBtn.disabled = false;
        guessBtn.textContent = "Gissa Siffran";
    }
});
