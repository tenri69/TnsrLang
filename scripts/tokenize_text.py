import os
import csv

text = """O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet."""

# Find all unique characters in the text (our "Vocabulary")
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mapping from character to integer and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }

print(f"Vocab size: {vocab_size}")

os.makedirs("data", exist_ok=True)

# Save vocabulary mapping for decoding later
with open("data/vocab.txt", "w") as f:
    for ch in chars:
        if ch == '\n':
            f.write("\\n\n")
        elif ch == ' ':
            f.write("<space>\n")
        else:
            f.write(ch + "\n")

# Encode text into integers
encoded_data = [stoi[c] for c in text]

# We will train the AI to predict the next character based on a sequence of past characters
seq_len = 8
X = []
Y = []

for i in range(len(encoded_data) - seq_len):
    X.append(encoded_data[i:i+seq_len])
    Y.append(encoded_data[i+1:i+seq_len+1]) # The target is just shifted by 1

# Save to TnsrLang compatible format (Header: rows cols \n Data...)
with open("data/shakespeare_X.csv", "w") as f:
    f.write(f"2 {len(X)} {seq_len}\n") # New format: dimensions, then shape
    for row in X:
        f.write(" ".join(map(str, row)) + "\n")

with open("data/shakespeare_Y.csv", "w") as f:
    f.write(f"2 {len(Y)} {seq_len}\n")
    for row in Y:
        f.write(" ".join(map(str, row)) + "\n")

print("Successfully created dataset for TnsrLang GPT!")
