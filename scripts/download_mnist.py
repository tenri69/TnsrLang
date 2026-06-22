import urllib.request
import gzip
import struct
import os

def download_and_convert():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]
    
    for f in files:
        if not os.path.exists(f):
            print(f"Laddar ner {f}...")
            urllib.request.urlretrieve(base_url + f, f)
            
    print("Konverterar till mnist_train.csv (4000 första bilderna)...")
    with gzip.open("train-labels-idx1-ubyte.gz", 'rb') as flbl:
        struct.unpack(">II", flbl.read(8))
        labels = list(flbl.read(4000))
        
    with gzip.open("train-images-idx3-ubyte.gz", 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        
        with open("mnist_train.csv", "w") as out:
            for i in range(4000):
                img_data = fimg.read(rows * cols)
                pixels = list(img_data)
                out.write(f"{labels[i]}," + ",".join(map(str, pixels)) + "\n")
                
    print("Färdigt! mnist_train.csv är skapad.")

if __name__ == "__main__":
    download_and_convert()
