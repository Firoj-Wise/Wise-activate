import os
import requests
from pathlib import Path

# Config
VERSION = "0.0.1-alpha.9"
BASE_URL = f"https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@{VERSION}/dist/"
DEST_DIR = Path("web/tflite")
FILES = [
    "tf-tflite.min.js",
    "tflite_web_api_cc.js",
    "tflite_web_api_cc.wasm",
    "tflite_web_api_cc_simd.js",
    "tflite_web_api_cc_simd.wasm"
]

def download_file(filename):
    url = BASE_URL + filename
    dest = DEST_DIR / filename
    print(f"Downloading {filename}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"Saved to {dest}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def main():
    if not DEST_DIR.exists():
        DEST_DIR.mkdir(parents=True)
    
    print(f"Downloading TFLite {VERSION} dependencies to {DEST_DIR}...")
    for f in FILES:
        download_file(f)
    print("Done.")

if __name__ == "__main__":
    main()
