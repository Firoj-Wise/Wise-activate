# 🚀 Training "Sundari" (Deepak/Deepa) on Google Colab

Since dataset generation and training are heavy for a local laptop, we will use Google Colab's free GPU/CPU power.

## **Step 1: Open Google Colab**
Go to: [colab.research.google.com](https://colab.research.google.com/) -> **New Notebook**.

---

## **Step 2: Setup & Clone (Copy-Paste this into Cell 1)**
We will clone your code. *If your repo is private, you'll need a token, but assuming public/local upload for now.* 
**The easiest way is to just drag-and-drop your `train.py` and `requirements.txt` if you don't want to sync git yet.**

**Option A (Drag and Drop Files):**
1. Click the **Folder Icon** 📁 on the left.
2. Drag and drop:
    - `dataset.zip` (The file I just made for you)
    - `wise-activate` folder (or just `training/train.py` and `requirements.txt`, but easiest to zip the whole code minus venv).

**Option B (Git Clone - Recommended for Code):**
```python
# 1. Clone the repo
!git clone https://github.com/YOUR_USERNAME/wise-activate.git
%cd wise-activate

# 2. Install Dependencies
!pip install -r requirements.txt
!pip install librosa tensorflow matplotlib
```

---

## **Step 3: Upload & Unzip Data (Copy-Paste into Cell 2)**
Since we generated the data locally (`dataset.zip`), we need to upload it.
1. Drag `dataset.zip` into the Colab Files sidebar.
2. Run this to unzip it exactly where `train.py` expects it.

```python
import os
import zipfile

# Unzip dataset.zip into 'data/'
# train.py expects: data/wake and data/background
with zipfile.ZipFile("../dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

print("✅ Data Unzipped! Checking structure...")
for root, dirs, files in os.walk("data"):
    print(f"{root}: {len(files)} files")
```

---

## **Step 4: Train the Model (Copy-Paste into Cell 3)**
This runs our improved `train.py` (with RIR, SpecAugment, and 3 Classes).

```python
# Make sure we are in the right directory
import sys
sys.path.append(".")

# Run Training
!python training/train.py

# note: At the end of training, you will now see a "Confusion Matrix".
# Check that the "Deepa" vs "Deepak" intersection is low!

```
*Wait ~10-15 minutes. You will see Epoch accuracy climbing.*

---

## **Step 5: Verify on Colab (Copy-Paste into Cell 4)**
Before downloading, let's test it on a random file from the test set to ensure it's not predicting "Background" for everything.

```python
import tensorflow as tf
import numpy as np
import librosa
import glob

# Load Model
interpreter = tf.lite.Interpreter(model_path="web/wakeword_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    # Extract MFCC (Simplified for test)
    if len(y) > 16000: y = y[:16000]
    else: y = np.pad(y, (0, 16000-len(y)))
    
    # Normalize
    peak = np.max(np.abs(y))
    y = y / (peak + 1e-6)

    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13, n_fft=512, hop_length=160, n_mels=40).T
    if mfcc.shape[0] < 101: mfcc = np.pad(mfcc, ((0, 101-mfcc.shape[0]), (0,0)))
    mfcc = mfcc[:101, :]
    
    input_data = np.expand_dims(mfcc, axis=0)[..., np.newaxis]
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    CLASSES = ["Background", "Deepa", "Deepak"]
    print(f"File: {file_path} -> Prediction: {CLASSES[np.argmax(output)]}  (Conf: {output})")

print("--- Testing DEEPA (Female) ---")
deepa_files = glob.glob("data/wake/deepa/**/*.wav")
if deepa_files: predict(deepa_files[0])

print("\n--- Testing DEEPAK (Male) ---")
deepak_files = glob.glob("data/wake/deepak/**/*.wav")
if deepak_files: predict(deepak_files[0])

print("\n--- Testing BACKGROUND ---")
bg_files = glob.glob("data/background/*.wav")
if bg_files: predict(bg_files[0])
```

---

## **Step 6: Download Model (Copy-Paste into Cell 5)**
If the tests pass (Correct Classifications), download the file!

```python
from google.colab import files
files.download('web/wakeword_model.tflite')
```

---

### **Next Steps Locally**
1. Place the downloaded `wakeword_model.tflite` into your local `web/` folder.
2. Run `python scripts/live_inference.py` to test with your microphone!
