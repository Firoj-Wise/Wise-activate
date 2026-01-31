import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import os
from pathlib import Path

# CONFIG
SAMPLE_RATE = 16000
DURATION = 2.0 
OUTPUT_DIR = Path("data/user_samples")

def record_sample(filename):
    print(f"\n🎙️  Recording (2s)... Say the wake word!")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    
    # Save
    filepath = OUTPUT_DIR / filename
    wav.write(filepath, SAMPLE_RATE, recording)
    print(f"✅ Saved: {filepath}")

def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
        
    print("="*50)
    print("USER VOICE ADAPTATION TOOL")
    print("="*50)
    print("The model needs to learn YOUR voice and mic characteristics.")
    print("We will record 20 samples per language (Total 60).")
    print("This allows for variations in tone, speed, and distance from mic.")
    print("--------------------------------------------------")
    
    # English
    print("\n--- SET 1: ENGLISH STYLE (20 samples) ---")
    for i in range(1, 21):
        filename = f"en_user_{i}.wav"
        input(f"[{i}/20] Press ENTER to record 'Hello Deepa'...")
        record_sample(filename)
        time.sleep(0.2)

    # Nepali
    print("\n--- SET 2: NEPALI STYLE (20 samples) ---")
    for i in range(1, 21):
        filename = f"ne_user_{i}.wav"
        input(f"[{i}/20] Press ENTER to record 'Namaste Deepak/Deepa'...")
        record_sample(filename)
        time.sleep(0.2)
        
    # Maithili
    print("\n--- SET 3: MAITHILI/MIXED STYLE (20 samples) ---")
    for i in range(1, 21):
        filename = f"mai_user_{i}.wav"
        input(f"[{i}/20] Press ENTER to record 'Deepak Sunnu' or 'Wake Up'...")
        record_sample(filename)
        time.sleep(0.2)

    print("\n✅ Done! Now we will retrain the model with your voice mixed in.")

if __name__ == "__main__":
    main()
