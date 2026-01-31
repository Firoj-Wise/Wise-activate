import sounddevice as sd
import scipy.io.wavfile as wav
import os
import time
import numpy as np

# Configuration
SAMPLE_RATE = 16000
DURATION = 2.0  # 2 seconds per clip (Longer than wake word to capture phrases)
OUTPUT_DIR = "data/user_negatives"
NUM_SAMPLES = 40  # We need a good amount of data

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"---------------------------------------------------------")
print(f"   HARD NEGATIVE RECORDER 🛑")
print(f"---------------------------------------------------------")
print(f"The model thinks EVERYTHING you say is a wake word.")
print(f"We need to teach it what is NOT a wake word.")
print(f"---------------------------------------------------------")
print(f"INSTRUCTIONS:")
print(f"1. You will record {NUM_SAMPLES} samples.")
print(f"2. Speak RANDOM sentences, humming, coughing, or other languages.")
print(f"3. DO NOT SAY 'Deepak', 'Deepa', or 'Namaste'.")
print(f"4. Examples: 'The weather is nice', 'I am coding', *Humming*, *Coughing*.")
print(f"---------------------------------------------------------")

input("Press Enter to Start Recording...")

for i in range(NUM_SAMPLES):
    print(f"\n[{i+1}/{NUM_SAMPLES}] Recording RANDOM NOISE/SPEECH... (Speak now!)")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()
    
    filename = os.path.join(OUTPUT_DIR, f"user_neg_{int(time.time())}_{i}.wav")
    wav.write(filename, SAMPLE_RATE, audio)
    print(f" Saved: {filename}")
    time.sleep(0.5)

print(f"\n---------------------------------------------------------")
print(f" DONE! Saved {NUM_SAMPLES} negative samples.")
print(f"---------------------------------------------------------")
