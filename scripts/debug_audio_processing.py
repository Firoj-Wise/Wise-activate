import librosa
import numpy as np
import sys

SAMPLE_RATE = 16000
SAMPLES = 16000

def debug_proc(file_path):
    print(f"--- Analyzing {file_path} ---")
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    print(f"Original Duration: {len(y)/SAMPLE_RATE:.2f}s | Max Vol: {np.max(np.abs(y)):.4f}")
    
    # Apply Train.py Logic
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    print(f"Trimmed Duration:  {len(y_trimmed)/SAMPLE_RATE:.2f}s | Max Vol: {np.max(np.abs(y_trimmed)):.4f}")
    
    if len(y_trimmed) == 0:
        print("CRITICAL: Trimmed to empty!")
        return

    # Center
    if len(y_trimmed) > SAMPLES:
        start = (len(y_trimmed) - SAMPLES) // 2
        y_final = y_trimmed[start:start+SAMPLES]
    else:
        padding = SAMPLES - len(y_trimmed)
        offset = padding // 2
        y_final = np.pad(y_trimmed, (offset, padding - offset), 'constant')
        
    print(f"Final Duration:    {len(y_final)/SAMPLE_RATE:.2f}s | Max Vol: {np.max(np.abs(y_final)):.4f}")
    
    # Check Normalization need
    # Train.py does NOT apply normalization in `load_audio`, only in `augment_audio` implicitly?
    # Wait, `augment_audio` in `train.py` has noise injection but not explicit peak normalization 
    # unless `librosa.load` does it? Librosa loads as floating point -1 to 1.
    
    # Let's compare to TTS file
    return np.max(np.abs(y_final))

user_vol = debug_proc("data/user_samples/ne_user_1.wav")
print("\n")
tts_vol = debug_proc("data/wake/en/gcp_994_en-US-Wavenet-A_1.0_0.0.wav")

print("\n--- Comparison ---")
print(f"User Final Vol: {user_vol:.4f}")
print(f"TTS Final Vol:  {tts_vol:.4f}")
print(f"Ratio (TTS/User): {tts_vol/user_vol:.2f}x")
