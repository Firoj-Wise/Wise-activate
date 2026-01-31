import argparse
import numpy as np
import tensorflow as tf
import librosa
import queue
import sys
import time

# --- Configuration ---
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # 0.25s chunks
WINDOW_SIZE = SAMPLE_RATE # 1 second window

# --- Language Map ---
LANGUAGES = ["Background", "Deepa", "Deepak"]

def main():
    try:
        import sounddevice as sd
    except ImportError:
        print("Please install sounddevice: pip install sounddevice numpy")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Real-Time Wake Word Tester (Verbose)")
    parser.add_argument("--model", type=str, default="web/wakeword_model.tflite", help="Path to TFLite model")
    args = parser.parse_args()

    # Load Model
    print(f"Loading Model: {args.model}...")
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    q = queue.Queue()
    
    def callback(indata, frames, time, status):
        q.put(indata.copy())

    print("\n🎤 LISTENING CONTINUOUSLY... (Press Ctrl+C to stop)")
    print(f"Classes: {LANGUAGES}")
    print("--------------------------------------------------------------------------------")
    
    audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, callback=callback):
            while True:
                while not q.empty():
                    data = q.get().flatten().astype(np.float32)
                    audio_buffer = np.roll(audio_buffer, -len(data))
                    audio_buffer[-len(data):] = data
                    
                    # 1. Peak Measure
                    peak = np.max(np.abs(audio_buffer))
                    
                    # 2. Adaptive Normalization
                    # ISSUE FIXED: Room noise (0.004) was being boosted to 1.0.
                    # New threshold: 0.02 (Speech is usually > 0.1, Noise is < 0.01)
                    
                    boost_tag = ""
                    if peak < 0.02: 
                         # Absolute silence
                         norm_buffer = audio_buffer 
                    else:
                         # Normalize to -1..1 range (Simulates loud TTS training data)
                         norm_buffer = audio_buffer / (peak + 1e-6)
                         boost_tag = "(BOOSTED)"

                    # 3. MFCC
                    mfcc = librosa.feature.mfcc(y=norm_buffer, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, hop_length=160, n_mels=40)
                    mfcc = mfcc.T 
                    
                    if mfcc.shape != (101, 13):
                        if mfcc.shape[0] > 101: mfcc = mfcc[:101, :]
                        else: continue # Skip

                    # 4. Inference
                    input_data = np.expand_dims(mfcc, axis=0)
                    input_data = np.expand_dims(input_data, axis=-1)
                    
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    # 5. Smart Sensitivity Logic
                    # Calculate Probability of "NOT BACKGROUND"
                    background_score = output_data[0]
                    wake_score = 1.0 - background_score
                    
                    status_line = f"Vol: {peak:.3f} {boost_tag} | [WakeProb: {wake_score:.2f}] | "
                    
                    for i, lang in enumerate(LANGUAGES):
                        status_line += f"{lang[:2]}: {output_data[i]:.2f} "
                    
                    # Trigger if > 85% chance AND Volume was high enough to be speech
                    # (Double check to prevent normalized silence from triggering)
                    if wake_score > 0.85 and peak > 0.02: 
                        # Identify dominant language
                        wake_probs = output_data[1:] # Exclude BG
                        dom_idx = np.argmax(wake_probs) + 1 # Shift back
                        detected_lang = LANGUAGES[dom_idx]
                        print(f"\n🔔 >>> DETECTED: {detected_lang.upper()} (Vol: {peak:.3f} | Conf: {wake_score:.2f}) <<<\n")
                    else:
                         sys.stdout.write(f"\r{status_line}")
                         sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped.")

if __name__ == "__main__":
    main()
