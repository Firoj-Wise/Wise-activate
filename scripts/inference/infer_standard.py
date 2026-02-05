import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time
from collections import deque

# === CONFIGURATION ===
MODEL_PATH = "web/wakeword_model.tflite" # Path to your standard model
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000  # 0.5s chunks
WINDOW_SIZE = 16000 # 1s window for inference
THRESHOLD = 0.85
CLASSES = [
    "Background", 
    "Deepa (EN)", "Deepa (NE)", "Deepa (MAI)",
    "Deepak (EN)", "Deepak (NE)", "Deepak (MAI)"
]

# Audio Buffer
audio_buffer = deque(maxlen=WINDOW_SIZE)

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_audio(audio_chunk):
    # Debug: Check signal health
    peak = np.max(np.abs(audio_chunk))
    
    # 1. Reject Silence
    if peak < 0.01:
        # Return zeros if too quiet
        return np.zeros((101, 13), dtype=np.float32), peak

    # 2. Normalize (Critical: Model expects -1 to 1 normalized audio)
    # sounddevice returns float32 by default, usually within -1.0 to 1.0
    # But often it's very quiet. Let's normalize to peak amplitude of 1.0 (AGC)
    # This matches training (librosa loads normalize)
    if peak > 0:
        audio_chunk = audio_chunk / peak # Scale to max volume
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio_chunk, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, hop_length=160).T
    
    # Ensure shape (101, 13) - pad or trim
    if mfcc.shape[0] < 101:
        pad_width = 101 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:101, :]
        
    return mfcc.astype(np.float32), peak

def run_inference():
    print(f"Loading model: {MODEL_PATH}")
    interpreter = load_tflite_model(MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape'] # Likely [1, 101, 13, 1]
    
    print("Listening... (Press Ctrl+C to stop)")
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_buffer.extend(indata[:, 0])

    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
            while True:
                if len(audio_buffer) == WINDOW_SIZE:
                    audio_data = np.array(audio_buffer)
                    
                    # Preprocess
                    mfcc, peak = preprocess_audio(audio_data)
                    
                    # If silent, skip inference to save CPU and reduce noise
                    if peak < 0.01:
                         print(f"Silence (Peak: {peak:.4f}) {' ' * 30}", end='\r')
                         time.sleep(0.1)
                         continue

                    # Reshape for model [1, 101, 13, 1]
                    input_data = mfcc.reshape(input_shape)
                    
                    # Inference
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    
                    # Check if classes match
                    current_classes = CLASSES
                    if len(output_data) != len(CLASSES):
                        if len(output_data) == 2:
                            current_classes = ["Background", "Wake Word"]
                        elif len(output_data) == 3:
                            current_classes = ["Background", "Deepa", "Deepak"]
                        else:
                            current_classes = [f"Class {i}" for i in range(len(output_data))]

                    # Predictions
                    max_idx = np.argmax(output_data)
                    confidence = output_data[max_idx]
                    
                    if max_idx > 0 and confidence > THRESHOLD:
                        print(f"\n>>> WAKE WORD DETECTED: {current_classes[max_idx]} ({confidence:.2f}) <<<")
                        # Optional: debounce / cooldown logic here
                        time.sleep(1.0) 
                        audio_buffer.clear() # Clear buffer to avoid double triggering
                    else:
                        # Show top 2 probabilities for debugging
                        top_2_idx = np.argsort(output_data)[-2:][::-1]
                        debug_str = " | ".join([f"{current_classes[i]}: {output_data[i]:.2f}" for i in top_2_idx])
                        print(f"Heard: {debug_str} | Peak: {peak:.2f} {' ' * 10}", end='\r')
                
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    run_inference()
