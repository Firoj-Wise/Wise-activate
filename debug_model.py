import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path

# Config (Must match train.py)
SAMPLE_RATE = 16000
DURATION = 1
SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 13
N_FFT = 512 
HOP_LENGTH = 160
MODEL_PATH = "web/wakeword_model.tflite"

def normalize_audio(y, target_rms=0.1, threshold=0.005):
    current_rms = np.sqrt(np.mean(y**2))
    if current_rms > threshold:
        y = y * (target_rms / current_rms)
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))
    return y

def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y = librosa.util.fix_length(y, size=SAMPLES)
        y = normalize_audio(y)
        return y
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_mfcc(y):
    # CRITICAL: n_mels=40 to match Meyda.js default
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=40)
    return mfcc.T

def run_tflite_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    # input_data shape: (101, 13) -> need (1, 101, 13, 1)
    input_tensor = input_data[np.newaxis, ..., np.newaxis].astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0] # [Prob(0), Prob(1)]

def main():
    print(f"Loading model: {MODEL_PATH}")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Test Files
    wake_file = "data/wake/greeting_00000.wav"
    noise_file = "data/background/noise_white_0000.wav"
    speech_neg = "data/background/speech_no_0000.wav"

    files = [("WAKE WORD", wake_file), ("NOISE", noise_file), ("SPEECH NEG", speech_neg)]

    for label, path in files:
        if not Path(path).exists():
            print(f"File not found: {path}")
            continue
            
        audio = load_audio(path)
        mfcc = extract_mfcc(audio)
        prediction = run_tflite_inference(interpreter, mfcc)
        
        score = prediction[1] * 100
        print(f"[{label}] {Path(path).name} -> Confidence: {score:.2f}% (Raw: {prediction})")

if __name__ == "__main__":
    main()
