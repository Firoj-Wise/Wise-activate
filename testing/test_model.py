import tensorflow as tf
import numpy as np
import librosa
import argparse
import os

# Constants
SAMPLE_RATE = 16000
DURATION = 1
SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160

# Language Map (must match training)
LANGUAGES = ["Background", "English", "Maithili", "Nepali"]

def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        # Fix length
        if len(y) > SAMPLES:
            y = y[:SAMPLES]
        else:
            y = librosa.util.fix_length(y, size=SAMPLES)
            
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=40)
        return mfcc.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def run_inference(model_path, audio_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Process Audio
    if os.path.isdir(audio_path):
        files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith('.wav')]
    else:
        files = [audio_path]
        
    print(f"{'-'*20}")
    print(f"Testing Model: {model_path}")
    print(f"{'-'*20}")
    
    for file in files:
        mfcc = preprocess_audio(file)
        if mfcc is None: continue
        
        # Prepare input data
        input_data = np.array(mfcc, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0) # Add batch dim
        input_data = np.expand_dims(input_data, axis=-1) # Add channel dim
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0]
        
        label_index = np.argmax(prediction)
        
        # Safety check for index
        if label_index < len(LANGUAGES):
             label = LANGUAGES[label_index]
        else:
             label = f"Unknown Class {label_index}"

        confidence = prediction[label_index] * 100
        
        print(f"File: {os.path.basename(file)}")
        print(f"Prediction: {label} ({confidence:.2f}%)")
        print(f"Raw Output: {prediction}")
        print(f"{'-'*20}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TFLite Wake Word Model")
    parser.add_argument("--model", type=str, default="web/wakeword_model.tflite", help="Path to .tflite model")
    parser.add_argument("--audio", type=str, required=True, help="Path to wav file or directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}")
        exit(1)
        
    run_inference(args.model, args.audio)