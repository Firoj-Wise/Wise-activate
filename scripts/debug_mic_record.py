import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
import tensorflow as tf
import os

# CONFIG
SAMPLE_RATE = 16000
DURATION = 2.0 # Record 2 seconds
OUTPUT_FILE = "debug_mic_test.wav"
MODEL_PATH = "web/wakeword_model.tflite"
LANGUAGES = ["Background", "English", "Maithili", "Nepali"]

def record_audio():
    print(f"\n🎙️  Recording for {DURATION} seconds... SPEAK NOW!")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("✅ Recording finished.")
    
    # Save raw for inspection
    wav.write(OUTPUT_FILE, SAMPLE_RATE, recording)
    print(f"Saved to {OUTPUT_FILE}")
    return recording.flatten()

def run_inference(audio_data):
    # Load Model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocessing (Match Training Logic exactly)
    # 1. PEAK NORMALIZE
    peak = np.max(np.abs(audio_data))
    if peak > 0:
        audio_data = audio_data / peak
        print(f"Normalized Audio (Peak was {peak:.4f})")
    else:
        print("Audio is silent!")
        return

    # 2. Slice 1 second (16000 samples)
    # We take the middle 1 second to capture the speech
    center = len(audio_data) // 2
    start = max(0, center - 8000)
    end = start + 16000
    
    # Pad if too short
    chunk = audio_data[start:end]
    if len(chunk) < 16000:
        chunk = np.pad(chunk, (0, 16000 - len(chunk)))

    # 3. MFCC
    mfcc = librosa.feature.mfcc(y=chunk, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, hop_length=160, n_mels=40)
    mfcc = mfcc.T # (101, 13)
    
    if mfcc.shape[0] > 101: mfcc = mfcc[:101, :]
    
    # 4. Infer
    input_data = np.expand_dims(mfcc, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    print("\n--- INFERENCE RESULTS ---")
    print(f"Background: {output_data[0]:.4f}")
    print(f"English:    {output_data[1]:.4f}")
    print(f"Maithili:   {output_data[2]:.4f}")
    print(f"Nepali:     {output_data[3]:.4f}")
    
    pred_idx = np.argmax(output_data)
    confidence = output_data[pred_idx]
    
    wake_score = 1.0 - output_data[0]
    
    print("-" * 30)
    print(f"PREDICTION: {LANGUAGES[pred_idx].upper()} ({confidence:.2f})")
    print(f"WAKE PROB:  {wake_score:.2f}")
    
    if wake_score > 0.6:
        print("✅ WAKE WORD DETECTED")
    else:
        print("❌ NO WAKE WORD")

if __name__ == "__main__":
    confirm = input("Press ENTER to start recording 'Hello Deepa'...")
    audio = record_audio()
    run_inference(audio)
