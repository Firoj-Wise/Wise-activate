import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time
from collections import deque

# === CONFIGURATION ===
MODEL_PATH = "web/wakeword_sequence.tflite" # Path to your SEQUENTIAL model
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # 0.25s chunks for faster response
WINDOW_SIZE = 16000 # 1s window
THRESHOLD = 0.8
CLASSES = ["Background", "Hello", "Namaste", "Deepa", "Deepak"]

# State Machine Variables
state = "IDLE" # IDLE -> WAITING_FOR_NAME -> TRIGGERED
last_hello_time = 0
HELLO_TIMEOUT = 4.0 # Increased timeout
detected_greeting = None # "Hello" or "Namaste"
greeting_cooldown = 0
last_inference_time = 0

# Audio Buffer
audio_buffer = deque(maxlen=WINDOW_SIZE)

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_audio(audio_chunk):
    # Debug: Check signal health
    peak = np.max(np.abs(audio_chunk))
    
    # 1. Reject Silence (Tuned threshold)
    if peak < 0.002:
        # Return zeros if too quiet
        return np.zeros((101, 13), dtype=np.float32), peak

    # 2. Normalize (AGC)
    if peak > 0:
        audio_chunk = audio_chunk / peak
    
    mfcc = librosa.feature.mfcc(y=audio_chunk, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, hop_length=160).T
    if mfcc.shape[0] < 101:
        pad_width = 101 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:101, :]
    return mfcc.astype(np.float32), peak

def run_inference():
    global state, last_hello_time, detected_greeting, greeting_cooldown
    
    print(f"Loading sequence model: {MODEL_PATH}")
    try:
        interpreter = load_tflite_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you run 'python training/train_sequence.py' yet?")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    print("State Machine Ready: Say 'Hello Deepa' or 'Namaste Deepak'")
    
    def audio_callback(indata, frames, time, status):
        audio_buffer.extend(indata[:, 0])

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        while True:
            current_time = time.time()
            
            # Timeout logic
            if state == "WAITING_FOR_NAME" and (current_time - last_hello_time > HELLO_TIMEOUT):
                print("\nTIMEOUT: Name not received in time. Resetting to IDLE.")
                state = "IDLE"
                detected_greeting = None

            if len(audio_buffer) == WINDOW_SIZE:
                audio_data = np.array(audio_buffer)
                mfcc, peak = preprocess_audio(audio_data)
                
                # Silence Gate
                if peak < 0.002:
                     if state == "WAITING_FOR_NAME":
                         remaining = HELLO_TIMEOUT - (current_time - last_hello_time)
                         print(f"Waiting for name... ({remaining:.1f}s) | Peak: {peak:.5f}", end='\r')
                     else:
                         print(f"Listening (Silent)... Peak: {peak:.5f}   ", end='\r')
                     time.sleep(0.05)
                     continue

                input_data = mfcc.reshape(input_shape)
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                
                max_idx = np.argmax(output_data)
                confidence = output_data[max_idx]
                label = CLASSES[max_idx]
                
                if confidence > THRESHOLD:
                    if state == "IDLE":
                        if label in ["Hello", "Namaste"]:
                            if current_time - greeting_cooldown > 1.0: # Prevent double triggers
                                print(f"\nGreeting Detected: {label} ({confidence:.2f})")
                                state = "WAITING_FOR_NAME"
                                detected_greeting = label
                                last_hello_time = current_time
                                # Do NOT clear buffer here! Let it slide naturally to catch "Hello-Deepa"
                        else:
                             print(f"Heard: {label} ({confidence:.2f}) | Peak: {peak:.4f}    ", end='\r')
                            
                    elif state == "WAITING_FOR_NAME":
                        if label in ["Deepa", "Deepak"]:
                            # SUCCESS LOGIC
                            full_phrase = f"{detected_greeting} {label}"
                            lang = "English" if "Hello" in detected_greeting else "Nepali/Hindi"
                            gender = "Female" if "Deepa" in label else "Male"
                            
                            print(f"\n>>> ACTIVATED: {full_phrase} [{lang}, {gender}] <<<")
                            
                            state = "IDLE"
                            detected_greeting = None
                            greeting_cooldown = current_time 
                            audio_buffer.clear() # Clear BUFFER only on success to prevent re-trigger
                        elif label == "Background":
                             remaining = HELLO_TIMEOUT - (current_time - last_hello_time)
                             print(f"Listening... ({remaining:.1f}s) | Heard: Background", end='\r')
                        elif label in ["Hello", "Namaste"]:
                             # Avoid re-triggering greeting while waiting for name
                             pass
                else:
                    if state == "WAITING_FOR_NAME":
                        remaining = HELLO_TIMEOUT - (current_time - last_hello_time)
                        print(f"Listening... ({remaining:.1f}s) | Heard: {label} ({confidence:.2f})", end='\r')
                    else:
                        print(f"Listening... Peak: {peak:.4f} | Heard: {label} ({confidence:.2f})   ", end='\r')

            time.sleep(0.05)

if __name__ == "__main__":
    try:
        run_inference()
    except KeyboardInterrupt:
        print("\nStopped.")
