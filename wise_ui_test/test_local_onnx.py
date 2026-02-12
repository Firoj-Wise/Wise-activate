import os
import sys
import numpy as np
import sounddevice as sd
import onnxruntime as ort

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_FILES = ["namaste_deepa.onnx", "hello_deepa.onnx"]   # Names of your trained models
THRESHOLD = 0.5                    # Detection threshold
CHUNK_SIZE = 1280                  # 80ms chunks (standard for openWakeWord)
SAMPLE_RATE = 16000                # Must be 16kHz

# ==========================================
# OPENWAKEWORD IMPORT CHECK
# ==========================================
try:
    from openwakeword.model import Model
    print("✅ openwakeword library found.")
except ImportError:
    print("❌ 'openwakeword' library not found!")
    print("   Please install it by running in terminal:")
    print("   pip install openwakeword")
    sys.exit(1)

# ==========================================
# MAIN CLASS
# ==========================================
class WakeWordTester:
    def __init__(self, model_paths):
        # validation
        for path in model_paths:
            if not os.path.exists(path):
                print(f"❌ Model not found at: {path}")
                sys.exit(1)
            print(f"Loading model: {path}...")
        
        # Load the model using openWakeWord
        try:
            # Check if using older version of openwakeword (wakeword_model_paths vs wakeword_models)
            import inspect
            sig = inspect.signature(Model.__init__)
            if "wakeword_model_paths" in sig.parameters:
                 print("ℹ️ Detected older openWakeWord API. Using 'wakeword_model_paths'")
                 self.oww_model = Model(
                    wakeword_model_paths=model_paths
                )
            else:
                 self.oww_model = Model(
                    wakeword_models=model_paths, 
                    inference_framework="onnx"
                )
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)

        self.buffer = np.zeros(CHUNK_SIZE, dtype=np.int16)
        print("✅ Models loaded successfully!")

    def start(self):
        print("\n🎤 LISTENING... (Press Ctrl+C to stop)")
        print(f"   Threshold: {THRESHOLD}")
        print("-" * 40)

        # Open microphone stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=CHUNK_SIZE,
            callback=self._audio_callback
        ):
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                print("\nStopped.")

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)

        # Convert int16 audio to float32 (norm -1 to 1) 
        # because openWakeWord expects numpy array of shape (N,)
        audio_data = indata.flatten()

        # Run inference
        # openWakeWord handles the buffering and feature extraction
        predictions = self.oww_model.predict(audio_data)

        # Get score for our model
        # predictions is a dict like {'namaste_deepa': 0.002, ...}
        for name, score in predictions.items():
            if score > THRESHOLD:
                is_hello = "hello_deepa" in name
                is_namaste = "namaste" in name
                
                # Determine display text and language
                if is_hello:
                    display_text = "Hello Deepa Activated.."
                    lang_text = "Language: English"
                elif is_namaste:
                    display_text = "Namaste Deepa Activated.."
                    lang_text = "Language: Nepali"
                else:
                    display_text = f"{name.upper()} Activated.."
                    lang_text = ""

                # Print consistent UI
                print(f"🔔 {display_text} (Score: {score:.4f}) {lang_text}")
            else:
                # Optional: Print raw scores to see if it's working
                if score > 0.01: # Only print if there's some activity
                    print(f"   {name}: {score:.4f}")

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_paths = []

    for model_name in MODEL_FILES:
        model_path = os.path.join(current_dir, model_name)
        # Check if model exists, if not maybe it's in web/
        if not os.path.exists(model_path):
            alt_path = os.path.join(current_dir, "..", "web", model_name)
            if os.path.exists(alt_path):
                model_path = alt_path
        resolved_paths.append(model_path)

    tester = WakeWordTester(resolved_paths)
    tester.start()
