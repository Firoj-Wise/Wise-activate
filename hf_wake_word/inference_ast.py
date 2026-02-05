from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import torch
import sys
from pathlib import Path

# Path to the trained model
MODEL_PATH = Path(__file__).resolve().parent / "ast_model"

if not MODEL_PATH.exists():
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please run train_ast.py first to fine-tune the model on your data.")
    sys.exit(1)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading model from {MODEL_PATH} on {device}...")

try:
    classifier = pipeline(
        "audio-classification", 
        model=str(MODEL_PATH), 
        device=device
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

def launch_fn(
    wake_word="Deepa (EN)", # Default to one of our classes
    prob_threshold=0.5,
    chunk_length_s=2.0,
    stream_chunk_s=0.25,
    debug=False,
):
    print(f"Valid labels: {list(classifier.model.config.label2id.keys())}")
    
    if wake_word not in classifier.model.config.label2id.keys():
        raise ValueError(
            f"Wake word '{wake_word}' not in set of valid class labels.\nPick a wake word in the set: {list(classifier.model.config.label2id.keys())}."
        )

    sampling_rate = classifier.feature_extractor.sampling_rate

    try:
        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s,
        )
    except Exception as e:
        print(f"Error initializing microphone: {e}")
        print("Ensure 'ffmpeg' is installed and added to your system PATH.")
        return

    print(f"Listening for wake word: '{wake_word}'...")
    print("Press Ctrl+C to stop.")
    
    try:
        for prediction in classifier(mic):
            # classifier returns a list of dicts like [{'label': 'Background', 'score': 0.99}, ...]
            # But sometimes pipeline returns list of list if batched. Here it flows individually.
            # The user snippet had: prediction = prediction[0]
            # Let's inspect what 'prediction' is typically. 
            # Audio classification pipeline usually returns a list of top-k predictions or just list of all scores.
            # If top_k is designated. By default top_k=5 usually.
            
            # Use top prediction
            top_prediction = prediction[0] 
            
            if debug:
                print(top_prediction)
            
            if top_prediction["label"] == wake_word:
                if top_prediction["score"] > prob_threshold:
                    print(f"\nWake word detected! ({top_prediction['score']:.2f})")
                    # In a real app, you would trigger the next stage here
                    # For now we just print and continue listening? 
                    # The user's snippet returns True, which exits the loop.
                    # We can do that or just keep listening (demo mode)
                    print("Triggering activation...")
                    # return True # Uncomment to exit on detection
            else:
                # Optional: Print current top prediction if not wake word
                if debug:
                    print(f"Heard: {top_prediction['label']} ({top_prediction['score']:.2f})", end='\r')

    except KeyboardInterrupt:
        print("\nStopped listening.")

if __name__ == "__main__":
    # Example usage:
    # You can change the wake_word to any of your 6 classes:
    # "Deepa (EN)", "Deepa (NE)", "Deepa (MAI)"
    # "Deepak (EN)", "Deepak (NE)", "Deepak (MAI)"
    
    launch_fn(wake_word="Deepa (EN)", debug=True)
