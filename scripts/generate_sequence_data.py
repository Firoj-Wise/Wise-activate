import os
import random
import time
from pathlib import Path
from tqdm import tqdm
from gcp_tts import get_gcp_client, generate_gcp_audio

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_SEQ_DIR = BASE_DIR / "data_sequence"

TARGET_COUNT = 200 # number of samples per word per voice type

# Voice configurations
# English Voices
VOICES_EN = [
    "en-IN-Wavenet-A", "en-IN-Wavenet-B", "en-IN-Wavenet-C", "en-IN-Wavenet-D",
    "en-US-Studio-M", "en-GB-Neural2-B", "en-US-Neural2-A"
]
# Nepali-ish voices (Hindi/Marathi used as proxy for diverse accents)
# Removed 'ne-NP-Standard-A' as it is not available in standard tier or misspelled
# Removed 'en-US-Journey-F' as it does not support pitch
VOICES_NE = [
    "hi-IN-Neural2-A", "hi-IN-Neural2-B", "hi-IN-Neural2-C", "hi-IN-Neural2-D",
    "bn-IN-Wavenet-A", "bn-IN-Wavenet-B" 
]

WORDS = {
    "hello": ["Hello"],
    "namaste": ["Namaste", "Namaskar"], 
    "deepa": ["Deepa"],
    "deepak": ["Deepak"]
}

def generate_samples():
    client = get_gcp_client()
    if not client:
        print("GCP Key not configured. Cannot generate synthetic data.")
        return

    print("Generating Sequence Component Data...")

    for category, phrases in WORDS.items():
        output_dir = DATA_SEQ_DIR / category
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Processing '{category}'...")
        
        # Check existing
        existing_files = list(output_dir.glob("syn_*.mp3")) + list(output_dir.glob("syn_*.wav"))
        count = len(existing_files)
        
        if count >= TARGET_COUNT:
            print(f"  Skipping '{category}': Already has {count} files.")
            continue
            
        print(f"  Resuming '{category}' from {count}/{TARGET_COUNT}...")
        
        # Use a mix of voices
        all_voices = VOICES_EN + VOICES_NE
        
        with tqdm(total=TARGET_COUNT, initial=count) as pbar:
            while count < TARGET_COUNT:
                voice = random.choice(all_voices)
                phrase = random.choice(phrases)
                
                # Variabilities
                pitch = random.uniform(-4.0, 4.0)
                rate = random.uniform(0.85, 1.15)
                
                filename = f"syn_{category}_{count}_{voice}_{int(time.time())}.mp3"
                path = output_dir / filename
                
                # Simple retry logic
                try:
                    # Journey and some Neural voices don't support pitch shifting
                    # If we use them, we should pass pitch=0
                    if "Journey" in voice:
                        success = generate_gcp_audio(phrase, voice, path, 0.0, rate)
                    else:
                        success = generate_gcp_audio(phrase, voice, path, pitch, rate)
                        
                    if path.exists(): # generate_gcp_audio writes file
                        count += 1
                        pbar.update(1)
                except Exception as e:
                    pass

if __name__ == "__main__":
    generate_samples()
