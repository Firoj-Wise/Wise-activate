import os
import random
import time
from pathlib import Path
from tqdm import tqdm
from gcp_tts import get_gcp_client, generate_gcp_audio

# Audio Processing
import librosa
import soundfile as sf
import numpy as np
from audiomentations import Compose, Gain, LowPassFilter, BandPassFilter, AddGaussianNoise

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "background" / "hard_negatives"

TARGET_COUNT_PER_WORD = 20

VOICES = [
    "en-IN-Wavenet-A", "en-IN-Wavenet-B", "en-IN-Wavenet-C", "en-IN-Wavenet-D",
    "hi-IN-Neural2-A", "hi-IN-Neural2-B", "hi-IN-Neural2-C", "hi-IN-Neural2-D",
    "en-US-Studio-M", "en-GB-Neural2-B"
]

CONFUSERS = {
    "deepak_like": [
        "Deep", "Pak", "Dipper", "Diva", "Dipa", "Depa", "Deeksha", 
        "Dipika", "Epoch", "Park", "Dark", "Dock", "Duck", "Peek"
    ],
    "hey_like": [
        "Hi", "Hay", "Bay", "Day", "Whey", "Haze", "Ray", "Say", 
        "High", "Height", "Hate", "Hail", "Hale"
    ],
    "namaste_like": [
        "Namaskar", "Stay", "Nasty", "Namas", "Master", "Musty", 
        "Misty", "Tea", "Sea", "Steer"
    ],
    "names": [
        "Ram", "Sita", "Hari", "Gita", "Shyam", "Ramesh", "Suresh", 
        "Mahesh", "Ganesh", "Laxmi", "Saraswati", "Bishnu", "Shiva",
        "Arjun", "Krishna", "Radha", "Mira", "Kabir", "Rahim", 
        "John", "David", "Michael", "Sarah", "Emily", "Jessica",
        "Rahul", "Rohan", "Priya", "Anjali", "Vikram", "Aditya",
        "Sanjay", "Vijay", "Ajay", "Sunil", "Anil", "Bimal",
        "Kamal", "Nimal", "Sunita", "Anita", "Binita", "Kavita",
        "Rita", "Mita", "Nita", "Gita", "Sita", "Rita"
    ]
}

# --- Augmentation Pipelines ---

# 1. Distant / Far-field (Muffled + Low Volume)
AUG_DISTANCE = Compose([
    LowPassFilter(min_cutoff_freq=200, max_cutoff_freq=800, p=1.0),
    Gain(min_gain_db=-20.0, max_gain_db=-10.0, p=1.0)
])

# 2. Bad Mic / Telephony (Bandpass + Noise)
AUG_BADMIC = Compose([
    BandPassFilter(min_center_freq=300, max_center_freq=3400, min_bandwidth_fraction=0.1, max_bandwidth_fraction=0.9, p=1.0),
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.02, p=0.8),
    Gain(min_gain_db=-5.0, max_gain_db=5.0, p=1.0)
])

def process_and_save_variant(original_path, pipeline, suffix):
    try:
        y, sr = librosa.load(original_path, sr=16000)
        y_aug = pipeline(samples=y, sample_rate=sr)
        
        new_path = str(original_path).replace(".mp3", f"_{suffix}.mp3")
        sf.write(new_path, y_aug, sr)
        return True
    except Exception as e:
        print(f"Failed to augment {suffix}: {e}")
        return False

def generate_hard_negatives():
    client = get_gcp_client()
    if not client:
        print("GCP Key not configured.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating Hard Negatives (+Variants) in {OUTPUT_DIR}...")
    
    total_generated = 0
    
    for category, words in CONFUSERS.items():
        print(f"Processing category: {category}")
        
        for word in tqdm(words):
            for i in range(TARGET_COUNT_PER_WORD):
                voice = random.choice(VOICES)
                pitch = random.uniform(-5.0, 5.0)
                rate = random.uniform(0.8, 1.2)
                
                # Base Filename
                filename = f"neg_{category}_{word}_{i}_{voice}.mp3"
                path = OUTPUT_DIR / filename
                
                # 1. Generate Clean (if not exists)
                if not path.exists():
                    try:
                        if "Journey" in voice:
                            success = generate_gcp_audio(word, voice, path, 0.0, rate)
                        else:
                            success = generate_gcp_audio(word, voice, path, pitch, rate)
                    except:
                        success = False
                else:
                    success = True # Originally existed

                if success and path.exists():
                    # 2. Generate "Distance" Variant
                    dist_path = OUTPUT_DIR / filename.replace(".mp3", "_dist.mp3")
                    if not dist_path.exists():
                        process_and_save_variant(path, AUG_DISTANCE, "dist")
                        total_generated += 1

                    # 3. Generate "Bad Mic" Variant
                    freq_path = OUTPUT_DIR / filename.replace(".mp3", "_freq.mp3")
                    if not freq_path.exists():
                        process_and_save_variant(path, AUG_BADMIC, "freq")
                        total_generated += 1
                        
                    if not path.exists(): # If it was newly created
                        total_generated += 1
                        
    print(f"Done! Generated {total_generated} NEW samples (including variants).")

if __name__ == "__main__":
    generate_hard_negatives()