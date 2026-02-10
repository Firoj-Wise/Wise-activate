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
DATA_SEQ_DIR = BASE_DIR / "data_sequence"

TARGET_COUNT = 300 # Increased from 200

# Voice configurations
VOICES_EN = [
    "en-IN-Wavenet-A", "en-IN-Wavenet-B", "en-IN-Wavenet-C", "en-IN-Wavenet-D",
    "en-US-Studio-M", "en-GB-Neural2-B", "en-US-Neural2-A"
]
VOICES_NE = [
    "hi-IN-Neural2-A", "hi-IN-Neural2-B", "hi-IN-Neural2-C", "hi-IN-Neural2-D",
    "bn-IN-Wavenet-A", "bn-IN-Wavenet-B" 
]

WORDS = {
    "hello": ["Hello", "Hullo", "Hellow", "Hey", "Hi There"],
    "namaste": ["Namaste", "Namaskar", "Namastey", "Namaskaar"], 
    "deepa": ["Deepa", "Deepaa", "Dipa"],
    "deepak": ["Deepak", "Dipak", "Deepuk"]
}

# --- Augmentation Pipelines (Same as Negatives) ---
AUG_DISTANCE = Compose([
    LowPassFilter(min_cutoff_freq=200, max_cutoff_freq=800, p=1.0),
    Gain(min_gain_db=-18.0, max_gain_db=-10.0, p=1.0)
])

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
        # print(f"Failed to augment {suffix}: {e}")
        return False

def generate_samples():
    client = get_gcp_client()
    if not client:
        print("GCP Key not configured. Cannot generate synthetic data.")
        return

    print("Generating Sequence Component Data (+Variants)...")

    for category, phrases in WORDS.items():
        output_dir = DATA_SEQ_DIR / category
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Processing '{category}'...")
        
        # Check existing count (approximate)
        existing_files = list(output_dir.glob("syn_*.mp3"))
        count = len(existing_files)
        
        # We want roughly TARGET_COUNT *base* files, which will result in 3x total files
        # So we count only the base files (without _dist or _freq) for the loop
        base_files = [f for f in existing_files if "_dist" not in f.name and "_freq" not in f.name]
        base_count = len(base_files)

        if base_count >= TARGET_COUNT:
            print(f"  Skipping '{category}': Already has {base_count} base files.")
            # Still might need to generate variants for existing files if missing
            # But for now let's assume if base count is met we move on or user can delete catalog
        else:
            print(f"  Resuming '{category}' base gen from {base_count}/{TARGET_COUNT}...")
            
        all_voices = VOICES_EN + VOICES_NE
        
        # 1. Generate Base Files
        with tqdm(total=TARGET_COUNT, initial=base_count, desc=f"{category} (Base)") as pbar:
            while base_count < TARGET_COUNT:
                voice = random.choice(all_voices)
                phrase = random.choice(phrases)
                
                # Robust Variabilities (Pitch/Rate)
                pitch = random.uniform(-4.0, 4.0)
                rate = random.uniform(0.85, 1.15)
                
                filename = f"syn_{category}_{base_count}_{voice}_{int(time.time())}.mp3"
                path = output_dir / filename
                
                try:
                    if "Journey" in voice:
                        success = generate_gcp_audio(phrase, voice, path, 0.0, rate)
                    else:
                        success = generate_gcp_audio(phrase, voice, path, pitch, rate)
                        
                    if path.exists():
                        base_count += 1
                        pbar.update(1)
                except Exception as e:
                    pass

        # 2. Generate Variants for ALL base files (new and old)
        # Re-list all base files
        current_base_files = [f for f in output_dir.glob("syn_*.mp3") if "_dist" not in f.name and "_freq" not in f.name]
        print(f"  Generating physical variants for {len(current_base_files)} base files...")
        
        for fpath in tqdm(current_base_files, desc=f"{category} (Variants)"):
            # Distance
            dist_path = output_dir / fpath.name.replace(".mp3", "_dist.mp3")
            if not dist_path.exists():
                process_and_save_variant(fpath, AUG_DISTANCE, "dist")
            
            # Bad Mic
            freq_path = output_dir / fpath.name.replace(".mp3", "_freq.mp3")
            if not freq_path.exists():
                process_and_save_variant(fpath, AUG_BADMIC, "freq")

if __name__ == "__main__":
    generate_samples()
