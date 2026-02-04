import asyncio
import os
import random
import hashlib
import numpy as np
import scipy.io.wavfile as wav
import edge_tts
from pydub import AudioSegment
from pathlib import Path
# Import negative sentences to generate speech for the "Background" class
from phrases import PHRASES, VOICES, NEGATIVE_SENTENCES_EN, NEGATIVE_SENTENCES_NE, NEGATIVE_SENTENCES_MAI
import gcp_tts

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "wake"
BACKGROUND_DIR = BASE_DIR / "data" / "background"

# Ensure directories exist
for lang in PHRASES.keys():
    for keyword in ["deepak", "deepa"]:
        (DATA_DIR / keyword / lang).mkdir(parents=True, exist_ok=True)
BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

# Massive GCP Voice List
GOOGLE_VOICES = {
    "en": [
        "en-IN-Wavenet-A", "en-IN-Wavenet-B", "en-IN-Wavenet-C", "en-IN-Wavenet-D",
        "en-US-Wavenet-A", "en-US-Wavenet-B", "en-US-Wavenet-C", "en-US-Wavenet-D",
        "en-US-Neural2-A", "en-US-Neural2-C", "en-US-Neural2-F"
    ],
    "ne": [
        "hi-IN-Wavenet-A", "hi-IN-Wavenet-B", "hi-IN-Neural2-A", "hi-IN-Neural2-B"
    ],
    "mai": [
        "hi-IN-Wavenet-A", "hi-IN-Neural2-A"
    ]
}

async def generate_edge_speech(text, voice, output_path, rate="+0%", pitch="+0Hz"):
    """Generates speech using edge-tts (Fast)."""
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(output_path)
        return True
    except Exception as e:
        return False

def convert_to_wav(mp3_path, wav_path):
    """Converts mp3 to wav (16kHz, mono) and deletes mp3."""
    try:
        audio = AudioSegment.from_file(mp3_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        os.remove(mp3_path)
        return True
    except:
        if os.path.exists(mp3_path): os.remove(mp3_path)
        return False

def get_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:10]

# --- Synthetic Noise Generators ---

def generate_white_noise(path, duration=1.0, sr=16000):
    samples = int(sr * duration)
    data = np.random.normal(0, 0.05, samples)
    data = (data * 32767).astype(np.int16)
    wav.write(path, sr, data)

def generate_brown_noise(path, duration=1.0, sr=16000):
    samples = int(sr * duration)
    data = np.cumsum(np.random.normal(0, 0.02, samples))
    data = (data / (np.max(np.abs(data)) + 1e-6) * 8000).astype(np.int16)
    wav.write(path, sr, data)

def generate_airplane_rumble(path, duration=1.0, sr=16000):
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    rumble = np.cumsum(np.random.normal(0, 0.08, samples))
    hum = 0.15 * np.sin(2 * np.pi * 75 * t)
    data = rumble + hum
    data = (data / (np.max(np.abs(data)) + 1e-6) * 10000).astype(np.int16)
    wav.write(path, sr, data)

def generate_train_noise(path, duration=1.0, sr=16000):
    samples = int(sr * duration)
    rumble = np.cumsum(np.random.normal(0, 0.06, samples))
    clack = np.zeros(samples)
    for i in [0.1, 0.35, 0.6, 0.85]:
        idx = int(i * samples)
        pulse = np.random.normal(0, 0.4, 300)
        env = np.exp(-np.linspace(0, 4, 300))
        clack[idx:idx+300] = pulse * env
    data = rumble + clack
    data = (data / (np.max(np.abs(data)) + 1e-6) * 9000).astype(np.int16)
    wav.write(path, sr, data)

def generate_typing_noise(path, duration=1.0, sr=16000):
    samples = int(sr * duration)
    data = np.zeros(samples)
    for _ in range(random.randint(4, 12)):
        idx = random.randint(0, samples - 500)
        click = np.random.normal(0, 0.3, random.randint(40, 150))
        env = np.exp(-np.linspace(0, 6, len(click)))
        data[idx:idx+len(click)] += click * env
    data = (data / (np.max(np.abs(data)) + 1e-6) * 15000).astype(np.int16)
    wav.write(path, sr, data)

def generate_office_noise(path, duration=1.0, sr=16000):
    samples = int(sr * duration)
    base = np.cumsum(np.random.normal(0, 0.015, samples))
    clicks = np.zeros(samples)
    for _ in range(random.randint(3, 8)):
        idx = random.randint(0, samples - 300)
        click = np.random.normal(0, 0.1, random.randint(20, 100))
        clicks[idx:idx+len(click)] += click
    floor = np.random.normal(0, 0.01, samples)
    data = base + clicks + floor
    data = (data / (np.max(np.abs(data)) + 1e-6) * 8000).astype(np.int16)
    wav.write(path, sr, data)

def generate_hum_noise(path, duration=1.0, sr=16000):
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    f0 = random.uniform(80, 150)
    data = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2*f0 * t) + 0.2 * np.sin(2 * np.pi * 3*f0 * t)
    jitter = 0.05 * np.cumsum(np.random.normal(0, 1, samples))
    data = data * np.exp(1j * jitter).real
    data = (data / (np.max(np.abs(data)) + 1e-6) * 15000).astype(np.int16)
    wav.write(path, sr, data)

def generate_scream_noise(path, duration=1.0, sr=16000):
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    f_mod = 1000 + 4000 * np.sin(2 * np.pi * 10 * t)
    data = np.sin(2 * np.pi * f_mod * t) * np.random.normal(0.8, 0.2, samples)
    data = (data / (np.max(np.abs(data)) + 1e-6) * 20000).astype(np.int16)
    wav.write(path, sr, data)

async def main():
    print("🚀 Starting HEAVY Hybrid Generation (Edge + GCP) + Negative Speech...")
    gcp_client = gcp_tts.get_gcp_client()

    # 1. Generate Wake Words (Positives)
    for lang_key, phrases in PHRASES.items():
        print(f"--- Generating {lang_key} Wake Words ---")
        repeat_count = 30 if lang_key in ["ne", "mai"] else 15
        count = 0
        
        for phrase in phrases:
            # Route to correct folder (Deepak vs Deepa)
            keyword = "deepak" if "deepak" in phrase.lower() or "दीपक" in phrase else "deepa"
            dest_dir = DATA_DIR / keyword / lang_key
            dest_dir.mkdir(parents=True, exist_ok=True)

            # A. EDGE TTS
            edge_voices = VOICES.get(lang_key, [])
            if edge_voices:
                for voice in edge_voices:
                    for rate in ["-20%", "+0%", "+20%"]:
                        for pitch in ["-2Hz", "+0Hz", "+2Hz"]:
                            for i in range(max(1, repeat_count // 5)):
                                unique_str = f"edge_{phrase}_{voice}_{rate}_{pitch}_{i}"
                                file_hash = get_hash(unique_str)
                                wav_path = dest_dir / f"edge_{file_hash}.wav"
                                if not os.path.exists(wav_path):
                                    mp3 = str(wav_path).replace(".wav", ".mp3")
                                    if await generate_edge_speech(phrase, voice, mp3, rate, pitch):
                                        if convert_to_wav(mp3, wav_path):
                                            count += 1
                                            if count % 200 == 0: print(f"{lang_key} (Edge): {count} samples...")

            # B. GCP TTS
            if gcp_client:
                gcp_voices = GOOGLE_VOICES.get(lang_key, [])
                for voice in gcp_voices:
                    for rate in [0.8, 1.0, 1.2]:
                        for pitch in [-2.0, 0.0, 2.0]:
                            for i in range(max(1, repeat_count // 5)):
                                unique_str = f"gcp_{phrase}_{voice}_{rate}_{pitch}_{i}"
                                file_hash = get_hash(unique_str)
                                wav_path = dest_dir / f"gcp_{file_hash}.wav"
                                if not os.path.exists(wav_path):
                                    mp3 = str(wav_path).replace(".wav", ".mp3")
                                    if gcp_tts.generate_gcp_audio(phrase, voice, mp3, pitch, rate):
                                        if convert_to_wav(mp3, wav_path):
                                            count += 1
                                            if count % 200 == 0: print(f"{lang_key} (GCP): {count} samples...")

    # 2. Generate Negative Speech (CRITICAL FIX: Teach model what speech is NOT a wakeword)
    print("--- Generating Negative Speech (Conversational & Tricky phrases) ---")
    neg_map = {
        "en": NEGATIVE_SENTENCES_EN, 
        "ne": NEGATIVE_SENTENCES_NE, 
        "mai": NEGATIVE_SENTENCES_MAI
    }
    
    # Store these in 'background' because they are negative samples (Class 0)
    neg_dir = BACKGROUND_DIR / "negative_speech"
    neg_dir.mkdir(parents=True, exist_ok=True)
    
    count_neg = 0
    for lang, sentences in neg_map.items():
        voices = VOICES.get(lang, [])
        if not voices: continue
        
        for phrase in sentences:
            for voice in voices:
                # Generate variations to bulk up the negative speech dataset
                for i in range(5): 
                    unique_str = f"neg_{lang}_{phrase}_{voice}_{i}"
                    file_hash = get_hash(unique_str)
                    wav_path = neg_dir / f"neg_{file_hash}.wav"
                    
                    if not os.path.exists(wav_path):
                        mp3 = str(wav_path).replace(".wav", ".mp3")
                        # Vary speed slightly for naturalness
                        rate = random.choice(["-10%", "+0%", "+10%"])
                        if await generate_edge_speech(phrase, voice, mp3, rate=rate):
                            if convert_to_wav(mp3, wav_path):
                                count_neg += 1
                                if count_neg % 50 == 0: print(f"Negative Speech: {count_neg} samples...")
    
    print(f"Total Negative Speech Generated: {count_neg}")

    # 3. Generate Background Noise
    print("--- Generating Background Noise (White, Office, Hum, Scream) ---")
    modes = ["white", "brown", "plane", "train", "typing", "office", "hum", "scream"]
    for i in range(1000):
        mode = random.choice(modes)
        path = BACKGROUND_DIR / f"noise_{i}_{mode}.wav"
        if os.path.exists(path): continue
        if mode == "white": generate_white_noise(path)
        elif mode == "brown": generate_brown_noise(path)
        elif mode == "plane": generate_airplane_rumble(path)
        elif mode == "train": generate_train_noise(path)
        elif mode == "typing": generate_typing_noise(path)
        elif mode == "office": generate_office_noise(path)
        elif mode == "hum": generate_hum_noise(path)
        elif mode == "scream": generate_scream_noise(path)
        if i % 100 == 0: print(f"Background: {i} samples...")

    print("\n[SUCCESS] Custom Balanced Dataset Ready.")

if __name__ == "__main__":
    asyncio.run(main())