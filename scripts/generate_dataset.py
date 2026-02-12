import asyncio
import os
import random
import hashlib
import numpy as np
import scipy.io.wavfile as wav
import edge_tts
import subprocess
from pathlib import Path
# Import negative sentences to generate speech for the "Background" class
from phrases import PHRASES, VOICES, NEG_MAP
import gcp_tts

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "wake"
BACKGROUND_DIR = BASE_DIR / "data" / "background"

# Ensure directories exist
for lang in PHRASES.keys():
    (DATA_DIR / "deepa" / lang).mkdir(parents=True, exist_ok=True)
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
    """Converts mp3 to wav (16kHz, mono) and deletes mp3 using ffmpeg."""
    try:
        # Ultra-fast conversion with ffmpeg
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(mp3_path), '-ar', '16000', '-ac', '1', str(wav_path)],
            capture_output=True, check=True
        )
        if os.path.exists(mp3_path): os.remove(mp3_path)
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

async def generate_task(semaphore, text, voice, dest_dir, prefix, stats, randomize=True):
    async with semaphore:
        if randomize:
            rate_val = random.randint(-25, 25)
            pitch_val = random.randint(-5, 5)
            rate = f"{rate_val:+d}%"
            pitch = f"{pitch_val:+d}Hz"
        else:
            rate, pitch = "+0%", "+0Hz"
        
        unique_str = f"{prefix}_{text}_{voice}_{rate}_{pitch}_{random.random()}"
        file_hash = get_hash(unique_str)
        wav_path = dest_dir / f"{prefix}_{file_hash}.wav"
        
        if not os.path.exists(wav_path):
            mp3 = str(wav_path).replace(".wav", ".mp3")
            if await generate_edge_speech(text, voice, mp3, rate, pitch):
                if convert_to_wav(mp3, wav_path):
                    stats["count"] += 1
                    if stats["count"] % 100 == 0:
                        print(f"Progress: {stats['count']} unique samples generated...")

async def main():
    print("Starting ULTIMATE Concurrent Generation (High Diversity)...")
    semaphore = asyncio.Semaphore(60) # High concurrency for speed
    stats = {"count": 0}
    tasks = []

    # 1. Wake Words (Positives)
    # Total targets: ~20k positives (10k EN, 10k NE)
    for lang_key, phrases in PHRASES.items():
        count_per_phrase = 20000 // len(phrases)
        dest_dir = DATA_DIR / "deepa" / lang_key
        dest_dir.mkdir(parents=True, exist_ok=True)
        edge_voices = VOICES.get(lang_key, [])
        for phrase in phrases:
            for i in range(count_per_phrase):
                voice = random.choice(edge_voices)
                tasks.append(generate_task(semaphore, phrase, voice, dest_dir, "edge", stats))

    # 2. Negative Speech (Hard Negatives)
    # Total targets: ~20k negatives
    neg_dir = BACKGROUND_DIR / "hard_negatives"
    neg_dir.mkdir(parents=True, exist_ok=True)
    for lang, sentences in NEG_MAP.items():
        voices = VOICES.get(lang, [])
        count_per_sentence = 20000 // (len(NEG_MAP) * len(sentences))
        for phrase in sentences:
            for i in range(count_per_sentence):
                voice = random.choice(voices)
                tasks.append(generate_task(semaphore, phrase, voice, neg_dir, f"neg_{lang}", stats))

    print(f"Total Speech Tasks Planned: {len(tasks)}")
    
    # 3. Background Noise (Parallel with speech)
    print("--- Queueing Background Noise ---")
    modes = ["white", "brown", "plane", "train", "typing", "office", "hum", "scream"]
    for i in range(5000):
        mode = random.choice(modes)
        path = BACKGROUND_DIR / f"noise_{i}_{mode}.wav"
        async def noise_task(p=path, m=mode):
            if not os.path.exists(p):
                if m == "white": generate_white_noise(p)
                elif m == "brown": generate_brown_noise(p)
                elif m == "plane": generate_airplane_rumble(p)
                elif m == "train": generate_train_noise(p)
                elif m == "typing": generate_typing_noise(p)
                elif m == "office": generate_office_noise(p)
                elif m == "hum": generate_hum_noise(p)
                elif m == "scream": generate_scream_noise(p)
                stats["count"] += 1
        tasks.append(noise_task())

    random.shuffle(tasks) # Mix them up so diversity is interlevead
    await asyncio.gather(*tasks)

    print(f"\n[SUCCESS] Dataset Ready. Total samples processed: {stats['count']}")

if __name__ == "__main__":
    asyncio.run(main())