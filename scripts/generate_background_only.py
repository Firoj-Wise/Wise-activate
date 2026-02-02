import asyncio
import os
import random
import edge_tts
from pydub import AudioSegment
from pathlib import Path
from phrases import VOICES, NEGATIVE_SENTENCES_EN, NEGATIVE_SENTENCES_NE, NEGATIVE_SENTENCES_MAI
import gcp_tts
import hashlib
import numpy as np
import scipy.io.wavfile as wav

# Google Voices Map (Premium/Standard)
GOOGLE_VOICES = {
    "en": [
        "en-IN-Standard-A", "en-IN-Standard-B", "en-IN-Standard-C", "en-IN-Standard-D",
        "en-IN-Wavenet-A", "en-IN-Wavenet-B", "en-IN-Wavenet-C", "en-IN-Wavenet-D",
        "en-US-Standard-A", "en-US-Wavenet-A"
    ],
    "ne": [
        "hi-IN-Wavenet-A", "hi-IN-Wavenet-B", "hi-IN-Wavenet-C", "hi-IN-Wavenet-D",
        "hi-IN-Standard-A", "hi-IN-Standard-B", "hi-IN-Standard-C", "hi-IN-Standard-D"
    ],
    "mai": [
        "hi-IN-Wavenet-A", "hi-IN-Wavenet-B", "hi-IN-Wavenet-C", "hi-IN-Wavenet-D",
        "hi-IN-Standard-A", "hi-IN-Standard-B", "hi-IN-Standard-C", "hi-IN-Standard-D"
    ]
}

BASE_DIR = Path(__file__).resolve().parent.parent
BACKGROUND_DIR = BASE_DIR / "data" / "background"
BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

async def generate_edge_speech(text, voice, output_path, speed="+0%", pitch="+0Hz"):
    """Generates speech using edge-tts."""
    try:
        communicate = edge_tts.Communicate(text, voice, rate=speed, pitch=pitch)
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"EdgeTTS Error: {e}")
        return False

def convert_to_wav(mp3_path, wav_path):
    """Converts mp3 to wav (16kHz, mono) and deletes mp3."""
    try:
        audio = AudioSegment.from_file(mp3_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        os.remove(mp3_path)
        return True
    except Exception as e:
        print(f"Error converting {mp3_path}: {e}")
        if os.path.exists(mp3_path): os.remove(mp3_path)
        return False

def get_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:10]

async def main():
    print("==================================================")
    print("   GENERATING BACKGROUND ONLY (Strict Purity)     ")
    print("==================================================")
    
    # Check GCP
    gcp_client = gcp_tts.get_gcp_client()
    if gcp_client:
        print("[OK] GCP Credentials found. Using Google Premium Voices.")
    else:
        print("[WARNING] No GCP Credentials. Skipping Google Voices.")

    # 1. Generate Hard Negatives (TTS)
    print("\n[Phrase-Based Negatives]")
    neg_count = 0
    
    # Combine lists
    neg_tasks = [
        (NEGATIVE_SENTENCES_EN, "en"), 
        (NEGATIVE_SENTENCES_NE, "ne"),
        (NEGATIVE_SENTENCES_MAI, "mai")
    ]
    
    rates = ["-25%", "+0%", "+25%"]
    pitches = ["-15Hz", "+0Hz", "+15Hz"]
    
    for sent_list, lang_code in neg_tasks:
        print(f"Processing {lang_code} negatives...")
        edge_voices = VOICES[lang_code]
        gcp_voice_list = GOOGLE_VOICES.get(lang_code, []) if gcp_client else []
        
        for sent in sent_list:
            selected_edge_voices = edge_voices 
            
            for voice in selected_edge_voices:
                for speed in rates:
                    for pitch in pitches:
                        unique_str = f"neg_edge_{sent}_{voice}_{speed}_{pitch}"
                        file_hash = get_hash(unique_str)
                        
                        filename = f"neg_edge_{file_hash}.mp3"
                        mp3_path = BACKGROUND_DIR / filename
                        wav_path = str(mp3_path).replace(".mp3", ".wav")
                        
                        if not os.path.exists(wav_path):
                            success = await generate_edge_speech(sent, voice, mp3_path, speed, pitch)
                            if success:
                                if convert_to_wav(mp3_path, wav_path):
                                    neg_count += 1
                                    if neg_count % 50 == 0: print(f"Negatives: {neg_count}")

            # GCP TTS: Generate variations (if available)
            if gcp_voice_list:
                for voice in gcp_voice_list:
                    gcp_variations = [(0.8, -3.0), (1.0, 0.0), (1.2, 3.0)]
                    
                    for rate, pitch in gcp_variations:
                        unique_str = f"neg_gcp_{sent}_{voice}_{rate}_{pitch}"
                        file_hash = get_hash(unique_str)
                        
                        filename = f"neg_gcp_{file_hash}.mp3"
                        mp3_path = BACKGROUND_DIR / filename
                        wav_path = str(mp3_path).replace(".mp3", ".wav")
                        
                        if not os.path.exists(wav_path):
                            success = gcp_tts.generate_gcp_audio(sent, voice, mp3_path, pitch=pitch, speaking_rate=rate)
                            if success:
                                if convert_to_wav(mp3_path, wav_path):
                                    neg_count += 1
    
    print(f"Finished Phrase-Based Negatives: {neg_count} files.")
    
    # 2. Generate Synthetic Background Noise (White, Pink, Brown, Typing)
    print("\n[Synthetic Noise Generation]")
    print("(White, Pink, Brown, Typing clicks)")
    
    NOISE_DIR = BACKGROUND_DIR / "synthetic_noise"
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    
    def save_noise(name, data):
        # Normalize to 16-bit PCM
        data = data / np.max(np.abs(data))
        data = (data * 32767).astype(np.int16)
        wav.write(NOISE_DIR / f"{name}.wav", 16000, data)
        
    sr = 16000
    duration = 1.0 # 1 second
    samples = int(sr * duration)
    
    noise_count = 0
    
    # A. White Noise (200 files)
    for i in range(200):
        white = np.random.normal(0, 1, samples)
        save_noise(f"white_{i}", white)
        noise_count += 1
        
    # B. Pink/Brown-ish Noise (Filtered White) (200 files)
    # Simple cumulative sum (Brownian)
    for i in range(200):
        brown = np.cumsum(np.random.normal(0, 1, samples))
        save_noise(f"brown_{i}", brown)
        noise_count += 1
        
    # C. "Typing" / Clicks (Random spikes) (200 files)
    for i in range(200):
        click_track = np.zeros(samples)
        num_clicks = np.random.randint(5, 20)
        for _ in range(num_clicks):
            idx = np.random.randint(0, samples)
            click_track[idx] = np.random.uniform(0.5, 1.0)
            # Add decay
            if idx+100 < samples:
                click_track[idx:idx+100] = click_track[idx] * np.exp(-np.linspace(0, 5, 100))
        save_noise(f"typing_{i}", click_track)
        noise_count += 1
        
    print(f"Generated {noise_count} synthetic noise files.")
    print("BACKGROUND DATASET COMPLETE.")

if __name__ == "__main__":
    asyncio.run(main())
