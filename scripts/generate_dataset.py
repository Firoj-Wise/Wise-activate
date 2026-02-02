import asyncio
import os
import random
import hashlib
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
from pathlib import Path
from phrases import PHRASES, NEGATIVE_SENTENCES_EN, NEGATIVE_SENTENCES_NE, NEGATIVE_SENTENCES_MAI
import gcp_tts

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "wake"
BACKGROUND_DIR = BASE_DIR / "data" / "background"

# Ensure directories exist
for lang in PHRASES.keys():
    for keyword in ["deepak", "deepa"]:
        (DATA_DIR / keyword / lang).mkdir(parents=True, exist_ok=True)
BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

# MASSIVE GCP VOICE LIST (Wavenet + Neural2 + Studio)
GOOGLE_VOICES = {
    "en": [
        "en-IN-Standard-A", "en-IN-Standard-B", "en-IN-Standard-C", "en-IN-Standard-D",
        "en-IN-Wavenet-A", "en-IN-Wavenet-B", "en-IN-Wavenet-C", "en-IN-Wavenet-D",
        "en-IN-Neural2-A", "en-IN-Neural2-B", "en-IN-Neural2-C", "en-IN-Neural2-D",
        "en-US-Standard-A", "en-US-Standard-B", "en-US-Standard-C", "en-US-Standard-D", "en-US-Standard-E",
        "en-US-Wavenet-A", "en-US-Wavenet-B", "en-US-Wavenet-C", "en-US-Wavenet-D", "en-US-Wavenet-E",
        "en-US-Neural2-A", "en-US-Neural2-C", "en-US-Neural2-D", "en-US-Neural2-E",
        "en-US-Studio-O", "en-US-Studio-Q"
    ],
    "ne": [
        # Using Hindi voices as High-Fidelity proxies for Nepali phonetics
        "hi-IN-Standard-A", "hi-IN-Standard-B", "hi-IN-Standard-C", "hi-IN-Standard-D",
        "hi-IN-Wavenet-A", "hi-IN-Wavenet-B", "hi-IN-Wavenet-C", "hi-IN-Wavenet-D",
        "hi-IN-Neural2-A", "hi-IN-Neural2-B", "hi-IN-Neural2-C", "hi-IN-Neural2-D"
    ],
    "mai": [
        # Using Hindi voices as High-Fidelity proxies for Maithili phonetics
        "hi-IN-Standard-A", "hi-IN-Standard-B", "hi-IN-Standard-C", "hi-IN-Standard-D",
        "hi-IN-Wavenet-A", "hi-IN-Wavenet-B", "hi-IN-Wavenet-C", "hi-IN-Wavenet-D",
        "hi-IN-Neural2-A", "hi-IN-Neural2-B", "hi-IN-Neural2-C", "hi-IN-Neural2-D"
    ]
}

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
    print("🚀 Starting GCP-ONLY High-Fidelity Dataset Generation...")
    
    gcp_client = gcp_tts.get_gcp_client()
    if not gcp_client:
        print("[CRITICAL] GCP Credentials not found. Check gcp_key.json!")
        return

    # 1. Generate Wake Words (Amplify voices via standard GCP SDK)
    for lang_key, phrases in PHRASES.items():
        print(f"--- Generating {lang_key} Wake Words (Massive Volume) ---")
        
        # MASSIVE VOLUME: 20x for NE/MAI, 10x for EN
        repeat_count = 20 if lang_key in ["ne", "mai"] else 10
        count = 0
        
        gcp_voice_list = GOOGLE_VOICES.get(lang_key, [])
        for phrase in phrases:
            # Determine keyword folder
            if any(x in phrase.lower() for x in ["deepak", "दीपक", "नमस्कार"]): keyword = "deepak"
            else: keyword = "deepa"

            dest_dir = DATA_DIR / keyword / lang_key
            dest_dir.mkdir(parents=True, exist_ok=True)

            for voice in gcp_voice_list:
                # FULL Variations Matrix: 3 rates x 3 pitches = 9 variations per voice/phrase
                rates = [0.85, 1.0, 1.15]
                pitches = [-2.0, 0.0, 2.0]
                
                # We use ALL 9 variations to maximize signal diversity
                for rate in rates:
                    for pitch in pitches:
                        for i in range(repeat_count):
                            unique_str = f"gcp_{phrase}_{voice}_{rate}_{pitch}_{i}_{lang_key}"
                            file_hash = get_hash(unique_str)
                            wav_path = dest_dir / f"gcp_{file_hash}.wav"
                            mp3_temp = str(wav_path).replace(".wav", ".mp3")
                            
                            # Skip if exists (idempotency)
                            if not os.path.exists(wav_path):
                                success = gcp_tts.generate_gcp_audio(
                                    phrase, voice, mp3_temp, pitch=pitch, speaking_rate=rate
                                )
                                if success:
                                    if convert_to_wav(mp3_temp, wav_path):
                                        count += 1
                                        if count % 200 == 0: print(f"GCP ({lang_key}): {count} samples generated...")

    # 2. Generate Basic Background Noise (Synthetic)
    print("--- Generating Synthetic Background Noise ---")
    sr = 16000
    duration = 1.0
    samples = int(sr * duration)
    
    for i in range(500): # 500 White
        data = np.random.normal(0, 1, samples)
        data = (data / (np.max(np.abs(data)) + 1e-6) * 32767).astype(np.int16)
        wav.write(BACKGROUND_DIR / f"white_{i}.wav", sr, data)
        
    for i in range(500): # 500 Brownian
        data = np.cumsum(np.random.normal(0, 1, samples))
        data = (data / (np.max(np.abs(data)) + 1e-6) * 32767).astype(np.int16)
        wav.write(BACKGROUND_DIR / f"brown_{i}.wav", sr, data)

    # 3. Generate Hard Negatives (Speech that is NOT wake word)
    print("--- Generating Hard Negatives (Speech) ---")
    for lang_key, sent_list in [("en", NEGATIVE_SENTENCES_EN), ("ne", NEGATIVE_SENTENCES_NE), ("mai", NEGATIVE_SENTENCES_MAI)]:
        gcp_voices = GOOGLE_VOICES.get(lang_key, [])[:5] # Use first 5 voices
        for sent in sent_list[:20]:
            for voice in gcp_voices:
                unique_str = f"neg_{sent}_{voice}"
                file_hash = get_hash(unique_str)
                wav_path = BACKGROUND_DIR / f"neg_{file_hash}.wav"
                mp3_temp = str(wav_path).replace(".wav", ".mp3")
                if not os.path.exists(wav_path):
                    if gcp_tts.generate_gcp_audio(sent, voice, mp3_temp):
                        convert_to_wav(mp3_temp, wav_path)

    print("\n[SUCCESS] High-Fidelity GCP-Only Dataset Ready.")

if __name__ == "__main__":
    asyncio.run(main())
