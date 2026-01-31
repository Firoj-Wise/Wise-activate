import asyncio
import os
import random
import edge_tts
from pydub import AudioSegment
from pathlib import Path
from phrases import PHRASES, VOICES, NEGATIVE_SENTENCES_EN, NEGATIVE_SENTENCES_NE, NEGATIVE_SENTENCES_MAI
import gcp_tts

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "wake"
BACKGROUND_DIR = BASE_DIR / "data" / "background"

# Ensure directories exist
for lang in PHRASES.keys():
    (DATA_DIR / lang).mkdir(parents=True, exist_ok=True)
BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

# Google Voices Map (Premium/Standard)
GOOGLE_VOICES = {
    "en": [
        "en-IN-Standard-A", "en-IN-Standard-B", "en-IN-Standard-C", "en-IN-Standard-D",
        "en-IN-Wavenet-A", "en-IN-Wavenet-B", "en-IN-Wavenet-C", "en-IN-Wavenet-D",
        "en-US-Standard-A", "en-US-Wavenet-A"
    ],
    "ne": [
        # Nepali voices unavailable in this region/account. Using Hindi (hi-IN) as high-quality proxy.
        # Phonetics are very similar (Devanagari script).
        "hi-IN-Wavenet-A", "hi-IN-Wavenet-B", "hi-IN-Wavenet-C", "hi-IN-Wavenet-D",
        "hi-IN-Standard-A", "hi-IN-Standard-B", "hi-IN-Standard-C", "hi-IN-Standard-D"
    ],
    "mai": [
        # Maithili is NOT supported in public GCP API. Using Hindi voices as best approximation.
        "hi-IN-Wavenet-A", "hi-IN-Wavenet-B", "hi-IN-Wavenet-C", "hi-IN-Wavenet-D",
        "hi-IN-Standard-A", "hi-IN-Standard-B", "hi-IN-Standard-C", "hi-IN-Standard-D"
    ]
}

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

async def main():
    print("Starting Hybrid Dataset Generation (Edge + Google)...")
    
    # Check GCP
    gcp_client = gcp_tts.get_gcp_client()
    if gcp_client:
        print("✅ GCP Credentials found. Using Google Premium Voices.")
    else:
        print("⚠️ No GCP Credentials. Skipping Google Voices.")

    # 1. Generate Wake Words
    for lang, phrases in PHRASES.items():
        print(f"--- Generating {lang} samples ---")
        count = 0
        
        # A. Edge TTS Generation
        edge_voices = VOICES[lang]
        for phrase in phrases:
            # DETERMINE KEYWORD FOLDER
            # "Deepak" (Male) vs "Deepa" (Female)
            if "deepak" in phrase.lower() or "दीपक" in phrase:
                keyword = "deepak"
            elif "deepa" in phrase.lower() or "दीपा" in phrase:
                keyword = "deepa"
            else:
                print(f"Skipping unknown phrase: {phrase}")
                continue
                
            dest_dir = DATA_DIR / keyword / lang
            dest_dir.mkdir(parents=True, exist_ok=True)

            for voice in edge_voices:
                # More aggressive variations to ensure distinct voices
                rates = ["-25%", "+0%", "+25%"]
                pitches = ["-15Hz", "+0Hz", "+15Hz"]
                
                for speed in rates:
                    for pitch in pitches:
                        filename = f"edge_{count}_{voice}_{speed}_{pitch}.mp3".replace("%", "").replace("+", "")
                        mp3_path = dest_dir / filename
                        wav_path = str(mp3_path).replace(".mp3", ".wav")
                        
                        if not os.path.exists(wav_path):
                            success = await generate_edge_speech(phrase, voice, mp3_path, speed, pitch)
                            if success and convert_to_wav(mp3_path, wav_path):
                                count += 1
                                if count % 10 == 0: print(f"Edge: {phrase} ({lang}) -> {keyword}")
                        
        # B. Google TTS Generation
        if gcp_client:
            gcp_voice_list = GOOGLE_VOICES.get(lang, [])
            for phrase in phrases:
                # DETERMINE KEYWORD FOLDER
                if "deepak" in phrase.lower() or "दीपक" in phrase:
                    keyword = "deepak"
                elif "deepa" in phrase.lower() or "दीपा" in phrase:
                    keyword = "deepa"
                else:
                    continue

                dest_dir = DATA_DIR / keyword / lang
                dest_dir.mkdir(parents=True, exist_ok=True)

                for voice in gcp_voice_list:
                    # Aggressive GCP variations
                    # Rate: 0.75 (slow), 1.0 (normal), 1.25 (fast)
                    # Pitch: -4.0 st (deeper), 0.0, +4.0 st (higher)
                    
                    variations = [
                        (0.75, -4.0), (1.0, 0.0), (1.25, 4.0)
                    ]
                    
                    for rate, pitch in variations:
                        filename = f"gcp_{count}_{voice}_{rate}_{pitch}.mp3"
                        mp3_path = dest_dir / filename
                        wav_path = str(mp3_path).replace(".mp3", ".wav")
                        
                        if not os.path.exists(wav_path):
                            success = gcp_tts.generate_gcp_audio(phrase, voice, mp3_path, pitch=pitch, speaking_rate=rate)
                            if success and convert_to_wav(mp3_path, wav_path):
                                count += 1
                                if count % 10 == 0: print(f"GCP: {phrase} ({lang}) -> {keyword}")

        print(f"Finished {lang}: {count} new files.")

    # 2. Generate Hard Negatives
    print("Generating Hard Negatives...")
    neg_count = 0
    
    # Combine lists
    neg_tasks = [
        (NEGATIVE_SENTENCES_EN, "en"), 
        (NEGATIVE_SENTENCES_NE, "ne"),
        (NEGATIVE_SENTENCES_MAI, "mai")
    ]
    
    for sent_list, lang_code in neg_tasks:
        edge_voices = VOICES[lang_code]
        gcp_voice_list = GOOGLE_VOICES.get(lang_code, []) if gcp_client else []
        
        for sent in sent_list:
            # 1 random Edge voice
            voice = random.choice(edge_voices)
            filename = f"neg_edge_{neg_count}.mp3"
            mp3_path = BACKGROUND_DIR / filename
            wav_path = str(mp3_path).replace(".mp3", ".wav")
            await generate_edge_speech(sent, voice, mp3_path)
            convert_to_wav(mp3_path, wav_path)
            neg_count += 1
            
            # 1 random GCP voice (if available)
            if gcp_voice_list:
                voice = random.choice(gcp_voice_list)
                filename = f"neg_gcp_{neg_count}.mp3"
                mp3_path = BACKGROUND_DIR / filename
                wav_path = str(mp3_path).replace(".mp3", ".wav")
                gcp_tts.generate_gcp_audio(sent, voice, mp3_path)
                convert_to_wav(mp3_path, wav_path)
                neg_count += 1

    print(f"Finished Negatives: {neg_count} files.")

if __name__ == "__main__":
    asyncio.run(main())
