import os
from google.cloud import texttospeech
from pathlib import Path

# Path to the service account key
KEY_PATH = Path(__file__).resolve().parent.parent / "gcp_key.json"

def get_gcp_client():
    """Returns a GCP TTS client if the key exists, else None."""
    if not KEY_PATH.exists():
        print(f"⚠️ GCP Key not found at {KEY_PATH}. Skipping Google Voices.")
        return None
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_PATH)
    return texttospeech.TextToSpeechClient()

def list_voices(language_code="en-US"):
    """Lists available voices for a language."""
    client = get_gcp_client()
    if not client: return []
    
    response = client.list_voices(language_code=language_code)
    voices = sorted([v.name for v in response.voices])
    return voices

def generate_gcp_audio(text, voice_name, output_path, pitch=0.0, speaking_rate=1.0):
    """
    Generates audio using Google Cloud TTS.
    pitch: -20.0 to 20.0 (semitones)
    speaking_rate: 0.25 to 4.0
    """
    client = get_gcp_client()
    if not client: return False

    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request
    # extract language code from voice name (e.g. "en-US-Wavenet-D" -> "en-US")
    lang_code = "-".join(voice_name.split("-")[:2])
    
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        name=voice_name
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        pitch=pitch,
        speaking_rate=speaking_rate
    )

    try:
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_path, "wb") as out:
            out.write(response.audio_content)
            
        return True
    except Exception as e:
        print(f"❌ GCP Error ({voice_name}): {e}")
        return False
