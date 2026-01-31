import os
from google.cloud import texttospeech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

def list_nepali_voices():
    client = texttospeech.TextToSpeechClient()
    response = client.list_voices()
    
    print("Searching for Nepali (ne-NP) voices...")
    found_any = False
    for voice in response.voices:
        for language_code in voice.language_codes:
            if "ne-NP" in language_code:
                print(f"✅ Found: {voice.name} | Gender: {voice.ssml_gender}")
                found_any = True
                
    if not found_any:
        print("❌ No 'ne-NP' voices found in your account/region.")

if __name__ == "__main__":
    try:
        list_nepali_voices()
    except Exception as e:
        print(f"Error: {e}")
