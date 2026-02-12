# Simplified Wake Phrases for Wake Word Detection
# 
# Names: Deepa (Female)

PHRASES = {
    # ENGLISH
    "en": [
        "Hello Deepa"
    ],
    
    # NEPALI
    "ne": [
        "नमस्ते दीपा"
    ]
}

# Voices for TTS generation
VOICES = {
    "en": ["en-US-AriaNeural", "en-US-GuyNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural"],
    "ne": ["ne-NP-HemkalaNeural", "ne-NP-SagarNeural", "hi-IN-SwaraNeural", "hi-IN-MadhurNeural"]
}

# ============================================================
# HARD NEGATIVE SENTENCES
# ============================================================
NEGATIVE_SENTENCES_EN = [
    # === Phonetically similar words ===
    "The water is deep.", "Go deeper.", "Deep end.",
    "Depart now.", "Department store.", "Depend on me.",
    "Deport the file.", "The depot is closed.",
    "Dipper.", "Dipika.", "Deeksha.", "Epoch.",
    "Halo.", "Pillow.", "Fellow.", "Yellow.",
    "Jelly.", "Belly.", "Jello.", "Below.",
    
    # === Common conversation ===
    "How are you?", "I'm going to work.", "See you later.",
    "Good morning.", "Thank you.", "Have a nice day.",
    "Turn on the lights.", "Play music.", "Stop.",
    "Hello dear.", "Hello everyone.", "Namaste."
]

NEGATIVE_SENTENCES_NE = [
    # === Similar names ===
    "दिलिप", "दिनेश", "दिपेन्द्र", "दीपक", "दीप", "दीपावली",
    
    # === Random Nepali ===
    "बत्ती बाल्नुहोस्", "गीत बजाउनुस्", 
    "खाना खानुभयो?", "घर जान्छु",
    "भोलि भेटौला", "नमस्कार", "नमस्ते साथी", "दीदी"
]

NEG_MAP = {
    "en": NEGATIVE_SENTENCES_EN,
    "ne": NEGATIVE_SENTENCES_NE
}
