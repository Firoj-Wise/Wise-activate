# Simplified Wake Phrases for Wake Word Detection
# 
# Names: Deepak (Male), Deepa (Female)

PHRASES = {
    # ENGLISH: Diverse spellings to capture different intonations
    "en": [
        "Hey Deepak", "Hey Deepa",
        "Hay Deepak", "Hay Deepa",
        "Haey Deepak", "Haey Deepa",
        "Hie Deepak", "Hie Deepa",
        "Heh Deepak", "Heh Deepa",
        "Deepak", "Deepa" # Names alone for robustness
    ],
    
    # NEPALI: Heavy variations to capture diverse Nepali accents/stress
    "ne": [
        "Namaste Deepak", "Namaste Deepa",
        "Namastay Deepak", "Namastay Deepa",
        "Nuh-must-ay Deepak", "Nuh-must-ay Deepa",
        "Namastey Deepak", "Namastey Deepa",
        "Namaskar Deepak", "Namaskar Deepa",
        "नमस्ते दीपक", "नमस्ते दीपा",
        "नमस्कार दीपक", "नमस्कार दीपा",
        "दीपक", "दीपा"
    ],
    
    # MAITHILI: Heavy variations for pronunciation diversity
    "mai": [
        "Pranam Deepak", "Pranam Deepa",
        "Pranaam Deepak", "Pranaam Deepa",
        "Pra-naam Deepak", "Pra-naam Deepa",
        "Prainam Deepak", "Prainam Deepa",
        "Pranam", "Pranaam",
        "प्रणाम दीपक", "प्रणाम दीपा",
        "दीपक", "दीपा"
    ]
}

# Voices for TTS generation
VOICES = {
    "en": ["en-US-AriaNeural", "en-US-GuyNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural"],
    "ne": ["ne-NP-HemkalaNeural", "ne-NP-SagarNeural", "hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
    "mai": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural", "en-IN-NeerjaNeural"]
}

# ============================================================
# HARD NEGATIVE SENTENCES
# ============================================================
# These are designed to teach the model what is NOT a wake word.

NEGATIVE_SENTENCES_EN = [
    # === Phonetically similar words ===
    "The water is deep.", "Go deeper.", "Deep end.",
    "Depart now.", "Department store.", "Depend on me.",
    "Six pack.", "Backpack.", "The park is open.",
    
    # === Common conversation ===
    "How are you?", "I'm going to work.", "See you later.",
    "Good morning.", "Thank you.", "Have a nice day.",
    "Turn on the lights.", "Play music.", "Stop."
]

NEGATIVE_SENTENCES_NE = [
    # === Similar names ===
    "दिलिप", "दिनेश", "दिपेन्द्र",
    
    # === Random Nepali ===
    "बत्ती बाल्नुहोस्", "गीत बजाउनुस्", 
    "खाना खानुभयो?", "घर जान्छु",
    "मलाई थाहा छैन", "भोलि भेटौला"
]

NEGATIVE_SENTENCES_MAI = [
    # === Random Maithili ===
    "हम घर जा रहल छी", "खाना खा लेलहुँ?",
    "आजुक दिन नीक अछि", "राति भ गेल"
]
