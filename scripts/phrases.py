# Simplified Wake Phrases for Wake Word Detection
# 
# STRICT PATTERN: Language-specific greeting PREFIX + Name
# The model should ONLY trigger on this exact pattern, not just the name alone.
#
# Names: Deepak (Male), Deepa (Female)

PHRASES = {
    # ENGLISH: "Hey there" / "Hello" + Name
    "en": [
        "Hey there Deepak", "Hey there Deepa",
        "Hello Deepak", "Hello Deepa",
        "Hey Deepak", "Hey Deepa",
        "Hi there Deepak", "Hi there Deepa",
    ],
    
    # NEPALI: "Namaste" + Name (नमस्ते + नाम)
    "ne": [
        "नमस्ते दीपक", "नमस्ते दीपा",           # Namaste Deepak/Deepa
        "नमस्कार दीपक", "नमस्कार दीपा",         # Namaskar Deepak/Deepa
    ],
    
    # MAITHILI: "Pranam" / "Ram-Ram" + Name
    "mai": [
        "प्रणाम दीपक", "प्रणाम दीपा",           # Pranam Deepak/Deepa
        "राम राम दीपक", "राम राम दीपा",         # Ram-Ram Deepak/Deepa
        "जय राम दीपक", "जय राम दीपा",           # Jai Ram Deepak/Deepa
    ]
}

# Voices for TTS generation
VOICES = {
    "en": ["en-US-AriaNeural", "en-US-GuyNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural"],
    "ne": ["ne-NP-SagarNeural", "ne-NP-HemkalaNeural"],
    "mai": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"]  # Hindi voices for Maithili
}

# ============================================================
# HARD NEGATIVE SENTENCES
# ============================================================
# These are designed to teach the model what is NOT a wake word.
# CRITICAL: Include the name ALONE (without prefix) as a negative!
# This prevents the model from triggering on just "Deepak" or "Deepa".

NEGATIVE_SENTENCES_EN = [
    # === CRITICAL: Name without prefix (should NOT trigger) ===
    "Deepak", "Deepa",
    "Deepak is here", "Deepa is here",
    "Call Deepak", "Call Deepa",
    "Where is Deepak?", "Where is Deepa?",
    "Deepak said so", "Deepa said so",
    "Talk to Deepak", "Talk to Deepa",
    "Deepak please help", "Deepa come here",
    "I saw Deepak", "I met Deepa",
    
    # === Wrong prefix (should NOT trigger) ===
    "Okay Deepak", "Okay Deepa",
    "Listen Deepak", "Listen Deepa",
    "Wake up Deepak", "Wake up Deepa",
    "Good morning Deepak", "Good morning Deepa",
    
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
    # === CRITICAL: Name without prefix (should NOT trigger) ===
    "दीपक", "दीपा",
    "दीपक आउनुस्", "दीपा आउनुस्",
    "दीपक कहाँ छ?", "दीपा कहाँ छ?",
    "दीपक सुन", "दीपा सुन",
    "दीपक उठ", "दीपा उठ",
    
    # === Wrong greeting (should NOT trigger) ===
    "ए दीपक", "ए दीपा",                # Hey Deepak (casual, not Namaste)
    "ओ दीपक", "ओ दीपा",                # Oh Deepak
    "अरे दीपक", "अरे दीपा",             # Are Deepak
    "दीपक सुन्नुहोस्", "दीपा सुन्नुहोस्",  # Deepak listen
    
    # === Similar names ===
    "दिलिप", "दिनेश", "दिपेन्द्र",
    
    # === Random Nepali ===
    "बत्ती बाल्नुहोस्", "गीत बजाउनुस्", 
    "खाना खानुभयो?", "घर जान्छु",
    "मलाई थाहा छैन", "भोलि भेटौला"
]

NEGATIVE_SENTENCES_MAI = [
    # === CRITICAL: Name without prefix (should NOT trigger) ===
    "दीपक", "दीपा",
    "दीपक सुनु", "दीपा सुनु",
    "दीपक एहिठाम आउ", "दीपा एहिठाम आउ",
    "है दीपक", "है दीपा",
    
    # === Wrong greeting (should NOT trigger) ===
    "ए दीपक", "ए दीपा",                # Hey (not Pranam)
    "अरे दीपक", "अरे दीपा",
    "हम दीपक बजेलहुँ", "हम दीपा बजेलहुँ",
    
    # === Random Maithili ===
    "हम घर जा रहल छी", "खाना खा लेलहुँ?",
    "आजुक दिन नीक अछि", "राति भ गेल"
]
