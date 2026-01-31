# Strict Grammar Phrases for Wake Word Detection
# Names: Deepak (Male), Deepa (Female)

PHRASES = {
    "en": [
        "Hello Deepak", "Hello Deepa",
        "Hey Deepak", "Hey Deepa",
        "Hi Deepak", "Hi Deepa",
        "Wake up Deepak", "Wake up Deepa",
        "Okay Deepak", "Okay Deepa",
        "Listen Deepak", "Listen Deepa",
        "Greetings Deepak", "Greetings Deepa",
        "Good morning Deepak", "Good morning Deepa"
    ],
    "ne": [
        "ए दीपक", "ए दीपा",          # Hey Deepak/Deepa
        "नमस्ते दीपक", "नमस्ते दीपा",  # Namaste Deepak/Deepa
        "नमस्कार दीपक", "नमस्कार दीपा", # Namaskar Deepak/Deepa
        "दीपक उठ", "दीपा उठ",        # Deepak/Deepa wake up (Informal)
        "दीपक जाग्नुहोस्", "दीपा जाग्नुहोस्", # Deepak/Deepa wake up (Formal)
        "दीपक सुन्नुहोस्", "दीपा सुन्नुहोस्", # Listen Deepak/Deepa (Formal)
        "ओ दीपक", "ओ दीपा",          # Oh Deepak/Deepa
        "सुन दीपक", "सुन दीपा"       # Listen Deepak/Deepa (Informal)
    ],
    "mai": [
        "दीपक सुनु", "दीपा सुनु",      # Listen Deepak/Deepa
        "अहाँ सुनु दीपक", "अहाँ सुनु दीपा", # You listen Deepak/Deepa (Respectful)
        "हे दीपक", "हे दीपा",          # Oh Deepak/Deepa
        "दीपक जगु", "दीपा जगु",        # Wake up Deepak/Deepa
        "प्रणाम दीपक", "प्रणाम दीपा",   # Pranam Deepak/Deepa
        "नमस्कार दीपक", "नमस्कार दीपा", # Namaskar Deepak/Deepa
        "दीपक उठु", "दीपा उठु"         # Deepak/Deepa wake up
    ]
}

# Voices map to ensure correct accent/language
# Using Hindi voices for Maithili as distinct Maithili TTS is rare
VOICES = {
    "en": ["en-US-AriaNeural", "en-US-GuyNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural"],
    "ne": ["ne-NP-SagarNeural", "ne-NP-HemkalaNeural"],
    "mai": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"]
}

# Common sentences for "Hard Negatives" (Background noise)
# Sentences that sound like speech but are NOT wake words
# Common sentences for "Hard Negatives" (Background noise)
# Includes phonetically similar words to test model boundaries ("Tricky Negatives")
# --- ROBUST NEGATIVE SAMPLES (Hard Protocol) ---
# These are designed to minimize false positives by training the model on what ISN'T a wake word.

NEGATIVE_SENTENCES_EN = [
    # 1. Phonetically Similar / "Tricky" Words (Deep-, De-, -pa, -pak)
    "The water is deep.", "Look at the deep end.", "Go deeper.",
    "The park is open.", "Car park.", "Theme park.",
    "Depart now.", "Department store.", "Deposit money.", "Depend on me.",
    "Debug this code.", "Defeat the boss.", "Defense mechanism.", "Default settings.",
    "Do pack your bags.", "Backpack.", "Ice pack.", "Six pack.",
    "Dee is here.", "Letter D.", "3D movie.",
    "The path is clear.", "Pass me the salt.", "Past time.",
    "Dip the cookie.", "Dipper.", "Diplomacy.",
    
    # 2. Common Smart Home Commands (Should NOT trigger)
    "Turn on the lights.", "Turn off the fan.", "Play music.", "Stop music.",
    "Next song.", "Volume up.", "Volume down.", "Set an alarm.",
    "What's the weather?", "Tell me a joke.", "Open YouTube.",
    "Call Mom.", "Send a message.", "Remind me later.",
    
    # 3. General Daily Conversation (Random)
    "How are you doing today?", "I am going to the office.", "Where is the bus stop?",
    "This food is delicious.", "I need a break.", "Let's watch a movie.",
    "Did you finish the project?", "I am very tired.", "Good night everyone.",
    "See you tomorrow.", "Have a nice day.", "What is your name?",
    "I don't know.", "Maybe later.", "Yes, of course.", "No problem."
]

NEGATIVE_SENTENCES_NE = [
    # 1. Phonetically Similar / "Tricky" Words
    "दिउँसो भेटौला", "दिदी आउनुभयो", "दिन राम्रो छ", "दिवस मनाउने", # Di- sounds
    "पक्का हो?", "पकौडा खानु", "पख है पख", "पखाल्नु", # Pak- sounds
    "दिपकल", "दिपेन्द्र", "दिलिप", "दिनेश", # Similar names
    "पापा आउनुभयो", "पानी दिनुस्", "पाल्पा जाने", # Pa- sounds
    
    # 2. Common Commands / Conversation
    "बत्ती बाल्नुहोस्", "बत्ती निभाउनुहोस्", "गीत बजाउनुस्", "चुप लाग",
    "ढोका खोल", "झ्याल बन्द गर", "खाना खानु भयो?",
    "मलाई सन्चो छैन", "आज अफिस बिदा छ", "पानी पर्यो",
    "बस आयो", "घर जान्छु", "के गर्दै हुनुहुन्छ?",
    "मलाई थाहा छैन", "भोलि भेटौला", " शुभ रात्री"
]

NEGATIVE_SENTENCES_MAI = [
    # Maithili Specific Negatives (Using Hindi/Maithili mix)
    "हम घर जा रहल छी", "खाना खा लेलहुँ?", "कि भेल?",
    "आजुक दिन नीक अछि", "हमरा नहि बुझना जाइत अछि",
    "दिदिया एल छथि", "दिन नीक छै", # Di- sounds
    "पानि पीय", "पावनि तिहार", # Pa- sounds
    "दरभंगा जेबै", "जनकपुर धाम",
    "राति भ गेल", "सुइत रहल छी"
]
