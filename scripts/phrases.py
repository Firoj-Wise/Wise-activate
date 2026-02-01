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
        "Good morning Deepak", "Good morning Deepa",
        "Are you there Deepak?", "Are you there Deepa?",
        "Can you hear me Deepak?", "Can you hear me Deepa?",
        "Deepak help me", "Deepa help me",
        "Deepak stop", "Deepa stop",
        "Deepak please", "Deepa please"
    ],
    "ne": [
        # Greetings & General
        "नमस्ते दीपक", "नमस्ते दीपा",          # Namaste
        "नमस्कार दीपक", "नमस्कार दीपा",        # Namaskar
        "शुभ प्रभात दीपक", "शुभ प्रभात दीपा",  # Good Morning
        "शुभ रात्रि दीपक", "शुभ रात्रि दीपा",  # Good Night
        "स्वागत छ दीपक", "स्वागत छ दीपा",      # Welcome
        "तपाईंलाई कस्तो छ दीपक?", "तपाईंलाई कस्तो छ दीपा?", # How are you?
        "के छ खबर दीपक?", "के छ खबर दीपा?",    # What's up?
        "सबै ठिक छ दीपक?", "सबै ठिक छ दीपा?",  # Is everything fine?
        "फेरि भेटौला दीपक", "फेरि भेटौला दीपा", # See you again
        "धन्यवाद दीपक", "धन्यवाद दीपा",        # Thank you

        # Commands & Wake Words
        "ए दीपक", "ए दीपा",                  # Hey
        "ओ दीपक", "ओ दीपा",                  # Oh
        "दीपक सुन्नुहोस्", "दीपा सुन्नुहोस्",        # Listen (Formal)
        "दीपक सुन", "दीपा सुन",              # Listen (Informal)
        "दीपक यता आउनुहोस्", "दीपा यता आउनुहोस्", # Come here
        "दीपक उठ्नुहोस्", "दीपा उठ्नुहोस्",        # Wake up
        "दीपक जाग्नुहोस्", "दीपा जाग्नुहोस्",       # Wake up (Formal)
        "दीपक छिटो आउनुस्", "दीपा छिटो आउनुस्",   # Come fast
        "दीपक काम गर", "दीपा काम गर",        # Do work
        "दीपक चुप लाग", "दीपा चुप लाग",       # Be quiet
        "दीपक बाहिर जाउ", "दीपा बाहिर जाउ",    # Go out
        "दीपक भित्र आउ", "दीपा भित्र आउ",      # Come in
        "दीपक ढोका खोल", "दीपा ढोका खोल",     # Open door
        "दीपक बत्ति बाल", "दीपा बत्ति बाल",     # Turn on light
        
        # Urgent / Triggers
        "हेलो दीपक", "हेलो दीपा",
        "अरे दीपक", "अरे दीपा",
        "ल दीपक", "ल दीपा"                   # Okay Deepak
    ],
    "mai": [
        # Greetings & General
        "प्रणाम दीपक", "प्रणाम दीपा",       # Pranam (Formal Hello)
        "नमस्कार दीपक", "नमस्कार दीपा",     # Namaskar
        "अहाँ फेर सँ कहू दीपक", "अहाँ फेर सँ कहू दीपा", # Say it again
        "की हाल चाल अछि दीपक?", "की हाल चाल अछि दीपा?", # How are you?
        "अहाँ कोना छी दीपक?", "अहाँ कोना छी दीपा?",    # How are you doing?
        "सभ कुशल मंगल दीपक?", "सभ कुशल मंगल दीपा?",   # Is everything fine?
        "स्वागत अछि दीपक", "स्वागत अछि दीपा",         # Welcome
        "शुभ प्रभात दीपक", "शुभ प्रभात दीपा",         # Good Morning
        "शुभ रात्रि दीपक", "शुभ रात्रि दीपा",         # Good Night
        
        # Commands (Wake Words Context)
        "दीपक सुनु", "दीपा सुनु",          # Listen
        "अहाँ सुनु दीपक", "अहाँ सुनु दीपा",     # You listen (Respectful)
        "हे दीपक", "हे दीपा",              # Oh Deepak/Deepa
        "दीपक उठु", "दीपा उठु",            # Wake up
        "दीपक जगु", "दीपा जगु",            # Wake up (Formal)
        "दीपक कनी सुनु", "दीपा कनी सुनु",     # Listen a bit
        "दीपक एहिठाम आउ", "दीपा एहिठाम आउ",   # Come here
        "दीपक घर जाउ", "दीपा घर जाउ",        # Go home
        "दीपक काज करू", "दीपा काज करू",       # Do work
        "दीपक रुकु", "दीपा रुकु",            # Stop
        "दीपक बदि जाउ", "दीपा बदि जाउ",       # Go away (Respectful)
        
        # Short / Quick Triggers
        "ए दीपक", "ए दीपा",
        "अरे दीपक", "अरे दीपा",
        "हँ दीपक", "हँ दीपा",
        "मैथिली बाजू दीपक", "मैथिली बाजू दीपा" # Speak Maithili
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
