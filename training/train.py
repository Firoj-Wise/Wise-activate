import os
import shutil
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
from sklearn.utils import class_weight

SAMPLE_RATE = 16000
DURATION = 1
SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 13
N_FFT = 512 
HOP_LENGTH = 160

BASE_DIR = Path(__file__).resolve().parent.parent
POSITIVE_DIR = BASE_DIR / "data" / "wake"
NEGATIVE_DIR = BASE_DIR / "data" / "background"
USER_NEGATIVE_DIR = BASE_DIR / "data" / "user_negatives"
MODEL_SAVE_PATH = BASE_DIR / "training" / "wakeword_model.h5"
TFLITE_SAVE_PATH = BASE_DIR / "web" / "wakeword_model.tflite"

# Defined Classes
CLASSES = ["Background", "Deepa", "Deepak"]

def normalize_audio(y):
    """
    Normalizes audio to Peak 1.0.
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / (peak + 1e-6)
    return y

def augment_audio(y, background_files=[]):
    """
    Applies random augmentations: Pitch, Speed, Noise, and Background Mixing.
    """
    options = ["pitch", "speed", "noise", "reverb", "mix", "none"]
    weights = [0.15, 0.15, 0.15, 0.2, 0.25, 0.1] # Added Reverb
    
    aug_type = np.random.choice(options, p=weights)
    
    if aug_type == "pitch":
        n_steps = np.random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)
    elif aug_type == "speed":
        rate = np.random.uniform(0.85, 1.15)
        y = librosa.effects.time_stretch(y, rate=rate)
    elif aug_type == "noise":
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape)
    elif aug_type == "reverb":
        y = add_reverb(y)
    elif aug_type == "mix" and len(background_files) > 0:
        # Load a random background file
        bg_file = np.random.choice(background_files)
        try:
            bg, _ = librosa.load(bg_file, sr=SAMPLE_RATE)
            bg = librosa.util.fix_length(bg, size=len(y))
            
            # Adjust volume of background (SNR between 5dB and 15dB)
            bg_vol = np.random.uniform(0.1, 0.4) 
            y = y + (bg * bg_vol)
        except:
            pass # Fail silently and return original if load fails
        
    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = librosa.util.fix_length(y, size=SAMPLES)
        
    y = normalize_audio(y)
    return y

def add_reverb(y):
    """
    Simulates Room Impulse Response (RIR) using exponential decay noise.
    The 'Secret Sauce' for Sim-to-Real gap.
    """
    # 1. Generate impulse response
    # Random length (0.2s to 0.5s)
    rir_len = int(SAMPLE_RATE * np.random.uniform(0.1, 0.3))
    t = np.linspace(0, 1, rir_len)
    # Exponential decay
    decay = np.exp(-t * np.random.uniform(5, 15)) 
    # White noise * decay
    rir = np.random.normal(0, 1, rir_len) * decay
    rir = rir / np.max(np.abs(rir)) # Normalize RIR
    
    # 2. Convolve
    # Scipy convolve is slow, let's use FFT based or simple mix?
    # For speed in training loop, we might want to just mix a delayed version?
    # Simple "Echo":
    delay = int(SAMPLE_RATE * np.random.uniform(0.01, 0.05))
    decay_factor = np.random.uniform(0.1, 0.4)
    
    if len(y) > delay:
        echo = np.roll(y, delay)
        echo[:delay] = 0
        y = y + echo * decay_factor
        
    return normalize_audio(y)

def apply_spec_augment(mfcc):
    """
    Zeros out blocks of time or frequency. 
    Forces model to rely on context.
    Input: (Time, MFCC_Feats) -> (101, 13)
    """
    # Time Masking
    if np.random.rand() < 0.5:
        t_mask = np.random.randint(5, 15)
        t_start = np.random.randint(0, mfcc.shape[0] - t_mask)
        mfcc[t_start:t_start+t_mask, :] = 0
        
    # Freq Masking
    if np.random.rand() < 0.5:
        f_mask = np.random.randint(1, 4)
        f_start = np.random.randint(0, mfcc.shape[1] - f_mask)
        mfcc[:, f_start:f_start+f_mask] = 0
        
    return mfcc

def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # 1. Trim Silence using dB threshold
        # This removes leading/trailing silence so we focus on the speech
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y_trimmed) == 0:
            return None
            
        # 2. Center Audio in the 1 second window
        if len(y_trimmed) > SAMPLES:
            # If too long, take the center
            start = (len(y_trimmed) - SAMPLES) // 2
            y_final = y_trimmed[start:start+SAMPLES]
        else:
            # If too short, pad symmetrically
            padding = SAMPLES - len(y_trimmed)
            offset = padding // 2
            y_final = np.pad(y_trimmed, (offset, padding - offset), 'constant')
            
        # 3. Normalize (Crucial for User vs TTS balance)
        y_final = normalize_audio(y_final)
        
        return y_final
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_mfcc(y):
    """
    Extracts MFCCs matching the client-side Meyda configuration.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=40)
    return mfcc.T

def load_dataset():
    X, y = [], []
    
    classes = ["Background", "Deepa", "Deepak"]
    class_indices = {"deepa": 1, "deepak": 2}
    num_classes = 3

    # Pre-load background file list for mixing
    print("Indexing background files for mixing...")
    bg_files = list(NEGATIVE_DIR.rglob("*.wav"))
    
    print("Loading positive samples...")
    # Load from language folders
    for lang_code, idx in class_indices.items():
        lang_dir = POSITIVE_DIR / lang_code
        if not lang_dir.exists():
            print(f"Warning: Directory {lang_dir} does not exist. Skipping.")
            continue
            
        files = list(lang_dir.rglob("*.wav"))
        print(f"Loading {len(files)} samples for {classes[idx]} ({lang_code})...")
        
        for file in files:
            audio = load_audio(file)
            if audio is not None:
                # Original
                X.append(extract_mfcc(audio))
                y.append(idx)
                # SUPER HEAVY AUGMENTATION (50x)
                # We want these real-world samples to be a significant chunk of the dataset
                # CRITICAL CHANGE: Do NOT add background noise to user samples. They already have it!
                # Only augment Pitch/Speed.
                for _ in range(50):
                    aug_audio = augment_audio(audio, background_files=[]) 
                    mfcc = extract_mfcc(aug_audio)
                    mfcc = apply_spec_augment(mfcc) # SpecAugment for Positives
                    X.append(mfcc)
                    y.append(idx)

    print("Loading USER NEGATIVES (Hard Negatives)...")
    user_neg_files = list(USER_NEGATIVE_DIR.rglob("*.wav"))
    print(f"Found {len(user_neg_files)} user negative samples.")
    
    for file in user_neg_files:
        audio = load_audio(file)
        if audio is not None:
             # Original
             X.append(extract_mfcc(audio))
             y.append(0) # Class 0
             
             # HEAVY Augmentation (50x) - NO BACKGROUND MIXING
             # Treat same as user positives to teach "Voice != Trigger"
             for _ in range(50):
                 aug_audio = augment_audio(audio, background_files=[]) 
                 mfcc = extract_mfcc(aug_audio)
                 mfcc = apply_spec_augment(mfcc) # THE SECRET SAUCE #2
                 X.append(mfcc)
                 y.append(0)

    print("Loading negative samples and applying HEAVY augmentation...")
    # Background / Negatives are Class 0
    neg_files = list(NEGATIVE_DIR.rglob("*.wav"))
    print(f"Found {len(neg_files)} base negative samples.")
    
    for file in neg_files:
        audio = load_audio(file)
        if audio is not None:
             # Original
            X.append(extract_mfcc(audio))
            y.append(0)
            
            # HEAVY Augmentation for Negatives (10x) to balance dataset
            # Positives: ~2800 * 4 = ~11,200
            # Negatives needs to be similar. ~1200 * 10 = ~12,000
            for _ in range(10):
                aug_audio = augment_audio(audio, background_files=bg_files)
                mfcc = extract_mfcc(aug_audio)
                mfcc = apply_spec_augment(mfcc) # SpecAugment for Negatives
                X.append(mfcc)
                y.append(0)

    print(f"Final Dataset Size: {len(X)}")
    return np.array(X), np.array(y), num_classes

def build_model(input_shape, num_classes):
    """
    Constructs FCN.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Block 1: Capture broad features
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2, 2), name='pool_1'), # (50, 6)
        
        # Block 2: Refine features (High Time Resolution)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'),
        tf.keras.layers.BatchNormalization(),
        # CRITICAL CHANGE: Pool Frequency (y) only, Keep Time (x) intact
        # This helps distinguish "Deepa" vs "Deepak" (the 'k' is a short time event)
        tf.keras.layers.MaxPool2D((1, 2), name='pool_2'), # (50, 3)
        
        # Block 3: Deep features
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_3'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'), # Prevent overfitting to TTS artifacts
        
        # Block 4: Compression
        tf.keras.layers.Conv2D(64, (1, 1), activation='relu', name='dense_conv'),
        tf.keras.layers.Dropout(0.5, name='dropout_2'),
        
        # Block 5: Class Projection
        tf.keras.layers.Conv2D(num_classes, (1, 1), activation='linear', name='output_conv'),
        
        # Aggregation: Average over time/freq
        tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool'),
        
        # Activation
        tf.keras.layers.Softmax(name='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("--- Starting Adaptive Training Pipeline ---")
    
    X, y, num_classes = load_dataset()
    print(f"Dataset Info: X shape={X.shape}, y shape={y.shape}")
    print(f"Number of Classes: {num_classes}")
    
    # SHUFFLE DATA (Crucial for correct Validation Split)
    # Without this, val_split takes the last 20% (which is all Background), leading to 0% accuracy.
    print("Shuffling dataset...")
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    X = X[..., np.newaxis]
    input_shape = X.shape[1:]
    
    model = build_model(input_shape, num_classes)
    model.summary()
    
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Boost Background Class (0) to reduce false positives
    # REMOVED: 10x boost was causing the model to ignore User Samples (Messy Audio -> Background)
    # if 0 in class_weight_dict:
    #     class_weight_dict[0] *= 10.0
    #     print(f"Boosted Background Weight: {class_weight_dict[0]}")

    print(f"Computed Class Weights: {class_weight_dict}")

    print("Training model...")
    history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1, class_weight=class_weight_dict)
    
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)

    # --- Validation Report ---
    print("\n--- Generating Confusion Matrix (Validation) ---")
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        
        # 1. Re-split to match training (not perfect but good for approximation)
        # Ideally we'd use train_test_split but here we used validation_split in fit()
        # So we'll evaluate on the WHOLE dataset (train+val) just to see if it learned.
        # Speed: Predict in batches
        y_pred_probs = model.predict(X, batch_size=32, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        cm = confusion_matrix(y, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=CLASSES))
        
        print(f"Total Background (0): {sum(y==0)}")
        print(f"Total Deepa (1): {sum(y==1)}")
        print(f"Total Deepak (2): {sum(y==2)}")

    except Exception as e:
        print(f"Skipping evaluation report: {e}")
    # -------------------------

    # Plotting Training History
    try:
        import matplotlib.pyplot as plt
        print("Generating training graph...")
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        
        graph_path = BASE_DIR / "training" / "training_history.png"
        plt.tight_layout()
        plt.savefig(graph_path)
        print(f"Saved training graph to {graph_path}")
    except Exception as e:
        print(f"Could not generate graph: {e}")
    
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    
    TFLITE_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TFLITE_SAVE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {TFLITE_SAVE_PATH}")
    print("--- Done ---")

if __name__ == "__main__":
    main()

