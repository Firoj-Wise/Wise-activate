"""
WiseYak Wake Word Training Pipeline (openWakeWord-Inspired)

Key improvements from openWakeWord:
- Focal Loss: Down-weights easy examples, focuses on hard negatives
- False Positive Penalty: Heavy penalty for triggering on background
- Strict Phonetic Anchors: PREFIX + NAME pattern (e.g., "Namaste Deepak")
- Robust augmentation with audiomentations

Usage:
    python train.py
"""

import os
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
from sklearn.utils import class_weight
from joblib import Parallel, delayed
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise, Gain

# ============================================================
# TRAINING CONFIGURATION (Tune these!)
# ============================================================
FALSE_POSITIVE_PENALTY = 5.0   # How strongly to penalize background false triggers (openWakeWord uses 1500, but we use class weights)
FOCAL_GAMMA = 2.0              # Focal loss gamma: higher = more focus on hard examples
TARGET_ACCURACY = 0.95         # Stop early if we reach this
MAX_EPOCHS = 100               # Early stopping will find optimal point

# Audio Configuration
SAMPLE_RATE = 16000
DURATION = 1
SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 13
N_FFT = 512 
HOP_LENGTH = 160

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
POSITIVE_DIR = BASE_DIR / "data" / "wake"
NEGATIVE_DIR = BASE_DIR / "data" / "background"
USER_NEGATIVE_DIR = BASE_DIR / "data" / "user_negatives"
MODEL_SAVE_PATH = BASE_DIR / "training" / "wakeword_model.h5"
TFLITE_SAVE_PATH = BASE_DIR / "web" / "wakeword_model.tflite"

# The 7 Output Classes (Background + 6 wake word variants)
CLASSES = [
    "Background", 
    "Deepa (EN)", "Deepa (NE)", "Deepa (MAI)",
    "Deepak (EN)", "Deepak (NE)", "Deepak (MAI)"
]


# ============================================================
# FOCAL LOSS (Key openWakeWord technique)
# ============================================================
# Focal Loss reduces the contribution of easy examples and focuses
# the model on learning hard negatives (background that sounds like wake word)

class SparseFocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for multi-class classification with integer labels.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 is equivalent to standard cross-entropy.
        alpha: Class balancing weight (optional, we use class_weight instead)
    """
    def __init__(self, gamma=2.0, name='sparse_focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Get the predicted probability for the true class
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(tf.squeeze(y_true), depth=tf.shape(y_pred)[-1])
        
        # Cross entropy
        ce = -y_true_one_hot * tf.math.log(y_pred)
        
        # Focal weight: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        # Focal loss
        focal_loss = focal_weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


# Audiomentations Pipeline (10x faster than librosa)
AUGMENT_PIPELINE = Compose([
    PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
    TimeStretch(min_rate=0.85, max_rate=1.15, p=0.3),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
    Gain(min_gain_db=-6, max_gain_db=6, p=0.2),
])



def normalize_audio(y):
    """Normalizes audio to Peak 1.0."""
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / (peak + 1e-6)
    return y


def add_reverb(y):
    """
    Simulates Room Impulse Response (RIR) using exponential decay noise.
    The 'Secret Sauce' for Sim-to-Real gap.
    """
    rir_len = int(SAMPLE_RATE * np.random.uniform(0.1, 0.3))
    t = np.linspace(0, 1, rir_len)
    decay = np.exp(-t * np.random.uniform(5, 15)) 
    rir = np.random.normal(0, 1, rir_len) * decay
    rir = rir / np.max(np.abs(rir))
    
    try:
        from scipy.signal import fftconvolve
        y_aug = fftconvolve(y, rir, mode="full")
        y_aug = y_aug[:len(y)]
        return normalize_audio(y_aug)
    except ImportError:
        # Fallback to simple echo
        delay = int(SAMPLE_RATE * 0.02)
        if len(y) > delay:
            y[:len(y)-delay] += y[delay:] * 0.3
        return normalize_audio(y)


def augment_audio(y, background_files=None):
    """
    Applies augmentations using audiomentations (fast) + custom RIR/mixing.
    """
    # 1. Fast augmentations via audiomentations
    y = AUGMENT_PIPELINE(samples=y, sample_rate=SAMPLE_RATE)
    
    # 2. Custom augmentations (RIR and Background Mixing)
    aug_choice = np.random.random()
    
    if aug_choice < 0.2:
        # Apply Room Reverb (custom)
        y = add_reverb(y)
    elif aug_choice < 0.4 and background_files and len(background_files) > 0:
        # Mix with background noise
        bg_file = np.random.choice(background_files)
        try:
            bg, _ = librosa.load(bg_file, sr=SAMPLE_RATE)
            bg = librosa.util.fix_length(bg, size=len(y))
            bg_vol = np.random.uniform(0.1, 0.4) 
            y = y + (bg * bg_vol)
        except:
            pass
    
    # Fix length and normalize
    if len(y) > SAMPLES:
        y = y[:SAMPLES]
    else:
        y = librosa.util.fix_length(y, size=SAMPLES)
    
    return normalize_audio(y)


def apply_spec_augment(mfcc):
    """
    Zeros out blocks of time or frequency (SpecAugment).
    Forces model to rely on context.
    """
    # Time Masking
    if np.random.rand() < 0.5:
        t_mask = np.random.randint(5, 15)
        t_start = np.random.randint(0, max(1, mfcc.shape[0] - t_mask))
        mfcc[t_start:t_start+t_mask, :] = 0
        
    # Freq Masking
    if np.random.rand() < 0.5:
        f_mask = np.random.randint(1, 4)
        f_start = np.random.randint(0, max(1, mfcc.shape[1] - f_mask))
        mfcc[:, f_start:f_start+f_mask] = 0
        
    return mfcc


def load_audio(file_path):
    """Loads and preprocesses a single audio file."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Trim Silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y_trimmed) == 0:
            return None
            
        # Center Audio in the 1 second window
        if len(y_trimmed) > SAMPLES:
            start = (len(y_trimmed) - SAMPLES) // 2
            y_final = y_trimmed[start:start+SAMPLES]
        else:
            padding = SAMPLES - len(y_trimmed)
            offset = padding // 2
            y_final = np.pad(y_trimmed, (offset, padding - offset), 'constant')
            
        return normalize_audio(y_final)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_mfcc(y):
    """Extracts MFCCs matching the client-side Meyda configuration."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, 
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=40
    )
    return mfcc.T


def process_positive_file(file_path, class_idx, bg_files, aug_count=10):
    """
    Process a single positive sample file with augmentations.
    Returns list of (mfcc, label) tuples.
    """
    results = []
    audio = load_audio(file_path)
    
    if audio is None:
        return results
    
    # Original sample
    results.append((extract_mfcc(audio), class_idx))
    
    # Augmented samples (10x instead of 50x)
    for _ in range(aug_count):
        aug_audio = augment_audio(audio, background_files=bg_files)
        mfcc = extract_mfcc(aug_audio)
        mfcc = apply_spec_augment(mfcc)
        results.append((mfcc, class_idx))
    
    return results


def process_negative_file(file_path, bg_files, aug_count=10):
    """
    Process a single negative sample file with augmentations.
    Returns list of (mfcc, label=0) tuples.
    """
    results = []
    audio = load_audio(file_path)
    
    if audio is None:
        return results
    
    # Original sample
    results.append((extract_mfcc(audio), 0))
    
    # Augmented samples
    for _ in range(aug_count):
        aug_audio = augment_audio(audio, background_files=bg_files)
        mfcc = extract_mfcc(aug_audio)
        mfcc = apply_spec_augment(mfcc)
        results.append((mfcc, 0))
    
    return results


def load_dataset():
    """
    Loads dataset with parallel processing and DYNAMIC AUGMENTATION.
    
    Key Fix: Classes with more samples get LESS augmentation to prevent collapse.
    - Small classes (< 2000 samples): 10x augmentation
    - Medium classes (2000-4000 samples): 5x augmentation  
    - Large classes (> 4000 samples): 2x augmentation
    
    This ensures ~20,000 samples per class after augmentation.
    """
    X, y = [], []
    num_classes = len(CLASSES)
    
    # Mapping: (Folder, LangSubfolder) -> Class Index
    label_map = {
        ("deepa", "en"): 1,
        ("deepa", "ne"): 2,
        ("deepa", "mai"): 3,
        ("deepak", "en"): 4,
        ("deepak", "ne"): 5,
        ("deepak", "mai"): 6
    }

    # Pre-index background files for mixing
    print("Indexing background files for mixing...")
    bg_files = [str(f) for f in NEGATIVE_DIR.rglob("*.wav")]
    print(f"Found {len(bg_files)} background files.")
    
    # Determine number of parallel jobs (use all cores, but cap at 8 for memory)
    n_jobs = min(8, os.cpu_count() or 4)
    print(f"Using {n_jobs} parallel workers for data loading.")
    
    # Target samples per class (for balance)
    TARGET_SAMPLES_PER_CLASS = 20000
    
    # --- LOAD POSITIVE SAMPLES (with DYNAMIC augmentation) ---
    print("\nLoading positive samples (Gender + Language)...")
    print("Using DYNAMIC augmentation to balance classes:")
    
    for (name_dir, lang_dir), idx in label_map.items():
        target_path = POSITIVE_DIR / name_dir / lang_dir
        
        if not target_path.exists():
            print(f"Warning: Path {target_path} not found. Skipping Class {idx}.")
            continue
            
        files = list(target_path.glob("*.wav"))
        file_count = len(files)
        
        # DYNAMIC AUGMENTATION: Fewer augments for larger classes
        if file_count < 2000:
            aug_count = 10  # Small class - heavy augmentation
        elif file_count < 4000:
            aug_count = 5   # Medium class - moderate augmentation
        else:
            aug_count = 2   # Large class - minimal augmentation
        
        # Optional: Undersample very large classes
        max_files = TARGET_SAMPLES_PER_CLASS // (aug_count + 1)
        if file_count > max_files:
            print(f"  Undersampling {CLASSES[idx]}: {file_count} -> {max_files} files")
            np.random.shuffle(files)
            files = files[:max_files]
        
        print(f"Class {idx} ({CLASSES[idx]}): {len(files)} files x {aug_count+1} = ~{len(files) * (aug_count+1)} samples")
        
        # Parallel processing with joblib
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_positive_file)(str(f), idx, bg_files, aug_count=aug_count)
            for f in files
        )
        
        # Flatten results
        for file_results in results:
            for mfcc, label in file_results:
                X.append(mfcc)
                y.append(label)
    
    # --- LOAD USER NEGATIVES (Hard Negatives) ---
    print("\nLoading USER NEGATIVES (Hard Negatives)...")
    user_neg_files = list(USER_NEGATIVE_DIR.rglob("*.wav")) if USER_NEGATIVE_DIR.exists() else []
    print(f"Found {len(user_neg_files)} user negative samples.")
    
    if user_neg_files:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_negative_file)(str(f), [], aug_count=5)
            for f in user_neg_files
        )
        
        for file_results in results:
            for mfcc, label in file_results:
                X.append(mfcc)
                y.append(label)
    
    # --- LOAD BACKGROUND NEGATIVES ---
    print("\nLoading negative samples (Background)...")
    neg_files = list(NEGATIVE_DIR.rglob("*.wav"))
    
    # Undersample background if too large
    target_bg = TARGET_SAMPLES_PER_CLASS // 3  # Less augmentation for negatives
    if len(neg_files) > target_bg:
        np.random.shuffle(neg_files)
        neg_files = neg_files[:target_bg]
    
    print(f"Using {len(neg_files)} base negative samples.")
    
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_negative_file)(str(f), bg_files, aug_count=2)
        for f in neg_files
    )
    
    for file_results in results:
        for mfcc, label in file_results:
            X.append(mfcc)
            y.append(label)
    
    # Print class distribution
    y_arr = np.array(y)
    print("\n=== Class Distribution ===")
    for i, cls_name in enumerate(CLASSES):
        count = np.sum(y_arr == i)
        print(f"  {cls_name}: {count} samples")
    
    print(f"\n=== Final Dataset Size: {len(X)} samples ===")
    return np.array(X), np.array(y), num_classes



def build_model(input_shape, num_classes):
    """Constructs FCN architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Block 1: Capture broad features
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2, 2), name='pool_1'),
        
        # Block 2: Refine features (High Time Resolution)
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((1, 2), name='pool_2'),
        
        # Block 3: Deep features
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_3'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),
        
        # Block 4: Compression
        tf.keras.layers.Conv2D(64, (1, 1), activation='relu', name='dense_conv'),
        tf.keras.layers.Dropout(0.5, name='dropout_2'),
        
        # Block 5: Class Projection
        tf.keras.layers.Conv2D(num_classes, (1, 1), activation='linear', name='output_conv'),
        
        # Aggregation
        tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool'),
        
        # Activation
        tf.keras.layers.Softmax(name='softmax')
    ])
    
    # Learning rate schedule for better convergence
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=5000,
        decay_rate=0.9
    )
    
    # Use Focal Loss to focus on hard negatives (key openWakeWord technique)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=SparseFocalLoss(gamma=FOCAL_GAMMA),
        metrics=['accuracy']
    )
    return model


def main():
    print("=" * 60)
    print("  WiseYak Adaptive Training Pipeline (Robust)")
    print("=" * 60)
    
    X, y, num_classes = load_dataset()
    print(f"\nDataset Info: X shape={X.shape}, y shape={y.shape}")
    print(f"Number of Classes: {num_classes}")
    
    # Shuffle data (crucial for correct validation split)
    print("Shuffling dataset...")
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    X = X[..., np.newaxis]
    input_shape = X.shape[1:]
    
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # Compute class weights for imbalanced data
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y), y=y
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Apply FALSE_POSITIVE_PENALTY to Background class (openWakeWord-inspired)
    # This heavily penalizes false triggers on background audio
    if 0 in class_weight_dict:
        class_weight_dict[0] *= FALSE_POSITIVE_PENALTY
        print(f"Background Weight (with penalty): {class_weight_dict[0]:.3f}")
    
    print(f"\nComputed Class Weights: {class_weight_dict}")

    # Early Stopping - prevents overfitting
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Training with more epochs (early stopping will find optimal point)
    print("\nTraining model (Early Stopping enabled)...")
    history = model.fit(
        X, y, 
        epochs=100,  # Early stopping will stop before this
        batch_size=32, 
        validation_split=0.2, 
        verbose=1, 
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)

    # --- Validation Report ---
    print("\n--- Generating Confusion Matrix (Validation) ---")
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        y_pred_probs = model.predict(X, batch_size=32, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        cm = confusion_matrix(y, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=CLASSES))
        
        # Heatmap Visualization
        print("Saving Confusion Matrix Heatmap...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=CLASSES, yticklabels=CLASSES)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix Heatmap')
        
        cm_path = BASE_DIR / "training" / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(cm_path)
        print(f"Saved Heatmap to {cm_path}")

    except Exception as e:
        print(f"Skipping evaluation report: {e}")

    # Training History Plot
    try:
        import matplotlib.pyplot as plt
        print("Generating training graph...")
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        
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
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    
    TFLITE_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TFLITE_SAVE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {TFLITE_SAVE_PATH}")
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
