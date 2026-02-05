"""
WiseYak Wake Word Training Pipeline (Robust & Leak-Free)

Key Features:
- STRICT Stratified Split: Files are split BEFORE augmentation to prevent leakage.
- Massive Balancing: All 7 classes are upsampled/augmented to have equal weight.
- Focal Loss & False Positive Penalty: Advanced rejection of background noise.
"""

import os
# import tensorflowjs as tfjs
import random
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise, Gain

# ============================================================
# CONFIGURATION
# ============================================================
FALSE_POSITIVE_PENALTY = 2.0   # Bias against false triggers
FOCAL_GAMMA = 2.0              # Focus on hard examples
MAX_EPOCHS = 100
TARGET_SAMPLES_PER_CLASS = 50000 # Massive volume

# Audio Constants
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
# TFJS_SAVE_DIR = BASE_DIR / "web" / "tfjs_model" 

CLASSES = [
    "Background", 
    "Deepa (EN)", "Deepa (NE)", "Deepa (MAI)",
    "Deepak (EN)", "Deepak (NE)", "Deepak (MAI)"
]

# ============================================================
# FOCAL LOSS & AUGMENTATION
# ============================================================

class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, name='sparse_focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(tf.squeeze(y_true), depth=tf.shape(y_pred)[-1])
        ce = -y_true_one_hot * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(focal_weight * ce, axis=-1))

AUGMENT_PIPELINE = Compose([
    PitchShift(min_semitones=-2, max_semitones=2, p=0.4),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.4),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
    Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
])

def extract_mfcc(y):
    """Extracts MFCCs matching client-side Meyda/librosa."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, 
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=40
    ).T
    return mfcc.astype(np.float32)

def load_audio(file_path):
    try:
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        # Fix length to exactly 1s
        if len(y) > SAMPLES: y = y[:SAMPLES]
        else: y = librosa.util.fix_length(y, size=SAMPLES)
        return y
    except: return None

# ============================================================
# PROCESSING PIPELINE
# ============================================================

def spec_augment(spec: np.ndarray, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.25):
    """
    Applied SpecAugment (Frequency and Time Masking) to the MFCC/Spectrogram.
    spec shape: (time_steps, n_mfcc)
    """
    spec = spec.copy() # Avoid modifying original
    time_steps, n_freq = spec.shape
    
    # 1. Frequency Masking
    for _ in range(num_mask):
        f = int(np.random.uniform(0, freq_masking_max_percentage) * n_freq)
        f0 = random.randint(0, n_freq - f)
        spec[:, f0:f0 + f] = 0

    # 2. Time Masking
    for _ in range(num_mask):
        t = int(np.random.uniform(0, time_masking_max_percentage) * time_steps)
        t0 = random.randint(0, time_steps - t)
        spec[t0:t0 + t, :] = 0
        
    return spec

def process_file_with_aug(file_path, label, target_count, bg_files):
    """Loads a file and generates multiple augmented versions to hit target_count."""
    results = []
    y = load_audio(file_path)
    if y is None: return []

    # Calculate how many augmentations per file
    for _ in range(target_count):
        # 1. Waveform Augmentation (Pitch, Time, Noise)
        y_aug = AUGMENT_PIPELINE(samples=y, sample_rate=SAMPLE_RATE)
        
        # 2. Background Mixing
        if bg_files and random.random() < 0.4: # Increased generic noise probability
            try:
                bg = load_audio(random.choice(bg_files))
                if bg is not None:
                    # Random volume for background
                    y_aug = y_aug + (bg * random.uniform(0.05, 0.5))
            except: pass
        
        # 3. Feature Extraction
        mfcc = extract_mfcc(y_aug)
        
        # 4. SpecAugment (Feature level masking)
        # Apply 50% of the time to keep some clean samples
        if random.random() < 0.5:
            mfcc = spec_augment(mfcc)

        results.append((mfcc, label))
    return results

def build_model(input_shape, num_classes):
    """
    DS-CNN (Depthwise Separable CNN) Architecture.
    Standard for Keyword Spotting on Edge Devices.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Initial Conv to expand features
    x = tf.keras.layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1))(inputs)
    x = tf.keras.layers.Conv2D(64, (10, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # DS-CNN Blocks
    # Seperates learning "spatial" correlations from "cross-channel" correlations
    for kernel_size, dilation in [(3, 1), (3, 1), (3, 2), (3, 2)]:
        # Depthwise
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, kernel_size), 
            padding='same', 
            dilation_rate=(dilation, 1),
            use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        # Pointwise
        x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), # Slightly higher LR for DS-CNN
        loss=SparseFocalLoss(gamma=FOCAL_GAMMA),
        metrics=['accuracy']
    )
    return model

def load_stratified_dataset():
    """
    1. Indexes all files.
    2. Splits them into Train/Val sets (No Leakage).
    3. Augments only the Training set.
    """
    label_map = {
        ("deepa", "en"): 1, ("deepa", "ne"): 2, ("deepa", "mai"): 3,
        ("deepak", "en"): 4, ("deepak", "ne"): 5, ("deepak", "mai"): 6
    }
    
    file_registry = {i: [] for i in range(7)} # 0-6

    # Index Positives
    for (name_dir, lang_dir), idx in label_map.items():
        path = POSITIVE_DIR / name_dir / lang_dir
        if path.exists():
            file_registry[idx] = [str(f) for f in path.glob("*.wav")]

    # Index Negatives (Background)
    file_registry[0] = [str(f) for f in NEGATIVE_DIR.rglob("*.wav")]
    user_neg = [str(f) for f in USER_NEGATIVE_DIR.rglob("*.wav")] if USER_NEGATIVE_DIR.exists() else []
    file_registry[0].extend(user_neg)

    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []

    print("\n--- Stratified Split & Leak-Free Loading ---")
    print(f"Searching for data in: {BASE_DIR}")
    
    n_jobs = min(8, os.cpu_count() or 4)

    total_files = 0
    for idx in range(7):
        files = file_registry[idx]
        total_files += len(files)
        if not files: 
            if idx == 0:
                print(f"Warning: No background files found in {NEGATIVE_DIR}")
            else:
                # Try to print which key failed
                lbl = [k for k, v in label_map.items() if v == idx]
                path = POSITIVE_DIR / lbl[0][0] / lbl[0][1] if lbl else "Unknown"
                print(f"Warning: No files found for Class {idx} ({CLASSES[idx]}) at {path}")
            continue

        # Stratified Split (80/20)
        train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)
        
        # Current counts
        train_count = len(train_files)
        val_count = len(val_files)
        print(f"Class {idx} ({CLASSES[idx]}): {train_count} train / {val_count} val files")

        # 1. Load Training Data (with massive augmentation)
        # Each file will be augmented (TARGET_SAMPLES / train_count) times.
        if train_count > 0:
            aug_per_file = max(1, TARGET_SAMPLES_PER_CLASS // train_count)
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_file_with_aug)(f, idx, aug_per_file, file_registry[0])
                for f in train_files
            )
            for res_list in results:
                for mfcc, label in res_list:
                    X_train_all.append(mfcc)
                    y_train_all.append(label)

        # 2. Load Validation Data (NO augmentation, pure test)
        if val_count > 0:
            results_val = Parallel(n_jobs=n_jobs)(
                delayed(process_file_with_aug)(f, idx, 1, []) # 1x, no bg mixing for pure val
                for f in val_files
            )
            for res_list in results_val:
                for mfcc, label in res_list:
                    X_val_all.append(mfcc)
                    y_val_all.append(label)

    if total_files == 0:
        raise ValueError(f"No audio files found! Check data paths.\nBase: {BASE_DIR}\nPositive: {POSITIVE_DIR}\nNegative: {NEGATIVE_DIR}")

    return (np.array(X_train_all), np.array(y_train_all), 
            np.array(X_val_all), np.array(y_val_all))

# Replaced by DS-CNN above
# def build_model(input_shape, num_classes):
# ...


def main():
    X_train, y_train, X_val, y_val = load_stratified_dataset()
    print(f"\nFinal Sets: Train={X_train.shape}, Val={X_val.shape}")

    # Shuffle Train
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx][..., np.newaxis], y_train[idx]
    X_val = X_val[..., np.newaxis]

    model = build_model(X_train.shape[1:], 7)
    
    # Class Weights (Balanced)
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw))
    cw_dict[0] *= FALSE_POSITIVE_PENALTY # Apply penalty
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=64, # Larger batch for speed
        class_weight=cw_dict,
        callbacks=callbacks
    )

    # 1. Export TFLite
    print("\nExporting TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_SAVE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to: {TFLITE_SAVE_PATH}")

    # # 2. Export TensorFlow.js (Graph Model)
    # print("\nExporting TensorFlow.js Model...")
    # # Requires: pip install tensorflowjs
    # tfjs.converters.save_keras_model(model, TFJS_SAVE_DIR)
    # print(f"Saved TFJS model to: {TFJS_SAVE_DIR}")

if __name__ == "__main__":
    main()
