"""
WiseYak SEQUENTIAL Wake Word Training
Classes: Background, Hello, Namaste, Deepa, Deepak
"""

import os
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
FALSE_POSITIVE_PENALTY = 2.0
FOCAL_GAMMA = 2.0
MAX_EPOCHS = 100
TARGET_SAMPLES_PER_CLASS = 15000 # Lower than main because we have cleaner data

SAMPLE_RATE = 16000
DURATION = 1 # 1 second is enough for single words
SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 13
N_FFT = 512 
HOP_LENGTH = 160

# Robustly find BASE_DIR
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent.parent
# Auto-fix for Colab/Nested envs
if not (BASE_DIR / "data_sequence").exists():
    for parent in CURRENT_FILE.parents:
        if (parent / "data_sequence").exists():
            BASE_DIR = parent
            break
    if not (BASE_DIR / "data_sequence").exists():
        # Fallback to CWD
        cwd = Path(os.getcwd())
        if (cwd / "data_sequence").exists():
            BASE_DIR = cwd

DATA_SEQ_DIR = BASE_DIR / "data_sequence"
NEGATIVE_DIR = BASE_DIR / "data" / "background" # Re-use background
MODEL_SAVE_PATH = BASE_DIR / "training" / "wakeword_sequence.h5"
TFLITE_SAVE_PATH = BASE_DIR / "web" / "wakeword_sequence.tflite"

CLASSES = ["Background", "Hello", "Namaste", "Deepa", "Deepak"]

# ============================================================
# UTILS (Reused)
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
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, 
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=40
    ).T
    return mfcc.astype(np.float32)

def load_audio(file_path):
    try:
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(y) > SAMPLES: y = y[:SAMPLES]
        else: y = librosa.util.fix_length(y, size=SAMPLES)
        return y
    except: return None

def spec_augment(spec: np.ndarray, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.25):
    spec = spec.copy()
    time_steps, n_freq = spec.shape
    for _ in range(num_mask):
        f = int(np.random.uniform(0, freq_masking_max_percentage) * n_freq)
        f0 = random.randint(0, n_freq - f)
        spec[:, f0:f0 + f] = 0
    for _ in range(num_mask):
        t = int(np.random.uniform(0, time_masking_max_percentage) * time_steps)
        t0 = random.randint(0, time_steps - t)
        spec[t0:t0 + t, :] = 0
    return spec

def process_file_with_aug(file_path, label, target_count, bg_files):
    results = []
    y = load_audio(file_path)
    if y is None: return []

    for _ in range(target_count):
        y_aug = AUGMENT_PIPELINE(samples=y, sample_rate=SAMPLE_RATE)
        if bg_files and random.random() < 0.4:
            try:
                bg = load_audio(random.choice(bg_files))
                if bg is not None:
                    y_aug = y_aug + (bg * random.uniform(0.05, 0.5))
            except: pass
        mfcc = extract_mfcc(y_aug)
        if random.random() < 0.5:
            mfcc = spec_augment(mfcc)
        results.append((mfcc, label))
    return results

# ============================================================
# DATA & MODEL
# ============================================================

def load_dataset():
    # Registry: 0=Bg, 1=Hello, 2=Namaste, 3=Deepa, 4=Deepak
    label_map = {
        "hello": 1, "namaste": 2, "deepa": 3, "deepak": 4
    }
    
    file_registry = {i: [] for i in range(5)} # 0-4

    # Index Positives
    for name, idx in label_map.items():
        path = DATA_SEQ_DIR / name
        if path.exists():
            file_registry[idx] = [str(f) for f in path.glob("*.*") if f.suffix in ['.wav', '.mp3']]
            print(f"Index {name}: {len(file_registry[idx])} files")

    # Index Negatives
    file_registry[0] = [str(f) for f in NEGATIVE_DIR.rglob("*.wav")]
    print(f"Index Background: {len(file_registry[0])} files")

    if sum(len(v) for v in file_registry.values()) == 0:
         raise ValueError(f"No files found in {DATA_SEQ_DIR} or {NEGATIVE_DIR}")

    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []
    n_jobs = min(8, os.cpu_count() or 4)

    for idx in range(5):
        files = file_registry[idx]
        if not files: 
            print(f"Warning: No files for class {CLASSES[idx]}")
            continue

        train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)
        
        train_count = len(train_files)
        aug_per_file = max(1, TARGET_SAMPLES_PER_CLASS // train_count) if train_count > 0 else 0
        
        print(f"Loading {CLASSES[idx]} (Train: {len(train_files)}, Val: {len(val_files)})")
        
        # Parallel Load
        if train_count > 0:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_file_with_aug)(f, idx, aug_per_file, file_registry[0])
                for f in train_files
            )
            for res_list in results:
                for mfcc, label in res_list:
                    X_train_all.append(mfcc)
                    y_train_all.append(label)
                    
        if len(val_files) > 0:
            results_val = Parallel(n_jobs=n_jobs)(
                delayed(process_file_with_aug)(f, idx, 1, [])
                for f in val_files
            )
            for res_list in results_val:
                for mfcc, label in res_list:
                    X_val_all.append(mfcc)
                    y_val_all.append(label)

    return (np.array(X_train_all), np.array(y_train_all), 
            np.array(X_val_all), np.array(y_val_all))

def build_model(input_shape, num_classes):
    # Same Robust DS-CNN
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1))(inputs)
    x = tf.keras.layers.Conv2D(64, (10, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    for kernel_size, dilation in [(3, 1), (3, 1), (3, 2), (3, 2)]:
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=(kernel_size, kernel_size), padding='same', dilation_rate=(dilation, 1), use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=SparseFocalLoss(gamma=FOCAL_GAMMA), metrics=['accuracy'])
    return model

def main():
    try:
        X_train, y_train, X_val, y_val = load_dataset()
    except Exception as e:
        print(e)
        return

    print(f"\nFinal Sets: Train={X_train.shape}, Val={X_val.shape}")

    # Shuffle Train
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx][..., np.newaxis], y_train[idx]
    X_val = X_val[..., np.newaxis]

    model = build_model(X_train.shape[1:], 5)
    
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw))
    cw_dict[0] *= FALSE_POSITIVE_PENALTY 
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=64, 
        class_weight=cw_dict,
        callbacks=callbacks
    )

    print("\nExporting TFLite...")
    # SAVE KERAS MODEL FIRST
    model.save(MODEL_SAVE_PATH)
    print(f"Saved Keras model to: {MODEL_SAVE_PATH}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_SAVE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to: {TFLITE_SAVE_PATH}")

if __name__ == "__main__":
    main()
