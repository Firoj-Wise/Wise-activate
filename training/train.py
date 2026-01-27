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
MODEL_SAVE_PATH = BASE_DIR / "training" / "wakeword_model.h5"
TFLITE_SAVE_PATH = BASE_DIR / "web" / "wakeword_model.tflite"

def normalize_audio(y, target_rms=0.1, threshold=0.005):
    """
    Normalizes the audio signal to a target RMS value, avoiding amplification of silence.
    
    This function implements a noise gate to ensure that only signal segments exceeding
    a specific threshold are normalized, preventing background noise from becoming 
    artificially loud. It also includes peak protection to prevent clipping.

    Args:
        y (np.array): Input audio time series.
        target_rms (float): Desired Root Mean Square amplitude.
        threshold (float): RMS threshold below which normalization is skipped.

    Returns:
        np.array: Normalized audio signal.
    """
    current_rms = np.sqrt(np.mean(y**2))
    
    # Noise Gate: Only normalize if signal > threshold to avoid amplifying background noise
    if current_rms > threshold:
        gain = target_rms / current_rms
        y = y * gain
        
        # Peak protection: Hard clip to [-1.0, 1.0] range to avoid distortion
        if np.max(np.abs(y)) > 1.0:
            y = y / np.max(np.abs(y))
            
    return y

def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y = librosa.util.fix_length(y, size=SAMPLES)
        # Normalization removed to rely on raw data diversity
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None



def extract_mfcc(y):
    """
    Extracts Mel Frequency Cepstral Coefficients (MFCCs) from the audio signal.

    This configuration is strictly calibrated to match the client-side feature extraction
    performed by Meyda in the web application. Deviating from these parameters will
    result in model inference mismatch.

    Parameters:
        - Sample Rate: 16000 Hz
        - n_mfcc: 13 features
        - n_fft: 512 samples
        - hop_length: 160 samples
        - n_mels: 40 (Critical for Meyda.js compatibility)

    Args:
        y (np.array): Audio time series.

    Returns:
        np.array: Transposed MFCC matrix (Time, N_MFCC).
    """
    # CRITICAL: n_mels is set to 40 explicitly to align with Meyda's default mel-filterbank construction.
    # While librosa defaults to 128, Meyda uses 40. This parameter is non-negotiable for cross-platform accuracy.
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=40)
    return mfcc.T

def load_dataset(balance_ratio=1.0):
    X, y = [], []
    
    print("Loading positive samples...")
    pos_files = list(POSITIVE_DIR.rglob("*.wav"))
    
    for file in pos_files:
        # Load original samples
        audio = load_audio(file)
        if audio is not None:
            features = extract_mfcc(audio)
            X.append(features)
            y.append(1)
            
    num_pos = len(X)
    print(f"Loaded {num_pos} positive samples.")
    
    print("Loading negative samples...")
    neg_files = list(NEGATIVE_DIR.rglob("*.wav"))
    
    # Use ALL negative samples (No balancing/limiting)
    np.random.shuffle(neg_files)
    print(f"Using ALL {len(neg_files)} negative samples for robust training.")
    
    for file in neg_files:
        audio = load_audio(file)
        if audio is not None:
            features = extract_mfcc(audio)
            X.append(features)
            y.append(0)

    print(f"Loaded {len(y) - num_pos} negative samples.")
    
    return np.array(X), np.array(y)

def build_model(input_shape):
    """
    Constructs a Fully Convolutional Neural Network (FCN) optimized for keyword spotting.

    Architecture Overview:
    1. Input Layer: Receives MFCC spectrograms.
    2. Convolutional Block 1: 32 filters, 3x3 kernel. Captures low-level spectral-temporal features.
    3. Max Pooling: Reduces dimensionality and adds translational invariance.
    4. Convolutional Block 2: 64 filters, 3x3 kernel. Captures higher-level distinct phonetic patterns.
    5. 'Dense' Convolution: 64 filters, 1x1 kernel. Acts as a fully connected layer but preserves spatial properties.
    6. Output Convolution: 2 filters (Wake Word vs. Background).
    7. Global Average Pooling: Aggregates features over the entire time dimension, enabling variable-length input support.
    8. Softmax: Probabilistic output.

    Args:
        input_shape (tuple): Shape of the input spectrogram (Time_Steps, N_MFCC).
        
    Returns:
        tf.keras.Model: compiled Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Block 1: Feature Extraction
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1'),
        tf.keras.layers.MaxPool2D((2, 2), name='pool_1'),
        
        # Block 2: Pattern Recognition
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'),
        tf.keras.layers.MaxPool2D((2, 2), name='pool_2'),

        # Block 3: Dimensionality Reduction & Interpretation (1x1 Conv replaces Dense)
        tf.keras.layers.Conv2D(64, (1, 1), activation='relu', name='dense_conv'),
        tf.keras.layers.Dropout(0.5, name='dropout'),
        
        # Block 4: Class Projection
        tf.keras.layers.Conv2D(2, (1, 1), activation='linear', name='output_conv'),
        
        # Aggregation: Reduces (Time, Freq, Channels) -> (Channels) basically (Batch, 2)
        tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool'),
        
        # Activation
        tf.keras.layers.Softmax(name='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("--- Starting Training Pipeline ---")
    
    X, y = load_dataset()
    print(f"Dataset Info: X shape={X.shape}, y shape={y.shape}")
    
    # Expand dims to add channel information (N, Time, MFCC, 1)
    X = X[..., np.newaxis]
    
    input_shape = X.shape[1:]
    model = build_model(input_shape)
    model.summary()
    
    # Compute Class Weights to handle imbalance (Positives >> Negatives)
    # This ensures the model treats both classes equally rather than "guessing" the majority class.
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Computed Class Weights: {class_weight_dict}")

    print("Training model...")
    # Increased epochs to 50
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1, class_weight=class_weight_dict)
    
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    
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
