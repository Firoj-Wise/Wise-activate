# WiseYak Wake Word Detection System ("Sundari")

## Overview

This repository contains the training and inference infrastructure for the **WiseYak** custom wake word detection system. The primary activation keyword is **"Sundari"**.

The system is designed to be lightweight and efficient, capable of running in a web browser using TensorFlow Lite (WASM) while maintaining high accuracy and low latency. The architecture utilizes a Fully Convolutional Neural Network (FCN) trained on Mel Frequency Cepstral Coefficients (MFCCs).

## Architecture

### Model Design
The core model is a **Fully Convolutional Neural Network (FCN)**. This architecture was selected for its parameter efficiency and ability to handle variable-length inputs (though fixed-length windows are used in production).

**Layer Configuration:**
1.  **Input**: Accepts a spectrogram of shape `(Time, MFCC_Features, 1)`.
2.  **Convolution Block 1**: 32 filters, 3x3 kernel, ReLU activation. Followed by 2x2 Max Pooling.
3.  **Convolution Block 2**: 64 filters, 3x3 kernel, ReLU activation. Followed by 2x2 Max Pooling.
4.  **Feature Compression**: A 1x1 Convolution with 64 filters acts as a dense layer substitute, preserving spatial context while reducing dimensionality.
5.  **Output Projection**: A final 1x1 Convolution projects features to 2 class channels (Wake Word vs. Background).
6.  **Global Average Pooling**: Aggregates the temporal dimension to produce a single prediction vector.
7.  **Softmax**: Produces the final probability distribution.

### Signal Processing Pipeline
Compatibility between the training environment (Python/Librosa) and the inference environment (JavaScript/Meyda) is critical.

*   **Sample Rate**: 16,000 Hz
*   **Feature Type**: MFCC (Mel Frequency Cepstral Coefficients)
*   **Fractions**: 13 Coefficients
*   **FFT Window**: 512 samples
*   **Hop Length**: 160 samples
*   **Mel Bands**: 40

**Critical Note**: The number of Mel bands (`n_mels`) is explicitly set to **40** in the training pipeline to match the default behavior of the `Meyda` JS library. Deviating from this parameter will cause significant accuracy degradation during inference.

## Training Pipeline

The training logic is encapsulated in `training/train.py`.

### Dataset Structure
*   **Positive Samples (`data/wake`)**: Contains recordings of the wake word "Sundari".
*   **Negative Samples (`data/background`)**: Contains background noise, silence, and non-target speech.

### Methodology
1.  **Preprocessing**: Audio is loaded at 16kHz. Silence is not trimmed to preserve environmental context.
2.  **Normalization**: A noise-gated RMS normalization strategy is applied to standardise input volume without amplifying floor noise.
3.  **Training**: The model is trained using `Sparse Categorical Crossentropy` loss and the `Adam` optimizer.
4.  **Export**: The trained Keras model (`.h5`) is quantized and converted to TensorFlow Lite (`.tflite`) format for web deployment.

## Inference Engine (`web/app.js`)

The web-based inference engine utilizes **TensorFlow.js (TFLite backend)** and **Meyda** for real-time processing.

### Workflow
1.  **Audio Capture**: The browser's `AudioContext` captures the microphone stream.
2.  **Downsampling**: If the hardware sample rate differs from 16kHz, a linear interpolation downsampler converts the stream.
3.  **Circular Buffer**: A rolling buffer holds exactly 1 second of audio (16,000 samples).
4.  **Feature Extraction**: `Meyda` extracts MFCC features from the buffer in sliding windows.
5.  **Inference**: The features are flattened and passed to the TFLite model.
6.  **Debounce Logic**: A cooldown period is enforced after a positive detection to prevent repeated triggering.

## Usage

### Training
Execute the training script to regenerate the model:
```bash
python training/train.py
```

### Web Inference
Serve the `web` directory using a static file server:
```bash
cd web
python -m http.server 8000
```
Access `http://localhost:8000` to test the wake word detection.

## Performance Considerations
*   **Latency**: The FCN architecture is extremely lightweight (< 1MB), ensuring sub-100ms inference times on standard mobile devices.
*   **False Positives**: The system is tuned to prioritize high precision. The threshold can be adjusted in `app.js` (default: 0.90).

## Production Recommendations (Dataset Expansion)

While the current model is functional and optimized for the existing dataset (~160 positive samples), achieving robust, production-grade performance across diverse acoustic environments requires a larger dataset.

**Recommended Data Scale:**
*   **Positive Samples**: 1,000+ unique recordings of "Sundari" from diverse speakers (gender, accent, age).
*   **Negative Samples**: 2,000+ variants of background noise and "confusing" words (e.g., words rhyming with "Sundari").

**Target Distribution:**
Maintaining a balanced or slightly skewed (2:1 Negative-to-Positive) dataset is recommended to minimize false positives in noisy environments. The current training pipeline includes `class_weight` balancing to mathematically compensate for dataset asymmetry.
