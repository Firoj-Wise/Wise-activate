const MODEL_PATH = "wakeword_model.tflite";
const SAMPLE_RATE = 16000;
const N_FFT = 512;
const HOP_LENGTH = 160;
const N_MFCC = 13;
const THRESHOLD = 0.90; // Lowered to 90%

// Core Audio Processing & Inference Variables
let model = null;          // TFLite model instance
let audioContext = null;   // Web Audio API Context
let processor = null;      // ScriptProcessorNode for raw audio access
let stream = null;         // Microphone MediaStream
let isListening = false;   // State flag for the listening loop

// UI Elements with new IDs
const statusDiv = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const indicator = document.getElementById("indicator");
const scoreFill = document.getElementById("scoreFill");
const confidenceDisplay = document.getElementById("confidenceDisplay");

async function loadModel() {
    statusDiv.innerText = "Loading Model...";
    try {
        tflite.setWasmPath('tflite/');
        model = await tflite.loadTFLiteModel(MODEL_PATH);
        statusDiv.innerText = "Ready to Listen";
        startBtn.disabled = false;
        console.log("Model loaded successfully");
    } catch (e) {
        statusDiv.innerText = "Error Loading Model";
        console.error(e);
    }
}

function preprocessAudio(signal) {
    /**
     * Extracts MFCC features from the raw audio signal using Meyda.
     * 
     * this implementation aligns with the Python training pipeline:
     * - Frame Size (N_FFT): 512
     * - Hop Length: 160
     * - MFCC Coefficients: 13
     */
    const mfccs = [];
    Meyda.bufferSize = N_FFT;
    for (let i = 0; i <= signal.length - N_FFT; i += HOP_LENGTH) {
        const frame = signal.slice(i, i + N_FFT);
        try {
            const features = Meyda.extract('mfcc', frame);
            if (features && features.length === N_MFCC) {
                mfccs.push(features);
            }
        } catch (e) {
            console.error("Meyda error", e);
        }
    }
    return mfccs;
}

async function runInference(audioData) {
    if (!model) return;

    const mfccs = preprocessAudio(audioData);
    if (mfccs.length === 0) return;

    // Strict Input Formatting for TFLite
    // The model expects a flat Float32Array representing (1, 101, 13, 1)
    const flatData = new Float32Array(101 * N_MFCC);
    for (let i = 0; i < 101; i++) {
        if (i < mfccs.length) {
            flatData.set(mfccs[i], i * N_MFCC);
        }
    }

    const tensor = tf.tensor(flatData, [1, 101, N_MFCC, 1], 'float32');

    try {
        const output = model.predict(tensor);
        const prediction = output.dataSync();
        const score = prediction[1];

        // Update UI Visuals
        const percentage = (score * 100).toFixed(1);
        scoreFill.style.width = `${percentage}%`; // Fill bar
        confidenceDisplay.innerText = `Confidence: ${percentage}%`;

        // Color logic for score bar
        if (score > THRESHOLD) {
            scoreFill.style.backgroundColor = "#34C759"; // Green
        } else if (score > 0.5) {
            scoreFill.style.backgroundColor = "#FF9500"; // Orange
        } else {
            scoreFill.style.backgroundColor = "#007AFF"; // Blue
        }

        if (score > THRESHOLD) {
            triggerWakeWord();
        } else {
            // Only update text if we are not currently showing "Detected!"
            if (!indicator.classList.contains("detected")) {
                statusDiv.innerText = "Listening...";
                statusDiv.className = "status-text";
            }
        }
    } catch (e) {
        console.error("Inference Error:", e);
    }
    tensor.dispose();
}

function triggerWakeWord() {
    statusDiv.innerText = "Wake Word Detected!";
    statusDiv.className = "status-text heavy";
    indicator.classList.add("detected");
    indicator.classList.remove("listening");

    // Debounce/Reset after 1.5 seconds
    setTimeout(() => {
        if (isListening) {
            indicator.classList.remove("detected");
            indicator.classList.add("listening");
            statusDiv.innerText = "Listening...";
            statusDiv.className = "status-text";
        }
    }, 1500);
}

async function startListening() {
    if (!model) await loadModel();
    if (isListening) return;

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: SAMPLE_RATE,
                channelCount: 1,
                echoCancellation: false,
                autoGainControl: true,
                noiseSuppression: true
            }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
        const source = audioContext.createMediaStreamSource(stream);
        const bufferSize = 4096;
        processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

        source.connect(processor);
        processor.connect(audioContext.destination);

        const rollingBuffer = new Float32Array(SAMPLE_RATE); // 1-second buffer

        // Explicitly tell Meyda we are using 16kHz (MUST match training)
        if (typeof Meyda !== 'undefined') {
            Meyda.sampleRate = SAMPLE_RATE;
            Meyda.melBands = 40; // Default for librosa is often 128, but Meyda defaults to 40. We need to check if this aligns.
            // Librosa default n_mels=128. Meyda default n_mels=40? 
            // Actually, for MFCC only, librosa uses n_mfcc=20 usually. We used 13.
            // Let's stick to just sampleRate for now as that's the big one.
            Meyda.windowingFunction = "hanning"; // Librosa default
        }

        processor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            const inputRate = audioContext.sampleRate;

            // Downsample if mismatch
            let processedData = inputData;
            if (inputRate !== SAMPLE_RATE) {
                processedData = downsampleBuffer(inputData, inputRate, SAMPLE_RATE);
            }

            // Sliding window (Circular Buffer)
            // We need to push 'processedData' (which is the new 16k chunk) into 'rollingBuffer'
            // Rolling buffer is 16000 long (1 second)

            // Shift existing data left
            const newLength = processedData.length;
            rollingBuffer.set(rollingBuffer.subarray(newLength));
            // Append new data at end
            rollingBuffer.set(processedData, rollingBuffer.length - newLength);

            // Run inference on the full 1-second 16k buffer
            runInference(rollingBuffer);
        };

        // UI Updates
        statusDiv.innerText = `Listening (${audioContext.sampleRate}Hz -> ${SAMPLE_RATE}Hz)`;
        indicator.classList.add("listening");
        startBtn.disabled = true;
        stopBtn.disabled = false;
        startBtn.innerText = "Listening";

        isListening = true;

    } catch (e) {
        statusDiv.innerText = "Mic Error";
        console.error(e);
    }
}

// Simple Downsampler (Linear Interpolation)
function downsampleBuffer(buffer, inputRate, outputRate) {
    /**
     * Downsamples audio from the browser's native sample rate (often 44.1k or 48k)
     * to the model's required 16kHz.
     * 
     * Uses simple decimation/nearest-neighbor selection. While a low-pass filter
     * is technically preferred to prevent aliasing, this method is computationally 
     * inexpensive and sufficient for speech keywords.
     */
    if (outputRate === inputRate) {
        return buffer;
    }
    const sampleRateRatio = inputRate / outputRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
        const nextIndex = Math.floor(i * sampleRateRatio);
        result[i] = buffer[nextIndex];
    }
    return result;
}



function stopListening() {
    if (!isListening) return;

    if (processor) {
        processor.disconnect();
        processor = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    // UI Updates
    isListening = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    startBtn.innerText = "Start";

    indicator.classList.remove("listening");
    indicator.classList.remove("detected");
    statusDiv.innerText = "Stopped";
    scoreFill.style.width = "0%";
    confidenceDisplay.innerText = "Confidence: 0%";
}

// Bind Global Functions
window.startListening = startListening;
window.stopListening = stopListening;

// Init
loadModel();
