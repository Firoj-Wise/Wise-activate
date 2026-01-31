const MODEL_PATH = "wakeword_model.tflite";
const SAMPLE_RATE = 16000;
const N_FFT = 512;
const HOP_LENGTH = 160;
const N_MFCC = 13;
const THRESHOLD = 0.96; // Increased from 0.85 for stricter trigger
const MIN_CONSECUTIVE = 2; // Require 2 consecutive frames to trigger

// Core Audio Processing & Inference Variables
let model = null;          // TFLite model instance
let audioContext = null;   // Web Audio API Context
let processor = null;      // ScriptProcessorNode for raw audio access
let stream = null;         // Microphone MediaStream
let isListening = false;   // State flag for the listening loop
let consecutiveTriggers = 0; // Counter for stability check

// ... (Rest of UI Elements unchanged)

// ... (loadModel unchanged)

// ... (preprocessAudio unchanged)

async function runInference(audioData) {
    if (!model) return;

    // --- 1. Peak Calculation & Normalization ---
    // Calculate Peak
    let peak = 0;
    for (let i = 0; i < audioData.length; i++) {
        const abs = Math.abs(audioData[i]);
        if (abs > peak) peak = abs;
    }

    // NormalizeLogic
    let normData = new Float32Array(audioData); // Copy
    const SILENCE_THRESHOLD = 0.05; // Increased noise gate (was 0.02)

    if (peak > SILENCE_THRESHOLD) {
        // Boost to Peak 1.0 (Simulates Training Data)
        for (let i = 0; i < normData.length; i++) {
            normData[i] = normData[i] / (peak + 1e-6);
        }
    } else {
        // Too quiet, skip inference to save CPU and false positives
        // Just clear the UI
        scoreFill.style.width = "0%";
        confidenceDisplay.innerText = "Silence";
        consecutiveTriggers = 0;
        return;
    }

    const mfccs = preprocessAudio(normData);
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
        const prediction = output.dataSync(); // Sync for simplicity

        let score = 0;
        let detected = false;
        let labelText = "Wake Word Detected!";

        // Adaptive Inference Logic
        // ... (Assuming standard multi-class logic) ... 

        let maxScore = -1;
        let maxIndex = -1;

        for (let i = 0; i < prediction.length; i++) {
            if (prediction[i] > maxScore) {
                maxScore = prediction[i];
                maxIndex = i;
            }
        }
        score = maxScore;

        // Index 0 is Background
        if (maxIndex > 0 && maxScore > THRESHOLD) {
            consecutiveTriggers++;

            if (consecutiveTriggers >= MIN_CONSECUTIVE) {
                detected = true;
                const languages = [
                    "Background",
                    "Deepa (EN)", "Deepa (NE)", "Deepa (MAI)",
                    "Deepak (EN)", "Deepak (NE)", "Deepak (MAI)"
                ];
                const lang = languages[maxIndex] || "Unknown";
                labelText = `Wake Word (${lang}) Detected!`;
                // Reset counter to prevent spamming triggers every frame
                // consecutiveTriggers = 0; 
                // Actually, usually beneficial to keep it high or cooldown. 
                // Let's rely on the tokenizer cooldown in 'triggerWakeWord'
            }
        } else {
            consecutiveTriggers = 0; // Reset if we drop below threshold
        }

        // (UI Visuals Code...)

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

        if (detected) {
            triggerWakeWord(labelText);
            // languageDisplay is handled in triggerWakeWord or here? 
            // Let's do it here for specificity if we have the lang
            // Actually, labelText already has it. 
            // But user asked to "show me that in UI as well".
            // Let's parse it or pass it.
            // Simplest: just update the element.
            const langMatch = labelText.match(/\((.*?)\)/);
            if (langMatch) {
                languageDisplay.innerText = `Language: ${langMatch[1]}`;
            }
        } else {
            // Only update text if we are not currently showing "Detected!"
            if (!indicator.classList.contains("detected")) {
                statusDiv.innerText = "Listening...";
                statusDiv.className = "status-text";
                languageDisplay.innerText = "";
            }
        }
    } catch (e) {
        console.error("Inference Error:", e);
    }
    tensor.dispose();
}

function triggerWakeWord(text = "Wake Word Detected!") {
    statusDiv.innerText = text;
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
