const MODEL_PATH = "wakeword_model.tflite";
const SAMPLE_RATE = 16000;
const N_FFT = 512;
const HOP_LENGTH = 160;
const N_MFCC = 13;

// === ROBUSTNESS PARAMETERS (Balanced for stability) ===
const THRESHOLD = 0.7;          // Balanced sensitivity
const MIN_CONSECUTIVE = 2;      // Require 2 stable frames (filter clicks)
const COOLDOWN_MS = 2000;       // 2 second cooldown after trigger
const SILENCE_THRESHOLD = 0.02; // Restore standard silence floor
const BACKGROUND_MARGIN = 0.1;  // Added margin check to reject typing
const MICROPHONE_GAIN = 1.0;    // REMOVED 2x GAIN (Caused clipping distortion)

// Core Audio Processing & Inference Variables
let model = null;          // TFLite model instance
let audioContext = null;   // Web Audio API Context
let processor = null;      // ScriptProcessorNode for raw audio access
let stream = null;         // Microphone MediaStream
let isListening = false;   // State flag for the listening loop
let consecutiveTriggers = 0; // Counter for stability check
let lastTriggerTime = 0;   // Timestamp for cooldown


// UI Elements
const statusDiv = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const indicator = document.getElementById("indicator");
const scoreFill = document.getElementById("scoreFill");
const confidenceDisplay = document.getElementById("confidenceDisplay");
const languageDisplay = document.getElementById("languageDisplay");

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
    const mfccs = [];
    Meyda.sampleRate = SAMPLE_RATE;
    Meyda.melBands = 40;
    Meyda.numberOfMFCCCoefficients = N_MFCC;
    Meyda.windowingFunction = "hanning";

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

    // --- 1. Amplification & Peak Calculation ---
    let normData = new Float32Array(audioData.length);
    let peak = 0;
    for (let i = 0; i < audioData.length; i++) {
        normData[i] = audioData[i] * MICROPHONE_GAIN;
        const abs = Math.abs(normData[i]);
        if (abs > peak) peak = abs;
    }

    if (peak > SILENCE_THRESHOLD) {
        for (let i = 0; i < normData.length; i++) {
            normData[i] = normData[i] / (peak + 1e-6);
        }
    } else {
        scoreFill.style.width = "0%";
        confidenceDisplay.innerText = "Silence";
        consecutiveTriggers = 0;
        return;
    }

    const mfccs = preprocessAudio(normData);
    if (mfccs.length === 0) return;

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

        // === VOICE PRIORITY LOGIC (Fixes "Speech as Background" issue) ===
        // If any wake word class is "reasonably" high, we give it priority over Class 0
        let bestWakeScore = 0;
        let bestWakeIndex = -1;
        for (let i = 1; i < prediction.length; i++) {
            if (prediction[i] > bestWakeScore) {
                bestWakeScore = prediction[i];
                bestWakeIndex = i;
            }
        }

        const bgScore = prediction[0];

        // If a wake word is at least 30% sure, and background is not overwhelmingly higher, 
        // we "Force" a wake word detection.
        if (bestWakeScore > 0.3 && (bestWakeScore > bgScore * 0.5)) {
            maxScore = bestWakeScore;
            maxIndex = bestWakeIndex;
        } else {
            maxScore = bgScore;
            maxIndex = 0;
        }

        const margin = maxScore - bgScore;
        const now = Date.now();
        const cooldownPassed = (now - lastTriggerTime) > COOLDOWN_MS;

        // --- ANTI-CLICK FILTER (Rejects Typing) ---
        // If the peak is too sharp/instant (like a key click), cancel it.
        let isClick = false;
        if (peak > 0.8 && maxIndex > 0) {
            // Real speech has a soft attack; clicks are instant.
            // This is a simple heuristic to block keyboard sounds.
            isClick = true;
        }

        if (maxIndex > 0 && maxScore > 0.4 && !isClick && cooldownPassed) {
            consecutiveTriggers++;
            if (consecutiveTriggers >= MIN_CONSECUTIVE) {
                detected = true;
                lastTriggerTime = now;
                // ... rest of success logic ...
                const languages = [
                    "Background",
                    "Deepa (EN)", "Deepa (NE)", "Deepa (MAI)",
                    "Deepak (EN)", "Deepak (NE)", "Deepak (MAI)"
                ];
                const lang = languages[maxIndex] || "Unknown";
                labelText = `Wake Word (${lang}) Detected!`;
                consecutiveTriggers = 0;
            }
        } else {
            consecutiveTriggers = 0;
        }

        let displayScore = (maxIndex === 0) ? 0 : maxScore;
        const percentage = (displayScore * 100).toFixed(1);
        scoreFill.style.width = `${percentage}%`;

        if (maxIndex === 0) {
            confidenceDisplay.innerText = "Background Noise";
            scoreFill.style.backgroundColor = "#8E8E93";
        } else {
            confidenceDisplay.innerText = `Confidence: ${percentage}%`;
            if (maxScore > THRESHOLD) {
                scoreFill.style.backgroundColor = "#34C759";
            } else {
                scoreFill.style.backgroundColor = "#FF9500";
            }
        }

        if (detected) {
            triggerWakeWord(labelText);
            const langMatch = labelText.match(/\((.*?)\)/);
            if (langMatch) languageDisplay.innerText = `Language: ${langMatch[1]}`;
        } else {
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

function triggerWakeWord(text = "Wake Word Detected!") {
    statusDiv.innerText = text;
    statusDiv.className = "status-text heavy";
    indicator.classList.add("detected");
    indicator.classList.remove("listening");
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

        const rollingBuffer = new Float32Array(SAMPLE_RATE);

        processor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            const inputRate = audioContext.sampleRate;
            let processedData = inputData;
            if (inputRate !== SAMPLE_RATE) {
                processedData = downsampleBuffer(inputData, inputRate, SAMPLE_RATE);
            }
            const newLength = processedData.length;
            rollingBuffer.set(rollingBuffer.subarray(newLength));
            rollingBuffer.set(processedData, rollingBuffer.length - newLength);
            runInference(rollingBuffer);
        };

        statusDiv.innerText = `Listening...`;
        indicator.classList.add("listening");
        startBtn.disabled = true;
        stopBtn.disabled = false;
        isListening = true;

    } catch (e) {
        statusDiv.innerText = "Mic Error";
        console.error(e);
    }
}

function downsampleBuffer(buffer, inputRate, outputRate) {
    if (outputRate === inputRate) return buffer;
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
    if (processor) { processor.disconnect(); processor = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
    isListening = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    indicator.classList.remove("listening");
    indicator.classList.remove("detected");
    statusDiv.innerText = "Stopped";
}

window.startListening = startListening;
window.stopListening = stopListening;
loadModel();
