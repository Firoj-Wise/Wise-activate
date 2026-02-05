const MODEL_PATH = "wakeword_sequence.tflite";
const SAMPLE_RATE = 16000;
const N_FFT = 512;
const HOP_LENGTH = 160;
const N_MFCC = 13;

// === SEQUENCE PARAMETERS ===
const THRESHOLD = 0.85;
const HELLO_TIMEOUT_MS = 4000; // 4s window 
const SILENCE_THRESHOLD = 0.002;
const MICROPHONE_GAIN = 5.0;

// State Machine
let state = "IDLE"; // "IDLE" | "WAITING_FOR_NAME"
let lastHelloTime = 0;
let detectedGreeting = null; // "Hello" or "Namaste"
let greetingCooldown = 0;

// Core Variables
let model = null;
let audioContext = null;
let processor = null;
let stream = null;
let isListening = false;

// UI Elements
const statusDiv = document.getElementById("status");
const scoreFill = document.getElementById("scoreFill");
const confidenceDisplay = document.getElementById("confidenceDisplay");
const indicator = document.getElementById("indicator");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

// Sequence Classes
const CLASSES = ["Background", "Hello", "Namaste", "Deepa", "Deepak"];

async function loadModel() {
    statusDiv.innerText = "Loading Sequence Model...";
    try {
        tflite.setWasmPath('tflite/');
        model = await tflite.loadTFLiteModel(MODEL_PATH);
        
        // Warmup
        try {
            const dummyInput = new Float32Array(1 * 101 * N_MFCC); // Flattened input
            const inputTensor = model.getInputs()[0];
            const buffer = model.getInputTensor(inputTensor).data;
            buffer.set(dummyInput);
            model.run();
            console.log("Warmup complete");
        } catch (e) { console.warn("Warmup skipped", e); }

        statusDiv.innerText = "Ready: Say 'Hello Deepa'";
        startBtn.disabled = false;
    } catch (e) {
        statusDiv.innerText = "Model Load Failed";
        console.error(e);
    }
}

function preprocessAudio(signal) {
    const mfccs = [];
    if (typeof Meyda === 'undefined') {
        console.error("Meyda not found");
        return [];
    }
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
        } catch (e) { }
    }
    return mfccs;
}

function runInference(audioData) {
    if (!model || !isListening) return;

    // 1. Amplification & Silence Check
    let normData = new Float32Array(audioData.length);
    let peak = 0;
    for (let i = 0; i < audioData.length; i++) {
        normData[i] = audioData[i] * MICROPHONE_GAIN;
        const abs = Math.abs(normData[i]);
        if (abs > peak) peak = abs;
    }

    const now = Date.now();
    
    // Timeout Check
    if (state === "WAITING_FOR_NAME" && (now - lastHelloTime > HELLO_TIMEOUT_MS)) {
        state = "IDLE";
        detectedGreeting = null;
        triggerVisualFeedback("TIMEOUT", "gray");
    }

    // Silence Handling
    if (peak < SILENCE_THRESHOLD) {
        if (state === "WAITING_FOR_NAME") {
            const remaining = ((HELLO_TIMEOUT_MS - (now - lastHelloTime)) / 1000).toFixed(1);
            statusDiv.innerText = `Waiting for Name... (${remaining}s)`;
            scoreFill.style.width = "50%";
            scoreFill.style.backgroundColor = "#ffeb3b"; // Yellow
        } else {
            statusDiv.innerText = "Listening (Silent)...";
            scoreFill.style.width = "1%";
            scoreFill.style.backgroundColor = "#ccc";
        }
        return;
    }

    // 2. Normalize (AGC)
    for (let i = 0; i < normData.length; i++) normData[i] /= (peak + 1e-6);

    // 3. Extract Features
    const mfccs = preprocessAudio(normData);
    if (mfccs.length === 0) return;

    // Flatten & Pad
    const flatData = new Float32Array(101 * N_MFCC);
    for (let i = 0; i < 101; i++) {
        if (i < mfccs.length) flatData.set(mfccs[i], i * N_MFCC);
    }

    // 4. Inference
    const inputTensor = model.getInputs()[0];
    const inputBuffer = model.getInputTensor(inputTensor).data;
    inputBuffer.set(flatData);

    model.run();

    const outputTensor = model.getOutputs()[0];
    const outputBuffer = model.getOutputTensor(outputTensor).data;
    const scores = Array.from(outputBuffer);

    // 5. Logic
    let maxScore = 0;
    let maxIndex = 0;
    for (let i = 0; i < scores.length; i++) {
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            maxIndex = i;
        }
    }
    
    const label = CLASSES[maxIndex];
    const confidence = (maxScore * 100).toFixed(0);

    // Debug Display
    confidenceDisplay.innerText = `${label} (${confidence}%)`;

    if (maxScore > THRESHOLD) {
        if (state === "IDLE") {
            if (label === "Hello" || label === "Namaste") {
                if (now - greetingCooldown > 2000) {
                    state = "WAITING_FOR_NAME";
                    detectedGreeting = label;
                    lastHelloTime = now;
                    triggerVisualFeedback(`Greetings: ${label}`, "orange");
                }
            }
        }
        else if (state === "WAITING_FOR_NAME") {
            if (label === "Deepa" || label === "Deepak") {
                const fullPhrase = `${detectedGreeting} ${label}`;
                triggerActivation(fullPhrase);
                
                // Reset
                state = "IDLE";
                detectedGreeting = null;
                greetingCooldown = now;
            }
        }
    }
}

function triggerVisualFeedback(text, color) {
    statusDiv.innerText = text;
    scoreFill.style.backgroundColor = color;
    scoreFill.style.width = "90%";
}

function triggerActivation(phrase) {
    statusDiv.innerText = `ACTIVATED: ${phrase}`;
    statusDiv.style.color = "green";
    statusDiv.style.fontWeight = "bold";
    indicator.classList.add("detected");
    setTimeout(() => {
        indicator.classList.remove("detected");
        statusDiv.style.color = "black";
        statusDiv.style.fontWeight = "normal";
    }, 3000);
}

// === BOILERPLATE AUDIO SETUP ===
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
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        source.connect(processor);
        processor.connect(audioContext.destination);

        const rollingBuffer = new Float32Array(SAMPLE_RATE); // 1s buffer

        processor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            
            // Circular Buffer Logic
            const newLength = inputData.length;
            rollingBuffer.set(rollingBuffer.subarray(newLength));
            rollingBuffer.set(inputData, rollingBuffer.length - newLength);
            
            runInference(rollingBuffer);
        };

        statusDiv.innerText = "Listening...";
        indicator.classList.add("listening");
        startBtn.disabled = true;
        stopBtn.disabled = false;
        isListening = true;

    } catch (e) { console.error(e); }
}

function stopListening() {
    if (!isListening) return;
    if (processor) { processor.disconnect(); processor = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    isListening = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    indicator.classList.remove("listening");
    statusDiv.innerText = "Stopped";
}

window.startListening = startListening;
window.stopListening = stopListening;
loadModel();
