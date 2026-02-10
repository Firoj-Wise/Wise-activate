#!/usr/bin/env python3
"""
====================================================================
WiseYak openWakeWord Training — Kaggle Notebook
====================================================================
Copy-paste each numbered section into a separate Kaggle notebook cell.
Requires:
  - Kaggle GPU (P100/T4)
  - Dataset: "openwakeword_features_ACAV100M_2000_hrs_16bit" uploaded
  - Dataset: "validation_set_features" uploaded
  - Dataset: Your wise-activate data/ folder uploaded as a dataset

Last Updated: 2026-02-10 (Pinned Dependencies & Sync'd Configs)
====================================================================
"""

# %%
# ============================================================
# CELL 1: Install Pinned Dependencies
# ============================================================
import subprocess, sys

deps = [
    "piper-tts==1.2.0", 
    "piper-phonemize-cross==0.1.0", 
    "webrtcvad==2.0.10",
    "mutagen==1.47.0", 
    "torchinfo==1.8.0", 
    "torchmetrics==1.2.0",
    "speechbrain==0.5.14", 
    "audiomentations==0.33.0",
    "torch-audiomentations==0.11.0", 
    "acoustics==0.2.6",
    "onnxruntime==1.22.1", 
    "ai_edge_litert==1.4.0", 
    "onnxsim==0.4.36",
    "onnx2tf==1.22.3", 
    "onnx_graphsurgeon==0.5.5", 
    "sng4onnx==1.0.4",
    "pronouncing==0.2.0", 
    "datasets==2.14.6",
    "deep-phonemizer==0.0.19",
    "pyyaml==6.0.1",
    "tqdm==4.66.1",
    "librosa==0.10.1",
    "scipy==1.11.3"
]

print("Installing pinned dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + deps, check=False)
print("Dependencies installed successfully.")

# %%
# ============================================================
# CELL 2: Clone Repos & Download Models
# ============================================================
import os
from pathlib import Path

WORK = Path("/kaggle/working")

# Clone piper-sample-generator
if not (WORK / "piper-sample-generator").exists():
    os.system(f"git clone https://github.com/rhasspy/piper-sample-generator {WORK / 'piper-sample-generator'}")

# Download Piper TTS model
piper_model = WORK / "piper-sample-generator" / "models" / "en_US-libritts_r-medium.pt"
if not piper_model.exists():
    piper_model.parent.mkdir(parents=True, exist_ok=True)
    os.system(f"wget -q -O {piper_model} 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'")

# Clone openWakeWord
if not (WORK / "openwakeword").exists():
    os.system(f"git clone https://github.com/dscripka/openwakeword {WORK / 'openwakeword'}")

# Install openWakeWord
os.system(f"pip install -q -e {WORK / 'openwakeword'} --no-deps")

# Download resource models
res_dir = WORK / "openwakeword" / "openwakeword" / "resources" / "models"
res_dir.mkdir(parents=True, exist_ok=True)
models = [
    ("embedding_model.onnx",   "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"),
    ("embedding_model.tflite", "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite"),
    ("melspectrogram.onnx",    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"),
    ("melspectrogram.tflite",  "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite"),
]
for fname, url in models:
    dest = res_dir / fname
    if not dest.exists():
        os.system(f"wget -q {url} -O {dest}")

# Add piper to Python path
sys.path.insert(0, str(WORK / "piper-sample-generator"))

print("Repos cloned, models downloaded.")

# %%
# ============================================================
# CELL 3: Patch openWakeWord Source (Critical Stability Fixes)
# ============================================================
import re, numpy as np

# --- Patch data.py: better phoneme diversity ---
data_py = WORK / "openwakeword" / "openwakeword" / "data.py"
if data_py.exists():
    text = data_py.read_text()
    old = "max_replace=max(0, len(phones)-5)"
    new = "max_replace=max(0, len(phones)-np.random.randint(2, len(phones)))"
    if old in text:
        data_py.write_text(text.replace(old, new))
        print("Patched data.py")

# --- Patch train.py ---
train_py = WORK / "openwakeword" / "openwakeword" / "train.py"
if train_py.exists():
    content = train_py.read_text()
    patched = False

    # PyTorch 2.6 weights_only fix
    if "_patched_torch_load" not in content:
        patch = (
            "\n# --- PyTorch 2.6 fix ---\n"
            "_original_torch_load = torch.load\n"
            "def _patched_torch_load(*args, **kwargs):\n"
            "    if 'weights_only' not in kwargs:\n"
            "        kwargs['weights_only'] = False\n"
            "    return _original_torch_load(*args, **kwargs)\n"
            "torch.load = _patched_torch_load\n\n"
        )
        content = content.replace("import torch\n", "import torch\n" + patch, 1)
        patched = True

    # Piper model_path injection
    if "if args.generate_clips is True:" in content and "model_path = os.path.join" not in content:
        block = (
            '    if args.generate_clips is True:\n'
            '        model_path = os.path.join(\n'
            '            config["piper_sample_generator_path"], "models", "en_US-libritts_r-medium.pt"\n'
            '        )\n'
            '        if not os.path.exists(model_path):\n'
            '            raise FileNotFoundError(f"Piper model not found at {model_path}")\n'
        )
        content = content.replace("    if args.generate_clips is True:", block)
        patched = True

    # Skip TFLite conversion
    tflite_old = (
        '        # Convert the model from onnx to tflite format\n'
        '        convert_onnx_to_tflite(os.path.join(config["output_dir"], config["model_name"] + ".onnx"),\n'
        '                               os.path.join(config["output_dir"], config["model_name"] + ".tflite"))'
    )
    tflite_new = (
        '        # Skipped TFLite conversion (compat issues)\n'
        '        # convert_onnx_to_tflite(...)\n'
        '        pass'
    )
    if tflite_old in content:
        content = content.replace(tflite_old, tflite_new)
        patched = True

    # Increase negative weight scaling
    if "max_negative_weight = max_negative_weight*2" in content:
        content = content.replace(
            "max_negative_weight = max_negative_weight*2",
            "max_negative_weight = max_negative_weight*5"
        )
        patched = True

    if patched:
        train_py.write_text(content)
        print("Patched train.py")
    else:
        print("train.py already patched")

# %%
# ============================================================
# CELL 4: Choose Model & Write Config
# ============================================================
# !! EDIT THIS CELL to choose which model to train !!
MODEL_CHOICE = "hello_namaste_deepa"
# Options: "hello_deepa" | "namaste_deepa" | "hello_namaste_deepa"

import yaml

CONFIGS = {
    "hello_deepa": {
        "model_name": "hello_deepa",
        "target_phrase": ["hello deepa"],
        "custom_negative_phrases": [
            "deport", "depart", "depend", "depot",
            "dipper", "dipika", "deeksha", "epoch",
            "halo", "pillow", "fellow", "yellow", "jelly", "belly",
            "jello", "below", "elbow",
            "dilip", "dinesh", "dipendra", "priya", "gita", "mira",
            "hello dear", "hello didi", "hello diva", "hello everyone",
            "namaste deepa", "namaste dear", "namaste didi",
        ],
    },
    "namaste_deepa": {
        "model_name": "namaste_deepa",
        "target_phrase": ["namaste deepa"],
        "custom_negative_phrases": [
            "deep", "deeper", "depart", "depend", "depot", "diva", "dipa",
            "depa", "dipper", "dipika", "deeksha", "epoch",
            "stay", "nasty", "master", "musty", "misty",
            "dilip", "dinesh", "dipendra", "ram", "sita", "hari", "shyam",
            "namaste dear", "namaste didi", "namaste dada", "namaste ji", "namaste everyone",
            "hello deepa", "hello dear", "hello didi",
        ],
    },
    "hello_namaste_deepa": {
        "model_name": "hello_namaste_deepa",
        "target_phrase": ["hello deepa", "namaste deepa"],
        "custom_negative_phrases": [
            "depend", "depot", "dipper", "dipika", "deeksha", "epoch",
            "fellow", "yellow", "jello", "below", "elbow",
            "stay", "nasty", "master", "musty", "misty",
            "dilip", "dinesh", "dipendra", "ram", "sita", "hari", "shyam",
            "priya", "gita", "mira",
            "hello dear", "hello didi", "hello diva",
            "namaste dear", "namaste didi", "namaste dada", "namaste ji",
            "hello everyone", "namaste everyone",
        ],
    },
}

# Common training params
common = {
    "n_samples": 50000,
    "n_samples_val": 5000,
    "tts_batch_size": 50,
    "augmentation_batch_size": 16,
    "piper_sample_generator_path": "/kaggle/working/piper-sample-generator",
    "output_dir": "/kaggle/working/my_custom_model",
    "rir_paths": ["/kaggle/working/mit_rirs"],
    "background_paths": ["/kaggle/working/audioset_16k", "/kaggle/working/fma"],
    "background_paths_duplication_rate": [1],
    "false_positive_validation_data_path": "/kaggle/input/wake-word/validation_set_features.npy",
    "feature_data_files": {
        "ACAV100M_sample": "/kaggle/input/acav100m-2000-16bit/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    },
    "augmentation_rounds": 1,
    "model_type": "dnn",
    "layer_size": 32,
    "batch_n_per_class": {"ACAV100M_sample": 4096, "adversarial_negative": 50, "positive": 50},
    "steps": 50000,
    "max_negative_weight": 5,
    "target_false_positives_per_hour": 0.2,
    "target_accuracy": 0.7,
    "target_recall": 0.5,
}

config = {**common, **CONFIGS[MODEL_CHOICE]}
config_path = WORK / "my_model.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Config written: {config_path}")
print(f"Model Target: {config['model_name']}")

# %%
# ============================================================
# CELL 5: Inject Multi-TTS Clips (Edge-TTS + Google TTS)
# ============================================================
import shutil

# Look for uploaded wise-activate data
data_sources = [
    Path("/kaggle/input/wise-activate-data/data"), 
    Path("/kaggle/input/wise-activate/data"),
    Path("/kaggle/working/wise-activate/data"),
]

data_dir = None
for src in data_sources:
    if src.exists():
        data_dir = src
        break

if data_dir:
    model_name = config["model_name"]
    output_clips = Path(config["output_dir"]) / model_name
    output_clips.mkdir(parents=True, exist_ok=True)

    keyword = "deepa" 
    source_root = data_dir / "wake" / keyword

    copied = 0
    if source_root.exists():
        for lang_dir in source_root.iterdir():
            if not lang_dir.is_dir(): continue
            for wav_file in lang_dir.glob("*.wav"):
                dest = output_clips / f"multitts_{lang_dir.name}_{wav_file.name}"
                if not dest.exists():
                    shutil.copy2(wav_file, dest)
                    copied += 1

    # Hard negatives (adversarial clips)
    hard_neg = data_dir / "background" / "hard_negatives"
    if hard_neg.exists():
        adv_dir = output_clips / "adversarial_clips"
        adv_dir.mkdir(exist_ok=True)
        for f in hard_neg.rglob("*"):
            if f.suffix in [".wav", ".mp3"]:
                dest = adv_dir / f"hardneg_{f.name}"
                if not dest.exists():
                    shutil.copy2(f, dest)
                    copied += 1

    print(f"Injected {copied} high-diversity clips from Edge/Google TTS.")
else:
    print("WARNING: No data found. Training will use Piper TTS only.")

# %%
# ============================================================
# CELL 6: Start 4-Phase Training Pipeline
# ============================================================
train_script = str(WORK / "openwakeword" / "openwakeword" / "train.py")
config_file = str(config_path)

# Phase 1: Generate synthetic clips
print("\n--- PHASE 1: Piper TTS Generation ---")
os.system(f"{sys.executable} {train_script} --training_config {config_file} --generate_clips")

# Phase 2: Augmentation
print("\n--- PHASE 2: Augmenting Clips ---")
os.system(f"{sys.executable} {train_script} --training_config {config_file} --augment_clips")

# Phase 3: Feature Extraction
print("\n--- PHASE 3: Extracting Features ---")
os.system(f"{sys.executable} {train_script} --training_config {config_file} --generate_features")

# Phase 4: Main Training
print("\n--- PHASE 4: Training Neural Network ---")
os.system(f"{sys.executable} {train_script} --training_config {config_file} --train")

# %%
# ============================================================
# CELL 7: Final Export & Verification
# ============================================================
output_dir = Path(config["output_dir"])
onnx_path = output_dir / f"{config['model_name']}.onnx"

if onnx_path.exists():
    size_kb = onnx_path.stat().st_size / 1024
    print(f"\nSUCCESS! Model saved to: {onnx_path}")
    print(f"Final Model Size: {size_kb:.1f} KB")
else:
    print("\nERROR: Model file not found. Check previous cells for failures.")

print("\nFiles in output directory:")
if output_dir.exists():
    for f in sorted(output_dir.rglob("*")):
        if f.is_file(): print(f"  {f.name} ({f.stat().st_size/1024:.1f} KB)")
