#!/usr/bin/env python3
"""
====================================================================
WiseYak openWakeWord Training — Google Colab Notebook
====================================================================
Copy-paste each numbered section into a separate Colab notebook cell.
Requires:
  - Colab GPU runtime (T4 recommended)
  - Your wise-activate data/ folder uploaded to Google Drive

Last Updated: 2026-02-10 (Pinned Dependencies & Sync'd Configs)
====================================================================
"""

# %%
# ============================================================
# CELL 1: Mount Drive & Install Pinned Dependencies
# ============================================================
import subprocess, sys, os
from pathlib import Path
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

WORK = Path("/content")
DRIVE_DATA = Path("/content/drive/MyDrive/wise-activate/data")

deps = [
    # "piper-tts==1.2.0",  <-- Removed from here, installed below with --no-deps
    "piper-phonemize-cross==1.2.0", 
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
    "librosa==0.10.1",
    "scipy==1.11.3"
]

print("Installing pinned dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + deps, check=False)

# Install piper-tts WITHOUT dependencies (avoids broken piper-phonemize~=1.1.0)
print("Installing piper-tts (no-deps)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "piper-tts", "--no-deps"], check=False)

print("Dependencies installed successfully.")

# %%
# ============================================================
# CELL 2: Clone Repos & Download Resource Models
# ============================================================

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

# Install openWakeWord (MUST happen after Cell 1 deps are installed,
# otherwise 'import openwakeword' fails on missing onnxruntime)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                str(WORK / 'openwakeword'), "--no-deps"], check=False)

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

# Set up features path
FEATURES_DIR = WORK / "features"
FEATURES_DIR.mkdir(exist_ok=True)

# Add piper to Python path
sys.path.insert(0, str(WORK / "piper-sample-generator"))

print("Repos cloned and models downloaded.")

# %%
# ============================================================
# CELL 3: Patch openWakeWord Source (Critical Stability Fixes)
# ============================================================
import re, numpy as np

# --- Patch data.py ---
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

# Download or detect features on Drive
# (User: upload acav100m features to Drive to avoid long downloads!)
acav_name = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
drive_acav = Path("/content/drive/MyDrive/wise-activate/features") / acav_name
local_acav = FEATURES_DIR / acav_name

if drive_acav.exists():
    print("Detected ACAV100M features on Drive. Linking...")
    os.system(f"ln -s {drive_acav} {local_acav}")
else:
    print("WARNING: ACAV100M features not on Drive. You'll need to download it (~2GB).")

# Validation features
val_name = "validation_set_features.npy"
drive_val = Path("/content/drive/MyDrive/wise-activate/features") / val_name
local_val = FEATURES_DIR / val_name
if drive_val.exists(): os.system(f"ln -s {drive_val} {local_val}")

common = {
    "n_samples": 50000, "n_samples_val": 5000,
    "tts_batch_size": 50, "augmentation_batch_size": 16,
    "piper_sample_generator_path": "/content/piper-sample-generator",
    "output_dir": "/content/my_custom_model",
    "rir_paths": ["/content/mit_rirs"],
    "background_paths": ["/content/audioset_16k", "/content/fma"],
    "background_paths_duplication_rate": [1],
    "false_positive_validation_data_path": str(local_val),
    "feature_data_files": {"ACAV100M_sample": str(local_acav)},
    "augmentation_rounds": 1,
    "model_type": "dnn", "layer_size": 32,
    "batch_n_per_class": {"ACAV100M_sample": 4096, "adversarial_negative": 50, "positive": 50},
    "steps": 50000, "max_negative_weight": 5, "target_false_positives_per_hour": 0.2,
    "target_accuracy": 0.7, "target_recall": 0.5,
}

config = {**common, **CONFIGS[MODEL_CHOICE]}
config_path = WORK / "my_model.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Config Ready: {config_path}")

# %%
# ============================================================
# CELL 5: Inject Multi-TTS Clips from Drive
# ============================================================
import shutil

keyword = "deepa"
source_root = DRIVE_DATA / "wake" / keyword
output_clips = Path(config["output_dir"]) / config["model_name"]
output_clips.mkdir(parents=True, exist_ok=True)

copied = 0
if source_root.exists():
    for lang_dir in source_root.iterdir():
        if not lang_dir.is_dir(): continue
        for wav_file in lang_dir.glob("*.wav"):
            dest = output_clips / f"multitts_{lang_dir.name}_{wav_file.name}"
            shutil.copy2(wav_file, dest)
            copied += 1

# Hard negatives
hard_neg = DRIVE_DATA / "background" / "hard_negatives"
if hard_neg.exists():
    adv_dir = output_clips / "adversarial_clips"
    adv_dir.mkdir(exist_ok=True)
    for f in hard_neg.rglob("*"):
        if f.suffix in [".wav", ".mp3"]:
            shutil.copy2(f, adv_dir / f"hardneg_{f.name}")
            copied += 1

print(f"Injected {copied} diverse TTS clips from Google Drive.")

# %%
# ============================================================
# CELL 6: Start 4-Phase Training
# ============================================================
train_script = str(WORK / "openwakeword" / "openwakeword" / "train.py")
config_file = str(config_path)

os.system(f"{sys.executable} {train_script} --training_config {config_file} --generate_clips")
os.system(f"{sys.executable} {train_script} --training_config {config_file} --augment_clips")
os.system(f"{sys.executable} {train_script} --training_config {config_file} --generate_features")
os.system(f"{sys.executable} {train_script} --training_config {config_file} --train")

# %%
# ============================================================
# CELL 7: Backup Model to Drive
# ============================================================
onnx_path = Path(config["output_dir"]) / f"{config['model_name']}.onnx"
drive_backup = Path("/content/drive/MyDrive/wise-activate/models")
drive_backup.mkdir(parents=True, exist_ok=True)

if onnx_path.exists():
    shutil.copy2(onnx_path, drive_backup / onnx_path.name)
    print(f"SUCCESS! Model backed up to Drive: {drive_backup / onnx_path.name}")
else:
    print("ERROR: Training failed, model not found.")
