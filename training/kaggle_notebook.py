#!/usr/bin/env python3
"""
====================================================================
WiseYak openWakeWord Training — Kaggle Notebook (v4 - Final Clean)
====================================================================
Copy-paste each numbered section into a separate Kaggle notebook cell.
Requires:
  - Kaggle GPU (P100 / T4)
  - Dataset 1: wise-activate data (zip of wake/ + background/ folders)
  - Dataset 2: ACAV100M features .npy  (optional but recommended)
====================================================================
"""

# %%
# ============================================================
# CELL 1 — Install Dependencies
# ============================================================
import subprocess, sys, os

deps = [
    "numpy>=2.0.0", "scipy>=1.13.1", "scikit-learn>=1.5.0",
    "webrtcvad==2.0.10", "mutagen==1.47.0",
    "audiomentations==0.33.0", "torch-audiomentations==0.11.0", "acoustics==0.2.6",
    "speechbrain==0.5.14", "torchinfo==1.8.0", "torchmetrics==1.2.0",
    "onnxruntime-gpu", "onnxsim", "onnx2tf",
    "onnx_graphsurgeon", "sng4onnx", "ai_edge_litert",
    "pronouncing==0.2.0", "deep-phonemizer==0.0.19",
    "datasets==2.14.6",
]

print("Installing dependencies...")
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q"] + deps,
    capture_output=True, text=True,
)
if r.returncode != 0:
    print(f"WARNING:\n{r.stderr[-800:]}")
else:
    print("All dependencies installed.")

try:
    import onnxruntime
    print(f"onnxruntime {onnxruntime.__version__}")
except ImportError:
    print("CRITICAL: onnxruntime not found")

# %%
# ============================================================
# CELL 2 — Clone openWakeWord & Locate ACAV100M
# ============================================================
from pathlib import Path
import shutil

WORK = Path("/kaggle/working")

# 1. Clone repo
if not (WORK / "openwakeword").exists():
    subprocess.run(["git", "clone", "https://github.com/dscripka/openwakeword",
                     str(WORK / "openwakeword")], check=True)
    shutil.rmtree(WORK / "openwakeword" / ".git", ignore_errors=True)

# 2. Install editable (no deps — we already installed them)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                str(WORK / "openwakeword"), "--no-deps"], check=False)

# 3. Download embedding + mel-spectrogram ONNX models
res_dir = WORK / "openwakeword" / "openwakeword" / "resources" / "models"
res_dir.mkdir(parents=True, exist_ok=True)
for fname in ["embedding_model.onnx", "melspectrogram.onnx"]:
    dest = res_dir / fname
    if not dest.exists():
        url = f"https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{fname}"
        subprocess.run(["wget", "-q", url, "-O", str(dest)], check=True)

# 4. Find ACAV100M features (.npy) — search all /kaggle/input
acav_path = ""
for root in [Path("/kaggle/input")]:
    if root.exists():
        for npy in root.rglob("*ACAV100M*.npy"):
            acav_path = str(npy)
            print(f"Found ACAV100M: {npy}")
            break
if not acav_path:
    wk = WORK / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    if wk.exists():
        acav_path = str(wk)
if not acav_path:
    print("ACAV100M not found — false-positive validation disabled")

# 5. Download false-positive validation features from HuggingFace (~500 MB)
val_features_path = WORK / "validation_set_features.npy"
if not val_features_path.exists():
    print("Downloading validation_set_features.npy from HuggingFace...")
    subprocess.run([
        "wget", "-q",
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
        "-O", str(val_features_path)
    ], check=False)
if val_features_path.exists() and val_features_path.stat().st_size > 1000:
    print(f"Validation features ready: {val_features_path.stat().st_size/1024/1024:.0f} MB")
    val_features_str = str(val_features_path)
else:
    print("Validation features download failed — FP validation disabled")
    val_features_str = ""

print("Setup complete.")

# %%
# ============================================================
# CELL 3 — Patch openWakeWord train.py (7 fixes)
# ============================================================
import re
import numpy as np

train_py = WORK / "openwakeword" / "openwakeword" / "train.py"
content = train_py.read_text()
patched = False

# 1. Remove top-level generate_samples import (we skip Piper TTS)
if "from generate_samples import generate_samples" in content:
    content = content.replace(
        "from generate_samples import generate_samples",
        "# from generate_samples import generate_samples  # patched",
    )
    patched = True

# 2. Guard piper_sample_generator_path KeyError
if 'config["piper_sample_generator_path"]' in content:
    content = content.replace(
        'sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))',
        'sys.path.insert(0, os.path.abspath(config.get("piper_sample_generator_path", ".")))',
    )
    patched = True

# 3. PyTorch 2.6 weights_only fix
if "_patched_torch_load" not in content:
    patch = (
        "\n# PyTorch 2.6 fix\n"
        "_original_torch_load = torch.load\n"
        "def _patched_torch_load(*args, **kwargs):\n"
        "    if 'weights_only' not in kwargs: kwargs['weights_only'] = False\n"
        "    return _original_torch_load(*args, **kwargs)\n"
        "torch.load = _patched_torch_load\n\n"
    )
    content = content.replace("import torch\n", "import torch\n" + patch, 1)
    patched = True

# 4. Skip TFLite conversion (replace call, keep definition)
if "convert_onnx_to_tflite" in content:
    content = re.sub(
        r'(?<!def )convert_onnx_to_tflite\(',
        'print("Skipping TFLite:", ',
        content,
    )
    patched = True

# 5. Increase negative weight multiplier
if "max_negative_weight = max_negative_weight*2" in content:
    content = content.replace(
        "max_negative_weight = max_negative_weight*2",
        "max_negative_weight = max_negative_weight*5",
    )
    patched = True

# 6. Diversify phoneme augmentation in data.py
data_py = WORK / "openwakeword" / "openwakeword" / "data.py"
if data_py.exists():
    dtxt = data_py.read_text()
    old = "max_replace=max(0, len(phones)-5)"
    new = "max_replace=max(0, len(phones)-np.random.randint(2, len(phones)))"
    if old in dtxt:
        data_py.write_text(dtxt.replace(old, new))
        print("Patched data.py (phoneme diversity)")

# 7. Bulletproof config — fill missing optional keys after YAML load
if "config = yaml." in content:
    safety = """
    # Bulletproof: fill missing optional keys
    _defaults = {
        "rir_paths": [], "background_paths": [], "background_paths_duplication_rate": [],
        "include_vad": False, "piper_sample_generator_path": "", "noise_paths": [],
        "feature_data_files": {}, "batch_n_per_class": {},
        "false_positive_validation_data_path": "", "augmentation_rounds": 3,
        "augmentation_batch_size": 128, "model_type": "dnn", "layer_size": 32,
        "target_false_positives_per_hour": 0.2, "custom_negative_phrases": [],
        "tts_batch_size": 50,
    }
    for _k, _v in _defaults.items():
        if _k not in config: config[_k] = _v
"""
    content = re.sub(
        r'(config = yaml\.(?:safe_)?load\(.*\))',
        r'\1' + safety,
        content,
    )
    patched = True

# 8. Guard false-positive validation loading (skip if file missing)
if 'X_val_fp = np.load(config["false_positive_validation_data_path"])' in content:
    old_fp_block = 'X_val_fp = np.load(config["false_positive_validation_data_path"])'
    new_fp_block = """_fp_path = config.get("false_positive_validation_data_path", "")
        if _fp_path and os.path.exists(_fp_path):
            X_val_fp = np.load(_fp_path)
        else:
            logging.warning("No false-positive validation data found, using dummy")
            X_val_fp = np.zeros((input_shape[0]+1, input_shape[1]), dtype=np.float32)"""
    content = content.replace(old_fp_block, new_fp_block)
    patched = True
    print("Patched: false-positive validation guard")

# 9. Multi-GPU DataParallel (use both T4 GPUs)
_dp_target = "self.to(self.device)\n        self.model.to(self.device)"
if _dp_target in content and "DataParallel" not in content:
    content = content.replace(
        _dp_target,
        _dp_target + "\n"
        "        if torch.cuda.device_count() > 1 and not isinstance(self.model, nn.DataParallel):\n"
        "            logging.info(f'Using {torch.cuda.device_count()} GPUs via DataParallel')\n"
        "            self.model = nn.DataParallel(self.model)",
    )
    # Unwrap DataParallel when saving checkpoints (deepcopy)
    if "copy.deepcopy(self.model)" in content:
        content = content.replace(
            "copy.deepcopy(self.model)",
            "copy.deepcopy(self.model.module if hasattr(self.model, 'module') else self.model)",
        )
    # Unwrap DataParallel when exporting to ONNX
    if "def export_model(self, model, model_name, output_dir):" in content:
        content = content.replace(
            "def export_model(self, model, model_name, output_dir):",
            "def export_model(self, model, model_name, output_dir):\n"
            "        model = model.module if hasattr(model, 'module') else model",
        )
    patched = True
    print("Patched: Multi-GPU DataParallel")

if patched:
    train_py.write_text(content)
    print("train.py patched (9 fixes)")
else:
    print("train.py already patched")

# %%
# ============================================================
# CELL 4 — Training Config
# ============================================================
# >>> EDIT THIS LINE to choose your model <<<
MODEL_CHOICE = "namaste_deepa"   # "hello_deepa" | "namaste_deepa" | "hello_namaste_deepa"

import yaml

MODELS = {
    "hello_deepa":          {"target_phrase": ["hello deepa"]},
    "namaste_deepa":        {"target_phrase": ["namaste deepa"]},
    "hello_namaste_deepa":  {"target_phrase": ["hello deepa", "namaste deepa"]},
}

config = {
    "model_name":  MODEL_CHOICE,
    "target_phrase": MODELS[MODEL_CHOICE]["target_phrase"],
    "n_samples":   50000,
    "n_samples_val": 5000,
    "output_dir":  "/kaggle/working/my_custom_model",
    "steps":       30000,
    "target_accuracy": 0.7,
    "max_negative_weight": 5,
    "augmentation_rounds": 3,
    "augmentation_batch_size": 128,  # larger batches to fill 2xT4 GPUs
    "model_type": "dnn",
    "layer_size": 32,
    "target_false_positives_per_hour": 0.2,
    "include_vad": False,
    # Dynamic — based on whether ACAV100M exists
    "feature_data_files":  {"ACAV100M_sample": acav_path} if acav_path else {},
    "batch_n_per_class":   ({"ACAV100M_sample": 2048, "adversarial_negative": 100, "positive": 50}
                            if acav_path else
                            {"adversarial_negative": 200, "positive": 50}),
    "false_positive_validation_data_path": val_features_str,
    # Robustness defaults
    "rir_paths": [], "background_paths": [], "background_paths_duplication_rate": [],
    "noise_paths": [], "piper_sample_generator_path": "",
}

config_path = WORK / "my_model.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f)
print(f"Config: {config['model_name']}  phrases={config['target_phrase']}")

# %%
# ============================================================
# CELL 5 — Inject Data into openWakeWord directory structure
# ============================================================
# train.py expects:
#   output_dir/model_name/{positive_train, positive_test}/*.wav
#   output_dir/model_name/{negative_train, negative_test}/*.wav
# ============================================================
import shutil, random

# Locate uploaded data
data_dir = None
for cand in [
    Path("/kaggle/input/wise-wake-deepa/data"),
    Path("/kaggle/input/wise-wake-deepa"),
]:
    if cand.exists() and (cand / "wake").exists():
        data_dir = cand
        break
    # One level deeper
    for sub in (cand.iterdir() if cand.exists() else []):
        if (sub / "wake").exists():
            data_dir = sub
            break
    if data_dir:
        break

assert data_dir, "Data not found! Check your Kaggle dataset name."

base = Path(config["output_dir"]) / config["model_name"]
dirs = {k: base / k for k in ["positive_train", "positive_test", "negative_train", "negative_test"]}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

# --- Positives (Adjusted for specific subfolders) ---
if config["model_name"] == "namaste_deepa":
    src_wake = data_dir / "wake" / "deepa" / "ne"
    print(f"Selecting ONLY 'namaste' data from: {src_wake}")
elif config["model_name"] == "hello_deepa":
    src_wake = data_dir / "wake" / "deepa" / "en"
    print(f"Selecting ONLY 'hello' data from: {src_wake}")
else:
    # "hello_namaste_deepa" or fallback
    src_wake = data_dir / "wake" / "deepa"
    print(f"Selecting ALL data from: {src_wake}")

pos_files = list(src_wake.rglob("*.wav")) if src_wake.exists() else []
random.shuffle(pos_files)
split = max(1, int(len(pos_files) * 0.9))

for i, f in enumerate(pos_files[:split]):
    shutil.copy2(f, dirs["positive_train"] / f"p{i}_{f.name}")
for i, f in enumerate(pos_files[split:]):
    shutil.copy2(f, dirs["positive_test"] / f"p{i}_{f.name}")
print(f"Positive: {split} train / {len(pos_files)-split} test")

# --- Negatives (background/hard_negatives) ---
src_neg = data_dir / "background" / "hard_negatives"
neg_files = [f for f in src_neg.rglob("*") if f.suffix in (".wav", ".mp3")] if src_neg.exists() else []
random.shuffle(neg_files)
nsplit = max(1, int(len(neg_files) * 0.9))

for i, f in enumerate(neg_files[:nsplit]):
    shutil.copy2(f, dirs["negative_train"] / f"n{i}_{f.name}")
for i, f in enumerate(neg_files[nsplit:]):
    shutil.copy2(f, dirs["negative_test"] / f"n{i}_{f.name}")
print(f"Negative: {nsplit} train / {len(neg_files)-nsplit} test")
print(f"Total: {len(pos_files)} positive + {len(neg_files)} negative")

# %%
# ============================================================
# CELL 6 — Train (augment then train)
# ============================================================
train_script = str(WORK / "openwakeword" / "openwakeword" / "train.py")

def run(args):
    cmd = [sys.executable, train_script, "--training_config", str(config_path)] + args
    print(f">>> {' '.join(args)}")
    subprocess.run(cmd, check=False)

run(["--augment_clips"])
run(["--train_model"])

# %%
# ============================================================
# CELL 7 — Export & Evaluate
# ============================================================
output_dir = Path(config["output_dir"])
onnx_path  = output_dir / f"{config['model_name']}.onnx"

if onnx_path.exists():
    print(f"Model: {onnx_path}  ({onnx_path.stat().st_size/1024:.1f} KB)")
else:
    print(f"Model not found at {onnx_path}")

# Optional: ROC + PR curves
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

try:
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0].name

    pos_scores, neg_scores = [], []
    for tag, bucket in [("positive", pos_scores), ("negative", neg_scores)]:
        for npy in sorted(output_dir.rglob(f"*{tag}*.npy"))[:2]:
            data = np.load(npy)
            for i in range(min(500, len(data))):
                pred = session.run(None, {inp: data[i:i+1].astype(np.float32)})
                bucket.append(float(pred[0][0][0]))

    if pos_scores and neg_scores:
        y = [1]*len(pos_scores) + [0]*len(neg_scores)
        s = pos_scores + neg_scores
        
        # 3 Plots: ROC, PR, Histogram
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y, s)
        ax[0].plot(fpr, tpr, lw=2, label=f"AUC={auc(fpr,tpr):.3f}")
        ax[0].plot([0,1],[0,1],"k--", alpha=0.5)
        ax[0].set_title("ROC Curve")
        ax[0].set_xlabel("False Positive Rate")
        ax[0].set_ylabel("True Positive Rate")
        ax[0].legend(loc="lower right")
        ax[0].grid(True, alpha=0.3)

        # 2. Precision-Recall
        prec, rec, _ = precision_recall_curve(y, s)
        ax[1].plot(rec, prec, lw=2, label=f"AP={average_precision_score(y,s):.3f}")
        ax[1].set_title("Precision-Recall")
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")
        ax[1].legend(loc="lower left")
        ax[1].grid(True, alpha=0.3)

        # 3. Score Distribution (Histogram)
        ax[2].hist(pos_scores, bins=30, alpha=0.6, label='Positives', density=True, color='green')
        ax[2].hist(neg_scores, bins=30, alpha=0.6, label='Negatives', density=True, color='red')
        ax[2].set_title("Score Distribution")
        ax[2].set_xlabel("Model Probability")
        ax[2].set_ylabel("Density")
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "metrics_enhanced.png")
        plt.show()
        
        # Text Summary
        print(f"Total Evaluated: {len(pos_scores)} positives, {len(neg_scores)} negatives")
except Exception as e:
    print(f"Eval skipped: {e}")

print("\nDone — download the .onnx model from the Output tab.")
