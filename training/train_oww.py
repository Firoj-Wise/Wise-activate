#!/usr/bin/env python3
"""
WiseYak openWakeWord Training Pipeline
=======================================
Designed for Kaggle/Colab (Linux). Orchestrates:
  1. Environment setup (clone repos, download models)
  2. Synthetic data generation via Piper TTS
  3. Injection of pre-generated multi-TTS clips (Edge-TTS, Google TTS)
  4. Feature extraction and model training
  5. ONNX export

Usage (Kaggle/Colab):
    python train_oww.py --config training/oww_configs/hey_deepak.yaml

Prerequisites:
    - Linux environment (Piper TTS requirement)
    - GPU recommended (CUDA)
    - Pre-generated multi-TTS clips in data/wake/ directory
"""

import os
import sys
import re
import shutil
import subprocess
import argparse
import yaml
import logging
import pathlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — always to file + console
# ---------------------------------------------------------------------------
root_logger = logging.getLogger()
if root_logger.handlers:
    root_logger.handlers.clear()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training_oww.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KAGGLE_WORKING   = Path("/kaggle/working")
OWW_REPO         = KAGGLE_WORKING / "openwakeword"
PIPER_REPO       = KAGGLE_WORKING / "piper-sample-generator"
PIPER_MODEL_NAME = "en_US-libritts_r-medium.pt"

OWW_RESOURCE_MODELS = [
    ("embedding_model.onnx",    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"),
    ("embedding_model.tflite",  "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite"),
    ("melspectrogram.onnx",     "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"),
    ("melspectrogram.tflite",   "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite"),
]


def run(cmd: str, check: bool = True):
    """Run a shell command, streaming output."""
    logging.info(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode


# ===================================================================
# STEP 0 — Environment Setup
# ===================================================================

def setup_environment():
    """Clone repos, install deps, download models."""

    # --- Piper ---
    if not PIPER_REPO.exists():
        run(f"git clone https://github.com/rhasspy/piper-sample-generator {PIPER_REPO}")
    piper_model = PIPER_REPO / "models" / PIPER_MODEL_NAME
    if not piper_model.exists():
        run(f"wget -O {piper_model} "
            f"'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/{PIPER_MODEL_NAME}'")

    # --- openWakeWord ---
    if not OWW_REPO.exists():
        run(f"git clone https://github.com/dscripka/openwakeword {OWW_REPO}")
    run(f"pip install -e {OWW_REPO} --no-deps", check=False)

    # Resource models
    res_dir = OWW_REPO / "openwakeword" / "resources" / "models"
    res_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in OWW_RESOURCE_MODELS:
        dest = res_dir / fname
        if not dest.exists():
            run(f"wget -q {url} -O {dest}")

    # --- Python deps ---
    deps = [
        "piper-tts", "piper-phonemize-cross", "webrtcvad",
        "mutagen==1.47.0", "torchinfo==1.8.0", "torchmetrics==1.2.0",
        "speechbrain==0.5.14", "audiomentations==0.33.0",
        "torch-audiomentations==0.11.0", "acoustics==0.2.6",
        "onnxruntime==1.22.1", "ai_edge_litert==1.4.0", "onnxsim",
        "onnx2tf", "onnx_graphsurgeon", "sng4onnx",
        "pronouncing==0.2.0", "datasets==2.14.6",
        "deep-phonemizer==0.0.19",
    ]
    run(f"pip install {' '.join(deps)}", check=False)

    # Add piper to path
    piper_str = str(PIPER_REPO)
    if piper_str not in sys.path:
        sys.path.insert(0, piper_str)

    logging.info("Environment setup complete.")


# ===================================================================
# STEP 1 — Patch openWakeWord source files
# ===================================================================

def patch_data_py():
    """Patch data.py for better phoneme augmentation diversity."""
    data_py = OWW_REPO / "openwakeword" / "data.py"
    if not data_py.exists():
        logging.warning(f"data.py not found at {data_py}, skipping patch.")
        return

    text = data_py.read_text()
    old = "max_replace=max(0, len(phones)-5)"
    new = "max_replace=max(0, len(phones)-np.random.randint(2, len(phones)))"
    if old in text:
        text = text.replace(old, new)
        data_py.write_text(text)
        logging.info("Patched data.py: phoneme augmentation diversity increased.")
    else:
        logging.info("data.py already patched or pattern not found.")


def patch_train_py():
    """Apply all critical patches to train.py."""
    train_py = OWW_REPO / "openwakeword" / "train.py"
    if not train_py.exists():
        logging.error(f"train.py not found at {train_py}")
        return

    content = train_py.read_text()
    patched = False

    # -- Patch 1: PyTorch 2.6 weights_only fix --
    if "_patched_torch_load" not in content:
        torch_patch = (
            "\n# --- WiseYak Patch: PyTorch 2.6 weights_only fix ---\n"
            "_original_torch_load = torch.load\n"
            "def _patched_torch_load(*args, **kwargs):\n"
            "    if 'weights_only' not in kwargs:\n"
            "        kwargs['weights_only'] = False\n"
            "    return _original_torch_load(*args, **kwargs)\n"
            "torch.load = _patched_torch_load\n"
            "# --- End Patch ---\n\n"
        )
        content = content.replace("import torch\n", "import torch\n" + torch_patch, 1)
        patched = True
        logging.info("Patched: PyTorch 2.6 weights_only fix.")

    # -- Patch 2: Add model_path arg to generate_samples calls --
    if "if args.generate_clips is True:" in content and "model_path = os.path.join" not in content:
        model_path_block = (
            '    if args.generate_clips is True:\n'
            '        # WiseYak: resolve Piper model path\n'
            '        model_path = os.path.join(\n'
            '            config["piper_sample_generator_path"], "models", "en_US-libritts_r-medium.pt"\n'
            '        )\n'
            '        if not os.path.exists(model_path):\n'
            '            raise FileNotFoundError(f"Piper model not found at {model_path}")\n'
        )
        content = content.replace("    if args.generate_clips is True:", model_path_block)
        patched = True
        logging.info("Patched: Piper model_path injection.")

    # Inject model= kwarg into all generate_samples() calls
    for pattern_str in [
        r'generate_samples\(\s*\n\s+text=config\["target_phrase"\],',
        r'generate_samples\(text=config\["target_phrase"\],',
        r'generate_samples\(text=adversarial_texts,',
    ]:
        for m in list(re.finditer(pattern_str, content)):
            region = content[m.start():min(m.start()+500, len(content))]
            if "model=model_path" not in region:
                replacement = m.group(0).rstrip(",") + "\n                model=model_path,"
                content = content[:m.start()] + replacement + content[m.end():]
                patched = True

    # -- Patch 3: Fix gradient accumulation stability --
    # Add accumulated_weights initialization
    old_accum = (
        '        accumulated_labels = torch.Tensor([]).to(self.device)\n'
        '        for step_ndx, data in tqdm(enumerate(X, 0), total=max_steps, desc="Training"):'
    )
    new_accum = (
        '        accumulated_labels = torch.Tensor([]).to(self.device)\n'
        '        accumulated_weights = torch.Tensor([]).to(self.device)\n'
        '        for step_ndx, data in tqdm(enumerate(X, 0), total=max_steps, desc="Training"):'
    )
    if old_accum in content and "accumulated_weights" not in content:
        content = content.replace(old_accum, new_accum)
        patched = True
        logging.info("Patched: Gradient accumulation weights init.")

    # -- Patch 4: Skip TFLite conversion (compat issues) --
    tflite_call = (
        '        # Convert the model from onnx to tflite format\n'
        '        convert_onnx_to_tflite(os.path.join(config["output_dir"], config["model_name"] + ".onnx"),\n'
        '                               os.path.join(config["output_dir"], config["model_name"] + ".tflite"))'
    )
    tflite_skip = (
        '        # WiseYak: Skip TFLite conversion (TensorFlow compat issues)\n'
        '        # convert_onnx_to_tflite(os.path.join(config["output_dir"], config["model_name"] + ".onnx"),\n'
        '        #                        os.path.join(config["output_dir"], config["model_name"] + ".tflite"))'
    )
    if tflite_call in content:
        content = content.replace(tflite_call, tflite_skip)
        patched = True
        logging.info("Patched: Skipped TFLite conversion.")

    # -- Patch 5: Increase negative weight scaling for sequence 2+3 --
    if "max_negative_weight = max_negative_weight*2" in content:
        content = content.replace(
            "max_negative_weight = max_negative_weight*2",
            "max_negative_weight = max_negative_weight*5"
        )
        patched = True
        logging.info("Patched: Increased negative weight scaling (2x -> 5x).")

    if patched:
        train_py.write_text(content)
        logging.info("All patches applied to train.py.")
    else:
        logging.info("train.py already fully patched.")


# ===================================================================
# STEP 2 — Inject pre-generated multi-TTS data
# ===================================================================

def inject_multi_tts_clips(config: dict, project_data_dir: Path):
    """
    Copy pre-generated Edge-TTS and Google TTS clips from the project's
    data/ directory into the openWakeWord output directory so they get
    processed as positive training examples.

    This is the key to spectrogram diversity:
    - Piper TTS generates one "signature"
    - Edge-TTS generates a different one (Neural voices)
    - Google TTS generates yet another (Wavenet/Neural2/Studio)

    The model learns to detect the WORD, not the TTS engine.
    """
    model_name = config["model_name"]
    output_clips = Path(config["output_dir"]) / model_name
    output_clips.mkdir(parents=True, exist_ok=True)

    # Determine which keyword we're training for
    # "hello_deepa" -> look in data/wake/deepa/
    # "hey_deepak"  -> look in data/wake/deepak/
    keyword = None
    if "deepak" in model_name:
        keyword = "deepak"
    elif "deepa" in model_name:
        keyword = "deepa"

    if keyword is None:
        logging.warning(f"Could not determine keyword from model_name '{model_name}'. "
                        f"Multi-TTS injection skipped.")
        return 0

    # Look for wav files in all language subdirectories
    source_root = project_data_dir / "wake" / keyword
    if not source_root.exists():
        logging.warning(f"Source directory {source_root} not found. Multi-TTS injection skipped.")
        return 0

    copied = 0
    for lang_dir in source_root.iterdir():
        if not lang_dir.is_dir():
            continue
        for wav_file in lang_dir.glob("*.wav"):
            dest = output_clips / f"multitts_{lang_dir.name}_{wav_file.name}"
            if not dest.exists():
                shutil.copy2(wav_file, dest)
                copied += 1

    # Also inject hard negatives as adversarial clips
    hard_neg_dir = project_data_dir / "background" / "hard_negatives"
    adversarial_dest = Path(config["output_dir"]) / model_name / "adversarial_clips"
    if hard_neg_dir.exists():
        adversarial_dest.mkdir(parents=True, exist_ok=True)
        for f in hard_neg_dir.rglob("*"):
            if f.suffix in [".wav", ".mp3"]:
                dest = adversarial_dest / f"hardneg_{f.name}"
                if not dest.exists():
                    shutil.copy2(f, dest)
                    copied += 1

    logging.info(f"Injected {copied} pre-generated multi-TTS clips into {output_clips}")
    return copied


# ===================================================================
# STEP 3 — Run training
# ===================================================================

def run_training(config_path: str, skip_generate: bool = False):
    """Execute the openWakeWord training pipeline."""
    train_script = OWW_REPO / "openwakeword" / "train.py"
    if not train_script.exists():
        logging.error(f"train.py not found at {train_script}")
        return

    steps = []

    if not skip_generate:
        # Phase 1: Generate synthetic clips
        steps.append(f"{sys.executable} {train_script} --training_config {config_path} --generate_clips")

    # Phase 2: Augment clips with room impulse responses and noise
    steps.append(f"{sys.executable} {train_script} --training_config {config_path} --augment_clips")

    # Phase 3: Extract features from augmented clips
    steps.append(f"{sys.executable} {train_script} --training_config {config_path} --generate_features")

    # Phase 4: Train the model
    steps.append(f"{sys.executable} {train_script} --training_config {config_path} --train")

    for i, cmd in enumerate(steps, 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"PHASE {i}/{len(steps)}")
        logging.info(f"{'='*60}")
        retcode = run(cmd, check=False)
        if retcode != 0:
            logging.error(f"Phase {i} failed with return code {retcode}")
            logging.error("Check training_oww.log for details.")
            return False

    logging.info("\n" + "=" * 60)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 60)
    return True


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WiseYak openWakeWord Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (setup + generate + train):
  python train_oww.py --config training/oww_configs/hey_deepak.yaml

  # Skip environment setup (already done):
  python train_oww.py --config training/oww_configs/hey_deepak.yaml --skip-setup

  # Skip clip generation (already have clips):
  python train_oww.py --config training/oww_configs/hey_deepak.yaml --skip-generate

  # Setup only (no training):
  python train_oww.py --setup-only
        """,
    )
    parser.add_argument("--config",       type=str, help="Path to YAML config file")
    parser.add_argument("--skip-setup",   action="store_true", help="Skip environment setup")
    parser.add_argument("--skip-generate",action="store_true", help="Skip Piper TTS clip generation")
    parser.add_argument("--setup-only",   action="store_true", help="Only set up environment, then exit")
    parser.add_argument("--data-dir",     type=str, default=None,
                        help="Path to project data/ directory for multi-TTS injection")
    args = parser.parse_args()

    # ----- Step 0: Environment -----
    if not args.skip_setup:
        logging.info("STEP 0: Setting up environment...")
        setup_environment()

    if args.setup_only:
        logging.info("Setup complete. Exiting (--setup-only).")
        return

    if not args.config:
        parser.error("--config is required unless using --setup-only")

    # ----- Load config -----
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.info(f"Loaded config: {config_path}")
    logging.info(f"  Model: {config['model_name']}")
    logging.info(f"  Target: {config['target_phrase']}")
    logging.info(f"  Steps:  {config['steps']}")

    # ----- Step 1: Patch source -----
    logging.info("STEP 1: Patching openWakeWord source...")
    patch_data_py()
    patch_train_py()

    # ----- Step 2: Inject multi-TTS clips -----
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is None:
        # Auto-detect: look relative to this script
        script_dir = Path(__file__).resolve().parent
        candidate = script_dir.parent / "data"
        if candidate.exists():
            data_dir = candidate
        else:
            # Try Kaggle input
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                # Look for uploaded dataset
                for d in kaggle_input.iterdir():
                    if (d / "data").exists():
                        data_dir = d / "data"
                        break

    if data_dir and data_dir.exists():
        logging.info(f"STEP 2: Injecting multi-TTS clips from {data_dir}...")
        inject_multi_tts_clips(config, data_dir)
    else:
        logging.warning("STEP 2: No data directory found. Skipping multi-TTS injection.")
        logging.warning("  The model will only train on Piper-generated clips.")
        logging.warning("  For best results, provide --data-dir pointing to your data/ folder.")

    # ----- Step 3: Train -----
    logging.info("STEP 3: Starting training pipeline...")
    success = run_training(str(config_path), skip_generate=args.skip_generate)

    if success:
        output_dir = config["output_dir"]
        model_name = config["model_name"]
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        if os.path.exists(onnx_path):
            size_kb = os.path.getsize(onnx_path) / 1024
            logging.info(f"\nModel exported: {onnx_path}")
            logging.info(f"Model size: {size_kb:.1f} KB")
        else:
            logging.warning(f"Expected ONNX model not found at {onnx_path}")
    else:
        logging.error("Training failed. Check logs above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
