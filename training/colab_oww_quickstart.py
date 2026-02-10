#!/usr/bin/env python3
"""
WiseYak openWakeWord — Colab/Kaggle Quick Start
================================================
Run this single file in a Colab/Kaggle notebook cell to set up
and train an openWakeWord model.

In a Colab notebook:
    !python training/colab_oww_quickstart.py --model deepak
    !python training/colab_oww_quickstart.py --model deepa

In a Kaggle notebook (after uploading your data as a dataset):
    !python /kaggle/working/wise-activate/training/colab_oww_quickstart.py --model deepak
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run(cmd: str):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", 
                        choices=["hello_deepa", "namaste_deepa", "hello_namaste_deepa"],
                        default="hello_namaste_deepa",
                        help="Which model to train: hello_deepa (EN), namaste_deepa (NE), or hello_namaste_deepa (mixed, default)")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip environment setup (already done)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip Piper TTS generation (clips already exist)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training steps (default: from config)")
    args = parser.parse_args()

    # Locate project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Check if we found the right directory
    if not (project_root / "data").exists() and not (project_root / "training").exists():
        # Try current working directory
        project_root = Path.cwd()
        if not (project_root / "training").exists():
            print("ERROR: Cannot find project root. Run from the project directory.")
            sys.exit(1)

    config_file = project_root / "training" / "oww_configs" / f"{args.model}.yaml"
    train_script = project_root / "training" / "train_oww.py"

    if not config_file.exists():
        print(f"ERROR: Config not found: {config_file}")
        sys.exit(1)

    if not train_script.exists():
        print(f"ERROR: Training script not found: {train_script}")
        sys.exit(1)

    # Override steps if requested
    if args.steps:
        import yaml
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        config["steps"] = args.steps
        # Write to a temp config
        temp_config = Path("/tmp") / f"hey_{args.model}_custom.yaml"
        with open(temp_config, "w") as f:
            yaml.dump(config, f)
        config_file = temp_config
        print(f"Overriding steps to {args.steps}")

    # Build command
    cmd = f"{sys.executable} {train_script} --config {config_file} --data-dir {project_root / 'data'}"

    if args.skip_setup:
        cmd += " --skip-setup"
    if args.skip_generate:
        cmd += " --skip-generate"

    print("=" * 60)
    print(f"WiseYak openWakeWord Training — {args.model.upper()}")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Config: {config_file}")
    print(f"Data dir: {project_root / 'data'}")
    print()

    run(cmd)

    print("\n" + "=" * 60)
    print("DONE! Check the output directory for .onnx model file.")
    print("=" * 60)


if __name__ == "__main__":
    main()
