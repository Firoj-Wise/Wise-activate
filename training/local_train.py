#!/usr/bin/env python3
"""
WiseYak Local Training Orchestrator
===================================
Tailored for Office OEM Servers. 
Assumes data/ is already populated with high-diversity samples.

Usage:
  python training/local_train.py --config training/oww_configs/hello_namaste_deepa.yaml
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="WiseYak Local Training Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--skip-setup", action="store_true", help="Skip dependency installation")
    parser.add_argument("--setup-only", action="store_true", help="Only verify/install environment")
    parser.add_argument("--skip-generate", action="store_true", default=True, 
                        help="Skip Piper TTS generation (default True for local since we have 65k samples)")
    
    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).resolve().parent.parent
    train_script = base_dir / "training" / "train_oww.py"
    config_path = Path(args.config).resolve()

    if not train_script.exists():
        print(f"[ERROR] Could not find {train_script}")
        sys.exit(1)

    # Build command
    cmd = [sys.executable, str(train_script), "--config", str(config_path)]
    
    if args.skip_setup:
        cmd.append("--skip-setup")
    if args.setup_only:
        cmd.append("--setup-only")
    if args.skip_generate:
        cmd.append("--skip-generate")

    # Point to the local data directory
    data_dir = base_dir / "data"
    if data_dir.exists():
        cmd.extend(["--data-dir", str(data_dir)])

    print(f"\n[INFO] Starting Local Training for {config_path.name}...")
    print(f"[INFO] Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n[SUCCESS] Local training session completed.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
