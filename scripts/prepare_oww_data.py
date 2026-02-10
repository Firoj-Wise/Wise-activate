#!/usr/bin/env python3
"""
Prepare data directory structure for openWakeWord training on Kaggle/Colab.

This script:
  1. Scans the project's data/wake/ and data/background/ directories
  2. Resamples all audio to 16kHz mono WAV (openWakeWord requirement)
  3. Reports statistics per keyword/language/TTS engine
  4. Validates spectrogram diversity across TTS engines

Usage:
    python scripts/prepare_oww_data.py
    python scripts/prepare_oww_data.py --check-only   # stats only, no conversion
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WAKE_DIR = DATA_DIR / "wake"
BG_DIR   = DATA_DIR / "background"

TARGET_SR = 16000


def get_tts_engine(filename: str) -> str:
    """Infer TTS engine from filename prefix."""
    name = filename.lower()
    if name.startswith("edge_"):
        return "Edge-TTS"
    elif name.startswith("gcp_"):
        return "Google-TTS"
    elif name.startswith("piper_") or name.startswith("multitts_"):
        return "Piper-TTS"
    elif name.startswith("neg_"):
        return "Negative-Speech"
    elif name.startswith("noise_"):
        return "Synthetic-Noise"
    elif name.startswith("hardneg_"):
        return "Hard-Negative"
    else:
        return "Unknown"


def check_and_resample(filepath: Path, dry_run: bool = False) -> dict:
    """Check audio file and resample to 16kHz mono if needed.
    
    Returns dict with: sr, duration, channels, resampled, error
    """
    result = {
        "sr": None, "duration": None, "channels": None,
        "resampled": False, "error": None, "engine": get_tts_engine(filepath.name)
    }
    
    try:
        y, sr = librosa.load(str(filepath), sr=None, mono=False)
        
        # Handle multi-channel
        if y.ndim > 1:
            channels = y.shape[0]
            y_mono = librosa.to_mono(y)
        else:
            channels = 1
            y_mono = y
        
        result["sr"] = sr
        result["duration"] = len(y_mono) / sr
        result["channels"] = channels
        
        needs_resample = (sr != TARGET_SR) or (channels != 1)
        
        if needs_resample and not dry_run:
            # Resample to 16kHz mono
            y_resampled = librosa.resample(y_mono, orig_sr=sr, target_sr=TARGET_SR)
            
            # Save as WAV (convert mp3 -> wav if needed)
            out_path = filepath.with_suffix(".wav")
            sf.write(str(out_path), y_resampled, TARGET_SR)
            
            # Remove original if it was mp3
            if filepath.suffix.lower() == ".mp3" and out_path != filepath:
                filepath.unlink()
            
            result["resampled"] = True
        elif needs_resample:
            result["resampled"] = "needed"
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


def compute_spectral_centroid(filepath: Path) -> float:
    """Compute mean spectral centroid as a diversity fingerprint."""
    try:
        y, sr = librosa.load(str(filepath), sr=TARGET_SR, duration=2.0)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        return float(np.mean(centroid))
    except:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Prepare data for openWakeWord training")
    parser.add_argument("--check-only", action="store_true", help="Only report stats, no conversion")
    parser.add_argument("--diversity", action="store_true", help="Compute spectral diversity per engine")
    args = parser.parse_args()

    print("=" * 60)
    print("openWakeWord Data Preparation")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print()

    # --- Positive Data ---
    print("POSITIVE DATA (wake words):")
    print("-" * 40)
    
    stats = defaultdict(lambda: defaultdict(int))
    total_positive = 0
    resampled = 0
    errors = 0
    diversity_data = defaultdict(list)

    for keyword_dir in sorted(WAKE_DIR.iterdir()) if WAKE_DIR.exists() else []:
        if not keyword_dir.is_dir():
            continue
        for lang_dir in sorted(keyword_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            
            for audio_file in lang_dir.glob("*"):
                if audio_file.suffix.lower() not in [".wav", ".mp3"]:
                    continue
                
                result = check_and_resample(audio_file, dry_run=args.check_only)
                engine = result["engine"]
                key = f"{keyword_dir.name}/{lang_dir.name}"
                stats[key][engine] += 1
                total_positive += 1
                
                if result["resampled"]:
                    resampled += 1
                if result["error"]:
                    errors += 1
                
                if args.diversity:
                    centroid = compute_spectral_centroid(audio_file)
                    diversity_data[engine].append(centroid)

    for category, engines in sorted(stats.items()):
        total_cat = sum(engines.values())
        engine_str = ", ".join(f"{e}: {c}" for e, c in sorted(engines.items()))
        print(f"  {category:20s}  {total_cat:5d} files  ({engine_str})")

    print(f"\n  Total positive: {total_positive}")
    if resampled:
        print(f"  Resampled: {resampled}")
    if errors:
        print(f"  Errors: {errors}")

    # --- Negative Data ---
    print("\nNEGATIVE DATA (background / hard negatives):")
    print("-" * 40)
    
    neg_stats = defaultdict(int)
    total_negative = 0
    
    if BG_DIR.exists():
        for audio_file in BG_DIR.rglob("*"):
            if audio_file.suffix.lower() not in [".wav", ".mp3"]:
                continue
            
            result = check_and_resample(audio_file, dry_run=args.check_only)
            engine = result["engine"]
            neg_stats[engine] += 1
            total_negative += 1
            
            if result["resampled"]:
                resampled += 1
            
            if args.diversity and engine in ["Negative-Speech", "Hard-Negative"]:
                centroid = compute_spectral_centroid(audio_file)
                diversity_data[engine].append(centroid)

    for engine, count in sorted(neg_stats.items()):
        print(f"  {engine:20s}  {count:5d} files")
    print(f"\n  Total negative: {total_negative}")

    # --- Diversity Report ---
    if args.diversity and diversity_data:
        print("\nSPECTRAL DIVERSITY (mean spectral centroid per engine):")
        print("-" * 40)
        for engine, centroids in sorted(diversity_data.items()):
            if centroids:
                arr = np.array(centroids)
                print(f"  {engine:20s}  mean={arr.mean():.0f}Hz  std={arr.std():.0f}Hz  "
                      f"range=[{arr.min():.0f}, {arr.max():.0f}]")
        
        # Check if we have sufficient diversity
        engine_means = {e: np.mean(c) for e, c in diversity_data.items() if len(c) > 10}
        if len(engine_means) >= 2:
            spread = max(engine_means.values()) - min(engine_means.values())
            if spread > 200:
                print(f"\n  Diversity score: GOOD (spread={spread:.0f}Hz across engines)")
            elif spread > 100:
                print(f"\n  Diversity score: MODERATE (spread={spread:.0f}Hz)")
            else:
                print(f"\n  Diversity score: LOW (spread={spread:.0f}Hz) — consider adding more TTS sources")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_positive + total_negative} audio files")
    print(f"  Positive: {total_positive}")
    print(f"  Negative: {total_negative}")
    ratio = total_negative / max(1, total_positive)
    if ratio < 5:
        print(f"  Neg:Pos ratio: {ratio:.1f}:1 (WARN: aim for >=10:1 for robust rejection)")
    else:
        print(f"  Neg:Pos ratio: {ratio:.1f}:1")
    print("=" * 60)


if __name__ == "__main__":
    main()
