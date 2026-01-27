import os
import shutil
from pathlib import Path

# Configuration
POSITIVE_DIR = Path("data/wake")
NEGATIVE_DIR = Path("data/background")
SOURCE_POSITIVE = Path("generated_greetings")
SOURCE_NEGATIVE = Path("negative_real")

def setup_directories():
    """Create destination directories if they don't exist."""
    if POSITIVE_DIR.exists():
        shutil.rmtree(POSITIVE_DIR)
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    if NEGATIVE_DIR.exists():
        shutil.rmtree(NEGATIVE_DIR)
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directories:\n- {POSITIVE_DIR}\n- {NEGATIVE_DIR}")

def copy_files(source: Path, destination: Path, file_pattern: str = "*", recursive: bool = False):
    """Copy files from source to destination."""
    if not source.exists():
        print(f"Warning: Source directory {source} does not exist.")
        return

    count = 0
    if recursive:
        files = source.rglob(file_pattern)
    else:
        files = source.glob(file_pattern)

    for file_path in files:
        if file_path.is_file():
            # Maintain structure if recursive?? 
            # User said "noises are all present inside negative_real folder... inside which as well we have sub folders"
            # For data/background, we usually flatten or keep structure? 
            # Let's keep structure relative to source for negative, flatten for positive if it's just files.
            
            # However, simpler approach for training usually expects flat or class-based. 
            # But 'data/background' implies a single class 'background'.
            # If I flat copy, names might collide.
            
            # Let's simple copy all files to destination.
            dest_file = destination / file_path.name
            
            # Handle potential name collisions
            if dest_file.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                dest_file = destination / f"{stem}_{count}{suffix}"
            
            shutil.copy2(file_path, dest_file)
            count += 1
            
    print(f"Copied {count} files from {source} to {destination}")

def main():
    print("Starting data organization...")
    setup_directories()
    
    # 1. Process Positive Samples
    print("\nProcessing Positive Samples...")
    # generated_greetings contains the wake word samples
    copy_files(SOURCE_POSITIVE, POSITIVE_DIR, "*.wav")
    
    # 2. Process Negative Samples
    print("\nProcessing Negative Samples...")
    # negative_real contains 'noise' and 'speech' subfolders
    # We want to pull all wav files from there into data/background
    # likely we want to recruit them recursively
    copy_files(SOURCE_NEGATIVE, NEGATIVE_DIR, "*.wav", recursive=True)

    print("\nData organization complete.")

if __name__ == "__main__":
    main()
