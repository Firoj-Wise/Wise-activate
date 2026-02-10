import shutil
import os
from pathlib import Path
from tqdm import tqdm

def zip_for_colab():
    base_dir = Path(__file__).resolve().parent.parent
    output_filename = base_dir / "wise_activate_colab.zip"
    
    # Folders to include 
    # (We exclude 'web' or 'node_modules' or '.venv' to save space)
    include_dirs = [
        "data", 
        "data_sequence", 
        "training",
        "scripts"
    ]
    
    # Create a temporary folder to organize the zip content
    temp_dir = base_dir / "temp_colab_pack"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    print(f"Packaging for Colab into: {output_filename}")
    
    try:
        total_size = 0
        file_count = 0
        
        for folder in include_dirs:
            src = base_dir / folder
            dst = temp_dir / folder
            
            if src.exists():
                print(f"  Copying {folder}...")
                shutil.copytree(src, dst)
                
                # Check size
                for root, _, files in os.walk(dst):
                    file_count += len(files)
                    for f in files:
                        total_size += os.path.getsize(os.path.join(root, f))
            else:
                print(f"  Warning: {folder} not found!")

        print(f"  Total Files: {file_count}")
        print(f"  Total Size: {total_size / (1024*1024):.2f} MB")
        
        print("  Zipping...")
        shutil.make_archive(str(output_filename).replace(".zip", ""), 'zip', temp_dir)
        
        print(f"Created {output_filename}")
        print("Upload this zip file to your Google Drive or Colab session.")
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    zip_for_colab()
