import os
import zipfile
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Configuration
ZIP_FILE_PATH = Path("data/downloads/ASVspoof2019_LA.zip")
EXTRACT_PATH = Path("data/downloads/extracted")
DATASET_PATH = Path("data/dataset")

def setup_directories():
    """Create necessary directories for train, dev, and eval splits"""
    if DATASET_PATH.exists():
        print(f"Cleaning up existing dataset directory: {DATASET_PATH}")
        shutil.rmtree(DATASET_PATH)
    
    subsets = ['train', 'dev', 'eval']
    categories = ['real', 'fake']
    
    for subset in subsets:
        for category in categories:
            path = DATASET_PATH / subset / category
            path.mkdir(parents=True, exist_ok=True)
            
    print(f"Created directory structure in: {DATASET_PATH}")

def extract_dataset():
    """Extract the zip file"""
    if not ZIP_FILE_PATH.exists():
        raise FileNotFoundError(f"Zip file not found: {ZIP_FILE_PATH}")
    
    print(f"Extracting {ZIP_FILE_PATH}...")
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        # Get total number of files for progress bar
        total_files = len(zip_ref.infolist())
        
        # Extract with progress
        for member in tqdm(zip_ref.infolist(), total=total_files, desc="Extracting"):
            zip_ref.extract(member, EXTRACT_PATH)
    print("Extraction complete.")

def organize_files():
    """Organize files into train/dev/eval directories based on protocols"""
    # ASVspoof 2019 LA structure usually has 'LA' as root inside zip
    base_dir = EXTRACT_PATH / "LA"
    
    if not base_dir.exists():
        print(f"Warning: Expected 'LA' directory not found in {EXTRACT_PATH}. Checking root...")
        base_dir = EXTRACT_PATH # Fallback
    
    # We will process train, dev, and eval sets
    subsets = ['train', 'dev', 'eval']
    
    processed_count = 0
    
    for subset in subsets:
        protocol_dir = base_dir / f"ASVspoof2019_LA_cm_protocols"
        audio_dir = base_dir / f"ASVspoof2019_LA_{subset}" / "flac"
        
        # Protocol file naming convention varies slightly
        protocol_file = protocol_dir / f"ASVspoof2019.LA.cm.{subset}.trn.txt"
        if not protocol_file.exists():
            # Try alternative name for eval
            protocol_file = protocol_dir / f"ASVspoof2019.LA.cm.{subset}.trl.txt"
            
        if not protocol_file.exists() or not audio_dir.exists():
            print(f"Skipping subset {subset}: Protocol or audio directory not found.")
            continue
            
        print(f"Processing {subset} set...")
        
        # Define destination paths for this subset
        subset_real_path = DATASET_PATH / subset / "real"
        subset_fake_path = DATASET_PATH / subset / "fake"
        
        # Read protocol
        # Format: SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID KEY
        # KEY is 'bonafide' (real) or 'spoof' (fake)
        try:
            # Try reading with 5 columns first
            df = pd.read_csv(protocol_file, sep=" ", header=None, names=["speaker", "filename", "system", "unused", "key"])
        except:
             # Fallback for eval file which might have different columns
            try:
                df = pd.read_csv(protocol_file, sep=" ", header=None, names=["speaker", "filename", "system", "key"])
            except:
                 print(f"Error reading protocol file: {protocol_file}")
                 continue
        
        # Iterate and move/copy files
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Organizing {subset}"):
            filename = row['filename'] + ".flac"
            src_file = audio_dir / filename
            
            if not src_file.exists():
                continue
                
            if row['key'] == 'bonafide':
                dest_file = subset_real_path / filename
            else:
                dest_file = subset_fake_path / filename
                
            # Copy file (using copy2 to preserve metadata)
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
                processed_count += 1
                
    print(f"Organization complete. Processed {processed_count} files.")

def main():
    print("="*50)
    print("ASVspoof 2019 Dataset Preparation")
    print("="*50)
    
    try:
        setup_directories()
        
        # Check if already extracted to save time
        if not (EXTRACT_PATH / "LA").exists():
            extract_dataset()
        else:
            print("Dataset appears to be already extracted. Skipping extraction.")
            
        organize_files()
        
        print("\nSUCCESS! Dataset is ready.")
        for subset in ['train', 'dev', 'eval']:
            real_count = len(list((DATASET_PATH / subset / "real").glob('*.flac')))
            fake_count = len(list((DATASET_PATH / subset / "fake").glob('*.flac')))
            print(f"{subset.upper()}: Real={real_count}, Fake={fake_count}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
