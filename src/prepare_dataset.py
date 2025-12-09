import os
import shutil
import pandas as pd
from pathlib import Path
import argparse

def setup_directories(base_dir):
    real_dir = os.path.join(base_dir, 'real')
    fake_dir = os.path.join(base_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    return real_dir, fake_dir

def process_protocol_file(protocol_path, audio_dir, target_base_dir):
    print(f"Processing protocol: {protocol_path}")
    
    # Read protocol file
    # Format: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
    # Example: LA_0079 LA_T_1138215 - - bonafide
    
    count_real = 0
    count_fake = 0
    missing = 0
    
    with open(protocol_path, 'r') as f:
        lines = f.readlines()
        
    real_dir, fake_dir = setup_directories(target_base_dir)
    
    total = len(lines)
    print(f"Found {total} entries. Starting organization...")
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        filename = parts[1]
        label = parts[4] # 'bonafide' or 'spoof'
        
        # Source file (ASVspoof 2019 uses .flac)
        src_file = os.path.join(audio_dir, filename + '.flac')
        
        if not os.path.exists(src_file):
            # Try .wav just in case
            src_file = os.path.join(audio_dir, filename + '.wav')
            if not os.path.exists(src_file):
                missing += 1
                continue
                
        if label == 'bonafide':
            dst_file = os.path.join(real_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)
            count_real += 1
        else:
            dst_file = os.path.join(fake_dir, os.path.basename(src_file))
            shutil.copy2(src_file, dst_file)
            count_fake += 1
            
        if i % 1000 == 0:
            print(f"Processed {i}/{total} files...")

    print(f"Done! Copied {count_real} real and {count_fake} fake files.")
    print(f"Missing files: {missing}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organize ASVspoof 2019 dataset')
    parser.add_argument('--protocol', required=True, help='Path to protocol file (e.g., ASVspoof2019.LA.cm.train.trn.txt)')
    parser.add_argument('--audio', required=True, help='Path to folder containing audio files (flac)')
    parser.add_argument('--output', default='data/dataset', help='Output directory')
    
    args = parser.parse_args()
    
    process_protocol_file(args.protocol, args.audio, args.output)
